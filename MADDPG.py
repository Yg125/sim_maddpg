import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import StepLR
import os 
from Configure import seed
np.random.seed(seed)
def onehot_from_logits(logits, eps=0.01):
    ''' 生成最优动作的独热（one-hot）形式 '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作,转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]],
                                       requires_grad=False).to(logits.device)
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])

def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)
    return F.softmax(y / temperature, dim=1)

def gumbel_softmax(logits, temperature=1.0):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    # 返回一个y_hard的独热量,但是它的梯度是y,我们既能够得到一个与环境交互的离散动作,又可以
    # 正确地反传梯度
    return y

class TwoLayerFC(nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(num_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = x.to(torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim, actor_lr, critic_lr, device):
        self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)
        self.actor.apply(self.init_weight)
        self.target_actor = TwoLayerFC(state_dim, action_dim,hidden_dim).to(device)
        self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)
        self.critic.apply(self.init_weight)
        self.target_critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)
        
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_scheduler = StepLR(self.actor_optimizer, step_size=10, gamma=0.998)
        self.critic_scheduler = StepLR(self.critic_optimizer, step_size=10, gamma=0.998)
        
    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.01) # 这里可能可以调整初始化参数
            # nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, mean=1, std=0.02)
            
    def take_action(self, state, env, temperature, explore=False):
        action = self.actor(state)
        mask = env.get_masked_actions(state)
        action = action * mask
        if explore:
            action = gumbel_softmax(action, temperature)
        else:
            action = onehot_from_logits(action)
        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

class MADDPG:
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dims, critic_input_dim, gamma, tau):
        self.agents = []
        self.env = env
        for i in range(env.agents):
            self.agents.append(
                DDPG(state_dims[i], action_dims[i], critic_input_dim,
                     hidden_dim, actor_lr, critic_lr, device))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device
        self.temperature = 1.0

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    def take_action(self, states, explore):
        states = [torch.tensor(states[i], dtype=torch.float, device=self.device).unsqueeze(0) for i in range(self.env.agents)]
        return [agent.take_action(state, self.env, self.temperature, explore) for agent, state in zip(self.agents, states)]

    def update(self, sample, i_agent):
        obs, act, rew, next_obs, done = sample
        cur_agent = self.agents[i_agent]

        cur_agent.critic_optimizer.zero_grad()
        all_target_act = []
        for pi, _next_obs in zip(self.target_policies, next_obs):
            next_action = pi(_next_obs)
            mask = self.env.get_masked_actions(_next_obs)
            next_action = next_action * mask
            all_target_act.append(onehot_from_logits(next_action))
        
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        target_critic_value = rew[i_agent].view(-1, 1) + self.gamma * cur_agent.target_critic(
                target_critic_input) * (1 - done[i_agent].view(-1, 1))
        critic_input = torch.cat((*obs, *act), dim=1)
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value, target_critic_value.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(cur_agent.critic.parameters(),max_norm=10.0)
        cur_agent.critic_optimizer.step()
        # cur_agent.critic_scheduler.step()
        cur_agent.actor_optimizer.zero_grad()
        
        cur_actor_out = cur_agent.actor(obs[i_agent])
        mask_cur_actor = self.env.get_masked_actions(obs[i_agent])
        cur_actor_out = cur_actor_out * mask_cur_actor
        cur_act_vf_in = gumbel_softmax(cur_actor_out, self.temperature)
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(onehot_from_logits(pi(_obs)*self.env.get_masked_actions(_obs)))
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out**2).mean() * 1e-3
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(cur_agent.actor.parameters(),max_norm=10.0)
        cur_agent.actor_optimizer.step()
        # cur_agent.actor_scheduler.step()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), f'{path}/agent_{i}_actor.pth')
            torch.save(agent.critic.state_dict(), f'{path}/agent_{i}_critic.pth')
        
        
    def load(self, path):
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(torch.load(f'{path}/agent_{i}_actor.pth'))
            agent.critic.load_state_dict(torch.load(f'{path}/agent_{i}_critic.pth'))
            agent.target_actor.load_state_dict(agent.actor.state_dict())
            agent.target_critic.load_state_dict(agent.critic.state_dict())
