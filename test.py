from matplotlib import pyplot as plt
from Configure import DAG, Args, Task, B_u, B_aver, B_c, B_e, TIME_STAMP, eta_vio, eta, Lambda, Q, beta, capacity, ccr, seed
from collections import deque
import numpy as np
from Env import ScheduleEnv, Server, Remote_cloud, server_capacity, comp, task_type, request_dict, interval_dict
import torch
import rl_utils
from MADDPG import MADDPG

class OnDoc_plus:
    def __init__(self, args):
        self.B_u = B_u
        self.B_aver = B_aver
        self.B_c = B_c
        self.B_e = B_e
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.TIME_STAMP = TIME_STAMP    
        self.eta_vio = eta_vio
        self.eta = eta
        self.servers = [Server(i, comp[i], server_capacity[i]) for i in range(self.n_agents)]
        self.cloud = Remote_cloud(self.n_agents, 7000) 
        self.Q = Q
        self.configuring_time = 30 # ms
        self.queues = [0 for _ in range(self.Q)]
        self.num_processors = self.n_agents + 1
        self.dags = [0 for _ in range(self.Q)]
        self.arrive_list = [0 for _ in range(self.Q)]
        self.ready_tasks = []
        self.virtual_time = 0.0        # 定义在线场景的虚拟时间，避免受机器运行时间干扰
        self.complete_task = []        # COFE算法中记录每个子任务的完成时间
        self.args = args
        self.graph = [0 for _ in range(self.Q)]
        self.comp_cost = [0 for _ in range(self.Q)]
        self.tasks = [0 for _ in range(self.Q)]
        self.processors = self.servers + [self.cloud]     # servers编号为0-4, cloud编号为5
        self.device = args.device
        self.request_list = request_dict[self.n_agents]
        self.interval_list = interval_dict[Lambda]
        self.task_type = task_type
    
    def receive_dag(self):                    # k is the index of the request/queue from 0
        virtual_time = 0
        tasks = [0 for _ in range(self.Q)]
        DAGS = np.load(f'./dag_infos/dag_info_random_ccr{ccr}.npy', allow_pickle=True)  # Read DAG from file
        type_count = 0
        for k in range(self.Q):
            self.dags[k] = DAG(k)
            self.dags[k].num_tasks, self.comp_cost[k], self.graph[k], deadline_heft = DAGS[k]  
            self.comp_cost[k] = np.hstack((self.comp_cost[k][:,:self.n_agents],self.comp_cost[k][:,-1].reshape(-1,1))) 
            self.dags[k].deadline = virtual_time + deadline_heft * beta   # 以heft算法的deadline为基准，增加15%的时间作为在线场景DAG的deadline
            self.dags[k].r = virtual_time    # ms
            self.arrive_list[k] = virtual_time
            num_tasks = self.dags[k].num_tasks
            tasks[k] = [Task(i,k,self.task_type[type_count+i]) for i in range(num_tasks)]
            type_count += num_tasks
            data_in = 0
            for j in range(self.dags[k].num_tasks):
                tasks[k][j].avg_comp = sum(self.comp_cost[k][j]) / self.num_processors
                if self.graph[k][0][j] != -1:
                    data_in += self.graph[k][0][j]
                else:
                    data_in += 0
            self.dags[k].t_offload = round(data_in * self.B_u / 10**6, 1)  # 任务由用户发送到服务器需要的offload时间
            interval = self.interval_list[k] * 1000
            virtual_time += interval
        self.tasks = tasks
      
    def advance_virtual_time(self, duration):
        if duration < 0:
            print('error1')
        self.virtual_time += duration
            
    def computeRank(self, task, k, computed_ranks):
        if task in computed_ranks:
            return computed_ranks[task]
        curr_rank = 0
        for succ in self.tasks[k]:
            if self.graph[k][task.id][succ.id] != -1:
                if succ.rank is None:
                    self.computeRank(succ, k, computed_ranks)
                curr_rank = max(curr_rank, round(self.graph[k][task.id][succ.id]*self.B_aver/10**6, 1) + succ.rank)
        task.rank = task.avg_comp + curr_rank
        computed_ranks[task] = task.rank
    
    def LT(self, k):
        for t in self.tasks[k]:
            t.lt = self.dags[k].r + (self.dags[k].deadline - self.dags[k].r) * (self.tasks[k][0].rank - t.rank) / self.tasks[k][0].rank
   
    def arrive(self, k):
        tasks = self.tasks[k]
        computed_ranks = {}
        self.computeRank(tasks[0], k, computed_ranks)
        self.LT(k)
        try:
            self.queues[k] = deque(tasks)
        except:
            print(1)
        self.ready_tasks.append(self.queues[k][0])

    def find_ready_tasks(self, t):
        successors = [succ for succ in self.queues[t.k] if self.graph[t.k][t.id][succ.id] != -1]          # 得到completed任务的直接后继节点
        for succ in successors:
            found_pre = False
            for pre in self.tasks[t.k]:
                if self.graph[t.k][pre.id][succ.id] != -1:
                    if self.virtual_time < pre.end:
                        found_pre = True
                        break
                    if self.virtual_time == pre.end and pre in self.complete_task:
                        found_pre = True
                        break
            if not found_pre:
                self.ready_tasks.append(succ)
        
    def check_ready(self):
        tar_est = float('inf')
        tar_vm = None 
        vm = None
        task = None 
        tar_p = None 
        if not self.ready_tasks:
            print('error')
        for t in self.ready_tasks:
            # p, est = self.get_tar(t)
            result = self.get_tar(t)
            if isinstance(result, tuple) and len(result) == 3:
                p, est, vm = result
            else:
                p, est = result
            if est < tar_est:
                tar_p = p
                tar_est = est
                task = t
                tar_vm = vm
        if tar_p == None:
            print('erorr')
        return task, tar_p, tar_est, tar_vm
       
    def schedule_task(self, task, p, est, vm=None):
        task.processor_id = p
        task.start = est
        task.end = task.start + self.comp_cost[task.k][task.id][p]
        try:
            self.queues[task.k].remove(task)
        except:
            print('DAG:{} Task {} is not in queue'.format(task.k,task.id))  
        if p in range(self.n_agents):
            self.processors[p].task_list.append(task)
        else:
            self.cloud.vms[vm].task_list.append(task)
        if task.id != self.dags[task.k].num_tasks - 1:
            self.complete_task.append(task)

        self.complete_task.sort(key=lambda x: x.end)
        try:
            self.ready_tasks.remove(task)
        except:
            print('DAG:{} Task {} is not in ready_tasks'.format(task.k,task.id))
   
    def get_est(self, t, p, k): 
        if (p.id in range(self.n_agents) and not p.service_list[t.service_id]) and (t.id != 0 and t.id != self.dags[t.k].num_tasks - 1):
            return float('inf')
        if p.id in range(self.n_agents):
            est = max(self.dags[k].r + self.dags[k].t_offload, self.virtual_time, (self.virtual_time//self.TIME_STAMP)*self.TIME_STAMP + p.service_migrate[t.service_id]*self.configuring_time)    # 初始化est时间为任务到达时间和offload时间之和
        else:
            est = max(self.dags[k].r + self.dags[k].t_offload, self.virtual_time)
        graph = self.graph[k]
        tasks = self.tasks[k]
        
        for pre in tasks:
           if graph[pre.id][t.id] != -1:  # if pre also done on p, no communication cost
                c = graph[pre.id][t.id] if pre.processor_id != p.id else 0
                if pre.processor_id in range(self.n_agents) and p.id in range(self.n_agents): 
                    est = max(est, pre.end + round(c*self.B_e/10**6, 1))  # ms
                else:
                    est = max(est, pre.end + round(c*self.B_c/10**6, 1))
        if p.id in range(self.n_agents) and not p.task_list:  # 在之前没有任务则直接返回任务依赖的EST
            return est
        elif p.id == self.n_agents:
            est_cloud = float('inf')
            vm_i = None 
            for i, vm in enumerate(self.cloud.vms):
                if not vm.task_list:
                    return (est, i)
                else:
                    if est_cloud > vm.task_list[-1].end:
                        est_cloud = vm.task_list[-1].end
                        vm_i = i
            return (max(est, est_cloud), vm_i)
        else:
            avail = p.task_list[-1].end # 否则需要返回当前processor任务list里最后一个任务的完成时间
            return max(est, avail)
    
    def get_tar(self, t):
        # input: object t & int k
        # return target processor's id and EST of target processor
        if t.id == 0: 
            tar_p = self.request_list[t.k]         # 随机从某个边缘服务器发出请求
            tar_est = self.get_est(t, self.processors[tar_p], t.k)
            return (tar_p, tar_est)
        
        elif t.id == self.tasks[t.k][-1].id:
            tar_p = self.tasks[t.k][0].processor_id
            tar_est = self.get_est(t, self.processors[tar_p], t.k)
            return (tar_p, tar_est)
        else:
            aft = float("inf")
            for processor in self.processors:
                result = self.get_est(t, processor, t.k)
                if isinstance(result, tuple):
                    est, vm_i = result
                else:
                    est = result
                eft = est + self.comp_cost[t.k][t.id][processor.id]
                if eft < aft:   # found better case of processor
                    aft = eft
                    tar_p = processor.id
                    tar_est = est
            if tar_p == self.n_agents:
                return (tar_p, tar_est, vm_i)
            else:
                return (tar_p, tar_est)

    def str(self):
        satisfy = 0
        task_e = 0
        task_c = 0
        for p in self.processors[0: self.n_agents]:
            task_e += len(p.task_list)
        for vm in self.processors[self.n_agents].vms:
            task_c += len(vm.task_list)
            # for t in vm.task_list:
                # print('id: {}, k: {}'.format(t.id,t.k))
        for k in range(self.Q):
            self.Makespan = max([t.end for t in self.tasks[k]]) - self.dags[k].r
            if self.Makespan < self.dags[k].deadline - self.dags[k].r:
                satisfy += 1
        # print("E = {}, C = {}".format(task_e, task_c))
        SR = satisfy / self.Q * 100
        return SR

def stack_array(x):
    # 将原本按“列”排列的元素改为按“行”排列 并转换为tensor
    rearranged = []
    for i in range(len(x[0])):
        rearranged.append([sub_x[i] for sub_x in x])
    # rearranged = [[sub_x[i] for sub_x in x] for i in range(len(x[0]))]
    return [torch.FloatTensor(np.vstack(aa)).to(device) for aa in rearranged]

num_episodes = 10000
buffer_size = 1000000
hidden_dim = 256
actor_lr = 1e-3
critic_lr = 1e-2
gamma = 0.95
tau = 1e-2
batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
update_interval = 100

env = ScheduleEnv()
args = Args()
args.set_env_info(env.get_info())

state_dims = []
action_dims = []
for action_space in range(env.agents):
    action_dims.append(env.n_actions)
for state_space in range(env.agents):
    state_dims.append(13)
critic_input_dim = sum(state_dims) + sum(action_dims)
               
torch.manual_seed(0)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim, state_dims, action_dims, critic_input_dim, gamma, tau)
ondoc_plus = OnDoc_plus(args)

def evaluate(maddpg, env, ondoc_plus):
    state = env.reset()
    ondoc_plus.receive_dag()
    for i in range(env.agents):      
        ondoc_plus.processors[i].task_list = []
        ondoc_plus.processors[i].service_list = [0,0,0,0,0]
        ondoc_plus.processors[i].service_migrate = [0,0,0,0,0]
    for vm in ondoc_plus.processors[env.agents].vms:
        vm.task_list = []
    arrive_len = len(ondoc_plus.arrive_list)
    k = 0
    ondoc_plus.virtual_time = 0.0
    ondoc_plus.arrive(k)
    k += 1
    step = -1 
    episode_reward = 0
    while (any(queue for queue in ondoc_plus.queues) or k < arrive_len):  # 调度还没有结束
        flag_1 ,flag_2= False, False
        if ondoc_plus.virtual_time % ondoc_plus.TIME_STAMP == 0:  # 每个时间戳执行一次动作
            actions = maddpg.take_action(state, explore=False)
            step += 1
            next_state, reward, done = env.step(actions)
            for i in range(env.agents):
                ondoc_plus.servers[i].service_list = next_state[i][0:5]
                ondoc_plus.servers[i].service_migrate = [1 if (y_1 == 1 and y_2 == 0) else 0 for y_1, y_2 in zip(next_state[i][0:5], state[i][0:5])]
            episode_reward += sum(reward)
            TIME_NEXT = ondoc_plus.TIME_STAMP * (step + 1)
        
        for task in ondoc_plus.complete_task:
            if task.processor_id in range(env.agents) and task.end > TIME_NEXT:
                next_state[task.processor_id][5+task.service_id] = 1
        
        while (not ondoc_plus.ready_tasks and ondoc_plus.complete_task) or (not ondoc_plus.ready_tasks and k < arrive_len):
            if not ondoc_plus.complete_task:
                if ondoc_plus.arrive_list[k] >= TIME_NEXT:
                    flag_1 = True
                    break
                ondoc_plus.advance_virtual_time(ondoc_plus.arrive_list[k] - ondoc_plus.virtual_time)
                ondoc_plus.arrive(k)
                k += 1
            elif k >= arrive_len:
                if ondoc_plus.complete_task[0].end >= TIME_NEXT:
                    flag_1 = True
                    break
                completed_task = ondoc_plus.complete_task.pop(0)
                ondoc_plus.advance_virtual_time(completed_task.end - ondoc_plus.virtual_time)
                ondoc_plus.find_ready_tasks(completed_task)
            elif ondoc_plus.arrive_list[k] <= ondoc_plus.complete_task[0].end:
                if ondoc_plus.arrive_list[k] >= TIME_NEXT:
                    flag_1 = True
                    break
                ondoc_plus.advance_virtual_time(ondoc_plus.arrive_list[k] - ondoc_plus.virtual_time)
                ondoc_plus.arrive(k)
                k += 1
            else:
                if ondoc_plus.complete_task[0].end >= TIME_NEXT:
                    flag_1 = True
                    break
                completed_task = ondoc_plus.complete_task.pop(0)
                ondoc_plus.advance_virtual_time(completed_task.end - ondoc_plus.virtual_time)
                ondoc_plus.find_ready_tasks(completed_task)
        if flag_1:
            ondoc_plus.advance_virtual_time(TIME_NEXT - ondoc_plus.virtual_time)
            state = next_state
            continue    
        if any(queue for queue in ondoc_plus.queues):
            task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()    # 每当有DAG加入或者任务完成调度之后都要调用
            
        while (len(ondoc_plus.complete_task) != 0 and ondoc_plus.complete_task[0].end <= tar_est) or (k < arrive_len and ondoc_plus.arrive_list[k] <= tar_est):
            if k < arrive_len:
                if len(ondoc_plus.complete_task) != 0:
                    if ondoc_plus.arrive_list[k] <= ondoc_plus.complete_task[0].end:
                        if ondoc_plus.arrive_list[k] >= TIME_NEXT:
                            flag_2 = True
                            break
                        ondoc_plus.advance_virtual_time(ondoc_plus.arrive_list[k] - ondoc_plus.virtual_time)
                        ondoc_plus.arrive(k)
                        k += 1
                        task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()
                    else:
                        if ondoc_plus.complete_task[0].end >= TIME_NEXT:
                            flag_2 = True
                            break
                        completed_task = ondoc_plus.complete_task.pop(0)
                        ondoc_plus.advance_virtual_time(completed_task.end - ondoc_plus.virtual_time)
                        ondoc_plus.find_ready_tasks(completed_task)
                        task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()
                else:
                    if ondoc_plus.arrive_list[k] >= TIME_NEXT:
                        flag_2 = True
                        break
                    ondoc_plus.advance_virtual_time(ondoc_plus.arrive_list[k] - ondoc_plus.virtual_time)
                    ondoc_plus.arrive(k)
                    k += 1
                    task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()
            else:
                if ondoc_plus.complete_task[0].end >= TIME_NEXT:
                    flag_2 = True
                    break
                completed_task = ondoc_plus.complete_task.pop(0)
                ondoc_plus.advance_virtual_time(completed_task.end - ondoc_plus.virtual_time)
                ondoc_plus.find_ready_tasks(completed_task)
                task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()
                
        if flag_2 or tar_est >= TIME_NEXT:
            ondoc_plus.advance_virtual_time(TIME_NEXT - ondoc_plus.virtual_time)
            state = next_state
            continue
        
        ondoc_plus.advance_virtual_time(tar_est - ondoc_plus.virtual_time)
        ondoc_plus.schedule_task(task, tar_p, tar_est, tar_vm)
        
        if task.processor_id in range(env.agents) and task.end >= TIME_NEXT:  # 判断调度之后是否会影响obs,state
            next_state[task.processor_id][5+task.service_id] = 1
        if task.processor_id in range(env.agents):
            next_state[task.processor_id][11] = task.end // ondoc_plus.TIME_STAMP
        # if task.id in range(7):
        #     if task.lt >= tar_est:
        #         episode_reward += 1
        #     else:
        #         episode_reward -= 1
        if task.id == ondoc_plus.dags[task.k].num_tasks - 1:
            if task.end > ondoc_plus.dags[task.k].deadline:
                episode_reward += -ondoc_plus.eta_vio
            # else:
                # episode_reward += ondoc_plus.eta

    while ondoc_plus.complete_task:     # 已经调度完成但是还需要等待执行
        if ondoc_plus.virtual_time % ondoc_plus.TIME_STAMP == 0:  # 每个时间戳执行一次动作
            actions = maddpg.take_action(state, explore=False)
            step += 1
            next_state, reward, done = env.step(actions)
            for i in range(env.agents):
                ondoc_plus.servers[i].service_list = next_state[i][0:5]
                ondoc_plus.servers[i].service_migrate = [1 if (y_1 == 1 and y_2 == 0) else 0 for y_1, y_2 in zip(next_state[i][0:5], state[i][0:5])]
            episode_reward += sum(reward)
            TIME_NEXT = ondoc_plus.TIME_STAMP * (step + 1)
        for task in ondoc_plus.complete_task:
            if task.processor_id in range(env.agents) and task.end > TIME_NEXT:
                next_state[task.processor_id][5+task.service_id] = 1
        if ondoc_plus.complete_task[0].end >= TIME_NEXT:
            ondoc_plus.advance_virtual_time(TIME_NEXT - ondoc_plus.virtual_time)
            state = next_state
            continue
        completed_task = ondoc_plus.complete_task.pop(0)
        ondoc_plus.advance_virtual_time(completed_task.end - ondoc_plus.virtual_time)
    done = [True for i in range(env.agents)]
    return episode_reward

total_step = 0
minimal_size = 4000
return_list = np.zeros(num_episodes)
i_episode = 0
while i_episode < num_episodes:
    # print(total_step)
    ondoc_plus.virtual_time = 0.0
    state = env.reset()
    ondoc_plus.receive_dag()
    for i in range(env.agents):        # 由于self.processors信息在environment.py中定义，所以这里需要重新初始化
        ondoc_plus.processors[i].task_list = []
        ondoc_plus.processors[i].service_list = [0,0,0,0,0]
        ondoc_plus.processors[i].service_migrate = [0,0,0,0,0]  
    for vm in ondoc_plus.processors[env.agents].vms:
        vm.task_list = []
    arrive_len = len(ondoc_plus.arrive_list)
    k = 0
    ondoc_plus.arrive(k)
    k += 1
    step = -1 
    next_state = None
    
    if replay_buffer.size() >= minimal_size:
        sample = replay_buffer.sample(batch_size)
        sample = [stack_array(x) for x in sample]
        for a_i in range(env.agents):
            maddpg.update(sample, a_i)
        maddpg.update_all_targets()
        
    while (any(queue for queue in ondoc_plus.queues) or k < arrive_len):  # 调度还没有结束
        flag_1 ,flag_2= False, False
        if ondoc_plus.virtual_time % ondoc_plus.TIME_STAMP == 0:  # 每个时间戳执行一次动作
            if next_state is not None:
                replay_buffer.add(state, actions, reward, next_state, done)  # 这里存储的是上一次动作之后state经过完整时隙后的信息               
                state = next_state
            actions = maddpg.take_action(state, explore=True)
            step += 1
            next_state, reward, done = env.step(actions)
            for i in range(env.agents):
                ondoc_plus.servers[i].service_list = next_state[i][0:5]
                ondoc_plus.servers[i].service_migrate = [1 if (y_1 == 1 and y_2 == 0) else 0 for y_1, y_2 in zip(next_state[i][0:5], state[i][0:5])]
            total_step += 1
            TIME_NEXT = ondoc_plus.TIME_STAMP * (step + 1)
        
        for task in ondoc_plus.complete_task:
            if task.processor_id in range(env.agents) and task.end > TIME_NEXT:
                next_state[task.processor_id][5+task.service_id] = 1
        
        while (not ondoc_plus.ready_tasks and ondoc_plus.complete_task) or (not ondoc_plus.ready_tasks and k < arrive_len):
            if not ondoc_plus.complete_task:
                if ondoc_plus.arrive_list[k] >= TIME_NEXT:
                    flag_1 = True
                    break
                ondoc_plus.advance_virtual_time(ondoc_plus.arrive_list[k] - ondoc_plus.virtual_time)
                ondoc_plus.arrive(k)
                k += 1
            elif k >= arrive_len:
                if ondoc_plus.complete_task[0].end >= TIME_NEXT:
                    flag_1 = True
                    break
                completed_task = ondoc_plus.complete_task.pop(0)
                ondoc_plus.advance_virtual_time(completed_task.end - ondoc_plus.virtual_time)
                ondoc_plus.find_ready_tasks(completed_task)
            elif ondoc_plus.arrive_list[k] <= ondoc_plus.complete_task[0].end:
                if ondoc_plus.arrive_list[k] >= TIME_NEXT:
                    flag_1 = True
                    break
                ondoc_plus.advance_virtual_time(ondoc_plus.arrive_list[k] - ondoc_plus.virtual_time)
                ondoc_plus.arrive(k)
                k += 1
            else:
                if ondoc_plus.complete_task[0].end >= TIME_NEXT:
                    flag_1 = True
                    break
                completed_task = ondoc_plus.complete_task.pop(0)
                ondoc_plus.advance_virtual_time(completed_task.end - ondoc_plus.virtual_time)
                ondoc_plus.find_ready_tasks(completed_task)
                
        if flag_1:
            ondoc_plus.advance_virtual_time(TIME_NEXT - ondoc_plus.virtual_time)
            continue    
        if any(queue for queue in ondoc_plus.queues):
            task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()    # 每当有DAG加入或者任务完成调度之后都要调用
        
            
        while (len(ondoc_plus.complete_task) != 0 and ondoc_plus.complete_task[0].end <= tar_est) or (k < arrive_len and ondoc_plus.arrive_list[k] <= tar_est):
            if k < arrive_len:
                if len(ondoc_plus.complete_task) != 0:
                    if ondoc_plus.arrive_list[k] <= ondoc_plus.complete_task[0].end:
                        if ondoc_plus.arrive_list[k] >= TIME_NEXT:
                            flag_2 = True
                            break
                        ondoc_plus.advance_virtual_time(ondoc_plus.arrive_list[k] - ondoc_plus.virtual_time)
                        ondoc_plus.arrive(k)
                        k += 1
                        task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()
                    else:
                        if ondoc_plus.complete_task[0].end >= TIME_NEXT:
                            flag_2 = True
                            break
                        completed_task = ondoc_plus.complete_task.pop(0)
                        ondoc_plus.advance_virtual_time(completed_task.end - ondoc_plus.virtual_time)
                        ondoc_plus.find_ready_tasks(completed_task)
                        task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()
                else:
                    if ondoc_plus.arrive_list[k] >= TIME_NEXT:
                        flag_2 = True
                        break
                    ondoc_plus.advance_virtual_time(ondoc_plus.arrive_list[k] - ondoc_plus.virtual_time)
                    ondoc_plus.arrive(k)
                    k += 1
                    task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()
            else:
                if ondoc_plus.complete_task[0].end >= TIME_NEXT:
                    flag_2 = True
                    break
                completed_task = ondoc_plus.complete_task.pop(0)
                ondoc_plus.advance_virtual_time(completed_task.end - ondoc_plus.virtual_time)
                ondoc_plus.find_ready_tasks(completed_task)
                task, tar_p, tar_est, tar_vm = ondoc_plus.check_ready()
                
        if flag_2 or tar_est >= TIME_NEXT:
            ondoc_plus.advance_virtual_time(TIME_NEXT - ondoc_plus.virtual_time)
            continue
        
        ondoc_plus.advance_virtual_time(tar_est - ondoc_plus.virtual_time)
        ondoc_plus.schedule_task(task, tar_p, tar_est, tar_vm)
        
        if task.processor_id in range(env.agents) and task.end >= TIME_NEXT:  # 判断调度之后是否会影响obs,state
            next_state[task.processor_id][5+task.service_id] = 1
        if task.processor_id in range(env.agents):
            next_state[task.processor_id][11] = task.end // ondoc_plus.TIME_STAMP
        #     if task.id in range(ondoc_plus.dags[task.k].num_tasks - 1):
        #         if task.lt >= tar_est:
        #             reward[task.processor_id] += 1
        #         else:
        #             reward[task.processor_id] -= 1
        # else:
        if task.id in range(ondoc_plus.dags[task.k].num_tasks - 1):
            if task.lt >= tar_est:
                for i in range(env.agents):
                    reward[i] += 1/env.agents
            else:
                for i in range(env.agents):
                    reward[i] += -1/env.agents
        if task.id == ondoc_plus.dags[task.k].num_tasks - 1:
            if task.end > ondoc_plus.dags[task.k].deadline:
                for i in range(env.agents):
                   # reward[i] += -1/env.agents
                    reward[i] += -ondoc_plus.eta_vio * 1/env.agents
            else:
                for i in range(env.agents):
                    reward[i] += ondoc_plus.eta * 1/env.agents

    while ondoc_plus.complete_task:     # 已经调度完成但是还需要等待执行
        if ondoc_plus.virtual_time % ondoc_plus.TIME_STAMP == 0:  # 每个时间戳执行一次动作
            actions = maddpg.take_action(state, explore=True)
            step += 1
            next_state, reward, done = env.step(actions)
            for i in range(env.agents):
                ondoc_plus.servers[i].service_list = next_state[i][0:5]
                ondoc_plus.servers[i].service_migrate = [1 if (y_1 == 1 and y_2 == 0) else 0 for y_1, y_2 in zip(next_state[i][0:5], state[i][0:5])]
            total_step += 1
            TIME_NEXT = ondoc_plus.TIME_STAMP * (step + 1)
        for task in ondoc_plus.complete_task:
            if task.processor_id in range(env.agents) and task.end > TIME_NEXT:
                next_state[task.processor_id][5+task.service_id] = 1
                
        if ondoc_plus.complete_task[0].end >= TIME_NEXT:
            ondoc_plus.advance_virtual_time(TIME_NEXT - ondoc_plus.virtual_time)
            replay_buffer.add(state, actions, reward, next_state, done)
            state = next_state
            continue
        completed_task = ondoc_plus.complete_task.pop(0)
        ondoc_plus.advance_virtual_time(completed_task.end - ondoc_plus.virtual_time)

    done = [True for i in range(env.agents)]
    replay_buffer.add(state, actions, reward, next_state, done)
    ondoc_plus.str()
    maddpg.temperature = max(maddpg.temperature * 0.9995, 0.1)
    # print(step)
    returns = evaluate(maddpg, env, ondoc_plus)
    return_list[i_episode] = returns
    i_episode += 1
    if (i_episode+1) % 10 == 0:
        print(f"Episode:{i_episode + 1},{returns} SR:{ondoc_plus.str()}%, critic_lr:{maddpg.agents[0].critic_optimizer.param_groups[0]['lr']}, tem:{maddpg.temperature}")
#        for name,param in maddpg.agents[0].critic.named_parameters():
#            print(f"Parameter:{name}, Gradient:{param.grad}") 
np.save(f'beta-{beta}-{seed}.npy', return_list)
# np.save(f'capacity-{capacity}-6.npy', return_list)
# np.save(f'Lambda-{Lambda}-69.npy', return_list)
# np.save(f'Es_{ondoc_plus.n_agents}-{seed}.npy', return_list)
# maddpg.save(f'./models-Lambda_{ondoc_plus.n_agents}')
# np.save(f'AC-{actor_lr}_CR-{critic_lr}_hidden-{hidden_dim}-{seed}.npy', return_list)
# maddpg.save(f'./models_default_AC-{actor_lr}_CR-{critic_lr}_hidden-{hidden_dim}-r')
