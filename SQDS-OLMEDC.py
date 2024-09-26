import copy
import itertools
import random
from Configure import DAG, Task, B_u, B_aver, B_c, B_e, Lambda, Q, beta, NUM_AGENTS, capacity, TIME_STAMP, ccr
from Env import Server, Remote_cloud, server_capacity, comp, request_dict, interval_dict, task_type
from collections import deque
import numpy as np
import multiprocessing
from multiprocessing import Pool
configuring_time = 30  # ms
random.seed(1)

class SQDS_OLMEDC:
    def __init__(self):
        self.B_u = B_u
        self.B_aver = B_aver
        self.B_c = B_c
        self.B_e = B_e
        self.Q = Q
        self.n_agents = NUM_AGENTS
        self.TIME_STAMP = TIME_STAMP  
        self.servers = [Server(i, comp[i], server_capacity[i]) for i in range(self.n_agents)]
        self.cloud = Remote_cloud(self.n_agents, 7000) 
        self.queues = [0 for _ in range(self.Q)]
        self.num_processors = self.n_agents + 1
        self.dags = [0 for _ in range(self.Q)]
        self.arrive_list = [0 for _ in range(self.Q)]
        self.virtual_time = 0.0 
        self.processors = self.servers + [self.cloud]
        self.graph = [0 for _ in range(self.Q)]
        self.comp_cost = [0 for _ in range(self.Q)]
        self.tasks = [0 for _ in range(self.Q)]
        self.request_list = request_dict[self.n_agents]     # Env中的request_list 避免调用全局变量
        self.interval_list = interval_dict[Lambda]
        self.task_type = task_type
        self.ready_tasks = []
        self.complete_task = []
        self.avail_action = [[[0, 0, 0, 0, 1],[0, 0, 0, 1, 0],[0, 0, 1, 0, 0],[0, 1, 0, 0, 0],[1, 0, 0, 0, 0]],
                        [[0, 0, 0, 1, 1],[0, 0, 1, 0, 1],[0, 0, 1, 1, 0],[0, 1, 0, 0, 1],[0, 1, 0, 1, 0],[0, 1, 1, 0, 0],[1, 0, 0, 0, 1],[1, 0, 0, 1, 0],[1, 0, 1, 0, 0],[1, 1, 0, 0, 0]],
                        [[0, 0, 1, 1, 1],[0, 1, 0, 1, 1],[0, 1, 1, 0, 1],[0, 1, 1, 1, 0],[1, 0, 0, 1, 1],[1, 0, 1, 0, 1],[1, 0, 1, 1, 0],[1, 1, 0, 0, 1],[1, 1, 0, 1, 0],[1, 1, 1, 0, 0]],
                        [[0, 1, 1, 1, 1],[1, 0, 1, 1, 1],[1, 1, 0, 1, 1],[1, 1, 1, 0, 1],[1, 1, 1, 1, 0]]]
    
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
    
    def policy(self, k, action):
        r = [0 for i in range(self.n_agents)]
        for i in range(self.n_agents):
            self.processors[i].service_list = action[i]
        TIME_NEXT = self.virtual_time + self.TIME_STAMP
        arrive_len = len(self.arrive_list)
        k = k
        flag_1 ,flag_2= False, False
        while (any(queue for queue in self.queues) or k < arrive_len): 
            while (not self.ready_tasks and self.complete_task) or (not self.ready_tasks and k < arrive_len):
                if not self.complete_task:
                    if self.arrive_list[k] >= TIME_NEXT:
                        flag_1 = True
                        break
                    self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                    self.arrive(k)
                    k += 1
                elif k >= arrive_len:
                    if self.complete_task[0].end >= TIME_NEXT:
                        flag_1 = True
                        break
                    completed_task = self.complete_task.pop(0)
                    self.advance_virtual_time(completed_task.end - self.virtual_time)
                    self.find_ready_tasks(completed_task)
                elif self.arrive_list[k] <= self.complete_task[0].end:
                    if self.arrive_list[k] >= TIME_NEXT:
                        flag_1 = True
                        break
                    self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                    self.arrive(k)
                    k += 1
                else:
                    if self.complete_task[0].end >= TIME_NEXT:
                        flag_1 = True
                        break
                    completed_task = self.complete_task.pop(0)
                    self.advance_virtual_time(completed_task.end - self.virtual_time)
                    self.find_ready_tasks(completed_task)
            if flag_1:
                self.advance_virtual_time(TIME_NEXT - self.virtual_time)
                break    
            if any(queue for queue in self.queues):
                task, tar_p, tar_est, tar_vm = self.check_ready()    # 每当有DAG加入或者任务完成调度之后都要调用
            
            while (len(self.complete_task) != 0 and self.complete_task[0].end <= tar_est) or (k < arrive_len and self.arrive_list[k] <= tar_est):
                if k < arrive_len:
                    if len(self.complete_task) != 0:
                        if self.arrive_list[k] <= self.complete_task[0].end:
                            if self.arrive_list[k] >= TIME_NEXT:
                                flag_2 = True
                                break
                            self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                            self.arrive(k)
                            k += 1
                            task, tar_p, tar_est, tar_vm = self.check_ready()
                        else:
                            if self.complete_task[0].end >= TIME_NEXT:
                                flag_2 = True
                                break
                            completed_task = self.complete_task.pop(0)
                            self.advance_virtual_time(completed_task.end - self.virtual_time)
                            self.find_ready_tasks(completed_task)
                            task, tar_p, tar_est, tar_vm = self.check_ready()
                    else:
                        if self.arrive_list[k] >= TIME_NEXT:
                            flag_2 = True
                            break
                        self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                        self.arrive(k)
                        k += 1
                        task, tar_p, tar_est, tar_vm = self.check_ready()
                else:
                    if self.complete_task[0].end >= TIME_NEXT:
                        flag_2 = True
                        break
                    completed_task = self.complete_task.pop(0)
                    self.advance_virtual_time(completed_task.end - self.virtual_time)
                    self.find_ready_tasks(completed_task)
                    task, tar_p, tar_est, tar_vm = self.check_ready()
                    
            if flag_2 or tar_est >= TIME_NEXT:
                self.advance_virtual_time(TIME_NEXT - self.virtual_time)
                break
            
            self.advance_virtual_time(tar_est - self.virtual_time)
            self.schedule_task(task, tar_p, tar_est, tar_vm)

            if task.processor_id in range(self.n_agents):
                
                if task.id in range(self.dags[task.k].num_tasks - 1):
                    if task.lt >= tar_est:
                        r[task.processor_id] += 1
                    else:
                        r[task.processor_id] -= 1
            else:
                if task.id in range(self.dags[task.k].num_tasks - 1):
                    if task.lt >= tar_est:
                        for i in range(self.n_agents):
                            r[i] += 1/self.n_agents
                    else:
                        for i in range(self.n_agents):
                            r[i] += -1/self.n_agents
            if task.id == self.dags[task.k].num_tasks - 1:
                if task.end > self.dags[task.k].deadline:
                    for i in range(self.n_agents):
                        r[i] += -20 * 1/self.n_agents
                else:
                    for i in range(self.n_agents):
                        r[i] += 10 * 1/self.n_agents
        return sum(r)
    
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
        # if task in self.queues[task.k]:
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
        # if p.id in range(self.n_agents):
        #     est = max(self.dags[k].r + self.dags[k].t_offload, self.virtual_time, (self.virtual_time//self.TIME_STAMP)*self.TIME_STAMP + p.service_migrate[t.service_id]*self.configuring_time)    # 初始化est时间为任务到达时间和offload时间之和
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
        print("E = {}, C = {}".format(task_e, task_c))
        SR = satisfy / self.Q * 100
        return SR

def Policy(args):
    k, sqds, actions = args
    r = {}
    for action in actions:
        sqds_copy = copy.deepcopy(sqds)
        r[sqds_copy.policy(k, action)] = action
    max_r = max(r)
    return max_r, r[max_r]

    
def run():
    multiprocessing.set_start_method('spawn')
    sqds_olmedc = SQDS_OLMEDC()
    sqds_olmedc.receive_dag()
    for i in range(sqds_olmedc.n_agents):        # 由于processors信息在environment.py中定义，所以这里需要重新初始化
        sqds_olmedc.processors[i].task_list = []
        sqds_olmedc.processors[i].service_list = random.sample(sqds_olmedc.avail_action[capacity-1], 1)[0]   # random initial
    for vm in sqds_olmedc.processors[sqds_olmedc.n_agents].vms:
        vm.task_list = [] 

    next_policy = np.zeros((sqds_olmedc.n_agents, 5))  # 下一个时隙的缓存策略
    k = 0
    arrive_len = len(sqds_olmedc.arrive_list)
    sqds_olmedc.virtual_time = 0.0
    sqds_olmedc.arrive(k)
    k += 1 
    step = 0
    arrive_k = k
    sqds_olmedc_copy = copy.deepcopy(sqds_olmedc)
    with Pool(50) as pool:
        while (any(queue for queue in sqds_olmedc.queues) or k < arrive_len):  # 调度还没有结束
            flag_1 ,flag_2= False, False
            if sqds_olmedc.virtual_time != 0 and sqds_olmedc.virtual_time % sqds_olmedc.TIME_STAMP == 0:  # 每个时间戳执行一次动作
                actions = []
                # print(next_policy)
                for i in range(sqds_olmedc.n_agents):
                    valid_actions = [action for action in sqds_olmedc.avail_action[capacity-1] if all(action[j] == 1 for j in range(5) if next_policy[i][j] == 1)]
                    actions.append(valid_actions)
                all_combinations = list(itertools.product(*actions))
                batch_size = 2000
                batched_combinations = [all_combinations[i:i + batch_size] for i in range(0, len(all_combinations), batch_size)]
                args_policy = [(arrive_k, sqds_olmedc_copy, batched_combinations[i]) for i in range(len(batched_combinations))]
                print(len(args_policy))
                results = pool.map(Policy, args_policy)
                r = [result[0] for result in results]
                policy = results[r.index(max(r))][1]
                print(policy)
                for i in range(sqds_olmedc.n_agents):
                    sqds_olmedc.processors[i].service_list = policy[i]
                arrive_k = k
                sqds_olmedc_copy = copy.deepcopy(sqds_olmedc)
                step += 1
                next_policy = np.zeros((sqds_olmedc.n_agents, 5))
                # for i in range(sqds_olmedc.n_agents):
                #     sqds_olmedc.servers[i].service_list = next_policy[i][0:]
                #     sqds_olmedc.servers[i].service_migrate = [1 if (y_1 == 1 and y_2 == 0) else 0 for y_1, y_2 in zip(next_policy[i][0:], policy[i][0:])]
            TIME_NEXT = sqds_olmedc.TIME_STAMP * (step + 1)
            
            for task in sqds_olmedc.complete_task:
                if task.processor_id in range(sqds_olmedc.n_agents) and task.end > TIME_NEXT:
                    next_policy[task.processor_id][task.service_id] = 1
            
            while (not sqds_olmedc.ready_tasks and sqds_olmedc.complete_task) or (not sqds_olmedc.ready_tasks and k < arrive_len):
                if not sqds_olmedc.complete_task:
                    if sqds_olmedc.arrive_list[k] >= TIME_NEXT:
                        flag_1 = True
                        break
                    sqds_olmedc.advance_virtual_time(sqds_olmedc.arrive_list[k] - sqds_olmedc.virtual_time)
                    sqds_olmedc.arrive(k)
                    k += 1
                elif k >= arrive_len:
                    if sqds_olmedc.complete_task[0].end >= TIME_NEXT:
                        flag_1 = True
                        break
                    completed_task = sqds_olmedc.complete_task.pop(0)
                    sqds_olmedc.advance_virtual_time(completed_task.end - sqds_olmedc.virtual_time)
                    sqds_olmedc.find_ready_tasks(completed_task)
                elif sqds_olmedc.arrive_list[k] <= sqds_olmedc.complete_task[0].end:
                    if sqds_olmedc.arrive_list[k] >= TIME_NEXT:
                        flag_1 = True
                        break
                    sqds_olmedc.advance_virtual_time(sqds_olmedc.arrive_list[k] - sqds_olmedc.virtual_time)
                    sqds_olmedc.arrive(k)
                    k += 1
                else:
                    if sqds_olmedc.complete_task[0].end >= TIME_NEXT:
                        flag_1 = True
                        break
                    completed_task = sqds_olmedc.complete_task.pop(0)
                    sqds_olmedc.advance_virtual_time(completed_task.end - sqds_olmedc.virtual_time)
                    sqds_olmedc.find_ready_tasks(completed_task)
                    
            if flag_1:
                sqds_olmedc.advance_virtual_time(TIME_NEXT - sqds_olmedc.virtual_time)
                continue    
            if any(queue for queue in sqds_olmedc.queues):
                task, tar_p, tar_est, tar_vm = sqds_olmedc.check_ready()    # 每当有DAG加入或者任务完成调度之后都要调用
            
                
            while (len(sqds_olmedc.complete_task) != 0 and sqds_olmedc.complete_task[0].end <= tar_est) or (k < arrive_len and sqds_olmedc.arrive_list[k] <= tar_est):
                if k < arrive_len:
                    if len(sqds_olmedc.complete_task) != 0:
                        if sqds_olmedc.arrive_list[k] <= sqds_olmedc.complete_task[0].end:
                            if sqds_olmedc.arrive_list[k] >= TIME_NEXT:
                                flag_2 = True
                                break
                            sqds_olmedc.advance_virtual_time(sqds_olmedc.arrive_list[k] - sqds_olmedc.virtual_time)
                            sqds_olmedc.arrive(k)
                            k += 1
                            task, tar_p, tar_est, tar_vm = sqds_olmedc.check_ready()
                        else:
                            if sqds_olmedc.complete_task[0].end >= TIME_NEXT:
                                flag_2 = True
                                break
                            completed_task = sqds_olmedc.complete_task.pop(0)
                            sqds_olmedc.advance_virtual_time(completed_task.end - sqds_olmedc.virtual_time)
                            sqds_olmedc.find_ready_tasks(completed_task)
                            task, tar_p, tar_est, tar_vm = sqds_olmedc.check_ready()
                    else:
                        if sqds_olmedc.arrive_list[k] >= TIME_NEXT:
                            flag_2 = True
                            break
                        sqds_olmedc.advance_virtual_time(sqds_olmedc.arrive_list[k] - sqds_olmedc.virtual_time)
                        sqds_olmedc.arrive(k)
                        k += 1
                        task, tar_p, tar_est, tar_vm = sqds_olmedc.check_ready()
                else:
                    if sqds_olmedc.complete_task[0].end >= TIME_NEXT:
                        flag_2 = True
                        break
                    completed_task = sqds_olmedc.complete_task.pop(0)
                    sqds_olmedc.advance_virtual_time(completed_task.end - sqds_olmedc.virtual_time)
                    sqds_olmedc.find_ready_tasks(completed_task)
                    task, tar_p, tar_est, tar_vm = sqds_olmedc.check_ready()
                    
            if flag_2 or tar_est >= TIME_NEXT:
                sqds_olmedc.advance_virtual_time(TIME_NEXT - sqds_olmedc.virtual_time)
                continue
            
            sqds_olmedc.advance_virtual_time(tar_est - sqds_olmedc.virtual_time)
            sqds_olmedc.schedule_task(task, tar_p, tar_est, tar_vm)
            
            if task.processor_id in range(sqds_olmedc.n_agents) and task.end >= TIME_NEXT:  # 判断调度之后是否会影响obs,state
                next_policy[task.processor_id][task.service_id] = 1

    str = sqds_olmedc.str()
    print(f"Q={sqds_olmedc.Q} ES={sqds_olmedc.n_agents} lambda={Lambda} beta={beta} capacity={capacity} ccr={ccr} SR={str}%")

if __name__ == '__main__':
    run()
    
    # capacity 16.5 32.5 46.5 60
    # agent    34.5 43.0 46.5 52.5 57.0
    # Lambda   66.0 50.5 46.5 44.5 41.0
    # beta     20.5 31 33.5 44.5 46.5 53.5 60.5 70.0 75.5
    # ccr      59.5 51.5 46.5 37.5 32