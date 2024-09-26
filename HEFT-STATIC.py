import random
from Configure import DAG, Task, B_u, B_aver, B_c, B_e, Lambda, Q, beta, capacity, NUM_AGENTS
from collections import deque
import numpy as np
from Env import Server, Remote_cloud, server_capacity, comp, task_type, request_dict, interval_dict
np.random.seed(1)

class HEFT:
    def __init__(self):
        self.B_u = B_u
        self.B_aver = B_aver
        self.B_c = B_c
        self.B_e = B_e
        self.Q = Q
        self.servers = [Server(i, comp[i], server_capacity[i]) for i in range(NUM_AGENTS)]
        self.cloud = Remote_cloud(NUM_AGENTS, 7000) 
        self.queues = [0 for _ in range(self.Q)]
        self.num_processors = NUM_AGENTS + 1
        self.dags = [0 for _ in range(self.Q)]
        self.arrive_list = [0 for _ in range(self.Q)]
        self.virtual_time = 0.0 
        self.processors = self.servers + [self.cloud]
        self.graph = [0 for _ in range(self.Q)]
        self.comp_cost = [0 for _ in range(self.Q)]
        self.tasks = [0 for _ in range(self.Q)]
        self.request_list = request_dict[NUM_AGENTS]     # Env中的request_list 避免调用全局变量
        self.interval_list = interval_dict[Lambda]
        self.task_type = task_type
        self.n_agents = NUM_AGENTS
        self.avail_action = [[[0, 0, 0, 0, 1],[0, 0, 0, 1, 0],[0, 0, 1, 0, 0],[0, 1, 0, 0, 0],[1, 0, 0, 0, 0]],
                    [[0, 0, 0, 1, 1],[0, 0, 1, 0, 1],[0, 0, 1, 1, 0],[0, 1, 0, 0, 1],[0, 1, 0, 1, 0],[0, 1, 1, 0, 0],[1, 0, 0, 0, 1],[1, 0, 0, 1, 0],[1, 0, 1, 0, 0],[1, 1, 0, 0, 0]],
                    [[0, 0, 1, 1, 1],[0, 1, 0, 1, 1],[0, 1, 1, 0, 1],[0, 1, 1, 1, 0],[1, 0, 0, 1, 1],[1, 0, 1, 0, 1],[1, 0, 1, 1, 0],[1, 1, 0, 0, 1],[1, 1, 0, 1, 0],[1, 1, 1, 0, 0]],
                    [[0, 1, 1, 1, 1],[1, 0, 1, 1, 1],[1, 1, 0, 1, 1],[1, 1, 1, 0, 1],[1, 1, 1, 1, 0]]]
        
    def receive_dag(self):                    # k is the index of the request/queue from 0
        virtual_time = 0
        tasks = [0 for _ in range(self.Q)]
        DAGS = np.load(f'./dag_infos/dag_info_random_es7.npy', allow_pickle=True)  # Read DAG from file
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
        
    def arrive(self, k):
        tasks = self.tasks[k]
        computed_ranks = {}
        self.computeRank(tasks[0], k, computed_ranks)
        self.tasks[k].sort(key = lambda x:x.rank, reverse=True)
        self.queues[k] = deque(tasks)
        self.schedule(k)
        
    def get_est(self, t, p, k):
        if (p.id in range(self.n_agents) and not p.service_list[t.service_id]) and (t.id != 0 and t.id != self.dags[t.k].num_tasks - 1):
            return float('inf')
        else:
            est = max(self.dags[k].r + self.dags[k].t_offload, self.virtual_time)
        graph = self.graph[k]
        tasks = self.tasks[k]
        for pre in tasks:
            if graph[pre.id][t.id] != -1:  # if pre also done on p, no communication cost
                c = graph[pre.id][t.id] if pre.processor_id != p.id else 0
                if pre.processor_id in range(self.n_agents) and p.id in range(self.n_agents):   # 前后两个都在边缘服务器上
                    est = max(est, pre.end + round(c*self.B_e/10**6, 1))  # ms
                else:
                    est = max(est, pre.end + round(c*self.B_e/10**6, 1))
        if p.id in range(self.n_agents):
            free_times = []
            if len(p.task_list) == 0:       # no task has yet been assigned to processor
                free_times.append([0, float('inf')])
            else:
                for i in range(len(p.task_list)):
                    if i == 0:
                        if p.task_list[i].start != 0:    # if p is not busy from time 0
                            free_times.append([0, p.task_list[i].start])
                    else:
                        free_times.append([p.task_list[i-1].end, p.task_list[i].start])
                free_times.append([p.task_list[-1].end, float('inf')])
            for slot in free_times:     # free_times is already sorted based on avaialbe start times
                if est < slot[0] and slot[0] + self.comp_cost[k][t.id][p.id] <= slot[1]:
                    return slot[0]
                if est >= slot[0] and est + self.comp_cost[k][t.id][p.id] <= slot[1]:
                    return est
        else:
            est_cloud = []
            vm_id = []
            for j, vm in enumerate(p.vms):
                free_times = []
                if len(vm.task_list) == 0:       # no task has yet been assigned to processor
                    free_times.append([0, float('inf')])
                else:
                    for i in range(len(vm.task_list)):
                        if i == 0:
                            if vm.task_list[i].start != 0:    # if p is not busy from time 0
                                free_times.append([0, vm.task_list[i].start])
                        else:
                            free_times.append([vm.task_list[i-1].end, vm.task_list[i].start])
                    free_times.append([vm.task_list[-1].end, float('inf')])
                for slot in free_times:     # free_times is already sorted based on avaialbe start times
                    if est < slot[0] and slot[0] + self.comp_cost[k][t.id][p.id] <= slot[1]:
                        est_cloud.append(slot[0])
                        vm_id.append(j)
                        break
                    if est >= slot[0] and est + self.comp_cost[k][t.id][p.id] <= slot[1]:
                        est_cloud.append(est)
                        vm_id.append(j)
                        break
            return min(est_cloud), vm_id[est_cloud.index(min(est_cloud))]

    def schedule(self, k):
        for t in self.tasks[k]:
            if t == self.tasks[k][0]:   # the one with highest rank
                p = self.request_list[t.k] # 随机从某个边缘服务器发出请求
                w = self.comp_cost[k][t.id][p]
                t.processor_id = p
                t.start = 0
                t.end = w
                self.processors[p].task_list.append(t)
            elif t.id == self.dags[k].num_tasks - 1:
                tar_p = self.tasks[k][0].processor_id
                tar_est = self.get_est(t, self.processors[tar_p], t.k)
                t.processor_id = tar_p
                t.start = tar_est
                t.end = tar_est + self.comp_cost[k][t.id][tar_p]
                self.processors[tar_p].task_list.append(t)
            else:
                aft = float("inf")
                for p in self.processors:
                    result = self.get_est(t, p, t.k)
                    if isinstance(result, tuple):
                        est, vm_id = result
                    else:
                        est = result
                    eft = est + self.comp_cost[k][t.id][p.id]
                    if eft < aft:   # found better case of processor
                        aft = eft
                        best_p = p.id
                if est < self.virtual_time:
                    print(1)
                t.processor_id = best_p
                t.start = aft - self.comp_cost[k][t.id][best_p]
                t.end = aft
                if best_p in range(NUM_AGENTS):
                    self.processors[best_p].task_list.append(t)
                    self.processors[best_p].task_list.sort(key = lambda x: x.start)
                else:
                    self.processors[best_p].vms[vm_id].task_list.append(t)
                    self.processors[best_p].vms[vm_id].task_list.sort(key = lambda x: x.start)
    
    def online_schedule(self):
        self.receive_dag()
        self.virtual_time = 0.0
        for i in range(self.n_agents):        # 由于processors信息在environment.py中定义，所以这里需要重新初始化
            self.processors[i].task_list = []
            self.processors[i].service_list = random.sample(self.avail_action[capacity-1], 1)[0]   # random initial
        for vm in self.processors[self.n_agents].vms:
            vm.task_list = [] 
        k = 0
        arrive_len = len(self.arrive_list)
        self.arrive(k)
        k += 1 
        while k < arrive_len:
            self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
            self.arrive(k)
            k += 1
        return self.str()
    
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
            # print(self.Makespan)
            if self.Makespan < self.dags[k].deadline - self.dags[k].r:
                satisfy += 1
        print("E = {}, C = {}".format(task_e, task_c))
        SR = satisfy / self.Q * 100
        print(SR)
        return SR

# result = np.zeros(200)
# for i in range(200):
    # random.seed(i)
heft_static = HEFT()
str = heft_static.online_schedule()
# result[i] = str
# print(f"Lambda={Lambda} ES={NUM_AGENTS} capacity={capacity} beta={beta} SR={np.mean(result)} max={np.max(result)} min={np.min(result)}")