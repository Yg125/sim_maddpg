import random
from Configure import DAG, Task, B_u, B_aver, B_c, B_e, Lambda, Q, beta, NUM_AGENTS, capacity
from Env import Server, Remote_cloud, server_capacity, comp, request_dict, interval_dict, task_type, ScheduleEnv
from collections import deque
import numpy as np

# 由于只在EST时将任务调度到目标服务器，所以不可能出现排队现象，只是在计算EST、EFT时需要考虑当前边缘服务器是否正在处理任务，有avail时间
# 远程云不用考虑，只需要计算依赖限制的EST，边缘服务器还需考虑avail时间，取最大值
configuring_time = 35  # ms
env = ScheduleEnv()
random.seed(155)
class COFE:
    def __init__(self):   
        self.B_u = B_u
        self.B_aver = B_aver
        self.B_c = B_c
        self.B_e = B_e
        self.Q = Q
        self.servers = [Server(i, comp[i], server_capacity[i]) for i in range(NUM_AGENTS)]
        self.cloud = Remote_cloud(NUM_AGENTS, 7000) 
        self.queues = []
        self.num_processors = NUM_AGENTS + 1
        self.dags = [0 for _ in range(self.Q)]
        self.ready_tasks = []
        self.virtual_time = 0.0        # 定义在线场景的虚拟时间，避免受机器运行时间干扰
        self.complete_task = []        # COFE算法中记录每个子任务的完成时间
        self.arrive_list = [0 for _ in range(self.Q)]
        self.virtual_time = 0.0 
        self.processors = self.servers + [self.cloud]
        self.graph = [0 for _ in range(self.Q)]
        self.comp_cost = [0 for _ in range(self.Q)]
        self.tasks = [0 for _ in range(self.Q)]
        self.request_list = request_dict[NUM_AGENTS]
        self.interval_list = interval_dict[Lambda]
        self.task_type = task_type
        
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

    def receive_dag(self):                    # k is the index of the request/queue from 0
        virtual_time = 0
        tasks = [0 for _ in range(self.Q)]
        DAGS = np.load(f'./dag_infos/dag_info_random_es7_new2.npy', allow_pickle=True)  # Read DAG from file
        type_count = 0
        for k in range(self.Q):
            self.dags[k] = DAG(k)
            self.dags[k].num_tasks, self.comp_cost[k], self.graph[k], deadline_heft = DAGS[k]   
            self.comp_cost[k] = np.hstack((self.comp_cost[k][:,:NUM_AGENTS],self.comp_cost[k][:,-1].reshape(-1,1)))
            self.dags[k].deadline = virtual_time + deadline_heft * beta   # 以heft算法的deadline为基准，增加15%的时间作为在线场景DAG的deadline
            self.dags[k].r = virtual_time    # ms
            self.arrive_list[k] = virtual_time
            num_tasks = self.dags[k].num_tasks
            tasks[k] = [Task(i,k,self.task_type[type_count+i],self.dags[k].deadline) for i in range(num_tasks)]
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
            
    def arrive(self, k):
        tasks = self.tasks[k]
        computed_ranks = {}
        self.computeRank(tasks[0], k, computed_ranks)
        tasks.sort(key = lambda x: x.rank, reverse=True)
        self.queues.append(deque(tasks))
        self.ready_tasks.append(self.queues[k][0])
        self.schedule_tasks()
        
    def schedule(self):
        # COFE:
        # 1. New Request Arrival: 当有新请求到达时，计算每个任务的rank，将v_0插入ready_tasks，并从V中删除，调用schedule(ready_task)
        # 2. Processing Task Completion: 当任务完成时，从V中删除，根据该任务的直接后继判断是否是ready_task，并将其加入，调用schedule(ready_task)
        # 3. Schedule Task: 对ready_tasks队列排序，先根据deadline排序，再根据rank排序中选择一个任务，计算其EST，选择一个processor，计算其EFT，将任务调度到processor上
        # schedule task at EST
        # otherwise, wait
        avail_action = [[[0, 0, 0, 0, 1],[0, 0, 0, 1, 0],[0, 0, 1, 0, 0],[0, 1, 0, 0, 0],[1, 0, 0, 0, 0]],
                        [[0, 0, 0, 1, 1],[0, 0, 1, 0, 1],[0, 0, 1, 1, 0],[0, 1, 0, 0, 1],[0, 1, 0, 1, 0],[0, 1, 1, 0, 0],[1, 0, 0, 0, 1],[1, 0, 0, 1, 0],[1, 0, 1, 0, 0],[1, 1, 0, 0, 0]],
                        [[0, 0, 1, 1, 1],[0, 1, 0, 1, 1],[0, 1, 1, 0, 1],[0, 1, 1, 1, 0],[1, 0, 0, 1, 1],[1, 0, 1, 0, 1],[1, 0, 1, 1, 0],[1, 1, 0, 0, 1],[1, 1, 0, 1, 0],[1, 1, 1, 0, 0]],
                        [[0, 1, 1, 1, 1],[1, 0, 1, 1, 1],[1, 1, 0, 1, 1],[1, 1, 1, 0, 1],[1, 1, 1, 1, 0]]]
        for i in range(NUM_AGENTS):        # 由于processors信息在environment.py中定义，所以这里需要重新初始化
            self.processors[i].task_list = []
            self.processors[i].service_list = random.sample(avail_action[capacity-1], 1)[0]   # random initial 
        for vm in self.processors[NUM_AGENTS].vms:
            vm.task_list = []
        k = 0
        self.virtual_time = 0.0
        self.arrive(k)
        k += 1
        while any(queue for queue in self.queues) or k < len(self.arrive_list):
            while len(self.complete_task) != 0 and self.virtual_time <= self.complete_task[0].end:
                while k < len(self.arrive_list) and self.virtual_time < self.arrive_list[k] and self.arrive_list[k] < self.complete_task[0].end:
                    # 这里可能还需要调整，因为EST可能会和arrive_list[k]有关
                    self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                    self.arrive(k)
                    k += 1
                self.advance_virtual_time(self.complete_task[0].end - self.virtual_time)
                complete_task = self.complete_task.pop(0)
                self.find_ready_tasks(complete_task)
                self.schedule_tasks()
            if not any(queue for queue in self.queues) and k < len(self.arrive_list):
                self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                self.arrive(k)
                k += 1
        
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
                                                
    def schedule_tasks(self):
        self.ready_tasks.sort(key=lambda x: (x.deadline, x.rank), reverse=True)
        # k_id_list = [(item.k, item.id) for item in self.ready_tasks]
        # print(k_id_list)
        for t in self.ready_tasks:
            result = self.get_tar(t)
            if isinstance(result, tuple) and len(result) == 3:
                p, est, vm = result
                self.processors[p].vms[vm].task_list.append(t)
            else:
                p, est = result
                self.processors[p].task_list.append(t)
            t.processor_id = p
            t.start = est
            t.end = t.start + self.comp_cost[t.k][t.id][p]
            
            self.queues[t.k].remove(t)       
            self.complete_task.append(t)
        self.complete_task.sort(key=lambda x: x.end)
        self.ready_tasks.clear()
        
    def get_est(self, t, p, k): 
        if (p.id in range(NUM_AGENTS) and not p.service_list[t.service_id]) and (t.id != 0 and t.id != self.dags[t.k].num_tasks - 1):
            return float('inf')
        else:
            est = max(self.dags[k].r + self.dags[k].t_offload, self.virtual_time)
        graph = self.graph[k]
        tasks = self.tasks[k]
        for pre in tasks:
            if graph[pre.id][t.id] != -1:  # if pre also done on p, no communication cost
                c = graph[pre.id][t.id] if pre.processor_id != p.id else 0
                if pre.processor_id in range(NUM_AGENTS) and p.id in range(NUM_AGENTS): 
                    est = max(est, pre.end + round(c*B_e/10**6, 1))  # ms
                else:
                    est = max(est, pre.end + round(c*B_c/10**6, 1))
        if p.id in range(NUM_AGENTS) and len(p.task_list) == 0:  # 在之前没有任务则直接返回任务依赖的EST
            return est
        elif p.id == NUM_AGENTS:
            est_cloud = float('inf')
            vm_i = None 
            for i in range(len(self.processors[NUM_AGENTS].vms)):
                if len(self.processors[NUM_AGENTS].vms[i].task_list) == 0:
                    return (est, i)
                else:
                    if est_cloud > self.processors[NUM_AGENTS].vms[i].task_list[-1].end:
                        est_cloud = self.processors[NUM_AGENTS].vms[i].task_list[-1].end
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
            if tar_p == NUM_AGENTS:
                return (tar_p, tar_est, vm_i)
            else:
                return (tar_p, tar_est)
        
    def str(self):
        satisfy = 0
        task_e = 0
        task_c = 0
        for p in self.processors[0:NUM_AGENTS]:
            task_e += len(p.task_list)
        for vm in self.processors[NUM_AGENTS].vms:
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
        
cofe = COFE()
cofe.receive_dag()
cofe.schedule()

str = cofe.str()
print(f"Q={cofe.Q} ES={NUM_AGENTS} lambda={Lambda} SR={str}%")