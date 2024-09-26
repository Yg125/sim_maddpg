import random
from Configure import DAG, Task, B_u, B_aver, B_c, B_e, Lambda, Q, beta, NUM_AGENTS, capacity
from Env import Server, Remote_cloud, server_capacity, comp, request_dict, interval_dict, task_type, ScheduleEnv
from collections import deque
import numpy as np

# 由于只在EST时将任务调度到目标服务器，所以不可能出现排队现象，只是在计算EST、EFT时需要考虑当前边缘服务器是否正在处理任务，有avail时间
# 远程云不用考虑，只需要计算依赖限制的EST，边缘服务器还需考虑avail时间，取最大值
configuring_time = 30  # ms
random.seed(200)
env = ScheduleEnv()
class OnDoc:
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
        self.arrive_list = [0 for _ in range(self.Q)]
        self.virtual_time = 0.0 
        self.processors = self.servers + [self.cloud]
        self.graph = [0 for _ in range(self.Q)]
        self.comp_cost = [0 for _ in range(self.Q)]
        self.tasks = [0 for _ in range(self.Q)]
        self.request_list = request_dict[NUM_AGENTS]     # Env中的request_list 避免调用全局变量
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
        DAGS = np.load(f'./dag_infos/dag_info_random.npy', allow_pickle=True)  # Read DAG from file
        type_count = 0
        for k in range(self.Q):
            self.dags[k] = DAG(k)
            self.dags[k].num_tasks, self.comp_cost[k], self.graph[k], deadline_heft = DAGS[k]   
            self.comp_cost[k] = np.hstack((self.comp_cost[k][:,:NUM_AGENTS],self.comp_cost[k][:,-1].reshape(-1,1)))
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
            
    def arrive(self, k):
        tasks = self.tasks[k]
        computed_ranks = {}
        self.computeRank(tasks[0], k, computed_ranks)
        self.queues.append(deque(tasks))

    def schedule(self):
        avail_action = [[[0, 0, 0, 0, 1],[0, 0, 0, 1, 0],[0, 0, 1, 0, 0],[0, 1, 0, 0, 0],[1, 0, 0, 0, 0]],
                        [[0, 0, 0, 1, 1],[0, 0, 1, 0, 1],[0, 0, 1, 1, 0],[0, 1, 0, 0, 1],[0, 1, 0, 1, 0],[0, 1, 1, 0, 0],[1, 0, 0, 0, 1],[1, 0, 0, 1, 0],[1, 0, 1, 0, 0],[1, 1, 0, 0, 0]],
                        [[0, 0, 1, 1, 1],[0, 1, 0, 1, 1],[0, 1, 1, 0, 1],[0, 1, 1, 1, 0],[1, 0, 0, 1, 1],[1, 0, 1, 0, 1],[1, 0, 1, 1, 0],[1, 1, 0, 0, 1],[1, 1, 0, 1, 0],[1, 1, 1, 0, 0]],
                        [[0, 1, 1, 1, 1],[1, 0, 1, 1, 1],[1, 1, 0, 1, 1],[1, 1, 1, 0, 1],[1, 1, 1, 1, 0]]]
        for processor in self.processors:        # 由于processors信息在environment.py中定义，所以这里需要重新初始化
            processor.task_list = []
            processor.service_list = random.sample(avail_action[capacity-1], 1)[0]   # random initial 
            # processor.service_end = [0,0,0,0,0]    # when the service is free
            # processor.service_start = [0,0,0,0,0]
            if processor.id == NUM_AGENTS:
                for vm in processor.vms:
                    vm.task_list = []
        index = 0
        self.virtual_time = 0.0
        self.arrive(index)
        index += 1
        while any(queue for queue in self.queues) or index < len(self.arrive_list):
            if any(queue for queue in self.queues):
                min_k, tar_p, tar_est, tar_vm = self.check_queues()    # 每当有DAG加入或者任务完成调度之后都要调用
            # 如果所有队列现在都为空，并且还有未处理的到达事件
            # 则将时间推移到下一个到达事件，并将请求加入队列
            if not any(queue for queue in self.queues) and index < len(self.arrive_list):
                self.advance_virtual_time(self.arrive_list[index] - self.virtual_time)
                self.arrive(index)
                index += 1
                min_k, tar_p, tar_est, tar_vm = self.check_queues()
            while (index < len(self.arrive_list) and self.virtual_time < self.arrive_list[index] and self.arrive_list[index] < tar_est):
                self.advance_virtual_time(self.arrive_list[index] - self.virtual_time)
                self.arrive(index)
                index += 1
            task = self.queues[min_k][0]
            # if tar_p in range(NUM_AGENTS):
            #     if (task.service_id not in self.processors[tar_p].service_list) and (task.id != 0 and task.id != self.dags[task.k].num_tasks - 1):   
            #         if sum(self.processors[tar_p].service_list) == capacity:
            #             replace = self.processors[tar_p].service_end.index(min(self.processors[tar_p].service_end[i] for i, v in enumerate(self.processors[tar_p].service_list) if v == 1))
            #             self.processors[tar_p].service_list[replace] = 0
            #             # self.processors[tar_p].service_end[replace] = 0
            #             # self.processors[tar_p].service_start[replace] = 0
            #         self.processors[tar_p].service_list[task.service_id] = 1
            #         # self.processors[tar_p].service_start[task.service_id] = self.virtual_time + configuring_time
                        
            #         # min_k, tar_p, tar_est, tar_vm = self.check_queues()
            self.advance_virtual_time(tar_est - self.virtual_time)
            self.schedule_task(task, tar_p, tar_est, tar_vm)
        
    def check_queues(self):
        tar_est = float('inf')
        min_k = None
        tar_vm = None
        vm = None
        for k in range(len(self.queues)):
            if len(self.queues[k]) > 0:
                result = self.get_tar(self.queues[k][0])
                if isinstance(result, tuple) and len(result) == 3:
                    p, est, vm = result
                else:
                    p, est = result
                if est < tar_est:
                    tar_p = p
                    tar_est = est
                    min_k = k
                    tar_vm = vm
            else:
                continue   
        return min_k, tar_p, tar_est, tar_vm
        
    def schedule_task(self, task, p, est, vm=None):
        queue = self.queues[task.k]   # 调度第min_k个DAG的头任务
        if queue:
            t = queue[0]
            t.processor_id = p
            t.start = est
            t.end = t.start + self.comp_cost[t.k][t.id][p]
        queue.popleft()
        if p in range(NUM_AGENTS):
            self.processors[p].task_list.append(task)
            # self.processors[p].service_end[task.service_id] = t.end
        else:
            self.processors[NUM_AGENTS].vms[vm].task_list.append(task)
         
    def get_est(self, t, p, k): 
        if (p.id in range(NUM_AGENTS) and not p.service_list[t.service_id]) and (t.id != 0 and t.id != self.dags[t.k].num_tasks - 1):
            return float('inf')
        else:
            est = max(self.dags[k].r + self.dags[k].t_offload, self.virtual_time)
        # est = self.dags[k].r + self.dags[k].t_offload    # 初始化est时间为任务到达时间和offload时间之和
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

result = np.zeros(200)
for i in range(200):
    random.seed(i)
    ondoc = OnDoc()
    ondoc.receive_dag()
    ondoc.schedule()
    str = ondoc.str()
    result[i] = str
print(f"Lambda={Lambda} ES={NUM_AGENTS} capacity={capacity} beta={beta} SR={np.mean(result)} max={np.max(result)} min={np.min(result)}")


# ondoc = OnDoc()
# ondoc.receive_dag()
# ondoc.schedule()

# str = ondoc.str()
# print(f"Q={ondoc.Q} ES={NUM_AGENTS} lambda={Lambda} SR={str}%")
