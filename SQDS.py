import random
from Configure import DAG, Task, B_u, B_aver, B_c, B_e, Lambda, Q, beta, NUM_AGENTS
from Env import Server, Remote_cloud, server_capacity, comp, request_dict, interval_dict, task_type, ScheduleEnv
from collections import deque
import numpy as np
configuring_time = 30  # ms
random.seed(1)
env = ScheduleEnv()
class OnDoc_plus:
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
        self.ready_tasks = []
        self.complete_task = []
        self.n_agents = NUM_AGENTS
        
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

    def LT(self, k):
        for t in self.dags[k].tasks:
            t.lt = self.dags[k].r + (self.dags[k].deadline - self.dags[k].r) * (self.dags[k].tasks[0].rank - t.rank) / self.dags[k].tasks[0].rank
            
    def receive_dag(self):                    # k is the index of the request/queue from 0
        virtual_time = 0
        tasks = [0 for _ in range(self.Q)]
        DAGS = np.load(f'./dag_infos/dag_info_random.npy', allow_pickle=True)  # Read DAG from file
        type_count = 0
        for k in range(self.Q):
            self.dags[k] = DAG(k)
            self.dags[k].num_tasks, self.comp_cost[k], self.graph[k], deadline_heft = DAGS[k]   
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
        self.LT(k)
        self.queues[k] = deque(tasks)
        self.ready_tasks.append(self.queues[k][0])
        
    # My algorithm is going to do as follows:
    # compute each task's l_k to sort probabilistic
    # compute EST for the task whose priority is highest
    # schedule tasks to servers according to l_k and EFT at EST to server_p

    def schedule(self): 
        k = 0
        arrive_len = len(self.arrive_list)
        self.virtual_time = 0.0
        self.arrive(k)
        k += 1 
        while any(queue for queue in self.queues) or k < arrive_len:
            while (not self.ready_tasks and self.complete_task) or (not self.ready_tasks and k < arrive_len):
                if not self.complete_task:
                    self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                    self.arrive(k)
                    k += 1
                elif k >= arrive_len:
                    completed_task = self.complete_task.pop(0)
                    self.advance_virtual_time(completed_task.end - self.virtual_time)
                    self.find_ready_tasks(completed_task)
                elif self.arrive_list[k] <= self.complete_task[0].end:
                    self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                    self.arrive(k)
                    k += 1
                else:
                    completed_task = self.complete_task.pop(0)
                    self.advance_virtual_time(completed_task.end - self.virtual_time)
                    self.find_ready_tasks(completed_task)
                    
            if any(queue for queue in self.queues):
                task, tar_p, tar_est, tar_vm = self.check_ready()    # 每当有DAG加入或者任务完成调度之后都要调用
            # 得到下一个要调度的ready_task的DAG编号、目标服务器编号、目标服务器的EST
            # 将时间推移到EST，调度ready_task，更新queues和processors，
            # 如果中间经过了某个DAG的到达时间则将其加入队列或者经过了某个任务的完成时间就需要加入ready_tasks
            # 虚拟时间一直往前走，只在EST调度任务，任务完成只添加进ready_task     
            while (len(self.complete_task) != 0 and self.complete_task[0].end <= tar_est) or (k < arrive_len and self.arrive_list[k] <= tar_est):
                if k < arrive_len:
                    if len(self.complete_task) != 0:
                        if self.arrive_list[k] <= self.complete_task[0].end:
                            self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                            self.arrive(k)
                            k += 1
                            task, tar_p, tar_est, tar_vm = self.check_ready()
                        else:
                            completed_task = self.complete_task.pop(0)
                            self.advance_virtual_time(completed_task.end - self.virtual_time)
                            self.find_ready_tasks(completed_task)
                            task, tar_p, tar_est, tar_vm = self.check_ready()
                    else:
                        self.advance_virtual_time(self.arrive_list[k] - self.virtual_time)
                        self.arrive(k)
                        k += 1
                        task, tar_p, tar_est, tar_vm = self.check_ready()
                else:
                    completed_task = self.complete_task.pop(0)
                    self.advance_virtual_time(completed_task.end - self.virtual_time)
                    self.find_ready_tasks(completed_task)
                    task, tar_p, tar_est, tar_vm = self.check_ready()
                    
            self.advance_virtual_time(tar_est - self.virtual_time)
            self.schedule_task(task, tar_p, tar_est, tar_vm)
     
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
        est = max(self.dags[k].r + self.dags[k].t_offload, self.virtual_time)    # 初始化est时间为任务到达时间和offload时间之和
        # est = self.dags[k].r + self.dags[k].t_offload    # 初始化est时间为任务到达时间和offload时间之和
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
        
ondoc = OnDoc_plus()
ondoc.receive_dag()
ondoc.schedule()

str = ondoc.str()
print(f"Q={ondoc.Q} ES={NUM_AGENTS} lambda={Lambda} SR={str}%")
