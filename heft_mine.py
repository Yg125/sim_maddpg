# 仿照HEFT算法将其应用在我的场景中
import numpy as np
from Configure import NUM_AGENTS, B_aver, B_c, B_e, B_u, NUM_TASKS
from Env import Server, Remote_cloud, server_capacity, comp

servers = [Server(i, comp[i], server_capacity[i]) for i in range(NUM_AGENTS)]
cloud = Remote_cloud(NUM_AGENTS, 7000)

np.random.seed(1)
class Task:
    def __init__(self, id, k):
        self.id = id
        self.k = k
        self.processor_id = None
        self.rank = None
        self.comp_cost = []
        self.avg_comp = None
        self.duration = {'start':None, 'end':None}

class HEFT:
    def __init__(self, k, input_list=None, verbose=False):
        self.k = k
        self.num_tasks, self.comp_cost, self.graph = input_list
        self.num_processors = NUM_AGENTS + 1
        if verbose:
            print("No. of Tasks: ", self.num_tasks)
            print("No. of processors: ", self.num_processors)
            print("Computational Cost Matrix:")
            for i in range(self.num_tasks):
                print(self.comp_cost[i])
            print("Graph Matrix:")
            for line in self.graph:
                print(line)
        
        self.tasks = [Task(i, k) for i in range(self.num_tasks)]
        # self.processors = [Processor(i) for i in range(self.num_processors)] 
        self.processors = servers + [cloud]      # servers编号为0-4, cloud编号为5
        for i in range(NUM_AGENTS):        # 由于self.processors信息在environment.py中定义，所以这里需要重新初始化
            self.processors[i].task_list = []
        for vm in self.processors[NUM_AGENTS].vms:
            vm.task_list = []
        for i in range(self.num_tasks):
            self.tasks[i].comp_cost = self.comp_cost[i]
            self.tasks[i].avg_comp = round(sum(self.comp_cost[i])/self.num_processors, 1)
        
        data_in = 0
        for i in range(self.num_tasks):
            if self.graph[0][i] != -1:
                data_in += self.graph[0][i]
            else:
                data_in += 0
        self.t_offload = round(data_in * B_u / 10**6, 1)
        
        self.__computeRanks(self.tasks[0])
        if verbose:
            for task in self.tasks:
                print("Task ", task.id, "-> Rank: ", task.rank)
        self.tasks.sort(key = lambda x: x.rank, reverse=True)
        
        self.__allotProcessor()
        
        self.makespan = max([t.duration['end'] for t in self.tasks])
        

    def __computeRanks(self, task):
        # Assume that task[0] is the initial task, as generated by TGFF
        # 原始的HEFT算法假设通信速率在处理器之间是相同的，这里我们计算rank时采用平均传输速率
        # 其中self.graph的内容是每条边的传输数据Byte
        curr_rank = 0
        for succ in self.tasks:
            if self.graph[task.id][succ.id] != -1:
                if succ.rank is None:
                    self.__computeRanks(succ)
                curr_rank = max(curr_rank, round(self.graph[task.id][succ.id]*B_aver/10**6, 1) + succ.rank)
        task.rank = task.avg_comp + curr_rank

    def __get_est(self, t, p):
        est = self.t_offload
        for pre in self.tasks:
            if self.graph[pre.id][t.id] != -1:  # if pre also done on p, no communication cost
                c = self.graph[pre.id][t.id] if pre.processor_id != p.id else 0
                if pre.processor_id in range(NUM_AGENTS) and p.id in range(NUM_AGENTS):   # 前后两个都在边缘服务器上
                    est = max(est, pre.duration['end'] + round(c*B_e/10**6, 1))  # ms
                else:
                    est = max(est, pre.duration['end'] + round(c*B_c/10**6, 1))
        if p.id in range(NUM_AGENTS):
            free_times = []
            if len(p.task_list) == 0:       # no task has yet been assigned to processor
                free_times.append([0, float('inf')])
            else:
                for i in range(len(p.task_list)):
                    if i == 0:
                        if p.task_list[i].duration['start'] != 0:    # if p is not busy from time 0
                            free_times.append([0, p.task_list[i].duration['start']])
                    else:
                        free_times.append([p.task_list[i-1].duration['end'], p.task_list[i].duration['start']])
                free_times.append([p.task_list[-1].duration['end'], float('inf')])
            for slot in free_times:     # free_times is already sorted based on avaialbe start times
                if est < slot[0] and slot[0] + t.comp_cost[p.id] <= slot[1]:
                    return slot[0]
                if est >= slot[0] and est + t.comp_cost[p.id] <= slot[1]:
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
                            if vm.task_list[i].duration['start'] != 0:    # if p is not busy from time 0
                                free_times.append([0, vm.task_list[i].duration['start']])
                        else:
                            free_times.append([vm.task_list[i-1].duration['end'], vm.task_list[i].duration['start']])
                    free_times.append([vm.task_list[-1].duration['end'], float('inf')])
                for slot in free_times:     # free_times is already sorted based on avaialbe start times
                    if est < slot[0] and slot[0] + t.comp_cost[p.id] <= slot[1]:
                        est_cloud.append(slot[0])
                        vm_id.append(j)
                        break
                    if est >= slot[0] and est + t.comp_cost[p.id] <= slot[1]:
                        est_cloud.append(est)
                        vm_id.append(j)
                        break
            return min(est_cloud), vm_id[est_cloud.index(min(est_cloud))]

    def __allotProcessor(self):
        for t in self.tasks:
            if t == self.tasks[0]:   # the one with highest rank
                p = np.random.choice(range(NUM_AGENTS)) # 随机从某个边缘服务器发出请求
                w = t.comp_cost[p]
                t.processor_id = p
                t.duration['start'] = 0
                t.duration['end'] = w
                self.processors[p].task_list.append(t)
            elif t.id == NUM_TASKS + 1:
                tar_p = self.tasks[0].processor_id
                tar_est = self.__get_est(t, self.processors[tar_p])
                t.processor_id = tar_p
                t.duration['start'] = tar_est
                t.duration['end'] = tar_est + t.comp_cost[tar_p]
                self.processors[tar_p].task_list.append(t)
            else:
                aft = float("inf")
                for p in self.processors:
                    result = self.__get_est(t, p)
                    if isinstance(result, tuple):
                        est, vm_id = result
                    else:
                        est = result
                    # print("Task: ", t.id, ", Proc: ", p.id, " -> EST: ", est)
                    eft = est + t.comp_cost[p.id]
                    if eft < aft:   # found better case of processor
                        aft = eft
                        best_p = p.id
                
                t.processor_id = best_p
                t.duration['start'] = aft - t.comp_cost[best_p]
                t.duration['end'] = aft
                if best_p in range(NUM_AGENTS):
                    self.processors[best_p].task_list.append(t)
                    self.processors[best_p].task_list.sort(key = lambda x: x.duration['start'])
                else:
                    self.processors[best_p].vms[vm_id].task_list.append(t)
                    self.processors[best_p].vms[vm_id].task_list.sort(key = lambda x: x.duration['start'])
                
    def __str__(self):
        print_str = ""
        for p in self.processors[0: NUM_AGENTS]:
            print_str += 'Processor {}:\n'.format(p.id)
            for t in p.task_list:
                print_str += 'Task {}: start = {}, end = {}\n'.format(t.id, t.duration['start'], t.duration['end'])
        for vm in self.processors[NUM_AGENTS].vms:
            for t in vm.task_list:
                print_str+='id: {}, k: {}'.format(t.id,t.k)
        print_str += "Makespan = {}\n".format(self.makespan)
        return print_str

if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()
    DAGS = np.load('dag_info_6.npy', allow_pickle=True)
    new_sch = HEFT(6, DAGS[6][0:3], verbose=True)
    print(new_sch)