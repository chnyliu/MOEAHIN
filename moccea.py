import random, copy
from utils import gnn_search_space, cstr_nc, delete_inf_metric_indi, delete_inf_metric
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


class Moccea(object):
    def __init__(self, logger, args, adjs_len):
        self.logger = logger
        self.pc = args.pc
        self.pm = args.pm
        self.T = args.T
        self.N = args.N
        self.steps = args.steps
        self.cstr = cstr_nc[args.dataset]
        self.adj_len = adjs_len
        self.arg = args

    def initial_pop(self, n):
        meta_graphs = []
        meta_graph_len = self.steps * (self.steps + 1) * self.adj_len // 2
        last_index = meta_graph_len - self.steps * self.adj_len  # 40-4*4=24

        while len(meta_graphs) < n:
            meta_graph = []
            for index in range(meta_graph_len):
                if index < last_index:
                    meta_graph.append(random.randint(0, 1))
                else:
                    if index % self.adj_len in self.cstr:
                        meta_graph.append(random.randint(0, 1))
                    else:
                        meta_graph.append(0)
            meta_graphs.append(meta_graph)
        graph_len = 0
        for index in range(meta_graph_len):
            if index < last_index:
                graph_len += 1
            else:
                if index % self.adj_len in self.cstr:
                    graph_len += 1
        gnn_archs = []
        gnn_arch_range = len(gnn_search_space)
        while len(gnn_archs) < n:
            gnn_arch = []
            for _ in range(meta_graph_len):
                gnn_arch.append(random.randint(1, gnn_arch_range))
            gnn_archs.append(gnn_arch)
        pops = []
        for i in range(n):
            pops.append([meta_graphs[i], gnn_archs[i]])
        return pops, meta_graphs, gnn_archs, graph_len

    def crossover_and_mutation(self, n, indis, pop_f0, graph_len, group, pop_me, pop_sp, group_name):
        pop_f0_len = len(pop_f0)
        pop_f0_copy = copy.deepcopy(pop_f0)
        step = n // pop_f0_len
        sub_group = []
        offspring = []
        pop_f0_index = 0
        while len(sub_group) < n:
            index1, parent1 = choose_one_parent(group, pop_me, pop_sp)
            index2, parent2 = choose_one_parent(group, pop_me, pop_sp)
            while index1 == index2:
                index2, parent2 = choose_one_parent(group, pop_me, pop_sp)
            child = self.crossover(parent1, parent2)

            group_first = []
            if group_name == 'meta':
                for i in range(pop_f0_len):
                    group_first.append(pop_f0_copy[i][0])
            else:
                for i in range(pop_f0_len):
                    group_first.append(pop_f0_copy[i][1])
            child[0] = self.mutation(group_first, graph_len, child[0], group_name)
            child[1] = self.mutation(group_first, graph_len, child[1], group_name)
            for _child in child:
                if len(sub_group) >= (pop_f0_index + 1) * step and pop_f0_index < pop_f0_len - 1:
                    pop_f0_index += 1
                if group_name == 'meta':
                    tmp = [_child, pop_f0[pop_f0_index][1]]
                else:
                    tmp = [pop_f0[pop_f0_index][0], _child]
                if tmp not in indis:
                    sub_group.append(_child)
                    offspring.append(tmp)
                    indis.append(tmp)
                if len(sub_group) == n:
                    break
        return sub_group, offspring

    def crossover(self, parent1, parent2):
        if random.random() < self.pc:
            point = random.randint(0, len(parent1) - 1)
            child = [copy.deepcopy(parent1[:point]) + copy.deepcopy(parent2[point:]),
                     copy.deepcopy(parent2[:point]) + copy.deepcopy(parent1[point:])]
        else:
            child = [copy.deepcopy(parent1), copy.deepcopy(parent2)]
        return child

    def mutation(self, _first, graph_len, child, group_name):
        group_first = copy.deepcopy(_first)
        max_similar = -1.
        nearest_id = None
        child_len = len(child)
        if group_name == 'meta':
            meta_graph_len = self.steps * (self.steps + 1) * self.adj_len // 2
            for i in range(len(group_first)):
                similar = (sum(x == y for x, y in zip(child, group_first[i])) - meta_graph_len + graph_len) / graph_len
                if similar > max_similar:
                    max_similar = similar
                    nearest_id = i
            last_index = meta_graph_len - self.steps * self.adj_len  # 40-4*4=24
            for i in range(child_len):
                if i < last_index:
                    if random.random() < self.pm:
                        if max_similar == 1.0:
                            child[i] = 1 - child[i]
                        else:
                            child[i] = group_first[nearest_id][i]
                else:
                    if i % self.adj_len in self.cstr and random.random() < self.pm:
                        if max_similar == 1.0:
                            child[i] = 1 - child[i]
                        else:
                            child[i] = group_first[nearest_id][i]

        elif group_name == 'gnn':
            gnn_elem_range = len(gnn_search_space)
            for i in range(child_len):
                if random.random() < self.pm:
                    tmp = child[i]
                    while tmp == child[i]:
                        child[i] = random.randint(1, gnn_elem_range)
        else:
            raise
        return child

    def predict_and_select_real(self, group, real_n, group_all, group_all_me, group_all_sp):
        if float('inf') in group_all_me:
            ava_group, ava_me, ava_sp, _, _ = delete_inf_metric(group_all, group_all_me, group_all_sp)
        else:
            ava_group, ava_me, ava_sp = group_all, group_all_me, group_all_sp
        inputs = np.array(ava_group)

        me_targets = np.array(ava_me)
        rf_predictor = RandomForestRegressor()
        rf_predictor.fit(inputs, me_targets)
        me_prediction = rf_predictor.predict(inputs)
        sub_pred_me = []
        for elem in group:
            sub_pred_me.append(rf_predictor.predict(np.array([elem])).item())

        sp_targets = np.array(ava_sp)
        rf_predictor = RandomForestRegressor()
        rf_predictor.fit(inputs, sp_targets)
        sp_prediction = rf_predictor.predict(inputs)
        for i in range(len(sp_prediction)):
            sp_prediction[i] = int(sp_prediction[i])
        sub_pred_sp = []
        for elem in group:
            sub_pred_sp.append(int(rf_predictor.predict(np.array([elem])).item()))

        select_index = self.select_index(real_n, sub_pred_me, sub_pred_sp)
        return sub_pred_me, sub_pred_sp, select_index, me_prediction, me_targets, sp_prediction, sp_targets

    def select_index(self, n, group_me, group_sp):
        group_me_len = len(group_me)
        assert n <= group_me_len
        if group_me_len == n:
            select_index = [x for x in range(0, n)]
            return select_index
        if n <= group_me_len // 2:
            select_length = n * 2
        else:  # >
            select_length = group_me_len
        random_range = []
        while len(random_range) < select_length:
            num = random.randint(0, group_me_len - 1)
            if num not in random_range:
                random_range.append(num)
        select_index = []
        for i in range(0, select_length, 2):
            objs = [[group_me[random_range[i]], group_sp[random_range[i]]],
                    [group_me[random_range[i+1]], group_sp[random_range[i+1]]]]
            objs = np.array(objs)
            num_obj = objs.shape[1]
            if dominate(objs[0], objs[1], num_obj):
                select_index.append(random_range[i])
            elif dominate(objs[1], objs[0], num_obj):
                select_index.append(random_range[i+1])
            else:
                if random.random() < 0.5:
                    select_index.append(random_range[i])
                else:
                    select_index.append(random_range[i+1])
        if n > group_me_len // 2:
            while len(select_index) < n:
                index1 = random.randint(0, group_me_len - 1)
                while index1 in select_index:
                    index1 = random.randint(0, group_me_len - 1)
                index2 = random.randint(0, group_me_len - 1)
                while index2 in select_index or index2 == index1:
                    index2 = random.randint(0, group_me_len - 1)
                objs = [[group_me[random_range[index1]], group_sp[random_range[index1]]],
                        [group_me[random_range[index2]], group_sp[random_range[index2]]]]
                objs = np.array(objs)
                num_obj = objs.shape[1]
                if dominate(objs[0], objs[1], num_obj):
                    select_index.append(random_range[index1])
                elif dominate(objs[1], objs[0], num_obj):
                    select_index.append(random_range[index2])
                else:
                    if random.random() < 0.5:
                        select_index.append(random_range[index1])
                    else:
                        select_index.append(random_range[index2])
        return select_index

    def get_pop_f0(self, pop, pop_me, pop_sp, t):
        _pop, _pop_me, _pop_sp = delete_inf_metric_indi(pop, pop_me, pop_sp)
        pop_f0 = []
        pop_f0_me = []
        pop_f0_sp = []
        pop_size = len(_pop)
        pop_objs = []
        for i in range(pop_size):
            pop_objs.append([_pop_me[i], _pop_sp[i]])
        pop_objs = np.array(pop_objs)
        f, ranks = fast_non_dominate_sort(pop_objs)
        self.logger.info(f"\n  first level size: {len(f[0])}")
        select_index = f[0]
        for index in select_index:
            pop_f0.append(_pop[index])
            pop_f0_me.append(_pop_me[index])
            pop_f0_sp.append(_pop_sp[index])
        for i in range(len(pop_f0)):
            self.logger.info(f"  metric: {pop_f0_me[i]:.6f}, sparsity: {pop_f0_sp[i]:.6f}")
            self.logger.info(f"  arch: {pop_f0[i]}")

        if t == self.arg.T:
            plt.plot(pop_objs[f[0], 1], pop_objs[f[0], 0], 'r*')
            plt.xlabel('Sparsity')
            plt.ylabel('Metric')
            plt.savefig(f'{self.arg.logger_path}/{self.arg.dataset}-{self.arg.time}/Pop-objs-final.png')
        else:
            for i in range(len(f) - 1, -1, -1):
                if i == 0:
                    plt.plot(pop_objs[f[i], 1], pop_objs[f[i], 0], 'r*')
                else:
                    plt.plot(pop_objs[f[i], 1], pop_objs[f[i], 0], 'bo')
            plt.xlabel('Sparsity')
            plt.ylabel('Metric')
            plt.savefig(f'{self.arg.logger_path}/{self.arg.dataset}-{self.arg.time}/Pop-objs-{t}.png')
        plt.close()
        return pop_f0

    def selection(self, pops, offs, groups, subs, pop_me, pop_sp, offs_me, offs_sp):
        _pops = pops + offs
        _group = groups + subs
        _object1 = pop_me + offs_me
        _object2 = pop_sp + offs_sp

        pop_size = len(_pops)
        pop_objs = []
        for i in range(pop_size):
            pop_objs.append([_object1[i], _object2[i]])
        pop_objs = np.array(pop_objs)
        f, ranks = fast_non_dominate_sort(pop_objs)

        select_num = pop_size // 2

        crowd_dis = crowd_distance_sort(pop_objs, f)
        select_index = []
        num = 0
        last_level = []
        select_f = []
        for level in f:
            if len(level) + num <= select_num:
                select_f.append(level)
                for index in level:
                    select_index.append(index)
                num += len(level)
            else:
                last_level = level
                break
        if len(select_index) == select_num:
            pass
        elif len(select_index) < select_num:
            index = np.argsort(crowd_dis[last_level, 0])
            temp = []
            for i in range(select_num - len(select_index)):
                select_index.append(last_level[index[index.shape[0] - i - 1]])
                temp.append(last_level[index[index.shape[0] - i - 1]])
            select_f.append(temp)

        new_pop, new_group, new_ob1, new_ob2 = [], [], [], []
        for index in select_index:
            new_pop.append(_pops[index])
            new_group.append(_group[index])
            new_ob1.append(_object1[index])
            new_ob2.append(_object2[index])

        return new_pop, new_group, new_ob1, new_ob2


def fast_non_dominate_sort(pop_objs):
    pop_size = pop_objs.shape[0]
    num_obj = pop_objs.shape[1]
    dominate_index = [[] for _ in range(pop_size)]
    is_dominated_num = np.zeros([pop_size, 1], dtype=int)
    ranks = np.zeros([pop_size, 1], dtype=int)
    first_level = []
    for i in range(pop_size):
        for j in np.delete(range(pop_size), i):
            if dominate(pop_objs[i], pop_objs[j], num_obj):
                if j not in dominate_index[i]:
                    dominate_index[i].append(j)
            elif dominate(pop_objs[j], pop_objs[i], num_obj):
                is_dominated_num[i] += 1
        if is_dominated_num[i] == 0:
            first_level.append(i)
            ranks[i] = 1
    rank = 1
    f = []
    while first_level:
        f.append(first_level)
        next_level = []
        for i in first_level:
            for j in dominate_index[i]:
                is_dominated_num[j] -= 1
                if is_dominated_num[j] == 0:
                    ranks[j] = rank + 1
                    next_level.append(j)
        rank += 1
        first_level = next_level
    return f, ranks


def dominate(obj_i, obj_j, obj_num):  # i dominates b
    if type(obj_i) is not np.ndarray:
        obj_i, obj_j = np.array(obj_i), np.array(obj_j)
    res = np.array([np.sign(k) for k in obj_i - obj_j])
    res_ngt0, res_eqf1 = np.argwhere(res <= 0), np.argwhere(res == -1)
    if res_ngt0.shape[0] == obj_num and res_eqf1.shape[0] > 0:
        return True
    return False


def crowd_distance_sort(pop_objs, f):
    pop_size = pop_objs.shape[0]
    num_obj = pop_objs.shape[1]
    crowd_dis = np.zeros([pop_size, 1])
    for level in f:  # 1/S
        length = len(level) - 1
        for i in range(num_obj):  # m
            index = np.argsort(pop_objs[level, i])  # SlogS
            sorted_obj = pop_objs[level][index]
            crowd_dis[level[index[0]]] = np.inf
            crowd_dis[level[index[-1]]] = np.inf
            obj_range_fi = sorted_obj[-1, i] - sorted_obj[0, i]
            for j in level:  # S
                k = np.argwhere(np.array(level)[index] == j)[:, 0][0]
                if 0 < index[k] < length:
                    if obj_range_fi == 0:
                        crowd_dis[j] += 0
                    else:
                        crowd_dis[j] += (sorted_obj[index[k] + 1, i] - sorted_obj[index[k] - 1, i]) / obj_range_fi
    return crowd_dis


def choose_one_parent(group, group_me, group_sp):
    count = len(group)
    index1, index2 = random.randint(0, count - 1), random.randint(0, count - 1)
    while index1 == index2:
        index2 = random.randint(0, count - 1)
    objs = [[group_me[index1], group_sp[index1]],
            [group_me[index2], group_sp[index2]]]
    objs = np.array(objs)
    num_obj = objs.shape[1]
    if dominate(objs[0], objs[1], num_obj):
        return index1, group[index1]
    elif dominate(objs[1], objs[0], num_obj):
        return index2, group[index2]
    else:
        if random.random() < 0.5:
            return index1, group[index1]
        else:
            return index2, group[index2]
