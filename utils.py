import logging, os, sys
import torch
import numpy as np
import scipy.sparse as sp
import copy

cstr_nc = {
    "DBLP": [1, 4],  # [1],
    "ACM": [0, 2, 4],
    "IMDB": [0, 2, 4]
}
gnn_search_space = ['GCN', 'GAT_1', 'GAT_4', 'SAGE_MAX', 'SAGE_MEAN', 'EDGE']


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_row(mx):
    """Row-normalize sparse matrix"""  # 行归一化稀疏矩阵
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx.tocoo()


class Log(object):
    def __init__(self, args):
        self._logger = None
        self.save = args.logger_path
        self.dataset = args.dataset
        self.time = args.time
        self.__get_logger()

    def __get_logger(self):
        if self._logger is None:
            logger = logging.getLogger("CIP")
            logger.handlers.clear()
            formatter = logging.Formatter('%(message)s')
            if not os.path.exists(f'{self.save}/{self.dataset}-{self.time}'):
                os.mkdir(f'{self.save}/{self.dataset}-{self.time}')
            save_name = f'{self.save}/{self.dataset}-{self.time}/{self.time}-{self.dataset}.txt'
            file_handler = logging.FileHandler(save_name)
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            self._logger = logger
            return logger
        else:
            return self._logger

    def info(self, _str):
        self.__get_logger().info(_str)

    def warn(self, _str):
        self.__get_logger().warning(_str)


class Callback:
    def __init__(self): pass

    def on_train_begin(self, *args, **kwargs): pass

    def on_train_end(self, *args, **kwargs): pass

    def on_epoch_begin(self, *args, **kwargs): pass

    def on_epoch_end(self, *args, **kwargs): pass

    def on_batch_begin(self, *args, **kwargs): pass

    def on_batch_end(self, *args, **kwargs): pass

    def on_loss_begin(self, *args, **kwargs): pass

    def on_loss_end(self, *args, **kwargs): pass

    def on_step_begin(self, *args, **kwargs): pass

    def on_step_end(self, *args, **kwargs): pass


class EarlyStoppingLoss(Callback):
    def __init__(self, patience=30, tol=0.001, min_epochs=200):
        super(EarlyStoppingLoss, self).__init__()
        self.patience = patience
        self.tol = tol
        self.best = np.inf
        self.best_epoch = -1
        self.wait = 0
        self.stopped_epoch = -1
        self.min_epochs = min_epochs

    def on_epoch_end(self, epoch, metric, epoch_loss):
        metric = max(0, metric - self.tol)

        if metric < self.best:
            self.best = min(metric + self.tol, self.best)
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience and epoch > self.min_epochs:
                self.stopped_epoch = epoch
                return True
        return False

def delete_inf_metric(group_elems, group_metric, group_sparsity, group_pred_metric=None,
                      group_pred_sparsity=None):  # //
    _group, _group_metric = copy.deepcopy(group_elems), copy.deepcopy(group_metric)
    _group_sparsity = copy.deepcopy(group_sparsity)
    if group_pred_metric is not None:
        _group_pred_metric = copy.deepcopy(group_pred_metric)
        _group_pred_sparsity = copy.deepcopy(group_pred_sparsity)
    available_group = []
    available_metric, available_pred_metric = [], []
    available_sparsity, available_pred_sparsity = [], []
    _group_len = len(_group)
    for i in range(_group_len):
        if _group_metric[i] != float('inf'):
            available_group.append(_group[i])
            available_metric.append(_group_metric[i])
            available_sparsity.append(_group_sparsity[i])
            if group_pred_metric is not None:
                available_pred_metric.append(_group_pred_metric[i])
                available_pred_sparsity.append(_group_pred_sparsity[i])
    return available_group, available_metric, available_sparsity, available_pred_metric, available_pred_sparsity

def delete_inf_metric_indi(ins, ins_me, ins_sp):
    _ins = copy.deepcopy(ins)
    _ins_me = copy.deepcopy(ins_me)
    _ins_sp = copy.deepcopy(ins_sp)
    ava_ins, ava_me, ava_sp = [], [], []
    _ins_len = len(_ins)
    for i in range(_ins_len):
        if _ins_me[i] != float('inf'):
            ava_ins.append(_ins[i])
            ava_me.append(_ins_me[i])
            ava_sp.append(_ins_sp[i])
    return ava_ins, ava_me, ava_sp
