import copy
import torch.nn.functional as F
import os, pickle
import torch
import numpy as np
from utils import sparse_mx_to_torch_sparse_tensor, normalize_row, EarlyStoppingLoss
import scipy.sparse as sp
from network_build import HGNN
from sklearn.metrics import f1_score
from torch_geometric.profile import *


def evaluate(output, labels, mask):
    predict = output[mask].max(1)[1].type_as(labels[mask])
    correct = predict.eq(labels[mask]).double()
    acc = correct.sum() / len(labels[mask])
    acc = acc.item()
    return acc


class ModelManager(object):
    def __init__(self, args, logger):
        self.main_args = args
        self.logger = logger
        self.epochs = args.epochs
        self.loss_fn = F.cross_entropy
        self.early_stop_manager = None
        self.is_use_early_stop = True if self.main_args.use_early_stop else False
        self.logger.info(f"Dataset: {args.dataset}")
        self.min_metric = 1e+10
        self.min_metric_indi = None
        self.min_metric_arch_test_acc = 0
        self.min_metric_sparsity = float('inf')
        self.data = {}
        self.params = {}

        # 数据集导入
        datadir = "data"
        prefix = os.path.join(datadir, args.dataset)
        with open(os.path.join(prefix, "node_features.pkl"), "rb") as f:
            node_feats = pickle.load(f)
            f.close()
        self.node_feats = torch.from_numpy(node_feats.astype(np.float32))  # Tensor:(18405, 334)

        node_types = np.load(os.path.join(prefix, "node_types.npy"))  # ndarray:(18405,) paper为0，author为1，conference为2
        self.params['num_node_types'] = max(node_types) + 1  # 3
        self.node_types = torch.from_numpy(node_types)  # 转为Tensor:(18405,)

        with open(os.path.join(prefix, "edges.pkl"), "rb") as f:
            edges = pickle.load(f)  # 共19645*2+14328*2=67946条边
            f.close()
        self.adjs_pt = []
        for mx in edges:
            # 将邻接矩阵添加自连接，后归一化
            self.adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(mx.astype(np.float32) + sp.eye(mx.shape[0], dtype=np.float32))))
        self.adjs_pt.append(sparse_mx_to_torch_sparse_tensor(sp.eye(edges[0].shape[0], dtype=np.float32).tocoo()))  # 添加单位矩阵
        self.params['adj_len'] = len(self.adjs_pt)
        self.logger.info("Loading {} adjs...".format(len(self.adjs_pt)))  # 添加0矩阵

        # 导入标签
        mask = {}
        with open(os.path.join(prefix, "labels.pkl"), "rb") as f:
            labels = pickle.load(f)
            f.close()
        mask['train_idx'] = torch.from_numpy(np.array(labels[0])[:, 0]).type(torch.long)  # 训练集编号 Tensor:(800,) [0,1,...]
        mask['train_target'] = torch.from_numpy(np.array(labels[0])[:, 1]).type(torch.long)  # 训练集标签 Tensor:(800,)
        mask['valid_idx'] = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.long)  # 验证集编号 Tensor:(400,) [800,...]
        mask['valid_target'] = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.long)  # 验证集标签 Tensor:(400,)
        mask['test_idx'] = torch.from_numpy(np.array(labels[2])[:, 0]).type(torch.long)
        mask['test_target'] = torch.from_numpy(np.array(labels[2])[:, 1]).type(torch.long)
        self.mask = mask

        self.params['lr'] = args.lr
        self.params['drop'] = args.drop
        self.params['wd'] = args.wd
        self.params['layer_num'] = args.steps
        self.params['in_dim'] = self.node_feats.size(1)
        self.params['hid_dim'] = args.hid
        self.params['out_dim'] = self.n_classes = mask['train_target'].max().item() + 1
        self.params['layer_agg'] = args.layer_agg
        self.params['act'] = args.act

    def train(self, indi, graph_len):
        meta_graph, gnn_arch = indi[0], indi[1]
        sparsity = sum(meta_graph) / graph_len
        train_epoch = self.epochs
        model = HGNN(meta_graph, gnn_arch, self.params['layer_num'], self.params['adj_len'],
                     self.params['num_node_types'], self.params['in_dim'],  self.params['hid_dim'],
                     self.params['out_dim'],  self.params['drop'], self.params['layer_agg'], self.params['act'])

        stop_epoch = 0
        try:
            early_stop_manager = None
            optimizer = torch.optim.Adam(params=model.parameters(), lr=self.params['lr'], weight_decay=self.params['wd'])
            if self.is_use_early_stop:
                early_stop_manager = EarlyStoppingLoss(patience=self.main_args.early_stop_size,
                                                       min_epochs=self.main_args.epochs // 2)
            model, metric, test_acc, stop_epoch = self.run_model(self.min_metric, self.logger, self.main_args, model,
                                                                 optimizer, self.loss_fn, self.node_feats, self.adjs_pt,
                                                                 self.node_types, self.mask, train_epoch,
                                                                 early_stop_manager)
            if metric < self.min_metric:
                self.min_metric = metric
                self.min_metric_arch_test_acc = test_acc
                self.min_metric_indi = copy.deepcopy(indi)
                self.min_metric_sparsity = sparsity
        except RuntimeError as e:
            metric = float('inf')
            sparsity = float('inf')
            test_acc = 0.0

            if "cpu" in str(e):
                raise
            elif "cuda" in str(e) or "CUDA" in str(e):
                self.logger.info(f"\t we met cuda OOM; error message: {e}")
            else:
                self.logger.info(f"\t other error: {e}")
                raise

        del model
        del optimizer

        torch.cuda.empty_cache()

        return metric, sparsity, test_acc, stop_epoch

    @staticmethod
    def run_model(min_metric, logger, main_args, model, optimizer, loss_fn, node_feats, adjs_pt, node_types, mask,
                  epochs, early_stop=None, show_info=False):
        model.cuda()
        node_feats.cuda()

        best_performance = 0
        best_metric = float("inf")

        stop_epoch = epochs
        for epoch in range(1, epochs + 1):
            model.train()
            logits = model(node_feats, adjs_pt, node_types)
            loss = loss_fn(logits[mask['train_idx']], mask['train_target'].cuda())
            train_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            logits = model(node_feats, adjs_pt, node_types)
            loss = loss_fn(logits[mask['valid_idx']], mask['valid_target'].cuda())
            metric = loss.item()  # use val loss as metric

            test_acc = f1_score(mask['test_target'].cpu().numpy(), torch.argmax(logits[mask['test_idx']], dim=-1).cpu().numpy(), average='macro')
            judge_state = metric < best_metric

            if judge_state:
                best_metric = metric
                best_performance = test_acc
                if best_metric < min_metric:
                    min_metric = best_metric
                    torch.save(model, f'{main_args.logger_path}/{main_args.dataset}-{main_args.time}/model.pth')

            if show_info:
                logger.info(
                    "Epoch {:03d} |Train Loss {:.6f} | Metric {:.6f} | Test_acc {:.6f}".format(
                        epoch, train_loss, metric, best_performance))

            if early_stop is not None:
                early_stop_method = early_stop.on_epoch_end(epoch, metric, train_loss)
                if early_stop_method:
                    stop_epoch = epoch
                    break

        return model, best_metric, best_performance, stop_epoch

    def get_best_indi(self):
        self.logger.info(f"  min metric: {self.min_metric:.6f}, sparsity: {self.min_metric_sparsity:.6f}, test_acc: {self.min_metric_arch_test_acc:.6f}")
        self.logger.info(f"  arch: {self.min_metric_indi}")
