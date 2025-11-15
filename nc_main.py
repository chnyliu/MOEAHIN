import argparse
import copy
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from model_manager import ModelManager
from preprocess import preprocess
from moccea import Moccea
from utils import Log

parser = argparse.ArgumentParser(description='Implementation of CCMG')
parser.add_argument('--dataset', type=str, default='IMDB', help='DBLP, ACM, IMDB')
parser.add_argument('--logger_path', type=str, default='logs')
parser.add_argument('--time', type=str, default=time.strftime('%Y%m%d-%H%M%S'))
parser.add_argument('--use_early_stop', type=bool, default=True)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--epochs', type=int, default=200)  # 400
parser.add_argument('--early_stop_size', type=int, default=30)
parser.add_argument('--N', type=int, default=100)  # 100
parser.add_argument('--T', type=int, default=20)  # 20
parser.add_argument('--evaluate_N', type=int, default=20)  # 20
parser.add_argument('--pc', type=float, default=0.9)
parser.add_argument('--pm', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
parser.add_argument('--drop', type=float, default=0.6, help='dropout rate')
parser.add_argument('--hid', type=int, default=64, help='hidden dimension')
parser.add_argument('--layer_agg', type=str, default='sum', help='layer aggregator')
parser.add_argument('--act', type=str, default='gelu', help='activation function')
parser.add_argument('--steps', type=int, default=4, help='number of intermediate states in the meta graph')  # 元图中的中间状态数
args = parser.parse_args()
if not os.path.exists(args.logger_path):
    os.mkdir(args.logger_path)


def init_process(args, logger):
    if not torch.cuda.is_available():
        logger.info('no gpu device available')
        sys.exit(1)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True


def train_and_valid(arg, logger, model_manager, pops, graph_len):
    vals_me = []
    vals_sp = []
    tests_acc = []
    indi_id = 0

    n = len(pops)
    for i in range(n):
        indi = copy.deepcopy(pops[i])
        np.random.seed(arg.seed)
        torch.manual_seed(arg.seed)
        torch.cuda.manual_seed_all(arg.seed)

        metric, sparsity, test_score, stop_epoch = model_manager.train(indi, graph_len)
        logger.info(
            f"\t arch_{indi_id + 1}, metric: {metric:.6f}, sparsity: {sparsity:.6f}, test_score: {test_score:.6f}, stop_epoch: {stop_epoch}")
        vals_me.append(metric)
        vals_sp.append(sparsity)
        tests_acc.append(test_score)
        indi_id += 1

    return model_manager, vals_me, vals_sp, tests_acc


def main():
    res = []
    preprocess(args.dataset)
    for _ in range(5):
        args.time = time.strftime('%Y%m%d-%H%M%S')
        args.seed = random.randint(0, 10000)
        logger = Log(args)
        init_process(args, logger)
        logger.info(f"==== begin_time: {args.time}")
        logger.info(f"args: {args}")
        logger.info(f"args use early stop: {args.use_early_stop}")
        logger.info(f"Pop size:{args.N}; Generation:{args.T}; Seed:{args.seed}")

        manager = ModelManager(args, logger)
        ccea = Moccea(logger, args, manager.params['adj_len'])

        ins = []
        ins_me = []
        ins_sp = []
        ins_ta = []
        meta_all = []
        meta_all_me = []
        meta_all_sp = []
        gnn_all = []
        gnn_all_me = []
        gnn_all_sp = []

        logger.info("++++++++++ " + time.strftime('%Y%m%d-%H%M%S'))
        logger.info('========== Randomly generating individual pool')
        pops, metas, gnns, graph_len = ccea.initial_pop(args.N)

        manager, pop_me, pop_sp, pop_ta = train_and_valid(args, logger, manager, pops, graph_len)

        ins += pops
        ins_me += pop_me
        ins_sp += pop_sp
        ins_ta += pop_ta

        meta_all += metas
        meta_all_me += pop_me
        meta_all_sp += pop_sp

        gnn_all += gnns
        gnn_all_me += pop_me
        gnn_all_sp += pop_sp

        logger.info(f"---- record the best arch")
        manager.get_best_indi()

        for generation in range(args.T-1):
            pop_f0 = ccea.get_pop_f0(copy.deepcopy(pops), copy.deepcopy(pop_me), copy.deepcopy(pop_sp), generation)
            logger.info(f"========== Search generation {generation + 1} start ==========")
            if (generation + 1) % 2:
                logger.info(f"---- evolve the meta graph group")
                logger.info(f"---- crossover and mutation to generate architecture pool")
                subs, offs = ccea.crossover_and_mutation(args.N, ins, pop_f0, graph_len, metas, pop_me, pop_sp, 'meta')
                (sub_pred_me, sub_pred_sp, select_index, me_pred, me_targ,
                 sp_pred, sp_targ) = ccea.predict_and_select_real(subs, args.evaluate_N, meta_all, meta_all_me,
                                                                          meta_all_sp)
            else:
                logger.info(f"---- evolve the gnn graph group")
                logger.info(f"---- crossover and mutation to generate group pool")
                subs, offs = ccea.crossover_and_mutation(args.N, ins, pop_f0, graph_len, gnns, pop_me, pop_sp, 'gnn')
                (sub_pred_me, sub_pred_sp, select_index, me_pred, me_targ,
                 sp_pred, sp_targ) = ccea.predict_and_select_real(subs, args.evaluate_N, gnn_all, gnn_all_me,
                                                                          gnn_all_sp)
            real_subs = []
            real_offs, surr_offs = [], []
            surr_subs = []
            surr_sub_pred_me, surr_sub_pred_sp = [], []
            for i in range(args.N):
                if i in select_index:
                    real_offs.append(offs[i])
                    real_subs.append(subs[i])
                else:
                    surr_offs.append(offs[i])
                    surr_subs.append(subs[i])
                    surr_sub_pred_me.append(sub_pred_me[i])
                    surr_sub_pred_sp.append(sub_pred_sp[i])
            logger.info("---- evaluate the selected real-evaluated group")

            manager, real_sub_me, real_sub_sp, real_sub_ta = train_and_valid(args, logger, manager, real_offs, graph_len)

            if (generation + 1) % 2:
                meta_all += real_subs
                meta_all_me += real_sub_me
                meta_all_sp += real_sub_sp
            else:
                gnn_all += real_subs
                gnn_all_me += real_sub_me
                gnn_all_sp += real_sub_sp

            ins += real_offs
            ins_me += real_sub_me
            ins_sp += real_sub_sp
            ins_ta += real_sub_ta


            logger.info("---- select the next population")
            offs = surr_offs + real_offs
            subs = surr_subs + real_subs
            sub_me = surr_sub_pred_me + real_sub_me
            sub_sp = surr_sub_pred_sp + real_sub_sp
            if (generation + 1) % 2:
                pops, metas, pop_me, pop_sp = ccea.selection(pops, offs, metas, subs, pop_me, pop_sp, sub_me, sub_sp)
            else:
                pops, gnns, pop_me, pop_sp = ccea.selection(pops, offs, gnns, subs, pop_me, pop_sp, sub_me, sub_sp)

            logger.info(f"---- record the best arch")
            manager.get_best_indi()
        _ = ccea.get_pop_f0(copy.deepcopy(pops), copy.deepcopy(pop_me), copy.deepcopy(pop_sp), args.T)
        logger.info(f"==== end_time:{time.strftime('%Y%m%d-%H%M%S')}")

        res.append(manager.min_metric_arch_test_acc)
        logger.info(f"==== evaluated individuals len: {len(ins)}")
    print(f'result: {np.mean(res):.6f}+/-{np.std(res):.6f}')
    print(res)


if __name__ == "__main__":
    main()
