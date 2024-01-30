# coding=utf-8
import sys
import os

import warnings

warnings.filterwarnings("ignore")
sys.path.append('..')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt

from service_time_prediction.meta_algos import MySNAILMetaAlgo
from service_time_prediction.learning_utils import GOATLogger, pad_collate_seq

import argparse
import torch
import torch.nn as nn
import numpy as np
import random

import time
import math
import json



def cal_max(inp):
    f_max = 0
    for loc in inp:
        for del_task in loc[0]:
            t_floor_n = 0
            for unit in del_task:
                t_floor_n += unit.shape[0]
            if t_floor_n > f_max:
                f_max = t_floor_n

    return f_max




def pre_process(data_set, dev=None):
    data_set = [data_set[t] for t in range(len(data_set))]
    meta_inp, spt_inp, spt_tgt, qry_inp, qry_tgt = zip(*data_set)

    spt_max = cal_max(spt_inp)
    qry_max = cal_max(qry_inp)

    print(f'spt_max_unit={spt_max},qry_max_unit={qry_max}')

    seq_inp_spt = [pad_collate_seq(t[0]) for t in spt_inp]
    seq_inp_qry = [pad_collate_seq(t[0]) for t in qry_inp]
    for t_idx in range(len(seq_inp_qry)):
        for i in range(len(seq_inp_qry[t_idx])):
            if i == 0:
                for j in range(len(seq_inp_qry[t_idx][i])):
                    seq_inp_qry[t_idx][i][j] = seq_inp_qry[t_idx][i][j].to(dev)
            else:
                for j in range(len(seq_inp_qry[t_idx][i])):
                    for k in range(len(seq_inp_qry[t_idx][i][j])):
                        seq_inp_qry[t_idx][i][j][k] = seq_inp_qry[t_idx][i][j][k]

    for t_idx in range(len(seq_inp_spt)):
        for i in range(len(seq_inp_spt[t_idx])):
            if i == 0:
                for j in range(len(seq_inp_spt[t_idx][i])):
                    seq_inp_spt[t_idx][i][j] = seq_inp_spt[t_idx][i][j].to(dev)
            else:
                for j in range(len(seq_inp_spt[t_idx][i])):
                    for k in range(len(seq_inp_spt[t_idx][i][j])):
                        seq_inp_spt[t_idx][i][j][k] = seq_inp_spt[t_idx][i][j][k]



    meta_inp, spt_inp, spt_tgt, qry_inp, qry_tgt = [tuple([x.to(dev) for x in t]) for t in meta_inp], \
        [tuple([spt_inp[t_idx][i].to(dev) if i != 0 else
                seq_inp_spt[t_idx] for i in
                range(len(spt_inp[t_idx]))]) for t_idx in
         range(len(spt_inp))], \
        [x.to(dev) for x in spt_tgt], \
        [tuple([qry_inp[t_idx][i].to(dev) if i != 0 else
                seq_inp_qry[t_idx] for i in
                range(len(qry_inp[t_idx]))]) for t_idx in
         range(len(qry_inp))], \
        [t.to(dev) for t in qry_tgt]


    data_set = (meta_inp, spt_inp, spt_tgt, qry_inp, qry_tgt)


    return data_set  # meta_inp, spt_inp, spt_tgt, qry_inp, qry_tgt


def buquan(data_set=None, NumMean=16, h_emb_thd=0.047, l_emb_thd=-1.0, set_mode='meta_train', dev=torch.device('cpu'),
           do_buquan=True, ds='I3', beta=1.2, alpha=8.0, pred_time=False, embenc=None, online=False,
           args=None, raw=False):  # 20. 1e-4 -1.
    global embeds
    dev = dev  # torch.device('cpu')
    # global data_set
    if data_set is None:
        print("read data_set_{}".format(f'_bbf_{set_mode.split("-")[-1]}_new.pth'))
        data_set = torch.load(args.data_root + args.ds_name + f'_bbf_{set_mode.split("-")[-1]}_new.pth')
        meta_inp, spt_inp, spt_tgt, qry_inp, qry_tgt = pre_process(data_set, dev=dev)

    else:
        meta_inp, spt_inp, spt_tgt, qry_inp, qry_tgt = data_set
    if raw:
        emb_name = f"embeds_raw_{ds}_{set_mode}.pth"
    else:
        emb_name = "{}_embeds{}_{}_ae_class_sep_seed255_lr0.001_batchsize512_beta{}_alpha{}.pth"
    if not online:
        if not raw:
            embeds = torch.load(
                emb_name.format(ds, '_Pred' if pred_time else '', set_mode,  beta, alpha),
                map_location=dev).to(dev)
        else:
            embeds = torch.load(emb_name, map_location=dev).to(dev)
    else:
        if embenc is not None:
            quant_inp, cate_inp = zip(*meta_inp)
            quant_inp, cate_inp = list(quant_inp), list(cate_inp)
            for i in range(len(cate_inp)):
                cate_inp[i] = cate_inp[i].reshape(1)
            quant_inp, cate_inp = torch.vstack(quant_inp), torch.cat(cate_inp, dim=0)
            embeds = embenc((quant_inp, cate_inp))
        else:
            print("embenc None!")

    print("emb_name=", emb_name)
    row_embeds = embeds.repeat(len(embeds), 1, 1)
    col_embeds = torch.unsqueeze(embeds, dim=1).repeat(1, len(embeds), 1)

    dis_mat = torch.norm(row_embeds - col_embeds, dim=-1, keepdim=False)
    dis_mat += torch.diag_embed(1. + torch.max(dis_mat).repeat(len(embeds)))

    mostlikely = torch.argmin(dis_mat, dim=1)
    mostlikely = mostlikely.cpu().detach().numpy()
    spt_size = [len(spt_inp[i][0][0]) for i in range(len(spt_inp))]


    print("avg spt_size=", np.mean(spt_size))
    # input()
    deltaNum = []

    for i in range(len(mostlikely)):


        if do_buquan and spt_size[i] < NumMean and h_emb_thd >= dis_mat[i][mostlikely[i]] > l_emb_thd:  # :False   # :
            spt_inp[i] = list(spt_inp[i])
            spt_inp[i][0] = list(spt_inp[i][0])


            spt_inp[i][0][0].extend(spt_inp[mostlikely[i]][0][0][:min(len(spt_inp[mostlikely[i]][0][0]), max(spt_size[i]
                                                                                                             ,
                                                                                                             NumMean))])
            spt_inp[i][0][1].extend(spt_inp[mostlikely[i]][0][1][:min(len(spt_inp[mostlikely[i]][0][1]), max(spt_size[i]
                                                                                                             ,
                                                                                                             NumMean))])

            spt_inp[i][1] = torch.cat([spt_inp[i][1], spt_inp[mostlikely[i]][1][:min(len(spt_inp[mostlikely[i]][1]),
                                                                                     max(spt_size[i], NumMean))]],
                                      dim=0)
            spt_inp[i][2] = torch.cat([spt_inp[i][2], spt_inp[mostlikely[i]][2][:min(len(spt_inp[mostlikely[i]][2]),
                                                                                     max(spt_size[i], NumMean))]],
                                      dim=0)
            spt_inp[i][2] = torch.tensor(spt_inp[i][2], dtype=torch.int64)
            spt_tgt[i] = torch.cat((spt_tgt[i], spt_tgt[mostlikely[i]][:min(max(spt_size[i], NumMean),
                                                                            len(spt_inp[mostlikely[i]][1]))]), dim=0)

            deltaNum.append(min(max(spt_size[i], NumMean), len(spt_inp[mostlikely[i]][1])))
            spt_inp[i][0] = tuple(spt_inp[i][0])
            spt_inp[i] = tuple(spt_inp[i])

    print("Len deltaNum=", len(deltaNum))
    print(deltaNum)

    data_set = zip(meta_inp, spt_inp, spt_tgt, qry_inp, qry_tgt)
    data_set = list(data_set)


    return data_set
    # print("hi")


def main():
    """
    Initialize everything and train
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='default')
    parser.add_argument('--n_epoch', type=int, default=8, help="How many times for task set iteration")
    parser.add_argument('--data_root', type=str, help="Path of data")
    parser.add_argument('--ds', type=str, default='I3', required=False)
    parser.add_argument('--n_eval', type=int, default=5, help="How many examples for evaluation")
    parser.add_argument('--meta_lr', type=float, default=3e-3)
    parser.add_argument('--task_batch_size', type=int, default=4)
    parser.add_argument('--val-freq', type=int, default=12, help="Validation frequency")
    parser.add_argument('--meta_algo', type=str, default='MetaSTP+2A+C+P', help="Algorithms")  #
    # model-based representation learning params
    parser.add_argument('--encoder_hidden', type=int, help='representation hidden dim', default=8)
    parser.add_argument('--encoder_out', type=int, help='representation out dim', default=8)
    parser.add_argument('--seq_length', type=int, help='seq_length', default=10)
    parser.add_argument('--key_size', type=int, help='key_size', default=8)
    parser.add_argument('--value_size', type=int, help='value_size', default=8)
    parser.add_argument('--nb_filters', type=int, help='nb_filters', default=16)

    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--model_id', type=str, help="Only for test")
    parser.add_argument('--seed', type=int, default=255, help="Seed, must set for test")

    parser.add_argument('--gpu_id', type=int, help="if set, specify gpu id, else use cpu")
    parser.add_argument('--exp_set', type=str, default='att-errs-keep')

    parser.add_argument('--train_h_thd', type=float, default=0.2)  # 1e-4 0.36 0.7093  0.9893 0.047
    parser.add_argument('--train_l_thd', type=float, default=-1.0)
    parser.add_argument('--val_h_thd', type=float, default=0.2)  # 1e-4  0.7093 0.36
    parser.add_argument('--val_l_thd', type=float, default=-1.0)
    parser.add_argument('--test_h_thd', type=float, default=0.2)  # 1e-4  0.7093 0.36
    parser.add_argument('--test_l_thd', type=float, default=-1.0)
    parser.add_argument('--trainNumMean', type=int, default=24)
    parser.add_argument('--valNumMean', type=int, default=24)
    parser.add_argument('--testNumMean', type=int, default=24)
    parser.add_argument('--beta', type=float, default=1.2)
    parser.add_argument('--alpha', type=float, default=8.)
    parser.add_argument('--thd', type=float, default=0.2)
    parser.add_argument('--num', type=int, default=14)
    parser.add_argument('--thr', type=int, default=10)

    args = parser.parse_args()
    args.trainNumMean = args.num
    args.valNumMean = args.num
    args.testNumMean = args.num
    args.train_h_thd = args.thd
    args.val_h_thd = args.thd
    args.test_h_thd = args.thd

    torch.set_num_threads(args.thr)

    do_buquan = True if '+C' in args.meta_algo else False
    pred_time = False if '-nPr' in args.meta_algo else True

    args.exp_set += f'  emb_para=(beta={args.beta},alpha={args.alpha}) emb pred ' \
                    f'NumMean={args.trainNumMean, args.valNumMean, args.testNumMean},'

    # tr_l_d={args.train_l_thd}
    if args.mode == 'train':
        args.model_id = time.strftime("%Y%m%d%H%M%S")
    print('exp_set={}'.format(args.exp_set))

    if args.seed is None:
        args.seed = random.randint(0, int(1e3))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.exp_set += f',seed={args.seed}'

    raw_feature = False
    if "+P" in args.meta_algo:
        pass
    else:
        raw_feature = True

    if not args.gpu_id is None:
        print("True gpu_id=", args.gpu_id, sep="")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        args.dev = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        print("False gpu_id=", args.gpu_id, sep="")
        args.dev = torch.device('cpu')

    args.data_root = '../data/result_DurPred/'
    base_dir = '../data/'
    with open(os.path.join(base_dir, 'params.json'), 'r') as f:
        params = json.load(f)
    dist_thresh, min_stay_time = params['preprocessing']['dist_thresh'], params['preprocessing']['min_stay_time']
    behavior_min_trips = 10
    behavior_min_rate = 0.8
    gt_cluster_dist_thresh = 30
    merge_cluster_dist_thresh = 10
    min_delvs = 2
    min_conf = 0.51

    dlinf_level = 'build'
    filter_dist = 50
    args.ds_name = 'deliveries_{}_{}_{}_S{}_R{}_{}_Dgt{}_merged_D{}_LQ{}_{}_filter{}'.format(
        args.ds, dist_thresh, min_stay_time, behavior_min_trips, behavior_min_rate, dlinf_level, gt_cluster_dist_thresh,
        merge_cluster_dist_thresh, min_delvs, min_conf, filter_dist)

    stop_iter = 58000 if args.ds == 'I3' and pred_time else 5000
    emb_prefix = args.ds + "_{}"
    emb_name = "{}_meta_train_ae_class_sep_seed255_lr0.001_batchsize512_beta{}_alpha{}.pth"
    emb_name = emb_prefix + emb_name.format('_Pred' if pred_time else '', args.beta, args.alpha)

    Y_scaler = torch.load(args.data_root + args.ds_name + '_bb_Yscaler_new.pth')
    criterion = nn.MSELoss().to(args.dev)
    if args.meta_algo.startswith('MetaSTP'):
        meta_algo = MySNAILMetaAlgo(args.encoder_hidden, args.encoder_out,
                                    (args.seq_length, args.key_size, args.value_size, args.nb_filters), criterion,
                                    Y_scaler, args, emb_name=emb_name, ft=args.ft)  # att_crt=att_crt,
        all_hyperparams = (
            args.n_eval, args.encoder_hidden, args.encoder_out, args.seq_length, args.key_size, args.value_size,
            args.nb_filters)
        # meta_algo=meta_algo.to(args.dev)
    else:
        raise Exception('Not Imple')

    args.save_path = os.path.join(args.data_root,
                                  'saved_model/{}/Meta/{}/H{}_seed{}_{}/'.format(
                                      args.ds_name, args.meta_algo, all_hyperparams, args.seed, args.model_id))
    os.makedirs(args.save_path, exist_ok=True)
    logger = GOATLogger(args)

    if args.mode == 'test':
        test_set = torch.load(args.data_root + args.ds_name + '_bbf_test_new.pth')
        meta_algo.load(args.save_path + 'final_model.pt')
        loss, mae, rmse, mape, smape = meta_algo.meta_test(test_set)
        print(
            "Test loss:{}\tTest MAE:{}\tTest RMSE:{}\tTest MAPE:{}\tTest SMAPE:{}".format(loss, mae, rmse, mape, smape))
        return
    print("train_set_raw=")
    train_set_raw = torch.load(args.data_root + args.ds_name + '_bbf_train_new.pth')
    train_set_raw = pre_process(train_set_raw, dev=args.dev)
    print("val_set_raw=")
    val_set_raw = torch.load(args.data_root + args.ds_name + '_bbf_val_new.pth')
    val_set_raw = pre_process(val_set_raw, dev=args.dev)
    print("test_set_raw=")
    test_set_raw = torch.load(args.data_root + args.ds_name + '_bbf_test_new.pth')
    test_set_raw = pre_process(test_set_raw, dev=args.dev)

    best_mae = float('inf')
    logger.loginfo("Start training")
    # epoch: 所有meta-train tasks过多少次
    # task_batch_size: 每个iter有多少个task
    # iter: 第x个batch的tasks
    # nb_tasks = len(train_set)

    # train_set=
    print("train_set_init.buquan=")
    train_set = buquan(train_set_raw, NumMean=args.trainNumMean, h_emb_thd=args.train_h_thd, l_emb_thd=args.train_l_thd,
                       dev=args.dev, beta=args.beta, alpha=args.alpha, pred_time=pred_time,
                       set_mode='meta_train', do_buquan=do_buquan, ds=args.ds, online=False,
                       raw=raw_feature)
    print("val_set_init.buquan=")
    val_set = buquan(val_set_raw, NumMean=args.valNumMean, h_emb_thd=args.val_h_thd, l_emb_thd=args.val_l_thd,
                     dev=args.dev, beta=args.beta, alpha=args.alpha, pred_time=pred_time,
                     set_mode='meta_val', do_buquan=do_buquan, ds=args.ds, online=False,
                     raw=raw_feature)
    print("test_set_init.buquan=")
    test_set = buquan(test_set_raw, NumMean=args.testNumMean, h_emb_thd=args.test_h_thd, l_emb_thd=args.test_l_thd,
                      dev=args.dev, beta=args.beta, alpha=args.alpha, pred_time=pred_time,
                      set_mode='meta_test', do_buquan=do_buquan, ds=args.ds, online=False,
                      raw=raw_feature)

    nb_tasks = len(train_set)
    nb_iters_per_epoch = math.ceil(nb_tasks / args.task_batch_size)
    tot_iters = nb_iters_per_epoch * args.n_epoch
    stop_f = False

    val_loss_draw, val_mae_draw, val_rmse_draw, val_mape_draw, x_iter = [], [], [], [], []
    # delete

    for epoch_idx in range(args.n_epoch):
        # shuffle tasks

        task_idxes = np.random.RandomState(seed=random.randint(0, int(1e3))).permutation(nb_tasks)
        # 每次取一个batch
        for epoch_iter_idx in range(nb_iters_per_epoch):
            iter_idx = epoch_idx * nb_iters_per_epoch + epoch_iter_idx
            iter_tasks = task_idxes[
                         epoch_iter_idx * args.task_batch_size:min((epoch_iter_idx + 1) * args.task_batch_size,
                                                                   nb_tasks)]
            tasks = [train_set[task_idx] for task_idx in iter_tasks]
            meta_inp, spt_inp, spt_tgt, qry_inp, qry_tgt = zip(*tasks)

            loss, mae, rmse, mape, smape = meta_algo.meta_train_batch(spt_inp, spt_tgt, qry_inp, qry_tgt, meta_inp)
            # t_trainbatch_outer_2 = time.perf_counter()
            # t_trainbatch_outer = t_trainbatch_outer_2 - t_trainbatch_outer_1
            print(
                "[{:4d}/{:4d}] loss: {:6.4f}, mae: {:6.3f}, rmse: {:6.3f}, mape: {:6.3f}, smape: {:6.3f}".format(
                    iter_idx, tot_iters, loss, mae, rmse, mape, smape))


            if iter_idx % args.val_freq == 0 and iter_idx != 0:


                loss, mae, rmse, mape, _ = meta_algo.meta_test(val_set)
                print(
                    "{} Iter idx:{}\tEval loss:{}\t{}\t{}\t{}".format( "val",
                                                                      iter_idx, loss, mae, rmse,
                                                                      mape))

                val_loss_draw.append(loss), val_mae_draw.append(mae), val_rmse_draw.append(rmse), val_mape_draw.append(
                    mape), x_iter.append(iter_idx)


                if mae < best_mae:
                    print('* Best mae so far *\n')
                    best_mae = mae
                    meta_algo.save('{}/final_model.pt'.format(args.save_path))


                logger.loginfo(
                    "{} Iter idx:{}\tEval loss:{} \t{} \t{} \t{}".format("val",
                                                                        iter_idx, loss, mae, rmse,
                                                                        mape))


                stop_f = False

            if stop_f:
                break

        print()
        if stop_f:
            break
    logger.loginfo('Done')


    meta_algo.load(args.save_path + 'final_model.pt')
    test_loss, test_mae, test_rmse, test_mape, test_smape = meta_algo.meta_test(test_set)
    test_result = f'{args.meta_algo} --ds {args.ds} {args.exp_set}\t{test_mae}\t{test_rmse}\t{test_mape}\t{test_loss}\t--thd {args.thd} --num {args.num} --rebu {args.rebu} --n_epoch {args.n_epoch} --batchsize {args.task_batch_size} --seed {args.seed} {args.model_id}\n'
    logger.loginfo(test_result)
    args.exp_set += f',tr_v_t_thd={args.train_h_thd, args.val_h_thd, args.test_h_thd},n_epoch={args.n_epoch},batchsize' \
                    f'={args.task_batch_size}'
    logger.loginfo(args.exp_set)
    with open('./exp_result.csv', 'a') as f:
        f.write(test_result)




if __name__ == '__main__':
    main()
