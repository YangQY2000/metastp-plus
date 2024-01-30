import math
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch


def read_data(data_root, ds_name):
    Y_scaler = torch.load(data_root + ds_name + '_bb_Yscaler_new.pth')

    train_set = torch.load(data_root + ds_name + '_bbf_train_new.pth')
    val_set = torch.load(data_root + ds_name + '_bbf_val_new.pth')

    # torch.from_numpy(X_agg_quant_scaler.transform())

    print("train_set.type=", type(train_set))
    print("len(train_set)=", len(train_set))
    train_set = [train_set[task_idx] for task_idx in range(len(train_set))]
    print("train_set.type=", type(train_set))
    meta_inp, spt_inp, spt_tgt, qry_inp, qry_tgt = zip(*train_set)
    spt_tgt, qry_tgt, meta_inp, spt_inp, qry_inp = list(spt_tgt), list(qry_tgt), list(meta_inp), list(spt_inp), list(
        qry_inp)
    _, avgs_out, raw_spt_tgt = [], [], []

    return spt_tgt, qry_tgt, meta_inp, spt_inp, qry_inp, Y_scaler


class GetRawNbCustomers():
    def __init__(self, bins, std_raw, min_raw, thrd=0.1):
        self.bins = bins
        self.std_raw = std_raw
        self.min_raw = min_raw
        self.thrd = thrd

    def toraw(self, after):
        float_tmp = (after - self.min_raw) / self.std_raw + 1
        raw = round(float_tmp)
        if math.fabs(float_tmp - float(raw)) > self.thrd:
            print("toraw error!")
            return -100
        else:
            return raw


def deal(spt_inp):
    random.seed = 100
    values = set()
    raw_nb = []
    for loc in spt_inp:
        units_seq, delv, timeslot = loc
        nb_customers_loc = delv[:, 1]
        nb_customers_loc = nb_customers_loc.detach().numpy().tolist()
        raw_nb.extend(nb_customers_loc)
        values.update(nb_customers_loc)
    mean_raw = sum(raw_nb) / len(raw_nb)
    print(
        f"mean={mean_raw},std={math.sqrt(sum([(raw_nb[i] - mean_raw) ** 2 for i in range(len(raw_nb))]) / (len(raw_nb) - 1))}")
    raw_nb = sorted(raw_nb, reverse=False)
    values = list(values)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(320, 90))
    plt.cla()

    ax[0].plot([25, -1], [0, 0])
    ax[0].scatter(x=raw_nb, y=[random.randint(0, 10000) / 1000. if i > 0 else 0. for i in range(len(raw_nb))])

    ax[1].plot([25, -1], [0, 0])
    y = [random.randint(0, 10000) / 1000. if i > 0 else 0. for i in range(len(values))]
    ax[1].scatter(x=values, y=y)

    for x0, y0 in zip(values, y):
        ax[1].plot([x0, x0], [0, y0])

    plt.savefig("inv_nb_customers")
    plt.show()

    values = sorted(values, reverse=False)

    delta_values = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    # print(delta_values)

    # print(len(values))
    # print(values)

    std_raw = sum(delta_values[:20]) / 20.
    min_raw = values[0]

    bins = [min_raw - std_raw / 2. + i * std_raw for i in range(55)]
    return bins, std_raw, min_raw


def new_avgs(spt_tgt, spt_inp, qry_tgt, qry_inp, Y_scaler, dev=None, ds='I3'):
    if dev is None:
        dev = torch.device('cpu')
    with open("{}_toraw.pkl".format(ds), 'rb') as f:
        toraw = pickle.load(f)
    for i in range(len(spt_inp)):
        spt_tgt[i] = torch.cat([spt_tgt[i], qry_tgt[i]], dim=0)
        spt_inp[i] = list(spt_inp[i])
        if isinstance(spt_inp[i][0], torch.Tensor):
            spt_inp[i][0] = [].extend(qry_inp[i][0])
        else:
            spt_inp[i][0].extend(qry_inp[i][0])
        spt_inp[i][1] = torch.cat([spt_inp[i][1], qry_inp[i][1]], dim=0)
        spt_inp[i][2] = spt_inp[i][2].type(dtype=torch.int64)
        spt_inp[i][2] = torch.cat([spt_inp[i][2], qry_inp[i][2]], dim=0)
        spt_inp[i] = tuple(spt_inp[i])
    raw_spt_tgt = []
    for i in range(len(spt_inp)):
        if len(spt_tgt[i]) > 0:
            raw_spt_tgt.append(Y_scaler.inverse_transform(spt_tgt[i].detach().cpu().numpy()))
        else:
            raw_spt_tgt.append(torch.tensor(data=[], dtype=torch.float))
            continue
    weighted_time = []
    raw_avgs = []

    d_cnt = 0

    for loc, tgt in zip(spt_inp, raw_spt_tgt):
        # print("d_cnt=",d_cnt)
        # if d_cnt==194:
        #     input("debug pause!")

        if len(tgt) > 0:
            units_seq, delv, timeslot = loc
            nb_customers_loc = delv[:, 1]
            nb_customers_loc = nb_customers_loc.detach().numpy()
            for i in range(len(nb_customers_loc)):
                nb_customers_loc[i] = toraw.toraw(nb_customers_loc[i])
            raw_avgs.append(np.mean(tgt / nb_customers_loc))
        else:
            raw_avgs.append(0.)
        d_cnt += 1

    raw_avgs = np.array(raw_avgs).reshape(-1, 1)  # np.concatenate(avgs,axis=0)
    X_avgs_scaler = StandardScaler()
    avgs = X_avgs_scaler.fit_transform(raw_avgs)
    std_z = torch.tensor(np.std(raw_avgs)).to(dev)
    # X_avgs_scaler.transform(avgs)
    avgs = [torch.tensor(avgs[i], dtype=torch.float).to(dev) for i in range(len(avgs))]
    return avgs, std_z, raw_avgs


def main(ds="I3"):
    data_root = '../data/result_DurPred/'

    dist_thresh = 20
    behavior_min_trips = 10
    behavior_min_rate = 0.8
    gt_cluster_dist_thresh = 30
    merge_cluster_dist_thresh = 10
    min_delvs = 2
    min_conf = 0.51
    min_stay_time = 30
    dlinf_level = 'build'
    filter_dist = 50

    ds_name = 'deliveries_{}_{}_{}_S{}_R{}_{}_Dgt{}_merged_D{}_LQ{}_{}_filter{}'.format(
        ds, dist_thresh, min_stay_time, behavior_min_trips, behavior_min_rate, dlinf_level, gt_cluster_dist_thresh,
        merge_cluster_dist_thresh, min_delvs, min_conf, filter_dist)

    spt_tgt, qry_tgt, meta_inp, spt_inp, qry_inp, Y_scaler = read_data(data_root=data_root, ds_name=ds_name)

    bins, std_raw, min_raw = deal(spt_inp=spt_inp)

    toraw = GetRawNbCustomers(bins=bins, std_raw=std_raw, min_raw=min_raw)

    with open("{}_toraw.pkl".format(ds), 'wb') as f:
        pickle.dump(toraw, f)


def try_process(ds="I3"):
    data_root = '../data/result_DurPred/'

    dist_thresh = 20
    behavior_min_trips = 10
    behavior_min_rate = 0.8
    gt_cluster_dist_thresh = 30
    merge_cluster_dist_thresh = 10
    min_delvs = 2
    min_conf = 0.51
    min_stay_time = 30
    dlinf_level = 'build'
    filter_dist = 50

    ds_name = 'deliveries_{}_{}_{}_S{}_R{}_{}_Dgt{}_merged_D{}_LQ{}_{}_filter{}'.format(
        ds, dist_thresh, min_stay_time, behavior_min_trips, behavior_min_rate, dlinf_level, gt_cluster_dist_thresh,
        merge_cluster_dist_thresh, min_delvs, min_conf, filter_dist)

    spt_tgt, qry_tgt, meta_inp, spt_inp, qry_inp, Y_scaler = read_data(data_root=data_root, ds_name=ds_name)
    new_avgs(spt_tgt, spt_inp, qry_tgt, qry_inp, Y_scaler, ds=ds)


if __name__ == '__main__':
    main(ds='I3')
    try_process(ds='I3')
