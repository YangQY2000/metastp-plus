# -*- coding: utf-8 -*-
import math
import random
import sys
import os

sys.path.append('..')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from service_time_prediction.inv_nb_cumers import new_avgs
import service_time_prediction.models as models
from sklearn.preprocessing import StandardScaler



def loaddata(data_root, ds_name, set_mode='meta_train', useTime=True, dev=None, ds='I3'):
    global post_fix
    Y_scaler = torch.load(data_root + ds_name + '_bb_Yscaler_new.pth')
    if set_mode == 'meta_train':
        post_fix = '_bbf_train_new.pth'
    elif set_mode == 'meta_val':
        post_fix = '_bbf_val_new.pth'
    elif set_mode == 'meta_test':
        post_fix = '_bbf_test_new.pth'
    data_set = torch.load(data_root + ds_name + post_fix)
    # val_set = torch.load(data_root + ds_name + '_bbf_val_new.pth')

    # torch.from_numpy(X_agg_quant_scaler.transform())

    print("train_set.type=", type(data_set))
    print("len(train_set)=", len(data_set))
    data_set = [data_set[task_idx] for task_idx in range(len(data_set))]
    print("train_set.type=", type(data_set))
    meta_inp, spt_inp, spt_tgt, qry_inp, qry_tgt = zip(*data_set)
    # print(spt_tgt)
    # input()
    spt_tgt, qry_tgt, meta_inp, spt_inp, qry_inp = list(spt_tgt), list(qry_tgt), list(meta_inp), list(spt_inp), list(
        qry_inp)

    # print(spt_tgt)
    # input()
    avgs, avgs_out, raw_spt_tgt = [], [], []
    new_flag = True
    if new_flag:
        avgs, std_z, raw_avgs = new_avgs(spt_tgt=spt_tgt, spt_inp=spt_inp, qry_inp=qry_inp, qry_tgt=qry_tgt,
                                         Y_scaler=Y_scaler, dev=dev, ds=ds)
    else:
        for i in range(len(spt_tgt)):
            spt_tgt[i] = torch.cat([spt_tgt[i], qry_tgt[i]], dim=0).to(dev)
            raw_spt_tgt.append(Y_scaler.inverse_transform(spt_tgt[i].detach().cpu().numpy()))
            avgs.append(sum(spt_tgt[i]) / len(spt_tgt[i]))
            avgs_out.append(torch.tensor(sum(raw_spt_tgt[i]) / len(spt_tgt[i])))

        # spt_tgt 进行逆变换，
        # spt_tgt=torch.vstack(spt_tgt)
        avgs_out = torch.vstack(avgs_out)
        torch.save(avgs_out, "avgs.pth")
        std_z = torch.std(avgs_out)
    # 显示一下tgt的值

    # 整体配对并存储，或者返回
    if useTime:
        for i in range(len(meta_inp)):
            meta_inp[i] = list(meta_inp[i])
            meta_inp[i][0] = torch.cat([meta_inp[i][0], avgs[i]], dim=0)
            meta_inp[i] = tuple(meta_inp[i])
    if dev == torch.device('cuda:0'):
        for i in range(len(meta_inp)):
            meta_inp[i] = list(meta_inp[i])
            meta_inp[i][0] = meta_inp[i][0].to(dev)
            meta_inp[i][1] = meta_inp[i][1].to(dev)
    return meta_inp, spt_inp, spt_tgt, qry_inp, avgs, std_z


def raw_feature(data_root, ds_name, set_mode, use_time, dev, ds):
    meta_inp, _, _, _, avgs, std_z = loaddata(data_root=data_root, ds_name=ds_name, set_mode=set_mode,
                                              useTime=use_time, dev=dev, ds=ds)
    # train_set = meta_inp
    quant_inp, cate_inp = zip(*meta_inp)  # Llist)
    quant_inp, cate_inp = list(quant_inp), list(cate_inp)

    for i in range(len(cate_inp)):
        cate_inp[i] = cate_inp[i].reshape(-1,1)
    quant_inp, cate_inp = torch.stack(quant_inp, dim=0), torch.cat(cate_inp, dim=0)
    cate_inp_scaler = StandardScaler()
    cate_inp = cate_inp_scaler.fit_transform(cate_inp)
    raw_feats = torch.cat([quant_inp, torch.tensor(cate_inp)], dim=-1)
    torch.save(raw_feats, f"embeds_raw_{ds}_{set_mode}.pth")
    return True


class AutoEncoder(nn.Module):
    def __init__(self, inp_dim, nb_poi_class=18, poi_emb_dim=2, encoder_size=None, decoder_size=None,
                 classifier_size=None,
                 embedding_size=2, negative_slope=0.2, lr=1e-4, likely_threshold=0.1, std_z=0.1, Y_scaler=None,
                 ds_name="I3", batch_size=32, class_sep=True, poi_class=True, useTime=True, pred_time=True,
                 sep_prd=True, dev=None):
        super().__init__()
        self.class_sep = class_sep
        self.poi_class = poi_class
        self.useTime = useTime
        self.std_z = std_z
        self.outp_dim = inp_dim
        self.sep_prd = sep_prd
        self.dev = dev
        self.pred_time = pred_time
        if self.dev is None:
            self.dev = torch.device('cpu')
        if useTime:
            inp_dim += 1
        if self.pred_time:
            self.outp_dim += 1
            if self.sep_prd:
                self.outp_dim -= 1
                self.pred_hidden = [4, 8, 4, 2, 1]
        else:
            self.pred_hidden = None
        if decoder_size is None:
            if poi_class:
                if class_sep:
                    decoder_size = [8, 16, 8, self.outp_dim]
                else:
                    decoder_size = [8, 16, 8, nb_poi_class + inp_dim]
            else:
                decoder_size = [8, 16, 8, inp_dim + poi_emb_dim]

        if encoder_size is None:
            encoder_size = [16, 8, 4, 2]
        if classifier_size is None:
            classifier_size = [8, 16, nb_poi_class]
        self.poi_embedding = nn.Embedding(num_embeddings=nb_poi_class, embedding_dim=poi_emb_dim).to(self.dev)
        self.encoder = models.MLP(inp_dim=inp_dim + poi_emb_dim, hidden_dims=encoder_size, out_act=True,
                                  negative_slope=negative_slope, act_type='LeakyReLU').to(self.dev)
        self.decoder = models.MLP(inp_dim=embedding_size, hidden_dims=decoder_size, out_act=False,
                                  act_type='LeakyReLU', negative_slope=negative_slope).to(self.dev)
        self.poi_classifier = models.MLP(inp_dim=embedding_size, hidden_dims=classifier_size, out_act=True,
                                         negative_slope=negative_slope, act_type='LeakyReLU').to(self.dev)
        if self.pred_time and self.sep_prd:
            self.mean_time_pred = models.MLP(inp_dim=embedding_size, hidden_dims=self.pred_hidden, out_act=False,
                                             act_type='LeakyReLU', negative_slope=negative_slope).to(self.dev)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-4)
        self.Y_scaler = Y_scaler
        self.inp_dim = inp_dim
        self.poi_emb_dim = poi_emb_dim
        self.nb_poi_class = nb_poi_class
        self.unit = torch.eye(self.nb_poi_class).to(self.dev)

    def forward(self, x):
        global output, loss_reconst, loss_reconst_unsum, loss_poi_class, preds
        quant_inp, cate_inp = x

        poi_class_tgt = torch.index_select(self.unit, dim=0, index=cate_inp)

        x_embeded = torch.cat([quant_inp, self.poi_embedding(cate_inp)], dim=-1)

        embedding = self.encoder(x_embeded)

        if not self.poi_class:
            output = self.decoder(embedding)
            loss_reconst_unsum = F.mse_loss(x_embeded, output, reduction='none')  # (inp, outp)
            loss_reconst = F.mse_loss(x_embeded, output, reduction='sum')
            loss_poi_class = torch.tensor([], dtype=torch.float)
        elif self.poi_class:
            if self.class_sep:
                quant_outp = self.decoder(embedding)
                poi_class_outp = self.poi_classifier(embedding)
                output = torch.cat([quant_outp, poi_class_outp], dim=-1)
                if self.pred_time:
                    if self.sep_prd:
                        preds = self.mean_time_pred(embedding)
                    else:
                        preds = quant_outp[:, -1]
                else:
                    preds = torch.tensor([], dtype=torch.float)
            else:
                output = self.decoder(embedding)
                quant_outp, poi_class_outp = output[:self.inp_dim, :], output[self.inp_dim:, :]
            if self.pred_time and not self.pred_hidden:
                preds = quant_outp[:, -1]
                quant_outp = quant_outp[:, :-1]

            loss_reconst = F.mse_loss(quant_inp, quant_outp, reduction='sum')
            loss_reconst_unsum = F.mse_loss(quant_inp, quant_outp, reduction='none')  # (inp, outp)
            loss_poi_class = F.cross_entropy(poi_class_outp, poi_class_tgt, reduction='sum')



        return embedding, output, loss_reconst, preds, torch.sum(loss_reconst_unsum, dim=0), loss_poi_class  # pred_time


def train_batch(inp, ans, model):
    model.train()

    quant_inp, cate_inp = zip(*inp)  # Llist)
    # quant_inpn, cate_inpn = zip(*Lnlist)
    quant_inp, cate_inp = list(quant_inp), list(cate_inp)  # quant_inpn,, cate_inpn, list(quant_inpn), list(cate_inpn)
    # 与排序后的对齐
    ans = enumerate(ans)


    idxs, ans = zip(*ans)
    idxs, ans = list(idxs), list(ans)


    quant_inp = [quant_inp[idx] for idx in idxs]
    cate_inp = [cate_inp[idx] for idx in idxs]
    for i in range(len(cate_inp)):
        cate_inp[i] = cate_inp[i].reshape(1)

    quant_inp, cate_inp = torch.stack(quant_inp, dim=0), torch.cat(cate_inp, dim=0)  # , quant_inpn, cate_inpn

    embs, recons_outp, loss_reconst, m_srv_t, loss_reconst_unsum, loss_poi_class = model((quant_inp, cate_inp))


    flag_srv_t = True if model.pred_time else False
    if flag_srv_t:

        ans = torch.cat(ans)

        loss_srv_t = F.mse_loss(torch.squeeze(m_srv_t), ans, reduction='sum')

        m_srv_t_raw = model.Y_scaler.inverse_transform(m_srv_t.detach().cpu().numpy())
        ans_raw = model.Y_scaler.inverse_transform(ans.reshape(-1, 1).detach().cpu().numpy())
        raw_rmse = np.var(m_srv_t_raw - ans_raw)
    else:
        loss_srv_t = torch.tensor(0., dtype=torch.float)
        raw_rmse = numpy.array([])
    return loss_srv_t, loss_reconst, embs, m_srv_t, loss_reconst_unsum, recons_outp, loss_poi_class, raw_rmse


@torch.no_grad()
def test_batch(inp, ans, model):
    global m_srv_t_raw, ans_raw
    model.eval()

    quant_inp, cate_inp = zip(*inp)  # Llist)
    # quant_inpn, cate_inpn = zip(*Lnlist)
    quant_inp, cate_inp = list(quant_inp), list(cate_inp)  # quant_inpn,, cate_inpn, list(quant_inpn), list(cate_inpn)
    # 与排序后的对齐
    ans = enumerate(ans)

    idxs, ans = zip(*ans)
    idxs, ans = list(idxs), list(ans)


    quant_inp = [quant_inp[idx] for idx in idxs]
    cate_inp = [cate_inp[idx] for idx in idxs]
    for i in range(len(cate_inp)):
        cate_inp[i] = cate_inp[i].reshape(1)

    quant_inp, cate_inp = torch.stack(quant_inp, dim=0), torch.cat(cate_inp, dim=0)  # , quant_inpn, cate_inpn
    # torch.stack(quant_inpn, dim=0), torch.cat(cate_inpn, dim=0)
    embs, recons_outp, loss_reconst, m_srv_t, loss_reconst_unsum, loss_poi_class = model((quant_inp, cate_inp))

    print("test:\n", loss_reconst_unsum)

    # TODO loss2的计算放到这里来做
    flag_srv_t = True if model.pred_time else False
    if flag_srv_t:

        ans = torch.cat(ans)

        loss_srv_t = F.mse_loss(torch.squeeze(m_srv_t), ans, reduction='sum')

        m_srv_t_raw = model.Y_scaler.inverse_transform(m_srv_t.detach().cpu().numpy())
        ans_raw = model.Y_scaler.inverse_transform(ans.reshape(-1, 1).detach().cpu().numpy())
        raw_rmse = np.std(m_srv_t_raw - ans_raw, ddof=0, keepdims=False)
    else:
        loss_srv_t = torch.tensor(0., dtype=torch.float)
        raw_rmse = numpy.array([])
        m_srv_t_raw = numpy.array([])
        ans_raw = numpy.array([])
    return loss_srv_t, loss_reconst, embs, m_srv_t, loss_reconst_unsum, recons_outp, loss_poi_class, raw_rmse, m_srv_t_raw, ans_raw


def main(batch_size=512, n_epoch=1002, seed=256, alpha=1., testfreq=1, print_train=True, print_test=True,
         negative_slope=0.2, lr=1e-3, print_grad=False, stopiter=2000, p_threshold=30, class_sep=True,
         beta=0.2, test_flag='meta_test', use_time=True, pred_time=True, dev=torch.device('cuda:0'), dataset_name='I3'):
    global mtest_meta_inp, mval_m_srv_t_raw, mval_ans_raw, mtest_m_srv_t_raw, mtest_ans_raw
    data_root = '../data/result_DurPred/'
    ds = dataset_name
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
    Y_scaler = torch.load(data_root + ds_name + '_bb_Yscaler_new.pth')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    meta_inp, _, _, _, avgs, std_z = loaddata(data_root=data_root, ds_name=ds_name, set_mode='meta_train',
                                              useTime=use_time, dev=dev, ds=ds)

    mval_meta_inp, _, _, _, mval_avgs, mval_std_z = loaddata(data_root=data_root, ds_name=ds_name, set_mode=test_flag,
                                                             useTime=use_time, dev=dev, ds=ds)
    # spt_inp spt_tgt qry_inp

    direct_flag = False
    if direct_flag:
        embeds = torch.tensor(avgs, dtype=torch.float)
        embeds = embeds.unsqueeze(-1)
        ones = torch.ones_like(embeds, dtype=torch.float)
        embeds = torch.cat((embeds, ones / 10.), dim=-1)
        torch.save(embeds, "embeds.pth")
        input()
        return

    # dataset split
    train_set = meta_inp
    std_ans_train = avgs

    test_set = mval_meta_inp
    std_ans_test = mval_avgs

    # parameters
    likely_threshold = 0.1
    # model init
    model = AutoEncoder(inp_dim=4, batch_size=batch_size, lr=lr,
                        negative_slope=negative_slope, likely_threshold=likely_threshold,
                        std_z=std_z.detach(), Y_scaler=Y_scaler, poi_class=True, class_sep=class_sep, useTime=use_time,
                        dev=dev, pred_time=pred_time)
    dicte = model.state_dict()
    names = []
    nb_tasks = len(train_set)
    print(f"nb_tasks=len(train_set)={len(train_set)}")
    nb_iters_per_epoch = math.ceil(nb_tasks / batch_size)
    tot_iters = nb_iters_per_epoch * n_epoch

    loss_train_reconst_unsumed, loss_test_reconst_unsumed, loss_test_poi_class_numpy, loss_test_srv_t_numpy, test_t_r_rmse_numpy = [], [], [], [], []
    embs_var, recons_outp_square = [], []
    x_train_ax, x_test_ax = [], []

    stop_flag = False

    for epoch_idx in range(n_epoch):
        # shuffle tasks
        # print(f"epoch_idx={epoch_idx} in n_epoch={n_epoch}")
        task_idxes = np.random.RandomState(seed=random.randint(0, int(1e3))).permutation(nb_tasks)  # random.randint(
        # 0, int(1e3))
        # 每次取一个batch

        for epoch_iter_idx in range(nb_iters_per_epoch):
            # print(f"\t{epoch_iter_idx} in {nb_iters_per_epoch}")
            iter_idx = epoch_idx * nb_iters_per_epoch + epoch_iter_idx
            iter_tasks = task_idxes[
                         epoch_iter_idx * batch_size:min((epoch_iter_idx + 1) * batch_size,
                                                         nb_tasks)]
            tasks = [train_set[task_idx] for task_idx in iter_tasks]
            std_ans_train_batch = [std_ans_train[task_idx] for task_idx in iter_tasks]
            # --------
            loss_srv_t, loss_rescont, embeds, preds, loss_rescont_unsumed, recons_outp, loss_poi_class, t_r_rmse = train_batch(
                tasks,
                std_ans_train_batch,
                model)
            loss_numpy = loss_rescont_unsumed.detach().cpu().numpy()
            loss_train_reconst_unsumed.append(loss_numpy.tolist())
            x_train_ax.append(iter_idx)

            loss = alpha * loss_srv_t + loss_rescont + beta * loss_poi_class
            # print(f"\tafter data processing train epoch {epoch_idx} batch {epoch_iter_idx}")

            if print_train:
                print(
                    f"\ttrain epoch {epoch_idx} batch {epoch_iter_idx} : loss={loss},loss_recons={loss_rescont},loss_time={loss_srv_t},loss_poi={loss_poi_class}")

            loss.backward()
            if print_grad:
                xx = 0
                for p in model.parameters():
                    if p.grad is not None:
                        print(f"{names[xx]} grad:\n{p.grad.data}\n")
                    xx += 1
            model.optimizer.step()
            model.optimizer.zero_grad()
            if iter_idx == stopiter:
                # meta_train

                class_sep_str = "class_sep_" if class_sep else ""
                torch.save(model.encoder,
                           "{}_spatEncoder{}{}_meta_train_ae_{}seed{}_lr{}_batchsize{}_beta{}_alpha{}.pth". \
                           format(ds, '_Pred' if pred_time else '',
                                  '_Time' if use_time else '', class_sep_str,
                                  seed, lr,
                                  batch_size, beta, alpha))
                torch.save(model.poi_embedding,
                           "{}_poiEmb{}{}_meta_train_ae_{}seed{}_lr{}_batchsize{}_beta{}_alpha{}.pth". \
                           format(ds, '_Pred' if pred_time else '',
                                  '_Time' if use_time else '', class_sep_str,
                                  seed, lr,
                                  batch_size, beta, alpha))
                loss_test_srv_t, loss_test_rescont, embeds, preds, loss_rescont_unsumed, recons_outp, loss_poi_class, mtrain_t_r_rmse, _, _ = test_batch(
                    meta_inp, avgs,
                    model)  # train_set, std_ans_train

                embed_name = "{}_embeds{}{}_meta_train_ae_{}seed{}_lr{}_batchsize{}_beta{}_alpha{}.pth".format(
                    ds, '_Pred' if pred_time else '',
                    '_Time' if use_time else '', class_sep_str,
                    seed, lr,
                    batch_size, beta, alpha
                )
                torch.save(embeds, embed_name)

                # meta_val
                mval_meta_inp, _, _, _, mval_avgs, mval_std_z = loaddata(data_root=data_root, ds_name=ds_name, ds=ds,
                                                                         useTime=use_time, set_mode='meta_val', dev=dev)
                loss_mval_srv_t, loss_mval_rescont, mval_embeds, mval_preds, loss_mval_rescont_unsumed, mval_recons_outp, loss_mval_poi_class, mval_t_r_rmse, mval_m_srv_t_raw, mval_ans_raw = test_batch(
                    mval_meta_inp, mval_avgs,
                    model)  # train_set, std_ans_train

                mval_embed_name = "{}_embeds{}{}_meta_val_ae_{}seed{}_lr{}_batchsize{}_beta{}_alpha{}.pth".format(
                    ds, '_Pred' if pred_time else '',
                    '_Time' if use_time else '', class_sep_str,
                    seed, lr,
                    batch_size,
                    beta, alpha)
                torch.save(mval_embeds, mval_embed_name)

                # meta_test
                mtest_meta_inp, _, _, _, mtest_avgs, mtest_std_z = loaddata(data_root=data_root, ds_name=ds_name,
                                                                            useTime=use_time, set_mode='meta_test',
                                                                            dev=dev, ds=ds)
                loss_mtest_srv_t, loss_mtest_rescont, mtest_embeds, mtest_preds, loss_mtest_rescont_unsumed, mtest_recons_outp, loss_mtest_poi_class, mtest_t_r_rmse, mtest_m_srv_t_raw, mtest_ans_raw = test_batch(
                    mtest_meta_inp, mtest_avgs,
                    model)  # train_set, std_ans_train

                mtest_embed_name = "{}_embeds{}{}_meta_test_ae_{}seed{}_lr{}_batchsize{}_beta{}_alpha{}.pth".format(
                    ds, '_Pred' if pred_time else '',
                    '_Time' if use_time else '',
                    class_sep_str,
                    seed, lr,
                    batch_size,
                    beta, alpha)
                torch.save(mtest_embeds, mtest_embed_name)

                stop_flag = True
                break
                # embeds_all=test_batch()
                # break
            if iter_idx != 0 and (iter_idx % testfreq == 0):  # (not epoch_iter_idx == 0)epoch_epoch_
                # print(f"go test {epoch_iter_idx},test_set size={len(test_set)}")
                loss_test_srv_t, loss_test_rescont, embeds, preds, loss_rescont_unsumed, recons_outp, loss_test_poi_class, test_t_r_rmse, _, _ = test_batch(
                    test_set,
                    std_ans_test,
                    model)
                loss_numpy = loss_rescont_unsumed.detach().cpu().numpy()
                loss_test_reconst_unsumed.append(loss_numpy.tolist())
                x_test_ax.append(iter_idx)
                embeds_numpy = embeds.detach().cpu().numpy()
                embeds_x_y = embeds_numpy.T
                embs_var.append((np.var(embeds_x_y[0]), np.var(embeds_x_y[1])))
                recons_outp_numpy = recons_outp.detach().cpu().numpy()
                recons_outp_square.append(np.mean(np.square(recons_outp_numpy), axis=0).tolist())
                loss_test_poi_class_numpy.append(loss_test_poi_class.detach().cpu().numpy())
                loss_test_srv_t_numpy.append(loss_test_srv_t.detach().cpu().numpy())
                test_t_r_rmse_numpy.append(test_t_r_rmse)

                loss_test = loss_test_rescont + alpha * loss_test_srv_t + beta * loss_test_poi_class
                if print_test:
                    print(
                        f"test epoch {epoch_idx} batch:{epoch_iter_idx}: loss={loss_test},loss_rescont={loss_test_rescont},loss_time={loss_test_srv_t},loss_poi_class={loss_test_poi_class}")
        if stop_flag:
            print("train finished!")
            break
    loss_test_draw = list(zip(*loss_test_reconst_unsumed))
    loss_train_draw = list(zip(*loss_train_reconst_unsumed))
    loss_test_srv_t_draw = list(loss_test_srv_t_numpy)
    embs_var_draw = list(zip(*embs_var))
    recons_outp_sq_draw = list(zip(*recons_outp_square))
    # recons_grndtrus_numpy = np.mean(np.square(np.array(test_set.cpu())), axis=0)
    loss_test_poi_class_draw = list(loss_test_poi_class_numpy)
    test_t_r_rmse_draw = list(test_t_r_rmse_numpy)


    colors = ["red", "blue", "green", "skyblue", "grey", "black"]
    labels_train = ["train dim" + str(i) for i in range(model.inp_dim + model.poi_emb_dim)]
    labels_test = ["test dim" + str(i) for i in range(model.inp_dim + model.poi_emb_dim)]

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(120, 40))
    plt.cla()

    dim_draw_outp = model.inp_dim if model.poi_class else model.inp_dim + model.poi_emb_dim

    for t in range(dim_draw_outp):
        ax[0, 0].plot(x_train_ax, loss_train_draw[t], color=colors[t], label=labels_train[t])
        ax[0, 1].plot(x_test_ax, loss_test_draw[t], color=colors[t], label=labels_test[t])
        ax[0, 2].plot(x_test_ax, recons_outp_sq_draw[t], color=colors[t], label=labels_test[t])
        # ax[3].
    for t in range(2):
        ax[1, 0].plot(x_test_ax, embs_var_draw[t], color=colors[t], label=labels_test[t])

    for t in range(1):  # model.nb_poi_class
        ax[1, 1].plot(x_test_ax, loss_test_poi_class_draw, color='yellow', label=labels_test[0])

    for t in range(1):  # model.nb_poi_class
        # ax[1, 2].plot(x_test_ax, loss_test_srv_t_draw, color='blue', label=labels_test[0])
        ax[1, 2].plot(x_test_ax, test_t_r_rmse_draw, color='blue', label=labels_test[0])

    plt.xlabel("iters")
    plt.ylabel("loss")
    plt.legend()

    classify_str = "poi_C" if model.poi_class else "poi_N"
    classify_str += "_Sep_" if model.poi_class and model.class_sep else "_Nsep_"
    plt.savefig(
        "ae_{}_{}loss_var_seed{}_lr{}_batchsize{}_beta{}_alpha{}.jpg".format(ds, classify_str, seed, lr, batch_size, beta,
                                                                                        alpha))
    plt.show()

    plt.cla()
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(320, 90))
    val_idx = [i for i in range(len(mval_meta_inp))]
    test_idx = [i for i in range(len(mtest_meta_inp))]
    ax[0].scatter(x=val_idx, y=mval_m_srv_t_raw, c='r')
    ax[0].scatter(x=val_idx, y=mval_ans_raw, c='g')
    for x0, y0 in zip(val_idx, mval_ans_raw):
        ax[0].plot([x0, x0], [0, y0], color='skyblue')

    ax[1].scatter(x=test_idx, y=mtest_m_srv_t_raw, c='r')
    ax[1].scatter(x=test_idx, y=mtest_ans_raw, c='g')
    for x0, y0 in zip(test_idx, mtest_ans_raw):
        ax[1].plot([x0, x0], [0, y0], color='skyblue')
    plt.show()


if __name__ == "__main__":
    # main(batch_size=512, n_epoch=100002, seed=255, alpha=6., testfreq=100, print_train=True, print_test=True,
    #     negative_slope=0.2, lr=1e-3, print_grad=False, stopiter=58000, p_threshold=30, class_sep=True,
    #     beta=1., test_flag='meta_train', use_time=False, pred_time=False, dev=torch.device('cpu'), dataset_name='I3')

    dataset_name = 'O3'
    data_root = '../data/result_DurPred/'
    ds = dataset_name
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

    raw_feature(data_root=data_root, ds_name=ds_name, set_mode="meta_train", use_time=True, dev=torch.device("cpu"),
                ds=ds)
    raw_feature(data_root=data_root, ds_name=ds_name, set_mode="meta_val", use_time=True, dev=torch.device("cpu"),
                ds=ds)
    raw_feature(data_root=data_root, ds_name=ds_name, set_mode="meta_test", use_time=True, dev=torch.device("cpu"),
                ds=ds)
