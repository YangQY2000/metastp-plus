import math
import time

import numpy as np
from service_time_prediction.models import MetaSTP
import torch
from tqdm import tqdm



def cal_metrics_new(output, target):
    # can be vectorized
    mae = 0.0
    rmse = 0.0
    mape = 0.0
    smape = 0.0
    for i in range(len(output)):
        abs_err = math.fabs(output[i] - target[i])
        mae += abs_err
        rmse += abs_err * abs_err
        mape += math.fabs((target[i] - output[i]) / target[i])
        smape += math.fabs(2 * (target[i] - output[i]) / (target[i] + output[i]))
    mae = mae / len(output)
    rmse = math.sqrt(rmse / len(output))
    mape = mape * 100 / len(output)
    smape = smape * 100 / len(output)
    return mae, rmse, mape, smape


class MetaAlgo:
    def __init__(self):
        pass

    def meta_train_batch(self, spt_inp, spt_tgt, qry_inp, qry_tgt, meta_inp):
        '''
        :param spt_inp: list of tensor spt_size x F
        :param spt_tgt: list of tensor spt_size x 1
        :param qry_inp: list of tensor qry_size x F
        :param qry_tgt: list of tensor qry_size x 1
        :return:
        '''
        pass

    def meta_test(self, test_set):
        '''
        :param test_set: list of tuple (spt_inp, spt_tgt, qry_inp, qry_tgt)
        :return:
        '''
        pass

    def save(self, save_path):
        pass

    def load(self, save_path):
        pass


class MySNAILMetaAlgo(MetaAlgo):
    def __init__(self, encoder_hidden, encoder_out, snail_params, criterion, tgt_scaler, args, emb_name, ft, att_crt=2):
        super(MySNAILMetaAlgo, self).__init__()
        self.dev = args.dev
        self.t_trainbatch_in = 0.0
        self.t_trainbatch_cal = 0.0
        self.t_trainbatch_eval = 0.0
        self.emb_name = emb_name
        self.ft = ft
        seq_length, key_size, value_size, nb_filters = snail_params
        if args.meta_algo.startswith('MetaSTP'):
            self.model = MetaSTP(encoder_hidden, encoder_out,
                                 seq_length, key_size, value_size, nb_filters, emb_name=self.emb_name,
                                 spatial_hiddens=[4, 2], model_name=args.meta_algo, dev=args.dev, att_crt=att_crt,
                                 ft=ft).to(args.dev)
            # self.model=self.model
        else:
            raise Exception('unknown model')
        self.meta_algo = args.meta_algo
        self.meta_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), args.meta_lr)
        self.criterion = criterion
        self.tgt_scaler = tgt_scaler
        nb_params = 0
        for p in self.model.parameters():
            nb_params += np.prod(p.size())
        print('# of params:{}'.format(nb_params))
        nb_encoder_params = 0
        for p in self.model.encoder.parameters():
            nb_encoder_params += np.prod(p.size())
        print('# of encoder params:{}'.format(nb_encoder_params))
        print('# of meta params:{}'.format(nb_params - nb_encoder_params))

    def meta_train_batch(self, spt_inp, spt_tgt, qry_inp, qry_tgt, meta_inp):
        t_trainbatch_in_1 = time.perf_counter()
        self.model.train()
        nb_tasks = len(spt_inp)

        pred_all = self.model(spt_inp, spt_tgt, qry_inp, meta_inp)
        target_all = torch.vstack(qry_tgt)
        loss = self.criterion(pred_all, target_all)  # 算loss
        self.meta_optim.zero_grad()
        loss.backward()
        self.meta_optim.step()

        t_trainbatch_brk = time.perf_counter()
        self.t_trainbatch_cal += t_trainbatch_brk - t_trainbatch_in_1

        pred_all = self.tgt_scaler.inverse_transform(pred_all.detach().cpu().numpy())
        target_all = self.tgt_scaler.inverse_transform(target_all.detach().cpu().numpy())
        mae, rmse, mape, smape = cal_metrics_new(pred_all, target_all)
        t_trainbatch_in_2 = time.perf_counter()
        self.t_trainbatch_eval += t_trainbatch_in_2 - t_trainbatch_brk
        self.t_trainbatch_in += t_trainbatch_in_2 - t_trainbatch_in_1

        return loss.item(), mae, rmse, mape, smape

    @torch.no_grad()
    def meta_test(self, test_set):
        # spt/qry_inp: seq,agg_quant,agg_cate,y
        # seq: seq_inp list (len:N, element: unit_list, len: nb_units, element: nb_floors x 5)
        # agg_quant_inp: N x 6
        # agg_cate_inp: N x 1
        self.model.eval()
        losses, output_all, target_all = [], [], []
        nb_tasks = len(test_set)
        task_batch_size = 1
        task_idxes = [i for i in range(nb_tasks)]
        nb_iters_per_epoch = math.ceil(nb_tasks / task_batch_size)
        # 每次取一个batch
        for epoch_iter_idx in tqdm(range(nb_iters_per_epoch)):
            iter_tasks = task_idxes[
                         epoch_iter_idx * task_batch_size:min((epoch_iter_idx + 1) * task_batch_size, nb_tasks)]
            tasks = [test_set[task_idx] for task_idx in iter_tasks]
            meta_inp, spt_inp, spt_tgt, qry_inp, qry_tgt = zip(*tasks)

            qry_pred = self.model(spt_inp, spt_tgt, qry_inp, meta_inp)
            qry_tgt = torch.vstack(qry_tgt)
            loss_q = self.criterion(qry_pred, qry_tgt)
            output_all.append(self.tgt_scaler.inverse_transform(qry_pred.detach().cpu().numpy()))
            target_all.append(self.tgt_scaler.inverse_transform(qry_tgt.detach().cpu().numpy()))
            losses.append(loss_q.item())
        output_all = np.concatenate(output_all, 0)
        target_all = np.concatenate(target_all, 0)
        mae, rmse, mape, smape = cal_metrics_new(output_all, target_all)
        loss = sum(losses) / len(losses)
        return loss, mae, rmse, mape, smape

    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def save_ft(self, save_path):
        if "-nE" not in self.meta_algo:
            torch.save(self.model.spatial_encoder.poi_type_embedding, save_path + "poiEmb.pt")
        torch.save(self.model.spatial_encoder.fc, save_path + "fc.pt")

    def load(self, save_path):
        self.model.load_state_dict(torch.load(save_path, map_location=self.dev))
