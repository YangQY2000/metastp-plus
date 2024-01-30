import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from service_time_prediction.blocks import QSAttentionBlock1, SelfAttentionBlock, TCBlock, QSAttentionBlock2
from service_time_prediction.ae_embdding import AutoEncoder


class MLP(nn.Module):
    def __init__(self, inp_dim, hidden_dims, act_type, out_act, dev=None, negative_slope=0.2):
        """
        :param inp_dim: input dim
        :param hidden_dims: list of hidden units (include the last layer)
        :param act_type: activation name
        :param out_act: whether use activation after the last layer
        """
        self.device = dev if dev else torch.device('cpu')
        super(MLP, self).__init__()
        layers = []
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(hidden_dims[i - 1] if i != 0 else inp_dim, h))
            activation = None if i == len(hidden_dims) - 1 and not out_act else act_type
            if activation is not None:
                if activation != "LeakyReLU":
                    layers.append(getattr(nn, activation)())
                else:
                    layers.append(nn.LeakyReLU(negative_slope=negative_slope))
        self.mlp = nn.Sequential(*layers).to(device=self.device)

    def forward(self, x):
        return self.mlp(x)


class SpatialKnowledgeEncoder(nn.Module):  # 空间先验知识编码
    def __init__(self, spatial_hiddens, use_gid=True, withE=True, withPtr=False, emb_name='', dev=None, ft=True):
        super(SpatialKnowledgeEncoder, self).__init__()
        self.withE = withE
        self.withPtr = withPtr
        self.use_gid = use_gid
        self.device = dev if dev else torch.device('cpu')
        quant_dim = 0
        poi_embed_dim = 0
        poi_embed_name = emb_name.format("poiEmb")
        encoder_name = emb_name.format("spatEncoder")
        if self.withE:
            nb_poi_class = 18
            poi_embed_dim = 2
            if self.withPtr:
                self.poi_type_embedding = torch.load(poi_embed_name)
                for p in self.poi_type_embedding.parameters():
                    p.requires_grad = ft
            else:
                self.poi_type_embedding = nn.Embedding(nb_poi_class, poi_embed_dim).to(device=self.device)
            quant_dim += 2
        if use_gid:
            quant_dim += 2
        if self.withPtr:
            self.fc = torch.load(encoder_name)
            for p in self.fc.parameters():
                p.requires_grad = ft
        else:
            self.fc = MLP(poi_embed_dim + quant_dim, spatial_hiddens, act_type='ReLU', out_act=True, dev=self.device)

        for para_name, para in self.named_parameters():
            print(f"{para_name}:\t{para.size()},{para.requires_grad}")

    def forward(self, meta_inp):  # 空间先验知识编码

        quant_inp, cate_inp = meta_inp

        # not check no e and not use grid
        if self.withE:
            if self.use_gid:
                fused_inp = torch.cat([quant_inp, self.poi_type_embedding(cate_inp)], dim=-1)
            else:
                fused_inp = torch.cat([quant_inp[-2:], self.poi_type_embedding(cate_inp)], dim=-1)
        else:
            if self.use_gid:
                fused_inp = quant_inp[:-2]
            else:
                raise Exception('undefined behavior')

        # print("fused_inp.size=", fused_inp.size())

        tmp = self.fc(fused_inp)
        return tmp


class DeliveryEventEncoder(nn.Module):
    def __init__(self, hidden_dim, out_dim,
                 use_t=True, use_seq=True, use_w=True, dropout=0.1, dev=None):
        super(DeliveryEventEncoder, self).__init__()
        self.use_t = use_t
        self.use_seq = use_seq
        self.use_w = use_w
        self.device = dev if dev else torch.device('cpu')
        if not self.use_w:
            self.use_seq = False

        if self.use_w:
            agg_quant_inp_dim = 6
        else:
            agg_quant_inp_dim = 0

        self.agg_feature_idx = {
            'parcel': [0, 6],
            'temporal': [6, 7]
        }

        if self.use_t:
            self.nb_tod_classes = 5
            tod_embed_dim = 3
            self.tod_embedding = nn.Embedding(self.nb_tod_classes, tod_embed_dim).to(device=self.device)
            agg_quant_inp_dim += self.agg_feature_idx['temporal'][1] - self.agg_feature_idx['temporal'][0]
        else:
            tod_embed_dim = 0

        if self.use_seq:
            seq_inp_dim = 5
            self.inp_embedding = nn.Linear(seq_inp_dim, hidden_dim, device=self.device)  # .to(device=self.device)
            self.nb_heads, self.nb_layers = 1, 1
            encoder_layers = TransformerEncoderLayer(hidden_dim, self.nb_heads, hidden_dim, dropout, device=self.device)
            self.transformer_encoder = TransformerEncoder(encoder_layers, self.nb_layers).to(device=self.device)
            self.unit_fc = nn.Sequential(
                nn.Linear(hidden_dim, 16),
                nn.ReLU()).to(device=self.device)
            self.fc = nn.Sequential(
                nn.Linear(16 + agg_quant_inp_dim + tod_embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
                nn.ReLU()
            ).to(device=self.device)
        else:
            self.fc = nn.Sequential(
                nn.Linear(agg_quant_inp_dim + tod_embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
                nn.ReLU()
            ).to(device=self.device)

    def forward(self, inp):
        (x_seq, data_length), x_agg_quant, x_agg_cate = inp
        # x_seq: [nb_units * nb_storey * features] list[(U x S x F)]
        # data_length: list of list outer list: batch, inner list: nb_units
        # x_agg_quant: batch,F
        # x_agg_cate: batch,
        out = []
        nb_batches = len(data_length)
        for i in range(nb_batches):
            if self.use_seq:
                unit_length = data_length[i]
                nb_units = len(unit_length)
                key_padding_mask = self.generate_key_padding_mask(data_length[i])
                attn_mask = self.generate_attn_mask(self.nb_heads, data_length[i])
                trans_out = self.transformer_encoder(self.inp_embedding(x_seq[i]).transpose(0, 1), mask=attn_mask,
                                                     src_key_padding_mask=key_padding_mask).transpose(0, 1)
                unit_outs = []
                for j in range(nb_units):
                    trans_out_j = trans_out[j][:unit_length[j]]
                    # storey-level sum pooling
                    unit_outs.append(torch.sum(trans_out_j, dim=0))

                building_seq_out = torch.sum(self.unit_fc(torch.vstack(unit_outs)), dim=0)
                fused_inp = [building_seq_out]
            else:
                fused_inp = []

            if self.use_w and self.use_t:
                fused_inp.append(x_agg_quant[i])
                fused_inp.append(torch.squeeze(self.tod_embedding(x_agg_cate[i][0])))
            elif self.use_w and not self.use_t:
                fused_inp.append(x_agg_quant[i][self.agg_feature_idx['parcel'][0]:self.agg_feature_idx['parcel'][1]])
            elif self.use_t and not self.use_w:
                fused_inp.append(
                    x_agg_quant[i][self.agg_feature_idx['temporal'][0]:self.agg_feature_idx['temporal'][1]])
                fused_inp.append(torch.squeeze(self.tod_embedding(x_agg_cate[i][0])))
            else:
                raise Exception('undefined')
            fused_inp = torch.cat(fused_inp, dim=0)
            out.append(self.fc(fused_inp))
        return torch.vstack(out)

    def generate_key_padding_mask(self, data_length):
        # N * Max_Length
        # False: unchanged True: ignored
        bsz = len(data_length)
        max_len = max(data_length)
        key_padding_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
        for i in range(bsz):
            key_padding_mask[i, data_length[i]:] = True
        return key_padding_mask.to(self.device)

    def generate_attn_mask(self, nb_heads, data_length):
        # 每一个位置都能看到其余位置的信息
        bsz = len(data_length)
        max_len = max(data_length)
        attn_mask = torch.ones((bsz * nb_heads, max_len, max_len), dtype=torch.bool)
        for i in range(bsz):
            attn_mask[i * nb_heads:(i + 1) * nb_heads, :, :data_length[i]] = False
        return attn_mask.to(self.device)


class MetaSTP(nn.Module):
    def __init__(self, encoder_hidden, encoder_out, seq_length, key_size, value_size, nb_filters,
                 spatial_hiddens, model_name=None, emb_name='', dev=None, att_crt=1, ft=True):
        super(MetaSTP, self).__init__()
        self.device = dev if dev else torch.device('cpu')
        self.model_name = model_name
        self.emb_name = emb_name
        self.att_crt = att_crt
        # safety check
        known_model_names = {'MetaSTP', 'MetaSTP+2A+C+P-nS', 'MetaSTP+2A+C+P-nT', 'MetaSTP+2A+C+P-nW',
                             'MetaSTP+2A+C+P-nSeq', 'MetaSTP+2A+C+P-nRes',
                             'MetaSTP+2A+C+P-nUF', 'MetaSTP+2A+C+P-nE', 'MetaSTP+2A+C-nS', 'MetaSTP+2A+C-nT',
                             'MetaSTP+2A+C-nW', 'MetaSTP+2A+C-nSeq', 'MetaSTP+2A+C-nRes',
                             'MetaSTP+2A+C-nUF', 'MetaSTP+2A+C-nE', 'MetaSTP+1A+C', 'MetaSTP+2A+C', 'MetaSTP+2A+C+P',
                             'MetaSTP+2A', 'MetaSTP+C', 'MetaSTP+C+P', 'MetaSTP+S+A+P', 'MetaSTP+S+A',
                             'MetaSTP+S+A+C+P', 'MetaSTP+S+A+C', 'MetaSTP+2A+C+P-nPr', 'MetaSTP+2A+P'}
        if model_name not in known_model_names:
            raise Exception('unknown model name:{}'.format(model_name))
        self.spatial_fuse = True
        if '-nS' in self.model_name:
            self.spatial_fuse = False
            self.encoder = DeliveryEventEncoder(encoder_hidden, encoder_out, dev=self.device)
        elif '-nT' in self.model_name:
            self.encoder = DeliveryEventEncoder(encoder_hidden, encoder_out, use_t=False, dev=self.device)
        elif '-nW' in self.model_name:
            self.encoder = DeliveryEventEncoder(encoder_hidden, encoder_out, use_w=False, use_seq=False,
                                                dev=self.device)
        elif '-nSeq' in self.model_name:
            self.encoder = DeliveryEventEncoder(encoder_hidden, encoder_out, use_seq=False, dev=self.device)
        else:
            self.encoder = DeliveryEventEncoder(encoder_hidden, encoder_out, dev=self.device)


        num_channels = encoder_out + 1  # spatial_hiddens[-1] + 1

        # num_channels += spatial_hiddens[-1]

        nb_convs = int(math.ceil(math.log(seq_length + 1, 2)))
        if 'A' in self.model_name and not '+S' in self.model_name:
            self.attback = False
            if self.spatial_fuse:
                self.attention1 = QSAttentionBlock1(num_channels, num_channels + spatial_hiddens[-1] - 1, key_size,
                                                    value_size,
                                                    dev=self.device)  # ,  ,
            else:
                self.attention1 = QSAttentionBlock1(num_channels, num_channels - 1, key_size,
                                                    value_size,
                                                    dev=self.device)
        elif '+S' in self.model_name:
            self.attback = True
            self.attention1 = SelfAttentionBlock(num_channels, key_size, value_size, dev=self.device,
                                                 att_dim=self.att_crt, mask_f=False)
        else:
            self.attback = True
            self.attention1 = SelfAttentionBlock(num_channels, key_size, value_size, dev=self.device,
                                                 att_dim=self.att_crt)
        num_channels += value_size
        self.tc1 = TCBlock(num_channels, seq_length + 1, nb_filters, dev=self.device)
        num_channels += nb_convs * nb_filters
        if '+2A' in self.model_name:
            self.attention2 = QSAttentionBlock2(num_channels, key_size, value_size, dev=self.device,
                                                att_dim=self.att_crt)
        else:
            self.attention2 = SelfAttentionBlock(num_channels, key_size, value_size, dev=self.device,
                                                 att_dim=self.att_crt)
        num_channels += value_size
        if self.spatial_fuse:
            self.spatial_encoder = SpatialKnowledgeEncoder(spatial_hiddens, withE=('-nE' not in self.model_name),
                                                           withPtr=('+P' in self.model_name), emb_name=self.emb_name,
                                                           dev=self.device, ft=ft)
            if '-nRes' in self.model_name:
                num_channels += spatial_hiddens[-1]
            else:
                internal_dim = 2
                self.fuse_layer = nn.Sequential(
                    nn.Linear(num_channels + spatial_hiddens[-1], internal_dim),
                    nn.ReLU(),
                    nn.Linear(internal_dim, num_channels)
                ).to(self.device)
        self.fc = nn.Linear(num_channels, 1).to(self.device)
        self.dev = dev

    def forward(self, spt_inp, spt_tgt, qry_inp, spatial_inp=None):  # 对一个meta batch，给出预测
        # 由于我们每个task的train可能数量不同，分开处理
        global spt_embed, loc_knowlg, spatial_embed
        nb_tasks = len(spt_inp)
        preds = []
        for i in range(nb_tasks):  # 对meta batch中的每个子task（每个地点）给出预测
            # 保险起见，最好qry的每一个和spt_embed concat
            spt_size = len(spt_inp[i][-1])
            qry_size = len(qry_inp[i][-1])
            if self.spatial_fuse:  # 是否需要空间先验知识融合
                # (2dim quant, 1cate) -> [qry_size, spt_size+1, spatial_out]
                loc_knowlg = self.spatial_encoder(spatial_inp[i])  # 一维向量,纯每个地点的embed
                spatial_embed = loc_knowlg.expand(qry_size, spt_size + 1, -1)
            # [spt_size,out]
            if spt_size > 0:  # Delivery Event Encoder 进行编码
                spt_embed = self.encoder(spt_inp[i])
            # [qry_size,out]
            qry_embed = self.encoder(qry_inp[i])
            # [qry_size, 1, 1]

            if self.attback:
                mask_qry_tgt = torch.Tensor(np.zeros((qry_size, 1, 1))).to(self.dev)
            else:
                mask_qry_tgt = torch.Tensor(np.zeros((qry_size, 1))).to(self.dev)

            if spt_size > 0:  # 对h_q和标签或mask进行连接

                if self.attback:
                    x = torch.cat(
                        [torch.cat([spt_embed.expand(qry_size, -1, -1), spt_tgt[i].expand(qry_size, -1, -1)], 2),
                         torch.cat([torch.unsqueeze(qry_embed, 1), mask_qry_tgt], 2)], 1)
                else:
                    spt_tgt_exp = spt_tgt[i].expand(len(spt_embed), 1)
                    spt = torch.cat([spt_embed, spt_tgt_exp], 1)
            else:
                if self.attback:
                    x = torch.cat([torch.unsqueeze(qry_embed, 1), mask_qry_tgt], 2)
                else:
                    spt = torch.tensor([])


            if self.spatial_fuse:
                qry = torch.cat([qry_embed, loc_knowlg.expand(qry_size, -1)], 1)
            else:
                qry = qry_embed
            if self.attback:
                x = self.attention1(x)
            else:
                x = self.attention1(spt, qry)

            # print("after first self-att x.size=", x.size())

            x = self.tc1(x)

            x = self.attention2(x)

            if self.spatial_fuse:
                if '-nRes' in self.model_name:
                    x = self.fc(torch.cat([x, spatial_embed], 2))
                else:
                    x = self.fc(torch.relu(x + self.fuse_layer(torch.cat([x, spatial_embed], 2))))
                    # print("after loc know fuse x.size=", x.size())
            else:
                # [qry_size,spt_size+1,1]
                x = self.fc(x)
            preds.append(x[:, -1, 0:1])
        return torch.cat(preds, 0)
