import torch
import torch.nn as nn
from layers import GCN, AvgReadout, AvgReadout2, Discriminator, Discriminator2
import pdb
import math
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


class DGI(nn.Module):
    def __init__(self, n_in, hidden_dims,
                 activation, muti_act_1, muti_act_2):
        super(DGI, self).__init__()

        self.n_h = hidden_dims[-1]
        # print("n_h:", self.n_h)
        self.n_in = n_in
        self.hidden_dims = hidden_dims
        # self.encoder = GCN(n_in, self.n_h, activation)
        # self.decoder = GCN(self.n_h, n_in, activation)

        # Encoder
        if len(self.hidden_dims) == 1:
            self.encoder = GCN(self.n_in, self.n_h, activation)
        else:
            self.encoder = torch.nn.ModuleList()
            self.encoder.append(GCN(self.n_in, self.hidden_dims[0], act=muti_act_1))
            for l in range(len(self.hidden_dims)-1):
                if l==(len(self.hidden_dims)-2):
                    self.encoder.append(GCN(self.hidden_dims[l], self.n_h, act=muti_act_2))
                else:
                    self.encoder.append(GCN(self.hidden_dims[l], self.hidden_dims[l + 1], act=muti_act_1))
        # print("encoder:", self.encoder)

        # Decoder
        if len(self.hidden_dims) == 1:
            self.decoder = GCN(self.n_h, self.n_in, activation)
        else:
            self.decoder = torch.nn.ModuleList()
            self.decoder.append(GCN(self.n_h, self.hidden_dims[-2], act=muti_act_2))
            for l in range(len(self.hidden_dims)-2, -1, -1):
                if l==0:
                    self.decoder.append(GCN(self.hidden_dims[l], self.n_in, act=muti_act_1))
                else:
                    self.decoder.append(GCN(self.hidden_dims[l], self.hidden_dims[l - 1], act=muti_act_1))

        # print("decoder:", self.decoder)

        self.read = AvgReadout()
        self.read2 = AvgReadout2()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(self.n_h)
        self.disc2 = Discriminator2(self.n_h)

    def encode(self, feat, adj, sparse):
        if len(self.hidden_dims) == 1:
            return self.encoder(feat, adj, sparse)
        # print("self.encoder:", self.encoder)
        for en in self.encoder:
            # print("start feat dim:", feat.shape)
            feat = en(feat, adj, sparse)
            # print("end feat dim:", feat.shape)
        return feat

    def decode(self, h, adj, sparse):
        if len(self.hidden_dims) == 1:
            return self.decoder(h, adj, sparse)

        for de in self.decoder:
            h = de(h, adj, sparse)
        return h

    def forward(self, feat, shuf_feat, aug1_feat, aug2_feat,
                adj, aug_adj1, aug_adj2, sparse, aug_type1, aug_type2):

        h_0 = self.encode(feat, adj, sparse)

        if aug_type1 == 'edge':
            h_1 = self.encode(feat, aug_adj1, sparse)
        elif aug_type1 == 'mask':
            h_1 = self.encode(aug1_feat, adj, sparse)
        elif aug_type1 == 'HS_image':
            h_1 = self.encode(feat, aug_adj1, sparse)
        else:
            assert False

        if aug_type2 == 'edge':
            h_3 = self.encode(feat, aug_adj2, sparse)
        elif aug_type2 == 'mask':
            h_3 = self.encode(aug2_feat, adj, sparse)
        elif aug_type2 == 'HS_image':
            h_3 = self.encode(feat, aug_adj2, sparse)

        else:
            assert False

        h_2 = self.encode(shuf_feat, adj, sparse)

        return h_0, h_2, h_1, h_3

    def decoder_head(self, h, adj, sparse):
        X_ = self.decode(h, adj, sparse)
        return X_

    def node_global_loss(self, h_0, h_2, h_1, h_3, lbl, msk=None, samp_bias1=None, samp_bias2=None):
        b_xent = nn.BCEWithLogitsLoss()

        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        c_3 = self.read(h_3, msk)
        c_3 = self.sigm(c_3)

        ret1 = self.disc(c_1, h_0, h_2, samp_bias1, samp_bias2)
        ret2 = self.disc(c_3, h_0, h_2, samp_bias1, samp_bias2)

        ret = ret1 + ret2
        loss_dgi = b_xent(ret, lbl)

        return loss_dgi

    def node_cluster_loss(self, h_0, h_2, h_1, h_3, lbl, msk=None, samp_bias1=None, samp_bias2=None):
        b_xent = nn.BCEWithLogitsLoss()

        c_1 = self.read2(h_1, msk)
        c_1 = self.sigm(c_1)

        c_3 = self.read2(h_3, msk)
        c_3 = self.sigm(c_3)

        ret1 = self.disc2(c_1, h_0, h_2, samp_bias1, samp_bias2)
        ret2 = self.disc2(c_3, h_0, h_2, samp_bias1, samp_bias2)

        ret = ret1 + ret2
        loss_node_cluster = b_xent(ret, lbl)
        # print(type(loss_node_cluster))

        return loss_node_cluster

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.encode(seq, adj, sparse)
        c = self.read(h_1, msk)
        X_ = self.decode(h_1, adj, sparse)

        return h_1, X_, c

if __name__ == '__main__':
    pass