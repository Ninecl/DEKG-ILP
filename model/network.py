from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn.functional as F
import torch.nn as nn
import torch
"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class DEKG_ILP(nn.Module):
    
    def __init__(self, params, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id

        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)
        self.rsf_rel_emb = nn.Embedding(self.params.num_rels, self.params.rsf_dim, sparse=False)
        self.rsf_emb = nn.Embedding(self.params.aug_num_rels, self.params.rsf_dim)
        self.gnn_out_dim = self.params.num_gcn_layers * self.params.emb_dim

        if self.params.add_ht_emb:
            self.fc_layer = nn.Linear(3 * self.gnn_out_dim + self.params.rel_emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(self.gnn_out_dim + self.params.rel_emb_dim, 1)


    def forward(self, graph_data, contrastive_data):
        (rct, g, rel_labels), (head_con_pos, head_con_neg, tail_con_pos, tail_con_neg) = graph_data, contrastive_data

        #----------------------------------------------------------------
        # global semantic information
        #----------------------------------------------------------------

        # calculate the rsf of head entity
        head_rct = rct[:, 0]
        head_rct_sum = torch.sum(head_rct, dim=1).unsqueeze(1)
        head_rsf = torch.sum(head_rct.unsqueeze(2) * self.rsf_emb.weight, dim=1) / head_rct_sum

        # calculate the rsf of tail entity
        tail_rct = rct[:, 1]
        tail_rct_sum = torch.sum(tail_rct, dim=1).unsqueeze(1)
        tail_rsf = torch.sum(tail_rct.unsqueeze(2) * self.rsf_emb.weight, dim=1) / tail_rct_sum

        rsf_output = torch.sum(head_rsf * self.rsf_rel_emb(rel_labels) * tail_rsf, dim=1, keepdim=True)

        #----------------------------------------------------------------
        # local topological information
        #----------------------------------------------------------------

        g.ndata['h'] = self.gnn(g)

        g_out = mean_nodes(g, 'repr')

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]

        if self.params.add_ht_emb:
            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               head_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               tail_embs.view(-1, self.params.num_gcn_layers * self.params.emb_dim),
                               self.rel_emb(rel_labels)], dim=1)
        else:
            g_rep = torch.cat([g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim), self.rel_emb(rel_labels)], dim=1)

        output = self.fc_layer(g_rep)

        #----------------------------------------------------------------
        # Contrastive learning loss
        #----------------------------------------------------------------

        # the positive examples of head entity
        head_con_pos = head_con_pos.squeeze()
        # get dim
        d0, d1, d2 = head_con_pos.shape
        head_con_pos_rct = head_con_pos.view((d0 * d1, d2))
        head_con_pos_sum = torch.sum(head_con_pos_rct, dim=1).unsqueeze(1)
        head_con_pos_rsf = torch.sum(head_con_pos_rct.unsqueeze(2) * self.rsf_emb.weight, dim=1) / head_con_pos_sum
        # the negative examples of head entity
        head_con_neg = head_con_neg.squeeze()
        head_con_neg_rct = head_con_neg.view((d0 * d1, d2))
        head_con_neg_sum = torch.sum(head_con_neg_rct, dim=1).unsqueeze(1)
        head_con_neg_rsf = torch.sum(head_con_neg_rct.unsqueeze(2) * self.rsf_emb.weight, dim=1) / head_con_neg_sum
        # the positive examples of tail entity
        tail_con_pos = tail_con_pos.squeeze()
        tail_con_pos_rct = tail_con_pos.view((d0 * d1, d2))
        tail_con_pos_sum = torch.sum(tail_con_pos_rct, dim=1).unsqueeze(1)
        tail_con_pos_rsf = torch.sum(tail_con_pos_rct.unsqueeze(2) * self.rsf_emb.weight, dim=1) / tail_con_pos_sum
        # the negative examples of head entity
        tail_con_neg = tail_con_neg.squeeze()
        tail_con_neg_rct = tail_con_neg.view((d0 * d1, d2))
        tail_con_neg_sum = torch.sum(tail_con_neg_rct, dim=1).unsqueeze(1)
        tail_con_neg_rsf = torch.sum(tail_con_neg_rct.unsqueeze(2) * self.rsf_emb.weight, dim=1) / tail_con_neg_sum

        # calculate the distance between positive and negative examples
        head_rsf = head_rsf.unsqueeze(1).repeat(1, d1, 1).view((d0 * d1, self.params.rsf_dim))
        tail_rsf = tail_rsf.unsqueeze(1).repeat(1, d1, 1).view((d0 * d1, self.params.rsf_dim))
        # head positive
        con_head_pos_dis = F.pairwise_distance(head_rsf, head_con_pos_rsf, p=2)
        # head negative
        con_head_neg_dis = F.pairwise_distance(head_rsf, head_con_neg_rsf, p=2)
        # tail positive
        con_tail_pos_dis = F.pairwise_distance(tail_rsf, tail_con_pos_rsf, p=2)
        # tail negative
        con_tail_neg_dis = F.pairwise_distance(tail_rsf, tail_con_neg_rsf, p=2)

        con_pos_dis = torch.cat((con_head_pos_dis, con_tail_pos_dis))
        con_neg_dis = torch.cat((con_head_neg_dis, con_tail_neg_dis))

        if self.params.remove_rsf:
            return output, con_pos_dis, con_neg_dis
        else:
            return output + rsf_output, con_pos_dis, con_neg_dis
