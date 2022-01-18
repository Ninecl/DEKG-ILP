from ast import parse
import os
import pickle
import random
import argparse
import logging
import json
import time

import multiprocessing as mp
from networkx.algorithms.link_analysis.hits_alg import hits
import scipy.sparse as ssp
from tqdm import tqdm
import networkx as nx
import torch
import numpy as np
import dgl

from utils.graph_utils import ssp_multigraph_to_dgl, incidence_matrix
from subgraph_extraction.graph_sampler import get_neighbor_nodes


def process_files(files, saved_relation2id, add_traspose_rels):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    relation2id = saved_relation2id

    triplets = {}

    ent = 0
    rel = 0

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in saved_relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], saved_relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(saved_relation2id)):
        idx = np.argwhere(triplets['graph'][:, 2] == i)
        adj_list.append(ssp.csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['graph'][:, 0][idx].squeeze(1), triplets['graph'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))

    # Add transpose matrices to handle both directions of relations.
    adj_list_aug = adj_list
    if add_traspose_rels:
        adj_list_t = [adj.T for adj in adj_list]
        adj_list_aug = adj_list + adj_list_t

    dgl_adj_list = ssp_multigraph_to_dgl(adj_list_aug)

    # 构建relation-specific特征
    num_entity = len(entity2id)
    num_relation = len(relation2id)
    rsf_list = np.zeros((num_entity, num_relation*2))
    for h, t, r in triplets['graph']:
        rsf_list[h][r] += 1
        rsf_list[t][r+num_relation] += 1

    return adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rsf_list


def get_kge_embeddings(dataset, kge_model):

    path = './experiments/kge_baselines/{}_{}'.format(kge_model, dataset)
    node_features = np.load(os.path.join(path, 'entity_embedding.npy'))
    with open(os.path.join(path, 'id2entity.json')) as json_file:
        kge_id2entity = json.load(json_file)
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}

    return node_features, kge_entity2id


def get_neg_samples_replacing_head_tail(test_links, adj_list, num_samples=50):

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    neg_triplets = []
    for i, (head, tail, rel) in enumerate(zip(heads, tails, rels)):
        neg_triplet = {'head': [[], 0], 'tail': [[], 0], 'rel': [[], 0]}
        neg_triplet['head'][0].append([head, tail, rel])

        num_samples_ent = num_samples if num_samples < n else n
        # num_samples_ent = num_samples

        while len(neg_triplet['head'][0]) < num_samples_ent:
            neg_head = head
            neg_tail = np.random.choice(n)

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])

        neg_triplet['tail'][0].append([head, tail, rel])
        while len(neg_triplet['tail'][0]) < num_samples_ent:
            neg_head = np.random.choice(n)
            neg_tail = tail
            # neg_head, neg_tail, rel = np.random.choice(n), np.random.choice(n), np.random.choice(r)

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])

        num_samples_rel = num_samples if num_samples < r else r
        # num_samples_rel = num_samples

        neg_triplet['rel'][0].append([head, tail, rel])
        while len(neg_triplet['rel'][0]) < num_samples_rel:
            neg_rel = np.random.choice(r)

            if adj_list[neg_rel][head, tail] == 0:
                neg_triplet['rel'][0].append([head, tail, neg_rel])

        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])
        neg_triplet['rel'][0] = np.array(neg_triplet['rel'][0])

        neg_triplets.append(neg_triplet)

    return neg_triplets


def get_neg_samples_replacing_head_tail_all(test_links, adj_list):

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    neg_triplets = []
    print('sampling negative triplets...')
    for i, (head, tail, rel) in tqdm(enumerate(zip(heads, tails, rels)), total=len(heads)):
        neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
        neg_triplet['head'][0].append([head, tail, rel])
        for neg_tail in range(n):
            neg_head = head

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])

        neg_triplet['tail'][0].append([head, tail, rel])
        for neg_head in range(n):
            neg_tail = tail

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])

        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

        neg_triplets.append(neg_triplet)

    return neg_triplets


def save_to_file(neg_triplets, id2entity, id2relation):

    with open(os.path.join('./data', params.dataset, 'ranking_head.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['head'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')

    with open(os.path.join('./data', params.dataset, 'ranking_tail.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['tail'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')


def save_to_pickle_file(file_path, neg_triplets):

    with open(file_path, "wb") as f:
        pickle.dump(neg_triplets, f)


def intialize_worker(model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id, rsf_list):
    global model_, adj_list_, dgl_adj_list_, id2entity_, params_, node_features_, kge_entity2id_, rsf_list_
    model_, adj_list_, dgl_adj_list_, id2entity_, params_, node_features_, kge_entity2id_, rsf_list_ = model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id, np.array(rsf_list)


def prepare_features(subgraph, n_labels, max_n_label, n_feats=None):
    # One hot encode the node label feature and concat to n_featsure
    n_nodes = subgraph.number_of_nodes()
    label_feats = np.zeros((n_nodes, max_n_label[0] + 1 + max_n_label[1] + 1))
    label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
    label_feats[np.arange(n_nodes), max_n_label[0] + 1 + n_labels[:, 1]] = 1
    n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
    subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

    head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
    tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
    n_ids = np.zeros(n_nodes)
    n_ids[head_id] = 1  # head
    n_ids[tail_id] = 2  # tail
    subgraph.ndata['id'] = torch.FloatTensor(n_ids)

    return subgraph


def get_subgraphs_from_dic(all_links, nodes_labels_dic, dgl_adj_list, max_node_label_value, id2entity, node_features=None, kge_entity2id=None):
    # dgl_adj_list = ssp_multigraph_to_dgl(adj_list)

    subgraphs = []
    r_labels = []

    for link in all_links:
        head, tail, rel = link[0], link[1], link[2]
        nodes, node_labels = nodes_labels_dic[f'{head} {tail} {rel}']
        subgraph = dgl_adj_list.subgraph(nodes)
        subgraph.edata['type'] = dgl_adj_list.edata['type'][subgraph.edata[dgl.EID]]
        subgraph.edata['label'] = torch.tensor(rel * np.ones(subgraph.edata['type'].shape), dtype=torch.long)

        if subgraph.has_edges_between(0, 1):
            edges_btw_roots = subgraph.edge_ids(0, 1)
        else:
            edges_btw_roots = torch.tensor([], dtype=torch.int64)

        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == rel)

        if rel_link.squeeze().nelement() == 0:
            # subgraph.add_edge(0, 1, {'type': torch.tensor([rel]), 'label': torch.tensor([rel])})
            subgraph.add_edges(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(rel).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(rel).type(torch.LongTensor)

        kge_nodes = [kge_entity2id[id2entity[n]] for n in nodes] if kge_entity2id else None
        n_feats = node_features[kge_nodes] if node_features is not None else None
        subgraph = prepare_features(subgraph, node_labels, max_node_label_value, n_feats)

        subgraphs.append(subgraph)
        r_labels.append(rel)

    batched_graph = dgl.batch(subgraphs)
    r_labels = torch.LongTensor(r_labels)

    return (batched_graph, r_labels)


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def node_label_new(subgraph, max_distance=1):
    # an implementation of the proposed double-radius node labeling (DRNd   L)
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

    # dist_to_roots[np.abs(dist_to_roots) > 1e6] = 0
    # dist_to_roots = dist_to_roots + 1
    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    # print(len(enclosing_subgraph_nodes))
    return labels, enclosing_subgraph_nodes


def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, node_information=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

    # could pack these two into a function
    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    if enclosing_sub_graph:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)

    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]

    labels, enclosing_subgraph_nodes = node_label_new(incidence_matrix(subgraph), max_distance=h)

    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    return pruned_subgraph_nodes, pruned_labels


def move_batch_data_to_device(links, data):
    g, r_label = data
    g = g.to(params_.device)
    r_label = r_label.to(params_.device)
    head = links[:, 0]
    tail = links[:, 1]
    heads_rsf = torch.LongTensor(rsf_list_[head]).unsqueeze(1)
    tails_rsf = torch.LongTensor(rsf_list_[tail]).unsqueeze(1)
    links_rsf = torch.cat((heads_rsf, tails_rsf), dim=1).to(device=params_.device)

    # 伪造一波contrastive数据，为了模型能正确计算
    batch_num = len(r_label)
    n1_conpos_pos = torch.LongTensor(np.ones((batch_num, params_.con_sample_num, len(rsf_list_[0])))).to(params_.device)
    n1_conneg_pos = torch.LongTensor(np.ones((batch_num, params_.con_sample_num, len(rsf_list_[0])))).to(params_.device)
    n2_conpos_pos = torch.LongTensor(np.ones((batch_num, params_.con_sample_num, len(rsf_list_[0])))).to(params_.device)
    n2_conneg_pos = torch.LongTensor(np.ones((batch_num, params_.con_sample_num, len(rsf_list_[0])))).to(params_.device)

    return (links_rsf, g, r_label), (n1_conpos_pos, n1_conneg_pos, n2_conpos_pos, n2_conneg_pos)


def get_rank(neg_links, nodes_labels_dic):
    head_neg_links = neg_links['head'][0]
    head_target_id = neg_links['head'][1]

    if head_target_id != 10000:
        graph_data = get_subgraphs_from_dic(head_neg_links, nodes_labels_dic, dgl_adj_list_, model_.gnn.max_label_value, id2entity_, node_features_, kge_entity2id_)
        graph_data, contrastive_data = move_batch_data_to_device(head_neg_links, graph_data)
        head_scores, _, _ = model_(graph_data, contrastive_data)
        head_scores = head_scores.squeeze(1).detach().cpu().numpy()
        head_rank = np.argwhere(np.argsort(head_scores)[::-1] == head_target_id) + 1
    else:
        head_scores = np.array([])
        head_rank = 10000

    tail_neg_links = neg_links['tail'][0]
    tail_target_id = neg_links['tail'][1]

    if tail_target_id != 10000:
        graph_data = get_subgraphs_from_dic(tail_neg_links, nodes_labels_dic, dgl_adj_list_, model_.gnn.max_label_value, id2entity_, node_features_, kge_entity2id_)
        graph_data, contrastive_data = move_batch_data_to_device(tail_neg_links, graph_data)
        tail_scores, _, _  = model_(graph_data, contrastive_data)
        tail_scores = tail_scores.squeeze(1).detach().cpu().numpy()
        tail_rank = np.argwhere(np.argsort(tail_scores)[::-1] == tail_target_id) + 1
    else:
        tail_scores = np.array([])
        tail_rank = 10000

    rel_neg_links = neg_links['rel'][0]
    rel_target_id = neg_links['rel'][1]

    if rel_target_id != 10000:
        graph_data = get_subgraphs_from_dic(rel_neg_links, nodes_labels_dic, dgl_adj_list_, model_.gnn.max_label_value, id2entity_, node_features_, kge_entity2id_)
        graph_data, contrastive_data = move_batch_data_to_device(rel_neg_links, graph_data)
        rel_scores, _, _ = model_(graph_data, contrastive_data)
        rel_scores = rel_scores.squeeze(1).detach().cpu().numpy()
        rel_rank = np.argwhere(np.argsort(rel_scores)[::-1] == rel_target_id) + 1
    else:
        rel_scores = np.array([])
        rel_rank = 10000

    return head_scores, head_rank, tail_scores, tail_rank, rel_scores, rel_rank


def extract_save_subgraph(args_):

    (n1, n2, rel) = args_
    nodes, labels = subgraph_extraction_labeling((n1, n2), rel, adj_list_, h=params_.hop, enclosing_sub_graph=params_.enclosing_sub_graph, max_node_label_value=model_.gnn.max_label_value)

    return f'{n1} {n2} {rel}', nodes, labels

    
def mp_subgraph_extraction_labeling(tripelts_dic):

    all_triplets = []
    for dic in tripelts_dic:
        all_triplets.append(dic['head'][0])
        all_triplets.append(dic['tail'][0])
        all_triplets.append(dic['rel'][0])
    
    all_triplets = np.array([j for i in all_triplets for j in i])

    with mp.Pool(processes=None) as p:

        h, t, r = all_triplets.transpose()
        nodes_labels_dic = dict()

        args_ = zip(h, t, r)

        for (key, nodes, node_labels) in tqdm(p.imap(extract_save_subgraph, args_), total=len(all_triplets)):
            nodes_labels_dic[key] = (nodes, node_labels)
    
    return nodes_labels_dic


def save_score_to_file(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation):

    with open(os.path.join('./data', params.dataset, 'grail_ranking_head_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], head_score in zip(neg_triplet['head'][0], all_head_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(head_score)]) + '\n')

    with open(os.path.join('./data', params.dataset, 'grail_ranking_tail_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], tail_score in zip(neg_triplet['tail'][0], all_tail_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(tail_score)]) + '\n')


def main(params):
    # 加载模型
    model = torch.load(params.model_path, map_location=params.device)
    
    adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rsf_list = process_files(params.file_paths, model.relation2id, params.add_traspose_rels)

    node_features, kge_entity2id = get_kge_embeddings(params.dataset, params.kge_model) if params.use_kge_embeddings else (None, None)

    # 负采样
    if params.mode == 'sample':
        save_pickle_file = os.path.join('./data', params.dataset, 'sample_ranking.pickle')
        if os.path.exists(save_pickle_file) and params.no_resample :
            f = open(save_pickle_file, 'rb')
            neg_triplets = pickle.load(f)
        else:
            neg_triplets = get_neg_samples_replacing_head_tail(triplets['links'], adj_list, params.num_negative_sampler)
            # save_to_file(neg_triplets, id2entity, id2relation)
            if params.save_file:
                save_to_pickle_file(save_pickle_file, neg_triplets)
    elif params.mode == 'all':
        neg_triplets = get_neg_samples_replacing_head_tail_all(triplets['links'], adj_list)
    
    # 全局初始化
    intialize_worker(model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id, rsf_list)

    # 获得所有neg_triplets对应的子图
    save_dict_file = os.path.join('./data', params.dataset, 'subgraph_dict.pickle')
    if os.path.exists(save_dict_file) and params.no_resample:
        f = open(save_dict_file, 'rb')
        nodes_lables_dic = pickle.load(f)
    else:
        nodes_lables_dic = mp_subgraph_extraction_labeling(neg_triplets)
        if params.save_file:
            save_to_pickle_file(save_dict_file, nodes_lables_dic)
    
    # 记录数据
    head_ranks = []
    tail_ranks = []
    rel_ranks = []
    all_ranks = []
    all_head_scores = []
    all_tail_scores = []

    # 对所有测试用例进行计算
    with torch.no_grad():

        model_.eval()

        for triplets in tqdm(neg_triplets):
            head_scores, head_rank, tail_scores, tail_rank, rel_scores, rel_rank = get_rank(triplets, nodes_lables_dic)
            head_ranks.append(head_rank)
            tail_ranks.append(tail_rank)
            rel_ranks.append(rel_rank)
            
            # all_head_scores += head_scores.tolist()
            # all_tail_scores += tail_scores.tolist()

        # 统计head_rank
        head_isHit1List = [x for x in head_ranks if x <= 1]
        head_isHit5List = [x for x in head_ranks if x <= 5]
        head_isHit10List = [x for x in head_ranks if x <= 10]
        head_hits_1 = len(head_isHit1List) / len(head_ranks)
        head_hits_5 = len(head_isHit5List) / len(head_ranks)
        head_hits_10 = len(head_isHit10List) / len(head_ranks)
        head_mrr = np.mean(1 / np.array(head_ranks))

        # 统计tail_rank
        tail_isHit1List = [x for x in tail_ranks if x <= 1]
        tail_isHit5List = [x for x in tail_ranks if x <= 5]
        tail_isHit10List = [x for x in tail_ranks if x <= 10]
        tail_hits_1 = len(tail_isHit1List) / len(tail_ranks)
        tail_hits_5 = len(tail_isHit5List) / len(tail_ranks)
        tail_hits_10 = len(tail_isHit10List) / len(tail_ranks)
        tail_mrr = np.mean(1 / np.array(tail_ranks))

        # 统计rel_rank
        rel_isHit1List = [x for x in rel_ranks if x <= 1]
        rel_isHit5List = [x for x in rel_ranks if x <= 5]
        rel_isHit10List = [x for x in rel_ranks if x <= 10]
        rel_hits_1 = len(rel_isHit1List) / len(rel_ranks)
        rel_hits_5 = len(rel_isHit5List) / len(rel_ranks)
        rel_hits_10 = len(rel_isHit10List) / len(rel_ranks)
        rel_mrr = np.mean(1 / np.array(rel_ranks))

        # 把三个列表拼起来就是all_rank
        all_ranks = head_ranks + rel_ranks + tail_ranks
        # 统计all_rank
        all_isHit1List = [x for x in all_ranks if x <= 1]
        all_isHit5List = [x for x in all_ranks if x <= 5]
        all_isHit10List = [x for x in all_ranks if x <= 10]
        all_hits_1 = len(all_isHit1List) / len(all_ranks)
        all_hits_5 = len(all_isHit5List) / len(all_ranks)
        all_hits_10 = len(all_isHit10List) / len(all_ranks)
        all_mrr = np.mean(1 / np.array(all_ranks))

    return {'all_mrr': all_mrr, 'all_hits_1': all_hits_1, 'all_hits_5': all_hits_5, 'all_hits_10': all_hits_10, 
            'head_mrr': head_mrr, 'head_hits_1': head_hits_1, 'head_hits_5': head_hits_5, 'head_hits_10': head_hits_10,
            'tail_mrr': tail_mrr, 'tail_hits_1': tail_hits_1, 'tail_hits_5': tail_hits_5, 'tail_hits_10': tail_hits_10,
            'rel_mrr': rel_mrr, 'rel_hits_1': rel_hits_1, 'rel_hits_5': rel_hits_5, 'rel_hits_10': rel_hits_10}

    # save_score_to_file(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation)


# 写一个解析函数，方便点
def analyse_result(dic, s):
    return dic[f'{s}_mrr'], dic[f'{s}_hits_1'], dic[f'{s}_hits_5'], dic[f'{s}_hits_10']


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Testing script for hits@10')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str,
                        help="Experiment name. Log file with this name will be created")
    parser.add_argument("--dataset", "-d", type=str,
                        help="Path to dataset")
    parser.add_argument("--mode", "-m", type=str, default="sample", choices=["sample", "all", "ruleN"],
                        help="Negative sampling mode")
    parser.add_argument("--use_kge_embeddings", "-kge", type=bool, default=False,
                        help='whether to use pretrained KGE embeddings')
    parser.add_argument("--kge_model", type=str, default="TransE",
                        help="Which KGE model to load entity embeddings from")
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')
    parser.add_argument("--hop", type=int, default=3,
                        help="How many hops to go while eextracting subgraphs?")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='Whether to append adj matrix list with symmetric relations?')
    parser.add_argument('--use_cuda', '-uc', type=bool, default=True,
                        help='Whether use cuda.')
    parser.add_argument('--device', '-de', type=int, default=0, choices=[-1, 0, 1, 2, 3],
                        help='Which gpu to use.')
    parser.add_argument('--num_negative_sampler', '-ns', type=int, default=50,
                        help='Number of negative sample for each link.')
    parser.add_argument('--no_resample', '-nrs', action='store_true',
                        help='Whether resample negative links.')
    parser.add_argument('--con_sample_num', type=int, default=10,
                        help='Number of negative sample for each link.')
    parser.add_argument('--save_file', '-sf', action='store_true',
                        help='是否要保存每次随机生成的数据')
    parser.add_argument('--model_name', '-mn', type=str, default='best_graph_classifier.pth',
                        help='修改使用具体使用哪个模型')
    parser.add_argument('--test_times', '-tt', type=int, default=1, 
                        help='测试几次并取平均')

    params = parser.parse_args()

    params.file_paths = {
        'graph': os.path.join('./data', params.dataset, 'train.txt'),
        'links': os.path.join('./data', params.dataset, 'test.txt')
    }

    params.ruleN_pred_path = os.path.join('./data', params.dataset, 'pos_predictions.txt')
    params.model_path = os.path.join('experiments', params.experiment_name, params.model_name)

    file_handler = logging.FileHandler(os.path.join('experiments', params.experiment_name, f'rank_test_{time.time()}.log'))

    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    logger.info('============================================')

    # 设置gpu
    if params.use_cuda and torch.cuda.is_available() and params.device >= 0:
        params.device = torch.device('cuda:%d' % params.device)
    else:
        params.device = torch.device('cpu')

    sum_all_mrr = []
    sum_all_hits_1 = []
    sum_all_hits_5 = []
    sum_all_hits_10 = []

    for i in range(params.test_times):
        logger.info(f"Test for the {i}st time.")
        result = main(params)
        for s in ['all', 'head', 'tail', 'rel']:
            mrr, hits_1, hits_5, hits_10 = analyse_result(result, s)
            if s == 'all':
                sum_all_mrr.append(mrr)
                sum_all_hits_1.append(hits_1)
                sum_all_hits_5.append(hits_5)
                sum_all_hits_10.append(hits_10)
            logger.info('{} RESULT: MRR | Hits@1 | Hits@5 | Hits@10 : {:.5f} | {:.5f} | {:.5f} | {:.5f}'.format(s.upper(), mrr, hits_1, hits_5, hits_10))

    mean_mrr = np.mean(sum_all_mrr)
    mean_hits_1 = np.mean(sum_all_hits_1)
    mean_hits_5 = np.mean(sum_all_hits_5)
    mean_hits_10 = np.mean(sum_all_hits_10)

    logger.info('Test {} times.\nMean result: MRR | Hits@1 | Hits@5 | Hits@10 : {:.5f} | {:.5f} | {:.5f} | {:.5f}'.format(params.test_times, mean_mrr, mean_hits_1, mean_hits_5, mean_hits_10))