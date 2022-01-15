import os
import math
import struct
import logging
import random
import pickle as pkl
import pdb
from tqdm import tqdm
import lmdb
import multiprocessing as mp
import numpy as np
import scipy.io as sio
import scipy.sparse as ssp
import sys
import torch
from scipy.special import softmax
from utils.dgl_utils import _bfs_relational
from utils.graph_utils import incidence_matrix, remove_nodes, ssp_to_torch, serialize, deserialize, get_edge_count, diameter, radius
import networkx as nx


def sample_neg(adj_list, edges, num_neg_samples_per_link=1, max_size=1000000, constrained_neg_prob=0):
    pos_edges = edges
    neg_edges = []

    # if max_size is set, randomly sample train links
    if max_size < len(pos_edges):
        perm = np.random.permutation(len(pos_edges))[:max_size]
        pos_edges = pos_edges[perm]

    # sample negative links for train/test
    n, r = adj_list[0].shape[0], len(adj_list)

    # distribution of edges across reelations
    theta = 0.001
    edge_count = get_edge_count(adj_list)
    rel_dist = np.zeros(edge_count.shape)
    idx = np.nonzero(edge_count)
    rel_dist[idx] = softmax(theta * edge_count[idx])

    # possible head and tails for each relation
    valid_heads = [adj.tocoo().row.tolist() for adj in adj_list]
    valid_tails = [adj.tocoo().col.tolist() for adj in adj_list]

    pbar = tqdm(total=len(pos_edges))
    while len(neg_edges) < num_neg_samples_per_link * len(pos_edges):
        neg_head, neg_tail, rel = pos_edges[pbar.n % len(pos_edges)][0], pos_edges[pbar.n % len(pos_edges)][1], pos_edges[pbar.n % len(pos_edges)][2]
        if np.random.uniform() < constrained_neg_prob:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(valid_heads[rel])
            else:
                neg_tail = np.random.choice(valid_tails[rel])
        else:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(n)
            else:
                neg_tail = np.random.choice(n)

        if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
            neg_edges.append([neg_head, neg_tail, rel])
            pbar.update(1)

    pbar.close()

    neg_edges = np.array(neg_edges)
    return pos_edges, neg_edges


def links2subgraphs(A, rsf, graphs, params, max_label_value=None):
    '''
    extract enclosing subgraphs, write map mode + named dbs
    '''
    intialize_worker(A, rsf, params, max_label_value)

    max_n_label = {'value': np.array([0, 0])}
    subgraph_sizes = []
    enc_ratios = []
    num_pruned_nodes = []

    BYTES_PER_DATUM = get_average_subgraph_size(100, list(graphs.values())[0]['pos'], A, params) * 1.5
    links_length = 0
    for split_name, split in graphs.items():
        links_length += (len(split['pos']) + len(split['neg'])) * 2
    map_size = links_length * BYTES_PER_DATUM

    env = lmdb.open(params.db_path, map_size=map_size, max_dbs=6)

    def extraction_helper(A, links, g_labels, split_env):

        with env.begin(write=True, db=split_env) as txn:
            txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)), byteorder='little'))

        with mp.Pool(processes=None, initializer=intialize_worker, initargs=(A, rsf, params, max_label_value)) as p:
            args_ = zip(range(len(links)), links, g_labels)
            for (str_id, datum) in tqdm(p.imap(extract_save_subgraph, args_), total=len(links)):
                max_n_label['value'] = np.maximum(np.max(datum['n_labels'], axis=0), max_n_label['value'])
                subgraph_sizes.append(datum['subgraph_size'])

                with env.begin(write=True, db=split_env) as txn:
                    txn.put(str_id, serialize(datum))

    for split_name, split in graphs.items():
        logging.info(f"Extracting enclosing subgraphs for positive links in {split_name} set")
        labels = np.ones(len(split['pos']))
        db_name_pos = split_name + '_pos'
        split_env = env.open_db(db_name_pos.encode())
        extraction_helper(A, split['pos'], labels, split_env)

        logging.info(f"Extracting enclosing subgraphs for negative links in {split_name} set")
        labels = np.zeros(len(split['neg']))
        db_name_neg = split_name + '_neg'
        split_env = env.open_db(db_name_neg.encode())
        extraction_helper(A, split['neg'], labels, split_env)

    max_n_label['value'] = max_label_value if max_label_value is not None else max_n_label['value']

    with env.begin(write=True) as txn:
        bit_len_label_sub = int.bit_length(int(max_n_label['value'][0]))
        bit_len_label_obj = int.bit_length(int(max_n_label['value'][1]))
        txn.put('max_n_label_sub'.encode(), (int(max_n_label['value'][0])).to_bytes(bit_len_label_sub, byteorder='little'))
        txn.put('max_n_label_obj'.encode(), (int(max_n_label['value'][1])).to_bytes(bit_len_label_obj, byteorder='little'))


def get_average_subgraph_size(sample_size, links, A, params):
    total_size = 0
    for (n1, n2, r_label) in links[np.random.choice(len(links), sample_size)]:
        nodes, n_labels, subgraph_size = subgraph_extraction_labeling((n1, n2), r_label, A, params.hop, params.enclosing_sub_graph, params.max_nodes_per_hop)
        n1_con_pos, n1_con_neg = contrastive_data_generate(n1, rsf_)
        n2_con_pos, n2_con_neg = contrastive_data_generate(n2, rsf_)
        datum = {'link': [n1, n2], 'nodes': nodes, 'r_label': r_label, 'g_label': 0, 'n_labels': n_labels, 'subgraph_size': subgraph_size,
                 'n1_con_pos': n1_con_pos, 'n1_con_neg': n1_con_neg, 'n2_con_pos': n2_con_pos, 'n2_con_neg': n2_con_neg}
        total_size += len(serialize(datum))

    return total_size / sample_size


def intialize_worker(A, rsf, params, max_label_value):
    global A_, rsf_, params_, max_label_value_
    A_, rsf_, params_, max_label_value_ = A, rsf, params, max_label_value


def extract_save_subgraph(args_):
    idx, (n1, n2, r_label), g_label = args_
    nodes, n_labels, subgraph_size = subgraph_extraction_labeling((n1, n2), r_label, A_, params_.hop, params_.enclosing_sub_graph, params_.max_nodes_per_hop)
    n1_con_pos, n1_con_neg = contrastive_data_generate(n1, rsf_)
    n2_con_pos, n2_con_neg = contrastive_data_generate(n2, rsf_)

    # max_label_value_ is to set the maximum possible value of node label while doing double-radius labelling.
    if max_label_value_ is not None:
        n_labels = np.array([np.minimum(label, max_label_value_).tolist() for label in n_labels])

    datum = {'link': [n1, n2], 'nodes': nodes, 'r_label': r_label, 'g_label': g_label, 'n_labels': n_labels, 'subgraph_size': subgraph_size,
             'n1_con_pos': n1_con_pos, 'n1_con_neg': n1_con_neg, 'n2_con_pos': n2_con_pos, 'n2_con_neg': n2_con_neg}

    str_id = '{:08}'.format(idx).encode('ascii')

    return (str_id, datum)


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

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

    labels, enclosing_subgraph_nodes = node_label(incidence_matrix(subgraph), max_distance=h)

    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]
    # pruned_subgraph_nodes = subgraph_nodes
    # pruned_labels = labels

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    subgraph_size = len(pruned_subgraph_nodes)

    return pruned_subgraph_nodes, pruned_labels, subgraph_size


def node_label(subgraph, max_distance=1):
    # implementation of the node labeling scheme described in the paper
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    return labels, enclosing_subgraph_nodes


def contrastive_data_generate(n, rsf):
    # 先计算平均rsf是多少
    n_rsf = rsf[n]
    no_zero_idx = np.where(n_rsf > 0)[0]
    n_rsf_no_zero = n_rsf[no_zero_idx]
    avg_rsf = int(sum(n_rsf_no_zero) / len(n_rsf_no_zero))
    avg_rsf = avg_rsf if avg_rsf > 1 else 1
    # 根据平均数计算采样范围
    sample_size = avg_rsf * 2 + 1

    # 生成对比学习正样本数据
    # 首先确认要更改的数据个数，根据参数指定，如果计算出来不足1则补为
    num_no_zero = len(no_zero_idx)
    num_change = int(num_no_zero * params_.con_change_percent)
    num_change = 1 if num_change < 1 else num_change

    # 正样本生成
    con_pos = []
    for i in range(0, params_.con_sample_num):
        change_idx = no_zero_idx[np.random.choice(np.array(range(num_no_zero)), num_change, False)]
        change_data = np.random.randint(1, sample_size, num_change)
        pos_rsf = n_rsf.copy()
        pos_rsf[change_idx] = change_data
        con_pos.append(pos_rsf)

    # 负样本生成
    con_neg = []
    for i in range(0, params_.con_sample_num):
        # 需要添加与删除的数据
        num_add = num_del = int(num_change / 2) if int(num_change / 2) >= 1 else 1

        candidate_add_idx = np.array(list(set(range(len(n_rsf))) - set(no_zero_idx)))
        add_idx = np.random.choice(candidate_add_idx, num_add, False)
        add_data = np.random.randint(1, sample_size, num_add)
        del_idx = no_zero_idx[np.random.choice(np.array(range(num_no_zero)), num_del, False)]
        del_data = np.zeros_like(del_idx)
        neg_rsf = n_rsf.copy()
        neg_rsf[del_idx] = del_data
        neg_rsf[add_idx] = add_data
        con_neg.append(neg_rsf)
        if sum(neg_rsf) == 0:
            print(n_rsf)
            print(add_idx)
            print(add_data)
            print(del_idx)
            print(del_data)
            print(neg_rsf)
    return np.array(con_pos), np.array(con_neg)
