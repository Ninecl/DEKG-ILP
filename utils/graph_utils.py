import statistics
import numpy as np
import scipy.sparse as ssp
import torch
import networkx as nx
import dgl
import pickle


def serialize(data):
    data_tuple = tuple(data.values())
    return pickle.dumps(data_tuple)


def deserialize(data):
    data_tuple = pickle.loads(data)
    [n1, n2], nodes, r_label, g_label, n_labels, subgraph_size, n1_con_pos, n1_con_neg, n2_con_pos, n2_con_neg = data_tuple

    return [n1, n2], nodes, r_label, g_label, n_labels, n1_con_pos, n1_con_neg, n2_con_pos, n2_con_neg


def get_edge_count(adj_list):
    count = []
    for adj in adj_list:
        count.append(len(adj.tocoo().row.tolist()))
    return np.array(count)


def incidence_matrix(adj_list):
    '''
    adj_list: List of sparse adjacency matrices
    '''

    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def ssp_to_torch(A, device, dense=False):
    '''
    A : Sparse adjacency matrix
    '''
    idx = torch.LongTensor([A.tocoo().row, A.tocoo().col])
    dat = torch.FloatTensor(A.tocoo().data)
    A = torch.sparse.FloatTensor(idx, dat, torch.Size([A.shape[0], A.shape[1]])).to(device=device)
    return A


def ssp_multigraph_to_dgl(graph, n_feats=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """

    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    for rel, adj in enumerate(graph):
        # Convert adjacency matrix to tuples for nx0
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)

    # make dgl graph
    g_dgl = dgl.from_networkx(g_nx, edge_attrs=['type'])
    # add node features
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl


def collate_dgl(samples):
    # The input `samples` is a list of pairs
    links_rsf_pos, graphs_pos, g_labels_pos, r_labels_pos, n1_conpos_pos, n1_conneg_pos, n2_conpos_pos, n2_conneg_pos, \
    links_rsf_neg, graphs_negs, g_labels_negs, r_labels_negs, n1_conpos_negs, n1_conneg_negs, n2_conpos_negs, n2_conneg_negs = map(list, zip(*samples))

    graphs_neg = [item for sublist in graphs_negs for item in sublist]
    g_labels_neg = [item for sublist in g_labels_negs for item in sublist]
    r_labels_neg = [item for sublist in r_labels_negs for item in sublist]
    links_rsf_neg = [item for sublist in links_rsf_neg for item in sublist]

    batched_graph_pos = dgl.batch(graphs_pos)
    batched_graph_neg = dgl.batch(graphs_neg)

    return (links_rsf_pos, batched_graph_pos, np.array(g_labels_pos), np.array(r_labels_pos)), (np.array(n1_conpos_pos), np.array(n1_conneg_pos), np.array(n2_conpos_pos), np.array(n2_conneg_pos)), \
           (links_rsf_neg, batched_graph_neg, np.array(g_labels_neg), np.array(r_labels_neg)), (np.array(n1_conpos_negs), np.array(n1_conneg_negs), np.array(n2_conpos_negs), np.array(n2_conneg_negs))


def move_batch_to_device_dgl(batch, device):
    (links_rsf_pos, graph_pos, g_labels_pos, r_labels_pos), (n1_conpos_pos, n1_conneg_pos, n2_conpos_pos, n2_conneg_pos), \
    (links_rsf_neg, graph_neg, g_labels_neg, r_labels_neg), (n1_conpos_negs, n1_conneg_negs, n2_conpos_negs, n2_conneg_negs) = batch

    # move tensor to device
    links_rsf_pos = torch.LongTensor(links_rsf_pos).to(device=device)
    links_rsf_neg = torch.LongTensor(links_rsf_neg).to(device=device)
    r_labels_pos = torch.LongTensor(r_labels_pos).to(device=device)
    r_labels_neg = torch.LongTensor(r_labels_neg).to(device=device)
    g_labels_pos = torch.LongTensor(g_labels_pos).to(device=device)
    g_labels_neg = torch.LongTensor(g_labels_neg).to(device=device)
    n1_conpos_pos = torch.LongTensor(n1_conpos_pos).to(device=device)
    n1_conneg_pos = torch.LongTensor(n1_conneg_pos).to(device=device)
    n2_conpos_pos = torch.LongTensor(n2_conpos_pos).to(device=device)
    n2_conneg_pos = torch.LongTensor(n2_conneg_pos).to(device=device)
    n1_conpos_negs = torch.LongTensor(n1_conpos_negs).to(device=device)
    n1_conneg_negs = torch.LongTensor(n1_conneg_negs).to(device=device)
    n2_conpos_negs = torch.LongTensor(n2_conpos_negs).to(device=device)
    n2_conneg_negs = torch.LongTensor(n2_conneg_negs).to(device=device)
    # move graph to device
    graph_pos = graph_pos.to(device=device)
    graph_neg = graph_neg.to(device=device)

    return (links_rsf_pos, graph_pos, r_labels_pos), (n1_conpos_pos, n1_conneg_pos, n2_conpos_pos, n2_conneg_pos), g_labels_pos, \
           (links_rsf_neg, graph_neg, r_labels_neg), (n1_conpos_negs, n1_conneg_negs, n2_conpos_negs, n2_conneg_negs), g_labels_neg


def send_graph_to_device(g, device):
    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device)

    # edges
    labels = g.edge_attr_schemes()
    for l in labels.keys():
        g.edata[l] = g.edata.pop(l).to(device)
    return g

#  The following three functions are modified from networks source codes to
#  accomodate diameter and radius for dirercted graphs


def eccentricity(G):
    e = {}
    for n in G.nbunch_iter():
        length = nx.single_source_shortest_path_length(G, n)
        e[n] = max(length.values())
    return e


def radius(G):
    e = eccentricity(G)
    e = np.where(np.array(list(e.values())) > 0, list(e.values()), np.inf)
    return min(e)


def diameter(G):
    e = eccentricity(G)
    return max(e.values())
