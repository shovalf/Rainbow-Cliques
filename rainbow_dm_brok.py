import pandas as pd
import networkx as nx
import numpy as np
import time
from itertools import permutations, combinations
from scipy.linalg import eigh
import random
import matplotlib.pyplot as plt
import sys


def create_graph(num):
    matrix = pd.read_csv("M_251189.CSV", header=None)
    matrix = matrix.drop([1094, 1095], axis=1)
    adj = matrix[1:]
    adj = adj[:-1]
    labels = matrix[:1].to_numpy()
    my_dict = {}
    for l in range(len(labels[0, :])):
        my_dict.update({l: int(labels[0, l])})
    g = nx.Graph()
    keys = list(my_dict.keys())
    for key in keys:
        g.add_node(key, label=my_dict[key])
    adj_np = adj.to_numpy()
    edges_dict = {k: [] for k in keys}
    for col in range(adj_np.shape[1]):
        b = adj_np[:, col]
        for i in range(b.shape[0]):
            if b[i] == 1:
                edges_dict[col].append(i)
    nodes = list(g.nodes())
    for n in nodes:
        for neigh in edges_dict[n]:
            g.add_edge(n, neigh)
    labels_clique = list(set(my_dict.values()))
    d = {j: [] for j in labels_clique}
    for vv in keys:
        d[my_dict[vv]].append(vv)
    lens = [len(h) for h in list(d.values())]
    avg_len = np.mean(lens)
    d, my_dict, g = add_class(g, avg_len, d, my_dict, num=num)
    # g = add_nodes_edges(g, my_dict)
    return g, my_dict, d, avg_len


def add_class(g, avg_len, d, my_dict, num=1):
    for it in range(num):
        nodes = list(g.nodes())
        new_class = max(list(d.keys())) + 1
        new_nodes = [j for j in range(max(nodes)+1, int(avg_len) + max(nodes))]
        for n in new_nodes:
            nodes.append(n)
        d.update({new_class: new_nodes})
        my_dict.update({k: new_class for k in new_nodes})
        g = add_nodes_edges(g, my_dict)
    return d, my_dict, g


def add_nodes_edges(g, my_dict):
    old_g = g.copy()
    keys = list(my_dict.keys())
    new_nodes = list(set(keys) - set(g.nodes()))
    g.add_nodes_from(new_nodes)
    p1 = old_g.number_of_edges() / (old_g.number_of_nodes() * (old_g.number_of_nodes() - 1) / 2)
    miss = [pair for pair in combinations(g.nodes(), 2) if not g.has_edge(*pair) and my_dict[pair[0]] != my_dict[pair[1]]]
    old = [pair for pair in combinations(old_g.nodes(), 2) if not old_g.has_edge(*pair) and my_dict[pair[0]] != my_dict[pair[1]]]
    final_miss = list(set(miss) - set(old))
    num_edges = int(p1 * (g.number_of_nodes() * (g.number_of_nodes() - 1) / 2))
    current_num_edges = g.number_of_edges()
    random.shuffle(final_miss)
    g.add_edges_from(final_miss[:num_edges - current_num_edges])
    return g


def find_max_vertices(labels_clique, labels, list_v, num=5):
    d = {j: [] for j in labels_clique}
    for vv in list_v:
        d[labels[vv]].append(vv)
    my_clique = []
    for ll in labels_clique:
        try:
            my = d[ll][:num]
        except:
            my = d[ll]
        my_clique.append(my)
    return my_clique


def run_dm(graph, labels, sg_sz):
    """
    An implementation of the algorithm of Deshpande and Montanari for clique recovery
    using approximate message passing (i.e. belief propagation)
    """
    # graphs, all_labels = graphs_loader(sz, p, sg_sz, subgraph)
    remaining_subgraph_vertices = []
    sz = graph.number_of_nodes()
    p = graph.number_of_edges() / (sz * (sz-1) / 2)

    start_time = time.time()
    w = nx.adjacency_matrix(graph).toarray()
    # w = nx.to_numpy_array(graph)
    for i, j in permutations(range(w.shape[0]), 2):
        if i != j and w[i, j] == 0:
            w[i, j] = -1
        elif w[i, j] == 1:
            w[i, j] = (1 - p) / p  # DM is adjusted to match its requirement that E[W] = 0.
    kappa = sg_sz / np.sqrt(sz)
    gamma_vectors = [np.ones((sz,))]
    gamma_matrices = [np.subtract(np.ones((sz, sz)), np.eye(sz))]
    t_star = 100
    # Belief Propagation iterations. The code here uses some matrices to shorten the time comparing to for loops #
    for t in range(t_star):
        helping_matrix = np.exp(gamma_matrices[t]) / np.sqrt(sz)
        log_numerator = np.log(1 + np.multiply(1 + w, helping_matrix))
        log_denominator = np.log(1 + helping_matrix)
        helping_for_vec = log_numerator - log_denominator
        gamma_vec = np.log(kappa) + np.sum(helping_for_vec, axis=1) - np.diag(helping_for_vec)
        gamma_mat = np.tile(gamma_vec, (sz, 1)) - helping_for_vec.transpose()
        gamma_vectors.append(gamma_vec)
        gamma_matrices.append(gamma_mat)
    sorted_vertices = np.argsort(gamma_vectors[t_star])
    list_v = list(sorted_vertices)
    list_v.reverse()
    clique, labels_clique, indices = [], [], []
    for v in range(len(list_v)):
        if v == 0:
            clique.append(list_v[v])
            labels_clique.append(labels[list_v[v]])
            indices.append(v)
        else:
            if labels[list_v[v]] not in labels_clique:
                clique.append(list_v[v])
                labels_clique.append(labels[list_v[v]])
                indices.append(v)
    c_n_hat = sorted_vertices[-2 * sg_sz:]
    # Without the cleaning stage which is similar to ours.
    remaining_subgraph_vertices.append(len([v for v in c_n_hat if labels[v]]))
    total_time = time.time() - start_time
    final = find_max_vertices(labels_clique, labels, list_v, num=10)
    return total_time, c_n_hat, final, np.mean(remaining_subgraph_vertices)


# Utils and Algorithms:
def subgraph_test(graph, subgraph, final_set, subgraph_vertices=None):
    """
    Examine whether the planted subgraph pattern was found. For G(k, q), this examination requires knowing the vertices
    that belong to the subgraph, otherwise we look for a specific pattern and do not care whether we found exactly the
    vertices of the planted subgraph.
    """
    if subgraph == "clique":
        return all([graph.has_edge(v1, v2) for v1, v2 in combinations(final_set, 2)])
    else:  # G(k, q). The only case we have the exact vertices we want and not a subgraph shape.
        return len(subgraph_vertices) == len(set(subgraph_vertices).intersection(set(final_set)))


def condition(s, updates, graph, subgraph):
    if subgraph in ["clique", "biclique", "dag-clique", "k-plex"]:
        return not subgraph_test(graph, subgraph, s) and updates < 50
    else:
        return updates < 50


def cleaning_algorithm(dm_candidates, graph, subgraph, sg_sz):
    dm_adjacency = nx.adjacency_matrix(graph, nodelist=dm_candidates).toarray()
    normed_dm_adj = (dm_adjacency + dm_adjacency.T) - 1 + np.eye(dm_adjacency.shape[0])  # Zeros on the diagonal
    _, eigenvec = eigh(normed_dm_adj, eigvals=(normed_dm_adj.shape[0] - 1, normed_dm_adj.shape[0] - 1))
    dm_next_set = [dm_candidates[v] for v in np.argsort(np.abs(eigenvec.ravel()))[-sg_sz:].tolist()]
    updates = 0
    while condition(dm_next_set, updates, graph, subgraph):
        connection_to_set = [len(set(graph.neighbors(v)).intersection(set(dm_next_set))) for v in graph]
        dm_next_set = np.argsort(connection_to_set)[-sg_sz:].tolist()
        updates += 1
    return dm_next_set, updates


def find_sub_graph(graph, clique):
    vertices = []
    for c in clique:
        for cc in c:
            vertices.append(cc)
    subgraph = graph.subgraph(vertices)
    return subgraph


def find_c_hat_subgraph(graph, clique):
    subgraph = graph.subgraph(list(clique))
    return subgraph


def find_cliques(graph, labels, label_to_node_, potential_clique, remaining_nodes, skip_nodes, t, depth=0):
    if len(remaining_nodes) == 0 and len(skip_nodes) == 0:
        my_labels = []
        for j in potential_clique:
            my_labels.append(labels[j])
        if len(potential_clique) == len(label_to_node_):
            if len(set(my_labels)) == len(label_to_node_):
                print('This is a clique:', potential_clique)
                final_time = time.time() - t
                print("final time: ", round(final_time, 2))
                # sys.exit()
            else:
                print("not colorful", potential_clique)
        else:
            print("not in good length", potential_clique)
        return 1

    found_cliques = 0
    for node in remaining_nodes:

        # Try adding the node to the current potential_clique to see if we can make it work.
        new_potential_clique = potential_clique + [node]
        new_remaining_nodes = [n for n in remaining_nodes if n in list(graph.neighbors(node))]
        new_skip_list = [n for n in skip_nodes if n in list(graph.neighbors(node))]
        found_cliques += find_cliques(graph, labels, label_to_node_, new_potential_clique, new_remaining_nodes,
                                      new_skip_list, t_, depth + 1)

        # We're done considering this node.  If there was a way to form a clique with it, we
        # already discovered its maximal clique in the recursive call above.  So, go ahead
        # and remove it from the list of remaining nodes and add it to the skip list.
        remaining_nodes.remove(node)
        skip_nodes.append(node)

    return found_cliques


def first_sol(g, node2label, label2node, t):
    total_time_, c_n_hat, final_clique_, avg_ = run_dm(g, node2label, 100)
    subgraph_ = find_sub_graph(g, final_clique_)
    total_cliques_ = find_cliques(subgraph_, node2label, label2node, [], list(subgraph_.nodes()), [], t)
    return total_cliques_


def second_sol(g, node2label, label2node, t):
    total_cliques_ = find_cliques(g, node2label, label2node, [], list(g.nodes()), [], t)
    return total_cliques_


a=0
t_ = time.time()
graph_, node_to_label, label_to_node, avg_len_label = create_graph(0)

colors = ["tab:orange", "tab:green", "tab:red", "tab:blue", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive"]

subgraph = graph_.subgraph([0, 5, 167, 189, 364, 398, 577, 744, 924])
# nx.draw_networkx(subgraph, with_labels=True, node_size=500, node_color=colors)
# plt.show()


# labels = list(label_to_node.keys())
# label2color = {labels[l]: colors[l] for l in range(len(labels))}
# nodes = list(graph_.nodes())
# node2color = {n: label2color[node_to_label[n]] for n in nodes}
# colors_list = list(node2color.values())
# nx.draw_networkx(graph_, with_labels=False, node_size=35, node_color=colors_list)
# plt.show()

# my = time.time() - t_
# print(my)
# t_ = time.time()
a = 2
if a==1:
    total_cliques = first_sol(graph_, node_to_label, label_to_node, t_)
else:
    total_cliques = second_sol(graph_, node_to_label, label_to_node, t_)
print("i stopped")
# total_cliquess = second_sol(graph_, node_to_label, label_to_node, t_)

# total_time, c_n_hat_, final_clique, avg = run_dm(graph_, node_to_label, 50)
# subgraph = find_sub_graph(graph_, final_clique)
# # subgraph = find_c_hat_subgraph(graph_, c_n_hat_)
# # b = nx.algorithms.clique.find_cliques(graph_)
# print("start")
# total_cliques = find_cliques(subgraph, node_to_label, label_to_node, [], list(subgraph.nodes()), [], t_)
# print("end")
# # total_cliques = find_cliques(graph_, node_to_label, label_to_node, [], list(graph_.nodes()), [])
# a=1
# # dm_next_set_, updates_ = cleansing_algorithm(c_n_hat_, graph_, "clique", 100)
# # print(dm_next_set_)
a = 1
print("finished")