import pandas as pd
import networkx as nx
import numpy as np
import time
from itertools import permutations, combinations
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import random
import matplotlib as mpl
from operator import itemgetter

import sys

mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 15


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
    if num != 0:
        d, my_dict, g = add_class(g, avg_len, d, my_dict, num=num)
    labels_clique = list(set(my_dict.values()))
    # print("there are {} labels".format(len(labels_clique)))
    # g = add_nodes_edges(g, my_dict)
    return g, my_dict, d, avg_len


def add_class(g, avg_len, d, my_dict, num=1):
    for it in range(num):
        nodes = list(g.nodes())
        new_class = max(list(d.keys())) + 1
        new_nodes = [j for j in range(max(nodes) + 1, int(avg_len) + max(nodes))]
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
    miss = [pair for pair in combinations(g.nodes(), 2) if
            not g.has_edge(*pair) and my_dict[pair[0]] != my_dict[pair[1]]]
    old = [pair for pair in combinations(old_g.nodes(), 2) if
           not old_g.has_edge(*pair) and my_dict[pair[0]] != my_dict[pair[1]]]
    final_miss = list(set(miss) - set(old))
    num_edges = int(p1 * (g.number_of_nodes() * (g.number_of_nodes() - 1) / 2))
    current_num_edges = g.number_of_edges()
    random.shuffle(final_miss)
    g.add_edges_from(final_miss[:num_edges - current_num_edges])
    return g


def is_clique(g, nodes, label_to_node_, node_to_label_, trip=False):
    h = g.subgraph(nodes)
    n = len(nodes)
    if int(h.number_of_edges()) == int(n * (n - 1) / 2):
        if len(nodes) == len(label_to_node_):
            if not trip:
                return True
            else:
                my_labels = []
                for l in nodes:
                    my_labels.append(node_to_label_[l])
                if len(set(my_labels)) == len(label_to_node_):
                    print(my_labels)
                    return True
                else:
                    return False
        else:
            return False
    else:
        return False


def greedy(clique, g, nodes, label_to_node_, node_to_label_, trip=False):
    if trip:
        for v in clique:
            not_connected = [n for n in nodes if not g.has_edge(n, v) and n != v]
            g.remove_nodes_from(not_connected)
            nodes = list(g.nodes())
        node2deg = {n: g.degree(n) for n in nodes if n not in clique}
        node_max_deg = max(node2deg, key=node2deg.get)
        clique.append(node_max_deg)
    while is_clique(g, clique, label_to_node_, node_to_label_, trip=trip) is False:
        v = clique[-1]
        not_connected = [n for n in nodes if not g.has_edge(n, v) and n != v]
        g.remove_nodes_from(not_connected)
        nodes = list(g.nodes())
        node2deg = {n: g.degree(n) for n in nodes if n not in clique}
        node_max_deg = max(node2deg, key=node2deg.get)
        clique.append(node_max_deg)
    print(clique)
    return clique


def bron_kerbosch(graph, labels, label_to_node_, potential_clique, remaining_nodes, skip_nodes, depth=0):
    if len(remaining_nodes) == 0 and len(skip_nodes) == 0:
        my_labels = []
        for j in potential_clique:
            my_labels.append(labels[j])
        if len(potential_clique) == len(label_to_node_):
            if len(set(my_labels)) == len(label_to_node_):
                print("success")
                #sys.exit()
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
        found_cliques += bron_kerbosch(graph, labels, label_to_node_, new_potential_clique, new_remaining_nodes,
                                       new_skip_list, depth + 1)

        # We're done considering this node.  If there was a way to form a clique with it, we
        # already discovered its maximal clique in the recursive call above.  So, go ahead
        # and remove it from the list of remaining nodes and add it to the skip list.
        remaining_nodes.remove(node)
        skip_nodes.append(node)

    return found_cliques


def bron_kerbosch_kanna(graph, labels, label_to_node_, potential_clique, remaining_nodes, skip_nodes, depth=0):
    if len(remaining_nodes) == 0 and len(skip_nodes) == 0:
        my_labels = []
        for j in potential_clique:
            my_labels.append(labels[j])
        if len(potential_clique) == len(label_to_node_):
            if len(set(my_labels)) == len(label_to_node_):
                print("success")
                #sys.exit()
        return 1

    for node in remaining_nodes:
        # Try adding the node to the current potential_clique to see if we can make it work.
        new_potential_clique = potential_clique + [node]
        new_remaining_nodes = [n for n in remaining_nodes if n in list(graph.neighbors(node))]
        new_skip_list = [n for n in skip_nodes if n in list(graph.neighbors(node))]
        if bron_kerbosch_kanna(graph, labels, label_to_node_, new_potential_clique, new_remaining_nodes,
                                       new_skip_list, depth + 1):
            return 1

        # We're done considering this node.  If there was a way to form a clique with it, we
        # already discovered its maximal clique in the recursive call above.  So, go ahead
        # and remove it from the list of remaining nodes and add it to the skip list.
        remaining_nodes.remove(node)
        skip_nodes.append(node)

    return 0


def dm(graph, labels, sg_sz):
    """
    An implementation of the algorithm of Deshpande and Montanari for clique recovery
    using approximate message passing (i.e. belief propagation)
    """
    # graphs, all_labels = graphs_loader(sz, p, sg_sz, subgraph)
    remaining_subgraph_vertices = []
    sz = graph.number_of_nodes()
    p = graph.number_of_edges() / (sz * (sz - 1) / 2)

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


def my_algorithm(graph, labels_to_node, nodes_to_label):
    """
    The algorithm rank the labels and build clique with this order (take less communicate labels first)
    :param graph: a graph
    :param labels_to_node: a dictionary of label and the nodes in this label
    :param nodes_to_label: a dictionary of node and it's label
    :return: a clique
    """
    # rank nodes and labels
    node_score, label_score = rank_nodes(graph, labels_to_node, nodes_to_label)
    sorted_labels = {k: v for k, v in sorted(label_score.items(), key=lambda item: item[1])}
    # sort nodes in label by their score
    for label in labels_to_node:
        labels_to_node[label] = sorted(labels_to_node[label], key=lambda node: node_score[node])
    # find clique
    labels = label_to_node.keys()
    label_ind = 0
    node_label_ind = 0
    n_labels = len(label_to_node)
    remaining_nodes = graph.nodes
    clique = []
    while len(clique) < n_labels:
        label = labels[label_ind]
        potential_nodes = [n for n in labels_to_node[label][node_label_ind] if n in remaining_nodes]
        node = potential_nodes[0]
        clique.append(node)
        remaining_nodes = [n for n in remaining_nodes if n in list(graph.neighbors(node))]
        #check if is empty
        if not len(remaining_nodes):
            node_label_ind = node_label_ind + 1
            if node_label_ind > len(labels_to_node[label]):
                label_ind -= 1
        else:
            label_ind += 1
            node_label_ind = 0
    return clique


def kanna_algorithm(graph, node_to_label, label_to_node_, labels_list, potential_clique, remaining_nodes, label_ind=0):

    """
     The algorithm rank the labels and build clique with this order (take less communicate labels first)
     :param graph: a graph
     :param nodes_to_label: a dictionary of node and it's label
     :param labels_to_node: a dictionary of label and the nodes in this label
     :param labels_list: a list of all labels in graph
     :param potential_clique: the builded clique until now
     :param remaining_nodes: the potential nodes to be in the clique
     :param label_ind: which label we add now
     :return: a clique that contain all labels
     """

    # check if success
    if len(potential_clique) == len(labels_list):
        return potential_clique
    # run of the nodes in this label with their rank
    potential_nodes_in_label = [n for n in label_to_node_[labels_list[label_ind]] if n in remaining_nodes]
    for node in potential_nodes_in_label:
        # Try adding the node to the current potential_clique to see if we can make it work.
        new_potential_clique = potential_clique + [node]
        new_remaining_nodes = [n for n in remaining_nodes if n in list(graph.neighbors(node))]
        clique_founded = kanna_algorithm(graph, node_to_label, label_to_node_,labels_list, new_potential_clique, new_remaining_nodes,
                                       label_ind + 1)
        if clique_founded:
            return clique_founded

        # We're done considering this node.  If there was a way to form a clique with it, we
        # already discovered its maximal clique in the recursive call above.  So, go ahead
        # and remove it from the list of remaining nodes.
        remaining_nodes.remove(node)
    return


def rank_nodes(graph, label_to_node, nodes_to_label):
    """
    The function gives score for each node - the score is the minimum number of neighborhoods that it has from certain label.
    And give each label a score that depends how much it connect to other labels.
    :param grapha: a graph
    :param label_to_node: dictionary of labels and all nodes with this label
    :param nodes_to_label: dictionary oo nodes and the label of each node
    :return: a dictionary of nodes with their score and labels with their score.
    """
    n_labels = max(label_to_node)
    nodes_score = {}
    labels_score = {}
    for node in graph.nodes:
        connected_labels = np.zeros(n_labels + 1, dtype= int)
        connected_nodes = list(graph.neighbors(node))
        for connected_node in connected_nodes:
            connected_labels[nodes_to_label[connected_node]] += 1
            nodes_score[node] = min(connected_labels[np.nonzero(connected_labels)])
    for label in label_to_node:
        labels_score[label] = np.mean(itemgetter(*label_to_node[label])(nodes_score))

    return nodes_score, labels_score


def remove_lonely_nodes(graph, label_to_node, nodes_to_label):
    """
    The function get a graph and labels and remove the nodes that hasn't neighbor of each label
    :param graph: a graph
    :param label_to_node: dictionary of lables and all nodes with this label
    :return: the graph without all those nodes
    """
    lonely_nodes = []
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        for label in label_to_node:
            if nodes_to_label[node] != label and not len(set(neighbors).intersection(label_to_node[label])) !=0:
                lonely_nodes.append(node)
                del nodes_to_label[node]
                label_to_node[label].remove(node)
                break
    # print("There are {} lonely nodes".format(len(lonely_nodes)))
    for node in lonely_nodes:
        graph.remove_nodes_from(node)
        # change numbers of left nodes
        graph = nx.relabel_nodes(graph, lambda x: x+1 if (x>node) else x)
    return graph, label_to_node, node_to_label, len(lonely_nodes)


def plot_times(times_list):
    plt.figure(0, figsize=(9, 7))
    classes = [i + 9 for i in range(len(times_list))]
    plt.xticks(np.arange(min(classes), max(classes) + 1, 1))
    plt.plot(classes, times_list, color="blue", linewidth=3)
    plt.title("Running Time VS Number of Classes")
    plt.xlabel("Number of classes")
    plt.ylabel("Running time [seconds]")
    plt.savefig("running_time.png")


def plot_all_times(times_list1, times_list2, times_list3, times_list4):
    plt.figure(0, figsize=(9, 7))
    classes = [i + 9 for i in range(len(times_list1))]
    plt.xticks(np.arange(min(classes), max(classes) + 1, 1))
    plt.plot(classes, times_list1, color="blue", linewidth=3, label="dm algorithm")
    plt.plot(classes, times_list2, color="red", linewidth=3, label="Bron–Kerbosch algorithm")
    plt.plot(classes, times_list3, color="green", linewidth=3, label="Greedy Algorithm")
    plt.plot(classes, times_list3, color="black", linewidth=3, label="New Algorithm")
    plt.title("Running Time VS Number of Classes")
    plt.xlabel("Number of classes")
    plt.ylabel("Running time [seconds]")
    plt.legend(fontsize='xx-large')
    plt.savefig("running_time.png")


if __name__ == '__main__':
    num_of_colours = 2
    times_dm = []
    times_bron_kerbosch = []
    times_greedy = []
    times_kanna_algorithm = []
    lonely_nodes = []
    graph_, node_to_label, label_to_node, avg_len_label = create_graph(0)
    for _ in range(num_of_colours):
        d, node_to_label, graph_ = add_class(graph_, avg_len_label, label_to_node, node_to_label, num=1)
        # graph_, label_to_node, node_to_label, n_lonely_nodes = remove_lonely_nodes(graph_, label_to_node, node_to_label)
        # lonely_nodes.append(n_lonely_nodes)
        t1 = time.time()
        total_time_, c_n_hat, final_clique_, avg_ = dm(graph_.copy(), node_to_label, 100)
        t2 = time.time()
        times_dm.append(t2 - t1)
        total_cliques_ = bron_kerbosch_kanna(graph_.copy(), node_to_label, label_to_node, [], list(graph_.nodes()), [])
        t3 = time.time()
        times_bron_kerbosch.append(t3-t2)
        clique = greedy([0], graph_.copy(), graph_.nodes, label_to_node, node_to_label, trip=False)
        t4 = time.time()
        times_greedy.append(t4-t3)

        # rank nodes and labels
        node_score, label_score = rank_nodes(graph_.copy(), label_to_node, node_to_label)
        sorted_labels = {k: v for k, v in sorted(label_score.items(), key=lambda item: item[1])}
        # sort nodes in label by their score
        for label in label_to_node:
            label_to_node[label] = sorted(label_to_node[label], key=lambda node: node_score[node])
        clique = kanna_algorithm(graph_, node_to_label, label_to_node, list(label_to_node.keys()), [], list(graph_.nodes), 0)
        t5 = time.time()
        times_kanna_algorithm.append((t5-t4))
    plot_all_times(times_dm, times_bron_kerbosch, times_greedy, times_kanna_algorithm)
    plt.show()