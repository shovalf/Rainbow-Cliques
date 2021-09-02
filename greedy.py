import pandas as pd
import networkx as nx
import numpy as np
import time
from itertools import permutations, combinations
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import random
import matplotlib as mpl
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


def is_clique(g, nodes, label_to_node_, node_to_label_, trip=False):
    h = g.subgraph(nodes)
    n = len(nodes)
    if int(h.number_of_edges()) == int(n*(n-1)/2):
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


def GC(clique, g, nodes, label_to_node_, node_to_label_, trip=False):
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


# def GC_triplets(clique, g, nodes, label_to_node_):
#     for v in clique:
#         not_connected = [n for n in nodes if not g.has_edge(n, v) and n != v]
#         g.remove_nodes_from(not_connected)
#         nodes = list(g.nodes())
#     while is_clique(g, clique, label_to_node_) is False:
#         v = clique[-1]
#         not_connected = [n for n in nodes if not g.has_edge(n, v) and n != v]
#         g.remove_nodes_from(not_connected)
#         nodes = list(g.nodes())
#         node2deg = {n: g.degree[n] for n in nodes if n not in clique}
#         node_max_deg = max(node2deg, key=node2deg.get)
#         clique.append(node_max_deg)
#     print(clique)
#     return clique


def plot_all_times(times_list1, times_list2):
    plt.figure(0, figsize=(9, 7))
    classes = [i + 9 for i in range(len(times_list1))]
    plt.xticks(np.arange(min(classes), max(classes) + 1, 1))
    plt.plot(classes, times_list1, color="blue", linewidth=3, label="Bron–Kerbosch algorithm")
    plt.plot(classes, times_list2, color="red", linewidth=3, label="Greedy Algorithm")
    plt.title("Running Time VS Number of Classes")
    plt.xlabel("Number of classes")
    plt.ylabel("Running time [seconds]")
    plt.legend(fontsize='xx-large')
    plt.savefig("running_time.png")


def plot_times(times_list):
    plt.figure(0, figsize=(9, 7))
    classes = [i + 9 for i in range(len(times_list))]
    plt.xticks(np.arange(min(classes), max(classes)+1, 1))
    plt.plot(classes, times_list, color="blue", linewidth=3)
    plt.title("Running Time VS Number of Classes")
    plt.xlabel("Number of classes")
    plt.ylabel("Running time [seconds]")
    plt.savefig("running_time.png")


# times1 = [0.14, 0.2, 0.26, 0.34, 0.42, 0.53, 0.63, 0.73, 0.94, 1.0, 1.14, 1.41, 1.41, 1.69, 1.76, 1.98, 2.23, 2.42, 2.68]
# times1 = [0.14, 0.2, 0.26, 0.34, 0.42, 0.53, 0.63, 0.73, 0.94, 1.0, 1.14, 1.41, 1.41, 1.69, 1.76, 1.98, 2.23]
# times2 = [0.07, 0.1, 0.15, 0.12, 0.15, 0.17, 0.22, 0.21, 0.25, 0.29, 0.32, 0.46, 0.53, 0.42, 0.5, 0.52, 2.04]
# plot_all_times(times2, times1)
times = []
for num in range(1):
    t_ = time.time()
    graph_, node_to_label, label_to_node, avg_len_label = create_graph(num)
    middle = time.time()
    print("time to create graph: ", middle - t_)
    nodes_ = list(graph_.nodes())
    clique = GC([0], graph_, nodes_, label_to_node, node_to_label, trip=False)
    triplets = [pair for pair in combinations(clique, 3)]
    final_clique = GC(list(triplets[0]), graph_, nodes_, label_to_node, node_to_label, trip=True)
    final_time = round(time.time() - middle, 2)
    print("final time: ", final_time)
    times.append(final_time)
# print(times)
# plot_times(times)




