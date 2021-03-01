from engine.multiset import *


def distance_c_emd(hist1,hist2):
    dist = 0.5 * (distance_r_count(hist1,hist2) + EMD_hists(hist1, hist2))
    return dist


def distance_r_count(hist1, hist2):
    hist1_sum = sum(hist1)
    hist2_sum = sum(hist2)
    if (hist1_sum == 0 and hist2_sum != 0) or (hist2_sum == 0 and hist1_sum != 0):
        return 1
    elif hist1_sum == 0 and hist2_sum == 0:
        return 0
    else:
        dist = 1 - ((min(hist1_sum, hist2_sum))/(math.sqrt(hist1_sum * hist2_sum)))
        return dist


# helper function for calculating length
def __same_length_lists(list1, list2):
    while len(list1) != len(list2):
        list1.append(0)
    return list1


def EMD_hists(hist1, hist2):
    #normalize the two histograms
    if (sum(hist1) == 0 and sum(hist2) != 0) or (sum(hist2) == 0 and sum(hist1) != 0):
        return 1
    elif sum(hist2) == 0 and sum(hist1) == 0:
        return 0

    else:
        hist1 = __normalize_list(hist1)
        hist2 = __normalize_list(hist2)
        dist = np.zeros(len(hist1))
        for i in range(len(hist1)-1):
            dist[i+1] = (hist1[i] + dist[i]) - hist2[i]
        summ = np.sum(abs(dist))
        return summ


# helper function to compute manhatten distance
def __compute_manhatten_distance(hist1, hist2):
    sum_list = []
    for x, y in zip(hist1, hist2):
        sum_list.append(abs(x - y))
    distance = sum(sum_list)
    return distance


def __normalize_list(list):
        norm = [float(i) / sum(list) for i in list]
        return norm


def calc_distance(hist_tree1, hist_tree2, spec, root):
    spec_nodes = [n for n in edge_dfs(spec, source=root)]
    dist = []
    for x,y in spec_nodes:
        curr_node_hist1 = hist_tree1[y]['hist']
        curr_node_hist2 = hist_tree2[y]['hist']
        num_siblings = get_siblings(spec, y) + 1
        temp_dist = 1/num_siblings * distance_c_emd(curr_node_hist1[0], curr_node_hist2[0]) #1/num_siblings
        dist.append(temp_dist)
    res = sum(dist)
    return res


def get_siblings(aGraph, aNode):
    try:
        parentEdge = [(u, v, d) for u, v, d in aGraph.edges(data=True) if v == aNode]
        # print(parentEdge)
        parent = parentEdge[0][0]
        # print(parent)
        siblings = [v for u, v in aGraph.out_edges(parent) if v != aNode]
        # print(siblings)
        return len(siblings)
    except:
        return 0