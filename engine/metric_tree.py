from engine.distance import *


# helper function to get random pair
def __get_random_pair(data, spec, root):
    dist = 0
    while dist == 0:
        v1,v2 = np.random.choice(data, 2, replace=False)
        v1_hist = v1.get_histogram()
        v2_hist = v2.get_histogram()
        dist = calc_distance(v1_hist, v2_hist, spec, root)
    return v1, v2


# helper function to split data according to distance to v1 and v2
def __split_data(data, v1, v2, spec, root):
    data_1 = []
    data_2 = []
    v1_hist = v1.get_histogram()
    v2_hist = v2.get_histogram()
    for g in data:
        g_hist = g.get_histogram()
        v1_len = calc_distance(g_hist, v1_hist, spec, root)
        v2_len = calc_distance(g_hist, v2_hist, spec, root)
        if v1_len < v2_len:
            data_1.append(g)
        elif v2_len < v1_len:
            data_2.append(g)
    return data_1, data_2


# function for calculating distance between two histograms
def __distance(v1,v2, spec, root):
    v1_hist = v1.get_histogram()
    v2_hist = v2.get_histogram()
    return calc_distance(v1_hist, v2_hist, spec, root)


def __sort_dist(val,h,spec,root):
    dist = __distance(val,h,spec,root)
    return dist


# helper function to build metric tree
def __mt_build(g, d_max, b_max, d, data, name, spec, root):
    if not (d == d_max or len(data) <= b_max):
        z1, z2 = __get_random_pair(data, spec, root)
        data_1, data_2 = __split_data(data, z1, z2, spec, root)
        __mt_build(g, d_max, b_max, d + 1, data_1, f'{name}_l', spec, root) # 1
        __mt_build(g, d_max, b_max, d + 1, data_2, f'{name}_r', spec, root) # 2
        g.add_node(f'{name}', left=f'{name}_l', right=f'{name}_r', z1=z1, z2=z2) #left=data_1, right=data_2
        g.add_edge(f'{name}', f'{name}_l')
        g.add_edge(f'{name}', f'{name}_r')
    else:
        g.add_node(f'{name}', bucket=data)


# function for building metric tree
def mt_build(tet, k, bucket_max, spec):
    g = nx.DiGraph()
    __mt_build(g, k, bucket_max, 0, tet, 0, spec, 'user')
    return g


# helper function for searching metric tree
def __mt_search(g, mn, h, k, leafs, spec, root):
    if mn in leafs:
        bucket = g.nodes(data=True)[mn]['bucket']
        dist = [__distance(h, b, spec, root) for b in bucket]
        # bucket_sorted = [x for _, x in sorted(zip(dist, bucket))]
        bucket_sorted = [x for _, x in sorted(zip(dist, bucket), key=lambda pair: pair[0])]
        if len(bucket_sorted) < k:
            bucket_sorted.pop(0)
            return bucket_sorted
        else:
            bucket_sorted.pop(0)
            return bucket_sorted[:k]
    dist1 = __distance(h, g.nodes(data=True)[mn]['z1'], spec, root)
    dist2 = __distance(h, g.nodes(data=True)[mn]['z2'], spec, root)
    if dist1 <= dist2:
        return __mt_search(g, g.nodes(data=True)[mn]['left'], h, k, leafs, spec, root)
    else:
        return __mt_search(g, g.nodes(data=True)[mn]['right'], h, k, leafs, spec, root)


# function for searching metric tree
def mt_search(g, user_tet, k, spec):
    leaf_nodes = [node for node in g.nodes if
                  (g.in_degree(node) != 0 and g.out_degree(node) == 0)]
    root = [node for node in g.nodes if (g.in_degree(node) == 0 and g.out_degree != 0)]
    res = __mt_search(g, root[0], user_tet, k + 1, leaf_nodes, spec, 'user')
    return res