from engine.multiset import *
from engine.data_setup import *


def get_genres():
    l = []
    for index, movie in data.iterrows():
         for genre in movie['genres']:
            l.append(genre)
    s = set(l)
    final_list = list(s)
    final_list.sort()
    return final_list


# creates an overall directed bipartite graph used for constructing TETs
def generate_bipartite_graph(genres):
    B = nx.DiGraph()
    B.add_nodes_from(rdata.userId, bipartite=0, user=True, free=True, type='user')
    B.add_nodes_from(updated_data.movieId, bipartite=1, movie=True, free=True, type='movie')
    B.add_nodes_from(updated_data.director, bipartite=1, director=True, free=True, type='director')
    for index,row in updated_data.iterrows():
        B.add_edge(row['movieId'], row['director'])
    B.add_edges_from([(uId, mId) for (uId, mId) in rdata[['userId', 'movieId']].to_numpy()])
    for index, movie in data.iterrows():
        for genre in movie['genres']:
            B.add_node(genre, bipartite=1, genre=True, free=False, type='genre')
            B.add_edge(movie['movieId'], genre)
    return B


# function used for loading and creating a tet-specification
def tet_specification(nodes, edges, freevars):
    g = nx.DiGraph()
    for n in nodes:
        if n in freevars:
            g.add_node(n, type=n, free=True)
        else:
            g.add_node(n, type=n, free=False)
    g.add_edges_from(edges)
    return g


def tet_specification2(nodes, edges, freevars,genres):
    g = nx.DiGraph()
    for n in nodes:
        if n in freevars:
            g.add_node(n, type=n, free=True)
        elif n == 'genre':
            for genre in genres:
                g.add_node(genre,type=n,free=False)
        else:
            g.add_node(n, type=n, free=False)
    for genre in genres:
        g.add_edge('movie',genre)
    g.add_edges_from(edges)
    return g


# function used to generate our TETs for each user (root) based on the overall graph, and our specification
def generate_tet(graph, root, spec):
    roots = [n for n, info in graph.nodes(data=True) if info.get(f'{root}')]
    complete = []
    for r in roots:
        complete.append(__generate_tet(graph, r, spec))
    return complete


# private helper function used to generate TETs
def __generate_tet(graph, user, spec):
    nodes = [n[-1] for n in dfs_edges(spec, source="user")]
    ms = Multiset()     # here we instantiate our TETs
    ms.add_root(user, len(graph.out_edges(user)))
    subgraph = [n for n in graph.neighbors(user) if n[0] == f'{nodes[0][0]}']
    for i in subgraph:
        nodess = [n for n in dfs_edges(graph, source=i)]    # perform DFS and loop over the pairs and add nodes to TET
        for x in nodess:
            if len(graph.out_edges(x[0])) > 0:
                type = graph.nodes(data=True)[x[0]].get('type')
                if graph.nodes(data=True)[x[0]].get('free'):
                    ms.add_node_w_freevar(x[0],1, type) #len(graph.out_edges(x[0]))
                else:
                    ms.add_node_w_count(x[0],1, type) #len((graph.out_edges(x[0])))
            if len(graph.out_edges(x[1])) == 0:
                type = graph.nodes(data=True)[x[1]].get('type')
                # if type == 'genre':
                #     ms.add_node_w_cross(x[1],type,graph.nodes(data=True)[x[1]]['cross'])
                # else:
                ms.add_node_w_count(x[1], 1, type)
        for (k, l) in nodess:
            ms.add_edge((k, l))
    for (e1, e2) in graph.edges(user):
        ms.add_edge((e1, e2))
    return ms


def __update_tet(t, rating_df):
    movies = [x for x,y in t.graph.nodes(data=True) if y['type'] == 'movie']
    for idx, m in enumerate(movies):
        rat = rating_df[rating_df.movieId == m].rating.item()
        t.add_node_w_count(f'r{idx}',rat, 'rating')
        t.add_edge((m,f'r{idx}'))
    director = [x for x,y in t.graph.nodes(data=True) if y['type'] == 'director']
    for idx, d in enumerate(director):
        if type(d) == str:
            award_amount = updated_actor[updated_actor.actorId == d].awards.item()
            if award_amount > 0:
                t.add_node_w_count(f'aw{idx}',award_amount, 'award')
                t.add_edge((d, f'aw{idx}'))


def update_tet(tet_multiset):
    for t in tet_multiset:
        root = [x for x, y in t.graph.nodes(data=True) if y.get('root')]
        temp_r = rdata[rdata['userId'] == root[0]]
        __update_tet(t, temp_r)


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
        temp_dist = 1/num_siblings * distance_c_emd(curr_node_hist1[0], curr_node_hist2[0])
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


# helper function to build metric tree
def __mt_build(g, d_max, b_max, d, data, name, spec, root):
    if not (d == d_max or len(data) <= b_max):
        z1, z2 = __get_random_pair(data, spec, root)
        data_1, data_2 = __split_data(data,z1,z2, spec, root)
        __mt_build(g, d_max, b_max, d + 1, data_1, 1, spec, root)
        __mt_build(g, d_max, b_max, d + 1, data_2, 2, spec, root)
        g.add_node(f'{d}_{name}', left=f'{d + 1}_1', right=f'{d + 1}_2', z1=z1, z2=z2) #left=data_1, right=data_2
        g.add_edge(f'{d}_{name}', f'{d + 1}_1')
        g.add_edge(f'{d}_{name}', f'{d + 1}_2')
    else:
        g.add_node(f'{d}_{name}', bucket=data)


# function for building metric tree
def mt_build(tet, k, bucket_max, spec):
    g = nx.DiGraph()
    __mt_build(g, k, bucket_max, 0, tet, 0, spec, 'user')
    return g


# function for calculating distance between two histograms
def __distance(v1,v2, spec, root):
    v1_hist = v1.get_histogram()
    v2_hist = v2.get_histogram()
    return calc_distance(v1_hist, v2_hist, spec, root)


# helper function for searching metric tree
def __mt_search(g, mn, h, k,leafs, spec, root):
    if mn in leafs:
        bucket = g.nodes(data=True)[mn]['bucket']
        return bucket
        #sort bucket according to h and return k
    dist1 = __distance(h, g.nodes(data=True)[mn]['z1'], spec, root)
    dist2 = __distance(h, g.nodes(data=True)[mn]['z2'], spec, root)
    if dist1 <= dist2:
        return __mt_search(g, g.nodes(data=True)[mn]['left'], h, k, leafs, spec, root)
    else:
        return __mt_search(g, g.nodes(data=True)[mn]['right'], h, k, leafs, spec, root)


# function for searching metric tree
def mt_search(t, g, k, spec):
    leaf_nodes = [node for node in g.nodes if
                  (g.in_degree(node) != 0 and g.out_degree(node) == 0)]
    root = [node for node in g.nodes if (g.in_degree(node) == 0 and g.out_degree != 0)]
    res = __mt_search(g, root[0], t[1], k, leaf_nodes, spec, 'user')
    return res

