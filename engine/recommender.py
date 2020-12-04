from engine.multiset import *
from engine.data_setup import *
import time

updated_data = pd.read_csv('movie.csv',converters={'cast': eval})
updated_actor = pd.read_csv('actor_data.csv',converters={'awards': eval})


# creates an overall directed bipartite graph used for constructing TETs
def generate_bipartite_graph():
    B = nx.DiGraph()
    B.add_nodes_from(rdata.userId, bipartite=0, user=True, free=True, type='user')
    B.add_nodes_from(updated_data.movieId, bipartite=1, movie=True, free=True, type='movie')
    B.add_nodes_from(updated_data.director, bipartite=1, director=True, free=True, type='director')
    for index,row in updated_data.iterrows():
        # for actor in row['cast']:
        #     B.add_edge(row['movieId'],actor)
        B.add_edge(row['movieId'],row['director'])
        #     B.add_node(director,bipartite=1, director=True)
        #     B.add_edge(row['movieId'],director)
    # for index,row in updated_actor.iterrows():
    #     for award in row['awards']:
    #         if not award is None:
    #             B.add_node(award,bipartite=1,award=True)
    #             B.add_edge(row['actorId'],award)
        # for actor in list(row['cast']):
        #     print(actor)
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
    ms = Multiset()
    ms.add_root(user, len(graph.out_edges(user)))
    subgraph = [n for n in graph.neighbors(user) if n[0] == f'{nodes[0][0]}']
    for i in subgraph:
        nodess = [n for n in dfs_edges(graph, source=i)]
        for x in nodess:
            if len(graph.out_edges(x[0])) > 0:
                type = graph.nodes(data=True)[x[0]].get('type')
                if graph.nodes(data=True)[x[0]].get('free'):
                    ms.add_node_w_freevar(x[0],1, type) #len(graph.out_edges(x[0]))
                else:
                    ms.add_node_w_count(x[0],1, type) #len((graph.out_edges(x[0])))
            if len(graph.out_edges(x[1])) == 0:
                type = graph.nodes(data=True)[x[1]].get('type')
                ms.add_node_w_count(x[1], 1, type)
        for (k, l) in nodess:
            ms.add_edge((k, l))
    for (e1, e2) in graph.edges(user):
        ms.add_edge((e1, e2))
    return ms


# function to generate the overall histogram for each user
def generate_histograms(l):
    h = []
    for x in l:
        root = [x for x, y in x.graph.nodes(data=True) if y.get('root')]
        h.append(x.graph.nodes(data=True)[root[0]]['value'])
    h_unique_set = set(h)
    h_unique_list = list(h_unique_set)
    h_unique_list.sort()
    hist, bin_edges = np.histogram(h, bins=h_unique_list)
    histogram = [list(hist), list(bin_edges)]
    return histogram


# helper function for calculating length
def __same_length_lists(list1, list2):
    while len(list1) != len(list2):
        list1.append(0)
    return list1


# function to calculate length / similarity between 1-dimensional histograms
def emd_1d_histogram_similarity(hist1, hist2):
    # hist1 and hist2 must have the same length
    dist = 0.0
    if len(hist1) < len(hist2):
        hist_w_padding = __same_length_lists(hist1, hist2)
        dist = __compute_manhatten_distance(hist_w_padding, hist2)
    elif len(hist1) > len(hist2):
        hist_w_padding = __same_length_lists(hist2, hist1)
        dist = __compute_manhatten_distance(hist_w_padding, hist1)
    else:
        dist = __compute_manhatten_distance(hist1, hist2)
    return dist


# helper function to compute manhatten distance
def __compute_manhatten_distance(hist1, hist2):
    sum_list = []
    for x, y in zip(hist1, hist2):
        sum_list.append(abs(x - y))
    distance = sum(sum_list)
    return distance


# helper function to get random pair
def __get_random_pair(data):
    dist = 0
    while dist == 0:
        v1,v2 = np.random.choice(data, 2, replace=False)
        v1_root = [x for x, y in v1.graph.nodes(data=True) if y.get('root')]
        v2_root = [x for x, y in v2.graph.nodes(data=True) if y.get('root')]
        v1_hist = v1.get_histogram(v1_root[0])
        v2_hist = v2.get_histogram(v2_root[0])
        dist = emd_1d_histogram_similarity(v1_hist[1], v2_hist[1])
    return v1, v2


# helper function to split data according to distance to v1 and v2
def __split_data(data, v1, v2):
    data_1 = []
    data_2 = []
    v1_root = [x for x, y in v1.graph.nodes(data=True) if y.get('root')]
    v2_root = [x for x, y in v2.graph.nodes(data=True) if y.get('root')]
    v1_hist = v1.get_histogram(v1_root[0])
    v2_hist = v2.get_histogram(v2_root[0])
    # print(f'v1 histogram: {v1_hist}')
    # print(f'v2 histogram: {v2_hist}')
    for g in data:
        root = [x for x, y in g.graph.nodes(data=True) if y.get('root')]
        g_hist = g.graph.nodes(data=True)[root[0]]['hist']
        v1_len = emd_1d_histogram_similarity(g_hist[1], v1_hist[1])
        v2_len = emd_1d_histogram_similarity(g_hist[1], v2_hist[1])
        # print(f'v1: {v1_len}, v2: {v2_len}')
        if v1_len < v2_len:
            data_1.append(g)
        elif v2_len < v1_len:
            data_2.append(g)
    return data_1, data_2


# helper function to build metric tree
def __mt_build(g, d_max, b_max, d, data, name):
    if not (d == d_max or len(data) <= b_max):
        z1, z2 = __get_random_pair(data)
        data_1, data_2 = __split_data(data,z1,z2)
        __mt_build(g, d_max, b_max, d + 1, data_1, 1)
        __mt_build(g, d_max, b_max, d + 1, data_2, 2)
        g.add_node(f'{d}_{name}', left=f'{d + 1}_1', right=f'{d + 1}_2', z1=z1, z2=z2) #left=data_1, right=data_2
        g.add_edge(f'{d}_{name}', f'{d + 1}_1')
        g.add_edge(f'{d}_{name}', f'{d + 1}_2')
    else:
        g.add_node(f'{d}_{name}', bucket=data)


# function for building metric tree
def mt_build(tet):
    g = nx.DiGraph()
    __mt_build(g, 3, 30, 0, tet, 0)
    return g


# function for calculating distance between two histograms
def __distance(v1,v2):
    v1_root = [x for x, y in v1.graph.nodes(data=True) if y.get('root')]
    v2_root = [x for x, y in v2.graph.nodes(data=True) if y.get('root')]
    v1_hist = v1.get_histogram(v1_root[0])
    v2_hist = v2.get_histogram(v2_root[0])
    return emd_1d_histogram_similarity(v1_hist[1], v2_hist[1])


# helper function for searching metric tree
def __mt_search(g, mn, h, k,leafs):
    if mn in leafs:
        bucket = g.nodes(data=True)[mn]['bucket']
        return bucket
        #sort bucket according to h and return k
    dist1 = __distance(h, g.nodes(data=True)[mn]['z1'])
    dist2 = __distance(h, g.nodes(data=True)[mn]['z2'])
    if dist1 <= dist2:
        return __mt_search(g,g.nodes(data=True)[mn]['left'],h,k,leafs)
    else:
        return __mt_search(g,g.nodes(data=True)[mn]['right'],h,k,leafs)


# function for searching metric tree
def mt_search(t, g, k):
    leaf_nodes = [node for node in g.nodes if
                  (g.in_degree(node) != 0 and g.out_degree(node) == 0)]
    root = [node for node in g.nodes if (g.in_degree(node) == 0 and g.out_degree != 0)]
    res = __mt_search(g, root[0], t[1], k, leaf_nodes)
    return res


# overall run function, where we run our 'pipeline'
def run():
    print('Running...')
    start_time_total = time.time()
    print('Formatting data...')
    run_data()
    print('Generating graph...')
    start_time = time.time()
    graph = generate_bipartite_graph()
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Building TET specification...')
    start_time = time.time()
    #speci = nested_list(["user", "movie", "genre"], [("user", "movie"), ("movie", "genre")])
    #speci = nested_list(["user,","movie","genre","actor","budget","award","rated"], [("user","movie"),("movie", "genre"),("movie","actor"),("movie","budget"),("actor","award"),("movie","rated")])
    speci = tet_specification(["user,", "movie", "genre", "director", ],
                              [("user", "movie"), ("movie", "genre"), ("movie", "director")], ["movie","user"])
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Generating TET according to graph and specification...')
    start_time = time.time()
    tet = generate_tet(graph, 'user', speci)
    print("--- %s seconds ---" % (time.time() - start_time))

    #TILFØJ RATINGS

    print('Counting TETs...')
    start_time = time.time()
    [g.count_tree() for g in tet]
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Performing Logistic Evaluation on TETs...')
    start_time = time.time()
    [g.logistic_eval(0, 1) for g in tet]
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Generating histograms...')
    start_time = time.time()
    [g.histogram() for g in tet]
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Generating overall histogram for all users...')
    start_time = time.time()
    hist = generate_histograms(tet)
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Building Metric Tree')
    start_time = time.time()
    mts = mt_build(tet)
    print(mts.nodes)
    print(mts.edges)
   #[print(mts[i].graph.nodes(data=True)) for i in range(5)]
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Searching Metric Tree')
    start_time = time.time()
    mts_res = mt_search(tet,mts,3)
    print(f'Amount of similiar users found: {len(mts_res)}')
    print(mts_res[0].graph.nodes(data=True))
    #[print(x.graph.nodes(data=True)) for x in mts_res]
    print("--- %s seconds ---\n" % (time.time() - start_time))

    print('|| ------ COMPLETE ------ ||')
    print('Total run time: %s seconds.' % (time.time() - start_time_total))
    print('Amount of users: %s.' % len(tet))
    print('Amount of bins in histogram: %s.' % len(hist[0]))
    print('|| ---------------------- ||\n')
    print('Top 5 users:')
    [print(tet[i].graph.nodes(data=True)) for i in range(5)]

run()

