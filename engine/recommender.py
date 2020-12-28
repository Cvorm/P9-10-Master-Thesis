from engine.multiset import *
from engine.data_setup import *
from collections import defaultdict
import itertools

def get_movies_from_id(movie_ids):
    movies = {}
    # for i in movie_ids:
    for index, movie in data.iterrows():
        if movie['movieId'] in movie_ids:
            movies[movie['title']] = movie['genres']
    return movies
    # for i, movie in data.iterrows():
    #     if movie['movieId'] ==
    #     movies[movie['title']] = movie['genres']
    #     # df.loc[df['column_name'] == some_value]


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
def generate_bipartite_graph(x_data):
    B = nx.DiGraph()
    print(x_data)
    B.add_nodes_from(x_data.userId, bipartite=0, user=True, free=True, type='user')
    B.add_nodes_from(updated_data.movieId, bipartite=1, movie=True, free=True, type='movie')
    B.add_nodes_from(updated_data.director, bipartite=1, director=True, free=True, type='director')
    for index,row in updated_data.iterrows():
        B.add_edge(row['movieId'], row['director'])
    B.add_edges_from([(uId, mId) for (uId, mId) in x_data[['userId', 'movieId']].to_numpy()])
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
    # director = [x for x,y in t.graph.nodes(data=True) if y['type'] == 'director']
    # for idx, d in enumerate(director):
    #     if type(d) == str:
    #         award_amount = updated_actor[updated_actor.actorId == d].awards.item()
    #         if award_amount > 0:
    #             t.add_node_w_count(f'aw{idx}',award_amount, 'award')
    #             t.add_edge((d, f'aw{idx}'))


def update_tet(tet_multiset,x_train):
    for t in tet_multiset:
        root = [x for x, y in t.graph.nodes(data=True) if y.get('root')]
        temp_r_df = x_train[x_train['userId'] == root[0]]
        __update_tet(t, temp_r_df)


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

def movie_dist(p1,p2):
    return p1 - p2

def get_movies_user(user, top_k_movies, interval1, interval2):
    movies = {}
    for x, y in user.graph.nodes(data=True):
        if type(x) is str and x[0] == 'm':
            for n in user.graph.neighbors(x):
                if type(n) is str and n[0] == 'r':
                    rat = user.graph.nodes(data=True)[n]['count']
                    if rat >= 3.5:
                        #movies[x] = y['value']
                        movies[x] = rat

            # if interval1 <= y['value'] <= interval2:
            #     movies[x] = y['value']
    mean = get_user_mean_value(user)
    #sort_movies = sorted(movies.items(), key=lambda e: movie_dist(e[1],mean))
    sort_movies = sorted(movies.items(), key=lambda k: k[1], reverse=True)
    res = sort_movies #[:top_k_movies]
    return res

def get_user_max_hist(user):
    for x, y in user.ht.nodes(data=True):
        if x == 'movie':
            histogram = y['hist']
            print(histogram)
    yes = max([(v, i) for i, v in enumerate(histogram[0])])
    yes2 = (histogram[1][yes[1]],histogram[1][yes[1]+1])


def get_user_mean_value(user):
    tmp_val = 0.0
    tmp_count = 0
    for x, y in user.graph.nodes(data=True):
        if type(x) is str and x[0] == 'm':
            tmp_val = tmp_val + y['value']
            tmp_count = tmp_count + 1
    if tmp_val == 0 or tmp_count == 0:
        return 0
    else:
        return tmp_val / tmp_count


def get_movies_in_user(user):
    tmp_list = []
    for x, y in user.graph.nodes(data=True):
        if type(x) is str and x[0] == 'm':
            tmp_list.append(x)
    return tmp_list

def get_movies(user_hist, other_users_hist, interval1, interval2, top_k_movies):
    movies = []
    for u in other_users_hist:
        temp_movies = get_movies_user(u, top_k_movies, interval1, interval2)
        for i in temp_movies:
            movies.append(i)
    no_duplicates = [list(v) for v in dict(movies).items()]
    mu = get_movies_in_user(user_hist)
    no_mas = [(x,y) for x,y in no_duplicates if x not in mu]
    print(f' no duplicates: {len(no_duplicates)}, no_mas : {len(no_mas)}')
    get_user_max_hist(user_hist)
    mean = get_user_mean_value(user_hist)
    # sort_movies = sorted(no_duplicates, key=lambda e: movie_dist(e[1], mean))
    sort_movies = sorted(no_mas, key=lambda k: k[1], reverse=True)
    res = sort_movies #[:top_k_movies]
    # user_movies = []
    # for x, y in user_hist.graph.nodes(data=True):
    #     if type(x) is str and x[0] == 'm':
    #         for n in user_hist.graph.neighbors(x):
    #             if type(n) is str and n[0] == 'r':
    #                 rat = user_hist.graph.nodes(data=True)[n]['count']
    #                 print(n, rat)
    #                 if rat >= 4.0:
    #                     movies.append(x)
    #         # if interval1 <= y['value'] <= interval2:
    #         #     user_movies.append(x)
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


def __sort_dist(val,h,spec,root):
    dist = __distance(val,h,spec,root)
    return dist


# helper function for searching metric tree
def __mt_search(g, mn, h, k, leafs, spec, root):
    if mn in leafs:
        bucket = g.nodes(data=True)[mn]['bucket']
        dist = [__distance(h, b, spec, root) for b in bucket]
        bucket_sorted = [x for _, x in sorted(zip(dist, bucket))]
        if len(bucket_sorted) < k:
            return bucket_sorted
        else:
            return bucket_sorted[:k]
    dist1 = __distance(h, g.nodes(data=True)[mn]['z1'], spec, root)
    dist2 = __distance(h, g.nodes(data=True)[mn]['z2'], spec, root)
    if dist1 <= dist2:
        return __mt_search(g, g.nodes(data=True)[mn]['left'], h, k, leafs, spec, root)
    else:
        return __mt_search(g, g.nodes(data=True)[mn]['right'], h, k, leafs, spec, root)


# function for searching metric tree
def mt_search(t, g, user_tet, k, spec):
    leaf_nodes = [node for node in g.nodes if
                  (g.in_degree(node) != 0 and g.out_degree(node) == 0)]
    root = [node for node in g.nodes if (g.in_degree(node) == 0 and g.out_degree != 0)]
    res = __mt_search(g, root[0], user_tet, k, leaf_nodes, spec, 'user')
    return res


def hitrate(topNpredictions, leftoutpredictions):
    hits = 0
    total = 0
    for leftout in leftoutpredictions:
        hit = False
        for movieId, predictedRating in topNpredictions:
            mid = leftout[0]
            if movieId == mid:
                hit = True
        if hit:
            hits += 1
        total += 1

    return hits / total


def get_movies_juujiro(user, k_movies):
    movies = {}
    for x, y in user.graph.nodes(data=True):
        if type(x) is str and x[0] == 'm':
            rat = get_rating(user, x)
            # movies[x] = y['value']
            movies[x] = rat
    no_duplicates = [list(v) for v in dict(movies).items()]
    mean = get_user_mean_value(user)
    sort_movies = sorted(no_duplicates, key=lambda k: k[1], reverse=True)
    #sort_movies = sorted(movies.items(), key=lambda e: movie_dist(e[1], mean))
    # res = sort_movies[:top_k_movies]
    # sort_movies = sorted(movies.items(), key=lambda k: k[1], reverse=True)
    res = sort_movies #[:k_movies]
    return res


def get_tet_user(tet,user):
    for t in tet:
        username = [x for x, y in t.graph.nodes(data=True) if y.get('root')]
        if username[0] == user:
            return t


def has_rating(user, movieid):
    for x, y in user.graph.nodes(data=True):
        if type(x) is str and x == movieid:
            for n in user.graph.neighbors(x):
                if type(n) is str and n[0] == 'r':
                    rat = user.graph.nodes(data=True)[n]['count']
                    return True
    return False


def get_rating(user, movieid):
    for x, y in user.graph.nodes(data=True):
        if type(x) is str and x == movieid:
            for n in user.graph.neighbors(x):
                if type(n) is str and n[0] == 'r':
                    rat = user.graph.nodes(data=True)[n]['count']
                    return rat
    return 0


def __recall(predictions, leftout, user, user_leftout, k=5, threshold=3.5):

    n_rel = sum((get_rating(user_leftout, mov) >= threshold) for (mov, _) in predictions)
    n_rec_k = sum((est >= threshold) for (_, est) in predictions[:k])
    n_rel_and_rec_k = sum(((get_rating(user_leftout,mov) >= threshold) and (est >= threshold))
                          for (mov, est) in predictions[:k])
    precisions = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
    recalls = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
    print(f'n_rel: {n_rel}, n_rec_k: {n_rec_k}, n_rel_and_rec_k: {n_rel_and_rec_k}')
    print(f'PRECISION :: {precisions}')
    print(f' RECALL ::  {recalls}')
    return precisions, recalls


def recall(tet_train, tet_test, metric_tree, mt_search_k, speci_test):
    tmp = []
    for user in tet_train:
        username = [x for x,y in user.graph.nodes(data=True) if y.get('root')]
        user_leftout = get_tet_user(tet_test,username[0])
        similar_users = mt_search(tet_train, metric_tree, user, mt_search_k, speci_test)
        predicted_movies = get_movies(user, similar_users, 0.8, 1, 20)
        user_test_movies = get_movies_juujiro(user_leftout, 20)
        tmp.append(__recall(predicted_movies, user_test_movies, user, user_leftout))
    precision = 0.0
    precision_count = 0
    rec = 0.0
    rec_count = 0

    for x,y in tmp:
        precision = precision + x
        rec = rec + y
        precision_count = precision_count + 1
        rec_count = rec_count + 1
    precision_res = precision / precision_count
    recall_res = rec / rec_count
    print(tmp)
    return precision_res,recall_res
