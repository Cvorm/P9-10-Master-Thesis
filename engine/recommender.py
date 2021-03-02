from engine.multiset import *
from engine.data_setup import *
from collections import defaultdict
import itertools
import ast

def create_movie_tet(spec, dataframe, source):
    roots = np.unique(dataframe.movieId)
    complete = []
    for r in roots:
        ms = Multiset()
        complete.append(__create_movie_tet(r, spec, ms, dataframe, source))
    return complete

def __create_movie_tet(movie, tet_spec, ms, dataframe, source):
    nodes = [n[-1] for n in dfs_edges(tet_spec, source=source)]
    ms.add_root_movie(movie, 1)
    this_movie = data[data['movieId'] == movie]
    print(this_movie)
    user_director = updated_actor[updated_actor.actorId.isin(this_movie['director'])]
    # print(user_director)
    for node in nodes:
        if node == 'has_imdb_rating':
            # print(this_movie.iloc[0]['rating'])
            ms.add_node_w_count(f'ir{this_movie["movieId"]}', float(this_movie['rating']), 'has_imdb_rating')
            ms.add_edge((str(this_movie['movieId']), f'ir{this_movie["movieId"]}'))
        elif node == 'has_director':
            ms.add_node_w_count(str(this_movie['director']), 1, 'has_director')
            ms.add_edge((str(this_movie['movieId']), str(this_movie['director'])))
        elif node == 'has_genres':
            tmp = this_movie['genres']
            print(tmp[0])
            # yes = str(tmp[0])
            # print(tmp)
            tmp = ast.literal_eval(tmp)
            tmp_count = len(tmp)
            # print(tmp_count)
            ms.add_node_w_count(f'hg{this_movie}', int(tmp_count), 'has_genres')
            ms.add_edge((str(this_movie['movieId']), f'hg{this_movie}'))
            for idx, genre in enumerate(tmp):
                ms.add_node_w_count(f'g{this_movie}{idx}', 1, str(genre))
                ms.add_edge((f'hg{this_movie}', f'g{this_movie}{idx}'))
        elif node == 'has_awards':
                ms.add_node_w_count(f'a{this_movie}', int(user_director['awards']), 'has_awards')
                ms.add_edge((str(user_director['actorId']), f'a{this_movie}'))
        elif node == 'has_nominations':
                ms.add_node_w_count(f'n{this_movie}', int(user_director['nominations']), 'has_nominations')
                ms.add_edge((str(user_director['actorId']), f'n{this_movie}'))

    return ms
    # for node in nodes:
    #     if node == 'has_user_rating':
    #         for y, x in user_ratings.iterrows():
    #             ms.add_node_w_count(f'ur{y}', int(x['rating']), 'has_user_rating')
    #             ms.add_edge((str(x['movieId']), f'ur{y}'))
    #     elif node == 'has_imdb_rating':
    #         for y, x in user_movie.iterrows():
    #             ms.add_node_w_count(f'ir{y}', float(x['rating']), 'has_imdb_rating')
    #             ms.add_edge((str(x['movieId']), f'ir{y}'))
    #     elif node == 'has_director':
    #         for y, x in user_movie.iterrows():
    #             ms.add_node_w_count(str(x['director']), 1, 'has_director')
    #             ms.add_edge((str(x['movieId']), str(x['director'])))
    #     elif node == 'has_genres':
    #         for y, x in user_movie.iterrows():
    #             tmp = x['genres'][0]
    #             tmp = ast.literal_eval(tmp)
    #             tmp_count = len(tmp)
    #             ms.add_node_w_count(f'hg{y}', int(tmp_count), 'has_genres')
    #             ms.add_edge((str(x['movieId']), f'hg{y}'))
    #             for idx, genre in enumerate(tmp):
    #                 ms.add_node_w_count(f'g{y}{idx}', 1, str(genre))
    #                 ms.add_edge((f'hg{y}', f'g{y}{idx}'))
    #     elif node == 'has_awards':
    #         for y, x in user_director.iterrows():
    #             ms.add_node_w_count(f'a{y}', int(x['awards']), 'has_awards')
    #             ms.add_edge((str(x['actorId']), f'a{y}'))
    #     elif node == 'has_nominations':
    #         for y, x in user_director.iterrows():
    #             ms.add_node_w_count(f'n{y}', int(x['nominations']), 'has_nominations')
    #             ms.add_edge((str(x['actorId']), f'n{y}'))
    # return ms

def create_user_tet(spec, dat, source):
    roots = np.unique(dat.userId)
    # roots = [n for n, info in graph.nodes(data=True) if info.get(f'{root}')]
    complete = []
    for r in roots:
        ms = Multiset()  # here we instantiate our TETs
        complete.append(__create_user_tet(r, spec, ms, dat, source))
    return complete


def __create_user_tet(user, tet_spec, ms, dat, source):
    nodes = [n[-1] for n in dfs_edges(tet_spec, source=source)]
    ms.add_root(user, 1)
    user_ratings = dat[dat['userId'] == user]
    # print(user_ratings)
    user_movie = data[data.movieId.isin(user_ratings['movieId'])]
    user_director = updated_actor[updated_actor.actorId.isin(user_movie['director'])]
    for node in nodes:
        if node == 'has_rated':
            for y,x in user_ratings.iterrows():
                ms.add_node_w_count(str(x['movieId']), 1, 'has_rated')
                ms.add_edge((user, str(x['movieId'])))
        elif node == 'has_user_rating':
            for y, x in user_ratings.iterrows():
                ms.add_node_w_count(f'ur{y}', int(x['rating']), 'has_user_rating')
                ms.add_edge((str(x['movieId']), f'ur{y}'))
        elif node == 'has_imdb_rating':
            for y, x in user_movie.iterrows():
                ms.add_node_w_count(f'ir{y}', float(x['rating']), 'has_imdb_rating')
                ms.add_edge((str(x['movieId']), f'ir{y}'))
        elif node == 'has_director':
            for y, x in user_movie.iterrows():
                ms.add_node_w_count(str(x['director']), 1, 'has_director')
                ms.add_edge((str(x['movieId']), str(x['director'])))
        elif node == 'has_genres':
            for y, x in user_movie.iterrows():
                tmp = x['genres'][0]
                # print(tmp)
                tmp = ast.literal_eval(tmp)
                tmp_count = len(tmp)
                # print(y)
                ms.add_node_w_count(f'hg{y}', int(tmp_count), 'has_genres')
                ms.add_edge((str(x['movieId']), f'hg{y}'))
                for idx, genre in enumerate(tmp):
                    ms.add_node_w_count(f'g{y}{idx}', 1, str(genre))
                    ms.add_edge((f'hg{y}', f'g{y}{idx}'))
        elif node == 'has_awards':
            for y, x in user_director.iterrows():
                ms.add_node_w_count(f'a{y}', int(x['awards']), 'has_awards')
                ms.add_edge((str(x['actorId']), f'a{y}'))
        elif node == 'has_nominations':
            for y, x in user_director.iterrows():
                ms.add_node_w_count(f'n{y}', int(x['nominations']), 'has_nominations')
                ms.add_edge((str(x['actorId']), f'n{y}'))
    return ms


def get_movies_from_id(movie_ids):
    movies = {}
    for index, movie in data.iterrows():
        if movie['movieId'] == movie_ids:
            movies[movie['title']] = movie['genres']
    return movies


def get_genres():
    l = []
    for index, movie in data.iterrows():
         for genre in movie['genres']:
            l.append(genre)
    s = set(l)
    final_list = list(s)
    final_list.sort()
    return final_list


# function used for loading and creating a tet-specification
def tet_specification(nodes, edges):
    g = nx.DiGraph()
    for n in nodes:
            g.add_node(n, type=n)
    g.add_edges_from(edges)
    return g


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
    # print(spec_nodes)
    dist = []
    for x,y in spec_nodes:
        curr_node_hist1 = hist_tree1[y]['hist']
        curr_node_hist2 = hist_tree2[y]['hist']
        num_siblings = get_siblings(spec, y) + 1
        temp_dist = 1/num_siblings * distance_c_emd(curr_node_hist1[0], curr_node_hist2[0]) #1/num_siblings
        dist.append(temp_dist)
    res = sum(dist)
    return res


def movie_dist(p1,p2):
    return p1 - p2


def get_movies_user(user):
    movies = {}
    for x, y in user.graph.nodes(data=True):
        if type(x) is str and x[0] == 'm':
            for n in user.graph.neighbors(x):
                if type(n) is str and n[:2] == 'ur': #user.graph[n].get['type'] == 'has_user_rating':
                    rat = user.graph.nodes(data=True)[n]['count']
                    movies[x] = rat
    sort_movies = sorted(movies.items(), key=lambda k: k[1], reverse=True)
    res = sort_movies #[:top_k_movies]
    return res


def get_movies_in_user(user):
    tmp_list = []
    for x, y in user.graph.nodes(data=True):
        if type(x) is str and x[0] == 'm':
            # rat = get_rating(user, x)
            # if rat >= 4:
            tmp_list.append(x)
    return tmp_list


def get_movies(user_hist, other_users_hist):
    movies = []
    for u in other_users_hist:
        temp_movies = get_movies_user(u)
        for i in temp_movies:
            movies.append(i)
    no_duplicates = [list(v) for v in dict(movies).items()]
    mu = get_movies_in_user(user_hist)
    no_mas = [(x,y) for x,y in no_duplicates if x not in mu]
    tmp_val = len(no_duplicates) - len(no_mas)
    if tmp_val == 0:
        sim_tmp = 1
    elif len(no_duplicates) == 0:
        sim_tmp = 0
    else:
        sim_tmp = tmp_val / len(no_duplicates)  # if len(no_duplicates) != 0 else
    sort_movies = sorted(no_mas, key=lambda k: k[1], reverse=True)
    res = sort_movies #[:top_k_movies]
    return res, sim_tmp


def get_siblings(aGraph, aNode):
     # print(aGraph.nodes())
     # print(aNode)
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


def get_tet_user(tet,user):
    for t in tet:
        username = [x for x, y in t.graph.nodes(data=True) if y.get('root')]
        if username[0] == user:
            return t


def get_rating(user, movieid):
    for x, y in user.graph.nodes(data=True):
        if type(x) is str and x == movieid:
            for n in user.graph.neighbors(x):
                if type(n) is str and n[:2] == 'ur': #user.graph[n].get['type'] == 'has_user_rating':
                    rat = user.graph.nodes(data=True)[n]['count']
                    return rat
    return 0


def __recall(predictions, user_leftout, k, threshold=4):
    n_rel = sum((get_rating(user_leftout, mov) >= threshold) for (mov, _) in predictions)
    n_rec_k = sum((est >= threshold) for (_, est) in predictions[:k])
    n_rel_and_rec_k = sum(((get_rating(user_leftout,mov) >= threshold) and (est >= threshold))
                          for (mov, est) in predictions[:k])
    precisions = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
    recalls = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
    # print(f'n_rel: {n_rel}, n_rec_k: {n_rec_k}, n_rel_and_rec_k: {n_rel_and_rec_k}')
    # print(f'PRECISION :: {precisions}')
    # print(f' RECALL ::  {recalls}')
    return precisions, recalls


def recall(tet_train, tet_test, metric_tree, mt_search_k, spec, k_movies):
    tmp = []
    sim_counter = 0
    sim_collector = 0.0
    for user in tet_train:
        username = [x for x,y in user.graph.nodes(data=True) if y.get('root')]
        user_leftout = get_tet_user(tet_test,username[0])
        similar_users = mt_search(metric_tree, user, mt_search_k, spec)
        predicted_movies, sim_test = get_movies(user, similar_users)
        sim_counter = sim_counter + 1
        sim_collector = sim_collector + sim_test
        tmp.append(__recall(predicted_movies, user_leftout, k_movies))
    precision = 0.0
    precision_count = 0
    rec = 0.0
    rec_count = 0
    sim = {sim_collector/sim_counter}
    for x,y in tmp:
        precision = precision + x
        rec = rec + y
        precision_count = precision_count + 1
        rec_count = rec_count + 1
    precision_res = precision / precision_count
    recall_res = rec / rec_count
    # print(tmp)
    return precision_res, recall_res, sim


def get_movies_juujiro(user):
    tmp_list = []
    for x, y in user.graph.nodes(data=True):
        if type(x) is str and x[0] == 'm':
            tmp_list.append(x)
    return tmp_list