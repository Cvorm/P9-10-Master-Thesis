import pandas as pd
import pickle
import numpy as np
import imdb
from engine.multiset import *
import time
# RENAME TO MDATA 4 MOVIE DATA
moviesDB = imdb.IMDb()
data = pd.read_csv('Data/movies.csv')
ratings = pd.read_csv('Data/ratings.csv')
links = pd.read_csv('Data/links.csv')
#kg = pd.DataFrame(columns=['head', 'relation', 'tail'])
rdata = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
adata = pd.DataFrame(columns=['actorId','awards'])
xdata = pd.DataFrame(columns=['movieId','actors','directors','budget'])
updated_data = pd.read_csv('movie.csv',converters={'cast': eval})
updated_actor = pd.read_csv('actor_data.csv',converters={'awards': eval})


def update_movie_data():
    actor_id_l = []
    for index, movie in data.iterrows():
        print(f'{index} / {len(data)}')
        #if index == 5: break
        s = str(movie['movieId'])
        sx = s[1:]
        temp = links[links['movieId'] == int(sx)]
        try:
            imovie = moviesDB.get_movie(temp['imdbId'])
            print(imovie)
        except:
            print("fail to get movie")
            continue
        try:
            director = ""
            for d in imovie['directors']:
                #print(d['name'])
                director = 'a' + str(d.personID)
            #print(data.at[index, 'director'])
        except:
            print('except')
        data.at[index, 'director'] = director

        cast_l = []
        try:
            for actor in imovie['cast']:
                cast_l.append('a' + str(actor.personID))
                actor_id_l.append(actor.personID)
        except:
            print('fail cast')
        #print(cast_l)
        data.at[index, 'cast'] = str(cast_l)
        box_l = []
        try:
            temp_list = []
            for x,y in imovie.get('box office').items():
                temp_list.append([x,y])
            box_l = temp_list
        except:
            print('fail box office')
        data.at[index, 'box'] = str(box_l)

    data.to_csv('movie.csv',index=False)
    return actor_id_l


def temp_func():
    dat = updated_data['director']
    tmp = []
    update_actor_data(dat)
    # for x in dat:
    #     for y in x:
    #         tmp.append(y)
    # res = set(tmp)
    # update_actor_data(list(res))


def update_actor_data(actor_list):
    s = set(actor_list)
    ss = list(s)
    adata['actorId'] = ss
    for index, actor in adata.iterrows():
        print(f'{index} \ {len(adata)}')
        awards = []
        j = str(actor['actorId'])
        print(j)
        jx = j[1:]
        temp_dict = {"Winner": 0, "Nominee": 0}
        try:
            award = moviesDB.get_person_awards(jx)
            for x,y in award['data'].items():
                #print(n,m)
                for a in y:
                    res = a.get('result')
                    if res == 'Winner':
                        temp_dict["Winner"] += 1
                    if res == 'Nominee':
                        temp_dict['Nominee'] += 1
                    awards.append(res)
            print(temp_dict)
            adata.at[index, 'awards'] = temp_dict
        except:
            print('fail')
            adata.at[index,'awards'] = temp_dict
    print(adata)
    adata.to_csv('actor_data.csv',index=False)


def update_data(movie,actor):
    if movie:
        actor_list = update_movie_data()
        if actor:
            update_actor_data(actor_list)


def format_data():
    rdata['userId'] = 'u' + ratings['userId'].astype(str)
    rdata['movieId'] = 'm' + ratings['movieId'].astype(str)
    rdata['rating'] = ratings['rating']
    data['genres'] = [str(m).split("|") for m in data.genres]
    data['movieId'] = 'm' + data['movieId'].astype(str)


def generate_bipartite_graph():
    B = nx.DiGraph()
    B.add_nodes_from(rdata.userId, bipartite=0, user=True, free=True, type='user')
    #B.add_nodes_from(rdata.movieId, bipartite=1, movie=True)
    B.add_nodes_from(updated_data.movieId, bipartite=1, movie=True, free=True, type='movie')
    #B.add_nodes_from(updated_actor.actorId,bipartite=1, actor=True)
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
        # print(movie)
        for genre in movie['genres']:
            # print(genre)
            B.add_node(genre, bipartite=1, genre=True, free=False, type='genre')
            B.add_edge(movie['movieId'], genre)
    #[print(x) for x in B.nodes if B.nodes(data=True)[x].get('actor')]
    return B


def nested_list(nodes, edges, freevars):
    g = nx.DiGraph()
    for n in nodes:
        if n in freevars:
            g.add_node(n, type=n, free=True)
        else:
            g.add_node(n, type=n, free=False)
    g.add_edges_from(edges)
    return g


def generate_tet(graph, root, spec):
    roots = [n for n, info in graph.nodes(data=True) if info.get(f'{root}')]
    complete = []
    for r in roots:
        complete.append(generate_tet_yes(graph, r, spec))
    #complete.append(generate_tet_yes(graph, roots[0], spec))
    return complete


def is_internal(element, ls):
    new = [x[0] for x in ls]
    if element in new:
        return True
    else:
        return False


def generate_tet_yes(graph, user, spec):
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
                    # add_node (x[1]) with count 1
                    ms.add_node_w_count(x[0],1, type) #len((graph.out_edges(x[0])))
            # if not(is_internal(x[1], nodess_reversed)):
            if len(graph.out_edges(x[1])) == 0:
                type = graph.nodes(data=True)[x[1]].get('type')
                # add_node (x[1]) with count 1
                ms.add_node_w_count(x[1], 1, type)
        for (k, l) in nodess:
            ms.add_edge((k, l))
    for (e1, e2) in graph.edges(user):
        ms.add_edge((e1, e2))
    return ms


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


def same_length_lists(list1, list2):
    while len(list1) != len(list2):
        list1.append(0)
    return list1


def emd_1d_histogram_similarity(hist1, hist2):
    #hist1 and hist2 must have the same length
    hist_w_padding = []
    dist = 0.0
    if len(hist1) < len(hist2):
        hist_w_padding = same_length_lists(hist1, hist2)
        dist = __compute_manhatten_distance(hist_w_padding, hist2)
    elif len(hist1) > len(hist2):
        hist_w_padding = same_length_lists(hist2, hist1)
        dist = __compute_manhatten_distance(hist_w_padding, hist1)
    else:
        dist = __compute_manhatten_distance(hist1, hist2)
    return dist


def __compute_manhatten_distance(hist1, hist2):
        sum_list = []
        for x, y in zip(hist1, hist2):
            sum_list.append(abs(x - y))
        distance = sum(sum_list)
        return distance


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


def __mt_build(g, d_max, b_max, d, data, name):
    if not (d == d_max or len(data) <= b_max):
        z1, z2 = __get_random_pair(data)
        data_1, data_2 = __split_data(data,z1,z2)
        __mt_build(g, d_max, b_max, d + 1, data_1, 1)
        __mt_build(g, d_max, b_max, d + 1, data_2, 2)
        g.add_node(f'{d}_{name}', left=data_1, right=data_2, z1=z1, z2=z2)
        g.add_edge(f'{d}_{name}', f'{d + 1}_1')
        g.add_edge(f'{d}_{name}', f'{d + 1}_2')
    else:
        g.add_node(f'{d}_{name}', bucket=data)


def mt_build(tet):
    g = nx.DiGraph()
    __mt_build(g, 3, 2, 0, tet, 0)
    return g


def run():
    print('Running...')
    start_time_total = time.time()
    print('Formatting data...')
    format_data()
    print('Generating graph...')
    start_time = time.time()
    graph = generate_bipartite_graph()
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Building TET specification...')
    start_time = time.time()
    #speci = nested_list(["user", "movie", "genre"], [("user", "movie"), ("movie", "genre")])
    #speci = nested_list(["user,","movie","genre","actor","budget","award","rated"], [("user","movie"),("movie", "genre"),("movie","actor"),("movie","budget"),("actor","award"),("movie","rated")])
    speci = nested_list(["user,", "movie", "genre", "director","awards"],
                        [("user", "movie"), ("movie", "genre"), ("movie", "director"),("director","awards")], ["director","movie","user"])
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Generating TET according to graph and specification...')
    start_time = time.time()
    tet = generate_tet(graph, 'user', speci)
    print("--- %s seconds ---" % (time.time() - start_time))

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
   #[print(mts[i].graph.nodes(data=True)) for i in range(5)]
    print("--- %s seconds ---\n" % (time.time() - start_time))
    print('|| ------ COMPLETE ------ ||')
    print('Total run time: %s seconds.' % (time.time() - start_time_total))
    print('Amount of users: %s.' % len(tet))
    print('Amount of bins in histogram: %s.' % len(hist[0]))
    print('|| ---------------------- ||\n')
    print('Top 5 users:')
    [print(tet[i].graph.nodes(data=True)) for i in range(5)]


#temp_func()
#format_data()
#update_data(True,True)
run()

# [print(q) for q in graph2[0].graph.nodes(data=True)]

# x for x, y in self.graph.nodes(data=True) if y.get('root')
# for xx in graph2:
#     root = [x for x, y in xx.graph.nodes(data=True) if y.get('root')]
#     print(root)
# print(graph2[0].graph.edges(data=True))
# print(graph2[0].graph.nodes(data=True))
# print(graph2)
