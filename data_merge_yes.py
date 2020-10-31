import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np
import imdb
import csv
import networkx as nx
from networkx import *
import nltk
from nltk import Tree
from multiset import *
#RENAME TO MDATA 4 MOVIE DATA
moviesDB = imdb.IMDb()
data = pd.read_csv('Data/movies.csv')
ratings = pd.read_csv('Data/ratings.csv')
kg = pd.DataFrame(columns=['head', 'relation', 'tail'])
rdata = pd.DataFrame(columns=['userId', 'movieId', 'rating'])


def generate_bipartite_graph():
    rdata['userId'] = 'u' + ratings['userId'].astype(str)
    rdata['movieId'] = 'm' + ratings['movieId'].astype(str)
    rdata['rating'] = ratings['rating']
    data['genres'] = [str(m).split("|") for m in data.genres]
    data['movieId'] = 'm' + data['movieId'].astype(str)

    B = nx.DiGraph()
    B.add_nodes_from(rdata.userId, bipartite=0, user=True)
    B.add_nodes_from(rdata.movieId, bipartite=1, movie=True)
    B.add_edges_from([(uId, mId) for (uId, mId) in rdata[['userId', 'movieId']].to_numpy()])
    for index, movie in data.iterrows():
        #print(movie)
        for genre in movie['genres']:
            #print(genre)
            B.add_node(genre, bipartite=0, genre=True)
            B.add_edge(movie['movieId'], genre)

    print(is_bipartite(B))
    return B


def nested_list(nodes, edges):
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    nx.set_node_attributes(g, 'null', 'type')
    for n in g.nodes:
        g.nodes(data=True)[n]['type'] = n
    return g


def foo(n,g):
    e = g.edges(n)
    g.nodes[n]['count'] += len(e)
    #print(g.nodes[n]) # = len(e)
    for (n1,n2) in e:
        g.nodes[n2]['count'] += len(e)
        #print(n2)
        foo(n2,g)
    #heads = node_connected_component(g,n)
    #print(heads)

# def count_of_counts(n, g, leafs, internals):
#     # counter = Counter()
#     e = g.edges(n)
#     g.nodes[n]['count'] += len(e)
#     for node in leafs:
#         g.nodes[node]['count'] = 1
#     for node in internals:
#         e2 = g.edges(node)
#         counter = Counter()
#         for (n1,n2) in e2:
#             counter[n2] += 1
#         g.nodes[node]['count'] = counter


def most_def(g, leafs, internals):
    # e = g.edges(n)
    # for node in internals:
    #     g.nodes[node]['count'] = Counter()
    for node in leafs:
        # print(node)
        counter = Counter()
        counter[node] += 1
        # counter = Counter([node])
        g.nodes[node]['count'] = counter
        # print(g.in_edges(node))
        # for (n1,n2) in g.in_edges(node):
        #     # counter2 = Counter()
        #     # counter2[n2] += 1
        #     g.nodes[n1]['count'] += 1
        #     # g.nodes[n1]['count'] += g.nodes[n2]['count']
    for n1, n2, data in g.out_edges(internals, data=True):
        # for (n1,n2) in g.in_edges(node):
        # print(n2)
        edges = g.edges(n1)
        counterr = Counter()
        for (m1,m2) in edges:
            counterr[m2] += 1
        g.nodes[n1]['count'] = counterr

    for j1, j2, data in g.in_edges(internals, data=True):
        # print(n1)
        edges = g.edges(j1)
        counterrr = Counter()
        for (k1, k2) in edges:
            counterrr[k2] += 1
        g.nodes[j1]['count'] = counterrr


def generate_tet(g,cn,spec): #,parent,current_node
    neighbors = [n for n in spec.neighbors(f'{cn}')]
    if len(neighbors) == 0:
        for x, info in g.nodes(data=True):
            if info.get(f'{cn}'):
                multi = Multiset("user1")
                multi.add_leaf(str(x))
                return multi
    else:
        for n in neighbors:
            ms = generate_tet(g,n,spec)
        for x, info in g.nodes(data=True):
            if info.get(f'{cn}'):
                ms.add_internal_w_leaves(x)
                return(ms)
    return ms



    #for hver node specification, add current node
    # for subtree in spec(data=True): #kør for subtrees
    #     print(subtree)
    #     if subtree.out_edges == 0:
    #         print('child')
    #         for n, info in g.nodes(data=True):
    #             if (info.get(f'{subtree}')):
    #                 ms += 1
    #         return ms
    #     else:
    #         ms_tmp = generate_tet(g,subtree,ms)
    #         for n, info in g.nodes(data=True):
    #             if (info.get(f'{subtree}')):
    #                 ms +=1
    #         return ms
    # return ms
            # for loop for all leaf nodes in G(spec)
            #add internal

        # if type(subtree) == nltk.tree.Tree:
        #     #print(subtree.label())
        #     for n, info in g.nodes(data=True):
        #         if info.get(f'{subtree.label()}'):
        #             a.append('a')
        #             #print(info)
        #         #print(subtree.label())
        #     generate_tet(g,subtree)
        #     #print(subtree)
    # for n, info in g.nodes(data=True):
    #     if (info.get('genre')):
    #         print(n)

# def generate_tet(g):
#     nx.set_node_attributes(g, 0, 'count')
#     users = [u for u in g.nodes if u[0] == 'u']
#     print(g.nodes(data=True))
#     leaf = [x for x in g.nodes() if g.out_degree(x) == 0 and g.in_degree(x) >= 1]
#     internal = [x for x in g.nodes() if g.out_degree(x) >= 1 and g.in_degree(x) >= 1]
#     print(leaf[0])
#     print(internal[0])
#     # for u in users:
#         # count_of_counts(u, g, leaf, internal)
#     most_def(g, leaf, internal)
#     #res = [idx for idx in test_list if idx[0].lower() == check.lower()]
#     # #users = {n for n, d in g.nodes(data=True) if d["bipartite"] == 0}
#     # for u in set1:
#     #     foo(u,g)
#     return g
def transform_data():
    relations = ['has_genre', 'directed_by', 'rated','country'] #'acted_by',
    with open('Data/knowledge-tree.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(('head', 'relation', 'tail'))
        for x in range(500):  # len(data)
            movieID = data['title'][x]
            id = data['movieId'][x]
            try:
                movie = moviesDB.get_movie(moviesDB.search_movie(movieID)[0].movieID)
            except:
                continue
            try:
                title = movie['title']
            except:
                title = 'null'
            print(f'.:{x + 1}/{len(data)}:. - {title}')
            for r in relations:
                if r == 'has_genre':
                    try:
                        genres = movie['genres']
                        for g in genres:
                            writer.writerow(('m' + str(id), 'has_genre', g))
                    except:
                        g = 'null'
                        writer.writerow(('m' + str(id), 'has_genre', g))
                if r == 'directed_by':
                    try:
                        director = movie['directors']
                        for d in director:
                            writer.writerow(('m' + str(id), 'has_director', d))
                    except:
                        d = 'null'
                        writer.writerow(('m' + str(id), 'has_director', d))
                if r == 'acted_by':
                    try:
                        cast = movie['cast']
                        for a in cast[:5]:
                            writer.writerow(('m' + str(id), 'has_actors', a))
                    except:
                        cast = 'null'
                        writer.writerow(('m' + str(id), 'has_actors', cast))
                if r == 'rated':
                    try:
                        rating = ratings['rating']
                        _movie = [x for x in ratings if x['movieID'] == id]
                        for mm in _movie:
                            #writer.writerow(('u' + str(int(rating['userId'][id])), 'has_rated', 'm' + str(id)))
                            writer.writerow(('u' + str(int(mm['userId'][id])), 'has_rated', 'm' + str(id)))
                    except:
                        writer.writerow(('u' + str(int(rating[id])), 'has_rated', 'm' + str(id)))
                if r == 'country':
                    try:
                        country = movie['countries']
                        for c in country:
                            writer.writerow(('m' + str(id), 'has_countries', c))
                    except:
                        c = 'null'
                        writer.writerow(('m' + str(id), 'has_countries', c))


def split_data():
    ds_init = pd.read_csv('Data\knowledge-tree.csv', delimiter='\t', encoding='utf-8')  # engine='python'
    ds = ds_init.sample(frac=1)  # shuffles the data
    bookmark = 0  # len(ds)
    for i in ['movie-train', 'movie-valid', 'movie-test']:
        with open('Data/%s.txt' % i, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            for j in range(round(len(ds) / 3) - 1):
                writer.writerow(ds.iloc[bookmark + j])
        bookmark = bookmark + round(len(ds) / 3 - 1)


# tree = Tree.fromstring("(user(movie(genre))(movie(genre)))")
# print(tree.leaves())
graph = generate_bipartite_graph()
speci = nested_list(["user","movie","genre"],[("user","movie"),("movie","genre")])
graph2 = generate_tet(graph,'user',speci)
print(graph2)

#[print(x) for x in graph2.nodes(data=True) if x[0] == "u2"]
#yes = graph2.nodes['m1']['count']

#print(yes)
# myesss = yes['count']
# print(myesss)
#print(graph2.nodes(data=True))
#transform_data()
#split_data()

# def get_data():
#     with open('Data/csvfile.csv', 'w',encoding='utf-8') as f:
#         writer = csv.writer(f, delimiter=',')
#         writer.writerow(('id', 'title', 'year', 'rating', 'genres', 'director', 'country'))
#         for x in range(0,25): # iteratates through each movie
#             id = data['movieId'][x]
#             movie = moviesDB.get_movie(id)
#             # TITLE
#             try: title = movie['title']
#             except: title = 'null'
#             # YEAR
#             try: year = movie['year']
#             except: year = 'null'
#             # RATING
#             try: rating = movie['rating']
#             except: rating = 'null'
#             # DIRECTORS
#             try: director = movie['directors']
#             except: director = 'null'
#             # COUNTRIES
#             try: country = movie['countries']
#             except: country = 'null'
#             # GENRES
#             try: genres = movie['genres']
#             except: genres = 'null'
#             # votes = movie['votes'],kind = movie['kind'],plot = movie['plot'],aka = movie['akas'],
#             # casting = movie['cast']
#             print(f'{x+1}/25\r', end="")
#             writer.writerow((id,title,year,rating,genres,director,country))
#     print()

# def combine_data():
#     all_filenames = ['Data\has_genre.csv', 'Data\has_rating.csv', 'Data\directed_by.csv']
#     combined_csv = pd.concat(
#         [pd.read_csv(f, encoding="ISO-8859-1", engine='python', delimiter='\t') for f in all_filenames])
#     combined_and_shuffled_csv = combined_csv.sample(frac=1)
#     combined_and_shuffled_csv.to_csv("Data\combined.csv", index=False, sep='\t')
