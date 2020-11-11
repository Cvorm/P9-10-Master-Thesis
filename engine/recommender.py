import pandas as pd
import imdb
from engine.multiset import *
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


def generate_tet(graph, root ,spec):
    roots = [n for n, info in graph.nodes(data=True) if info.get(f'{root}')]
    complete = []
    complete.append(generate_tet_yes(graph, roots[0], spec))
    return complete


def is_internal(element, list):
     new = [x[0] for x in list]
     if (element in new):
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
                # add_node (x[1]) with count 1
                ms.add_node_w_count(x[0], len((graph.out_edges(x[0]))))
            #if not(is_internal(x[1], nodess_reversed)):
            if len(graph.out_edges(x[1])) == 0:
                # add_node (x[1]) with count 1
                ms.add_node_w_count(x[1], 1)
        for (k,l) in nodess:
            ms.add_edge((k,l))
    for (e1,e2) in graph.edges(user):
        ms.add_edge((e1,e2))
    return ms


graph = generate_bipartite_graph()
speci = nested_list(["user","movie","genre"],[("user","movie"),("movie","genre")])
graph2 = generate_tet(graph,'user',speci)
print(graph2[0].graph.nodes(data=True))
graph2[0].count_tree()
graph2[0].logistic_eval(0,1)
graph2[0].histogram()
print(graph2[0].graph.edges(data=True))
print(graph2[0].graph.nodes(data=True))
#print(graph2)




