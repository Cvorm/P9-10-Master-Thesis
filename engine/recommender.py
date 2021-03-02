from engine.multiset import *
from engine.data_setup import *
from collections import defaultdict
import itertools
import ast
pd.set_option('display.max_columns', None)


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
    print(this_movie.head())
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
            for y, x in this_movie.iterrows():
                # tmp = y['genres'][0]
                tmp = x['genres'][0]
                tmp = ast.literal_eval(tmp)
                tmp_count = len(tmp)
                # print("++", y, tmp_count)
                ms.add_node_w_count(f'hg{y}', int(tmp_count), 'has_genres')
                ms.add_edge((str(x['movieId']), f'hg{y}'))
                for idx, genre in enumerate(tmp):
                    # print(idx, genre)
                    ms.add_node_w_count(f'g{y}{idx}', 1, str(genre))
                    ms.add_edge((f'hg{y}', f'g{y}{idx}'))
                # print(tmp)
                # yes = str(tmp[0])
                # print(tmp)
                # tmp = ast.literal_eval(tmp)
                # tmp_count = len(tmp)
                # # print(tmp_count)
                # ms.add_node_w_count(f'hg{this_movie}', int(tmp_count), 'has_genres')
                # ms.add_edge((str(this_movie['movieId']), f'hg{this_movie}'))
                # for idx, genre in enumerate(tmp):
                #     ms.add_node_w_count(f'g{this_movie}{idx}', 1, str(genre))
                #     ms.add_edge((f'hg{this_movie}', f'g{this_movie}{idx}'))
        elif node == 'has_awards':
                ms.add_node_w_count(f'a{this_movie["movieId"]}', int(user_director['awards']), 'has_awards')
                ms.add_edge((str(user_director['actorId']), f'a{this_movie["movieId"]}'))
        elif node == 'has_nominations':
                ms.add_node_w_count(f'n{this_movie["movieId"]}', int(user_director['nominations']), 'has_nominations')
                ms.add_edge((str(user_director['actorId']), f'n{this_movie["movieId"]}'))
        elif node == 'has_votes':
            ms.add_node_w_count(f'hv{this_movie["movieId"]}', int(this_movie['votes']), 'has_votes')
            ms.add_edge((str(this_movie['movieId']), f'hv{this_movie["movieId"]}'))


    return ms

def create_user_tet(spec, dat):
    roots = np.unique(dat.userId)
    # roots = [n for n, info in graph.nodes(data=True) if info.get(f'{root}')]
    complete = []
    for r in roots:
        ms = Multiset()  # here we instantiate our TETs
        complete.append(__create_user_tet(r, spec, ms, dat))
    return complete


def __create_user_tet(user, tet_spec, ms, dat):
    nodes = [n[-1] for n in dfs_edges(tet_spec, source="user")]
    ms.add_root(user, 1)
    user_ratings = dat[dat['userId'] == user]
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
                tmp = ast.literal_eval(tmp)
                tmp_count = len(tmp)
                # print(y, tmp_count)
                ms.add_node_w_count(f'hg{y}', int(tmp_count), 'has_genres')
                ms.add_edge((str(x['movieId']), f'hg{y}'))
                for idx, genre in enumerate(tmp):
                    # print(idx, genre)
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


