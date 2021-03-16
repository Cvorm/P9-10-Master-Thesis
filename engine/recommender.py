from engine.multiset import *
from engine.data_setup import *
import ast


def create_user_book_tet(spec, dat):
    print('colons')
    print(dat.columns)
    roots = np.unique(dat.UserID)
    complete = []
    for r in roots:
        ms = Multiset()  # here we instantiate our TETs
        complete.append(__create_user_book_tet(r, spec, ms, dat))
    return complete
# call function for creating user TETs for MovieLens


def __create_user_book_tet(user, tet_spec, ms, dat):
    nodes = [n[-1] for n in dfs_edges(tet_spec, source="user")]
    ms.add_root(user, 1)
    user_ratings = dat[dat['UserID'] == user]
    user_info = users[users['UserID'] == user]
    user_book = [books.ISBN.isin(user_ratings['ISBN'])]
    # user_book['BookRating'] = user_book['BookRating'].fillna(0).astype(int)

    for node in nodes:
        if node == 'has_rated':
            for y,x in user_ratings.iterrows():
                ms.add_node_w_count(str(x['ISBN']), 1, 'has_rated')
                ms.add_edge((user, str(x['ISBN'])))
        elif node == 'rating':
            for y, x in user_ratings.iterrows():
                rat = int(x['BookRating']) * 0.1
                ms.add_node_w_count_w_val(f'ur{y}', rat, int(x['BookRating']), 'rating')
                ms.add_edge((str(x['ISBN']), f'ur{y}'))
        elif node == 'location':
            for y, x in user_info.iterrows():
                tmp = x['country']
                # print(tmp)
                ms.add_node_w_count(f'hl{y}', 1, 'location')
                ms.add_edge((str(x['UserID']), f'hl{y}'))
                if not tmp == None:
                    for idx, country in enumerate(tmp):
                        ms.add_node_w_count(f'l{y}{idx}', 1, str(country))
                        ms.add_edge((f'hl{y}', f'l{y}{idx}'))
        elif node == 'age':
            for y, x in user_info.iterrows():
                ag = int(x['Age']) * 0.0111
                ms.add_node_w_count(f'a{y}', ag, 'age')
                ms.add_edge((user, f'a{y}'))
    return ms




def create_user_movie_tet(spec, dat):
    roots = np.unique(dat.userId)
    # roots = [n for n, info in graph.nodes(data=True) if info.get(f'{root}')]
    complete = []
    for r in roots:
        ms = Multiset()  # here we instantiate our TETs
        complete.append(__create_user_movie_tet(r, spec, ms, dat))
    return complete


# function that handles the construction of a TET for a user for MovieLens
def __create_user_movie_tet(user, tet_spec, ms, dat):
    nodes = [n[-1] for n in dfs_edges(tet_spec, source="user")]
    ms.add_root(user, 1)
    user_ratings = dat[dat['userId'] == user]
    user_movie = data[data.movieId.isin(user_ratings['movieId'])]
    user_movie['rating'] = user_movie['rating'].fillna(0).astype(int)
    user_director = updated_actor[updated_actor.actorId.isin(user_movie['director'])]

    for node in nodes:
        if node == 'has_rated':
            for y,x in user_ratings.iterrows():
                ms.add_node_w_count(str(x['movieId']), 1, 'has_rated')
                ms.add_edge((user, str(x['movieId'])))
        elif node == 'has_user_rating':
            for y, x in user_ratings.iterrows():
                rat = int(x['rating']) * 0.2
                ms.add_node_w_count_w_val(f'ur{y}', rat, int(x['rating']), 'has_user_rating')
                ms.add_edge((str(x['movieId']), f'ur{y}'))
        elif node == 'has_imdb_rating':
            for y, x in user_movie.iterrows():
                # rat = int(x['rating']) * 0.1
                ms.add_node_w_count(f'ir{y}', float(x['rating']), 'has_imdb_rating')
                ms.add_edge((str(x['movieId']), f'ir{y}'))
        elif node == 'has_votes':
            for y, x in user_movie.iterrows():
                ms.add_node_w_count(f'hv{y}', float(x['votes']), 'has_votes')
                ms.add_edge((str(x['movieId']), f'hv{y}'))
        elif node == 'has_director':
            for y, x in user_movie.iterrows():
                ms.add_node_w_count(str(x['director']), 1, 'has_director')
                ms.add_edge((str(x['movieId']), str(x['director'])))
        elif node == 'has_genres':
            for y, x in user_movie.iterrows():
                tmp = x['genres']
                tmp = ast.literal_eval(tmp)
                tmp_count = len(tmp)
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


# function that returns the total list of genres for all movies
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


