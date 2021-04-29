from engine.multiset import *
from engine.data_setup import *
from engine.distance import *
from engine.evaluation import *
from collections import defaultdict
from matrix_fac import *
import itertools
import pandas as pd
# from pymf import *
# import pymf
from functools import partial
from sklearn.decomposition import NMF

from engine.recommender import *
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import pickle
import mapply as mpp


def dist_func_par(x, y, tet_dict, spec):
    # print(x, y)
    # print(int(x[1:]))
    if x == y:
        return 0
    elif int(x[1:]) > int(y[1:]):
        return 0
    else:
        # hist1 = tet_dict[int(x[1:])-1]
        # hist2 = tet_dict[int(y[1:])-1]
        hist1 = tet_dict[x]
        hist2 = tet_dict[y]
        lul = hist1.get_histogram()
        lul2 = hist2.get_histogram()
        dist = calc_distance(lul, lul2, spec, 'movie')
        return dist

def f(movie_element, tet_dict, list_movies, spec):
    # arr = np.zeros(len(movie_element))
    list_res = []
    movies = []
    for x in list_movies:
        if x == movie_element:
            movies.append(x)
            break
        else:
            movies.append(x)

    for i, x in enumerate(movies):
        hist1 = tet_dict[x]
        hist2 = tet_dict[movie_element]
        lul = hist1.get_histogram()
        lul2 = hist2.get_histogram()
        # arr[i] = calc_distance(lul, lul2, spec, 'movie')
        list_res.append(calc_distance(lul, lul2, spec, 'movie'))

    tuple = (movie_element, list_res)
    return tuple
    # print(movies)
    # print(movie_element, tet_dict[movie_element])
    # print(movie_element)

# def dist_func_ny(tet, num_movies):
#     res =
if __name__ == '__main__':
    file = open("movie_tets.obj", 'rb')
    tet = pickle.load(file)
    file.close()

    # sorted_movies = sort_items(movies)
    # print(sorted_movies)
    sorted_tets = sort_items_prefix(tet, "m")
    tet_dict = {}
    for i in sorted_tets:
        yes = func_get_movie(i)
        tet_dict[yes] = i
    list_movies = []
    list_tets = []
    for x in tet_dict:
        list_movies.append(x)
        list_tets.append(tet_dict[x])
    # for j in tet_dict:
    #     print(j)

    specification_moviessss = [
        ["movie", "has_genres", "has_votes", "has_imdb_rating", "has_user_rating", "has_director", "has_awards",
         "has_nominations",
         'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
         'Film-Noir',
         'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
        [("movie", "has_genres"), ("movie", "has_votes"), ("movie", "has_imdb_rating"), ("movie", "has_user_rating"),
         ("movie", "has_director"),
         ("has_director", "has_awards"), ("has_director", "has_nominations"),
         ("has_genres", "Action"), ("has_genres", "Adventure"), ("has_genres", "Animation"), ("has_genres", "Children"),
         ("has_genres", "Comedy"),
         ("has_genres", "Crime"), ("has_genres", "Documentary"), ("has_genres", "Drama"), ("has_genres", "Fantasy"),
         ("has_genres", "Film-Noir"),
         ("has_genres", "Horror"), ("has_genres", "IMAX"), ("has_genres", "Musical"), ("has_genres", "Mystery"),
         ("has_genres", "Romance"),
         ("has_genres", "Sci-Fi"), ("has_genres", "Thriller"), ("has_genres", "War"), ("has_genres", "Western")]]
    spec2 = tet_specification(specification_moviessss[0], specification_moviessss[1])

    num_movies = len(tet_dict)
    print(num_movies)
    pool = Pool(mp.cpu_count()-1)
    results = (pool.map(partial(f, tet_dict=tet_dict, list_movies=list_movies, spec=spec2), list_movies))

    filehandler = open("Matrix_lists.obj", "wb")
    pickle.dump(results, filehandler)
    filehandler.close()

    file = open("C:\\Users\\caspe\\PycharmProjects\\P9-10-Master-Thesis\\engine\\Matrix_lists.obj", 'rb')
    tet = pickle.load(file)
    file.close()

    data = [x[1] for x in tet]
    movies = []
    for x in tet:
        movies.append(x[0])

    df = pd.DataFrame(data=data, index=movies, columns=movies).fillna(0)
    cols = df.columns.values.tolist()
    rows = list(df.index)

    X = df.to_numpy()
    X = X + X.T - np.diag(np.diag(X))
    # # print(X)
    #
    mirrored = pd.DataFrame(data=X, index=rows, columns=cols)
    mirrored.to_csv("TEST_TEST_TEST_TEST_TEST.csv", sep='\t')