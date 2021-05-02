from engine.multiset import *
from engine.data_setup import *
from engine.distance import *
from engine.evaluation import *
from collections import defaultdict
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

specification_moviessss = [["movie", "has_genres", "has_votes", "has_imdb_rating", "has_user_rating", "has_director", "has_awards", "has_nominations",
                        'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                        'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
                       [("movie", "has_genres"), ("movie", "has_votes"), ("movie", "has_imdb_rating"), ("movie", "has_user_rating"),
                        ("movie", "has_director"),
                        ("has_director", "has_awards"), ("has_director", "has_nominations"),
                        ("has_genres", "Action"),  ("has_genres", "Adventure"), ("has_genres", "Animation"),  ("has_genres", "Children"),  ("has_genres", "Comedy"),
                        ("has_genres", "Crime"),  ("has_genres", "Documentary"), ("has_genres", "Drama"),  ("has_genres", "Fantasy"),  ("has_genres", "Film-Noir"),
                        ("has_genres", "Horror"),  ("has_genres", "IMAX"), ("has_genres", "Musical"),  ("has_genres", "Mystery"),  ("has_genres", "Romance"),
                        ("has_genres", "Sci-Fi"),  ("has_genres", "Thriller"), ("has_genres", "War"),  ("has_genres", "Western")]]
spec2 = tet_specification(specification_moviessss[0], specification_moviessss[1])
# from engine.recommender import __distance

# spec2 = [["user,", "movie", "genre", "director", "rating", "award"],
#         [("user", "movie"), ("movie", "director"), ("movie", "rating"), ("director", "award")],
#         ["movie", "user", "director"]]
#
# genres = get_genres()
# speci_test = tet_specification2(spec2[0], spec2[1], spec2[2], genres)

# spec2 = [["user,", "movie", "genre", "director", "rating", "award"],
#          [("user", "movie"), ("movie", "director"), ("movie", "rating"), ("director", "award")],
#          ["movie", "user", "director"]]
#
# genres = get_genres()
# speci_test = tet_specification2(spec2[0], spec2[1], spec2[2], genres)
# print(speci_test.nodes())

# x_train, x_test = run_data()
# train_test = x_train.append(x_test)
#
# # print(yes)
# # print(x_train)
# # print(x_test)
# print(train_test)
# df = train_test.drop(columns=['rating', 'timestamp'])
# users = df.drop_duplicates(subset = ["userId"])
# users_list = users["userId"].tolist()
#
# items = df.drop_duplicates(subset = ["movieId"])
# items_list = items["movieId"].tolist()
# items_list.sort()
# # print(items_list)
# # print(users_list)
# # print(users)
# user_user = pd.DataFrame(index=users_list, columns=users_list).fillna(0)
# item_item = pd.DataFrame(index=items_list, columns=items_list).fillna(0)
# user_item = pd.DataFrame(index=items_list, columns=users_list).fillna(0)
# print(user_user.head(5))
# print(item_item.head(5))
# print(user_item.head(5))

# num_cpu = 8

def dist_func(x, y, tet_dict, spec):
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

# def parrallel_df_item_item(tet, spec, num_cpu):
#     if __name__ == '__main__':
#         movies = []
#         for t in tet:
#             movie = [x for x, y in t.graph.nodes(data=True) if y.get('root')]
#             # print(movie)
#             movies.append(movie[0])
#         # for t in test_tet:
#         #     movie = [x for x, y in t.graph.nodes(data=True) if y.get('root')]
#         #     # print(movie)
#         #     movies.append(movie[0])
#
#         sorted_movies = sort_items(movies)
#         # print(sorted_movies)
#         sorted_tets = sort_items_prefix(tet, "m")
#
#         item_item = pd.DataFrame(index=sorted_movies, columns=sorted_movies).fillna(0.0)
#         split_size = math.floor(item_item.shape[1]/num_cpu)
#         list_dfs = []
#         for i in range(num_cpu-1):
#             list_dfs.append(item_item.iloc[:, split_size*i:split_size*(i+1)])
#         list_dfs.append(item_item.iloc[:,(num_cpu-1)*split_size:])
#
#         print("jungle gap")
#
#         pool = Pool(8)
#         dataa = pd.concat(pool.map(partial(dist_func, sorted_tets=sorted_tets, spec=spec), list_dfs[0]))
#         pool.close()
#         pool.join()
#
#         return dataa


# if __name__ == '__main__':
#
#     file = open("movie_tets.obj", 'rb')
#     tet = pickle.load(file)
#     file.close()
#     movies = []
#
#     for t in tet:
#         movie = [x for x, y in t.graph.nodes(data=True) if y.get('root')]
#         # print(movie)
#         movies.append(movie[0])
#     # for t in test_tet:
#     #     movie = [x for x, y in t.graph.nodes(data=True) if y.get('root')]
#     #     # print(movie)
#     #     movies.append(movie[0])
#
#     sorted_movies = sort_items(movies)
#     # print(sorted_movies)
#     sorted_tets = sort_items_prefix(tet, "m")
#
#     item_item = pd.DataFrame(index=sorted_movies, columns=sorted_movies).fillna(0.0)
#     split_size = math.floor(item_item.shape[1]/num_cpu)
#     list_dfs = []
#     for i in range(num_cpu-1):
#         list_dfs.append(item_item.iloc[:, split_size*i:split_size*(i+1)])
#     list_dfs.append(item_item.iloc[:,(num_cpu-1)*split_size:])
#
#     print("jungle gap")
#
#     pool = Pool(8)
#     dataa = pd.concat(pool.map(partial(dist_func, sorted_tets=sorted_tets, spec=spec2), list_dfs[0]))
#     pool.close()
#     pool.join()

        # return dataa

# def lulul(tet, spec, num_cpu):
#     if __name__ == '__main__':
#         pool = Pool(8)
#         dataa = pd.concat(pool.map(partial(dist_func, sorted_tets=sorted_tets, spec=spec), list_dfs[0]))
#         pool.close()
#         pool.join()
#         return dataa

def split_df(df, num_cpu):

    list_dfs = []
    split_size = math.floor(df.shape[1] / num_cpu)
    for i in range(num_cpu-1):
        list_dfs.append(df.iloc[:, split_size*i:split_size*(i+1)])
    list_dfs.append(df.iloc[:,(num_cpu-1)*split_size:])
    return list_dfs

# def parallelize(func, num_of_processes=8):
#     file = open("movie_tets.obj", 'rb')
#     tet = pickle.load(file)
#     file.close()
#
#     movies = []
#
#     for t in tet:
#         movie = [x for x, y in t.graph.nodes(data=True) if y.get('root')]
#         # print(movie)
#         movies.append(movie[0])
#
#     sorted_movies = sort_items(movies)
#     # print(sorted_movies)
#     sorted_tets = sort_items_prefix(tet, "m")
#
#     item_item = pd.DataFrame(index=sorted_movies, columns=sorted_movies).fillna(0.0)
#     split_size = math.floor(item_item.shape[1]/num_of_processes)
#     list_dfs = []
#     for i in range(num_of_processes-1):
#         list_dfs.append(item_item.iloc[:, split_size*i:split_size*(i+1)])
#     list_dfs.append(item_item.iloc[:,(num_of_processes-1)*split_size:])
#
#     # data_split = np.array_split(data, num_of_processes)
#     if __name__ == '__main__':
#         pool = Pool(num_of_processes)
#         data = pd.concat(pool.map(func, list_dfs))
#         pool.close()
#         pool.join()
#         return data
#
# def run_on_subset(func, data_subset):
#     return data_subset.apply(func, axis=1)
#
# def parallelize_on_rows(func, num_of_processes=8):
#     return parallelize(partial(run_on_subset, func), num_of_processes)
#
# file = open("movie_tets.obj", 'rb')
# tet = pickle.load(file)
# file.close()
#
# movies = []
#
# for t in tet:
#     movie = [x for x, y in t.graph.nodes(data=True) if y.get('root')]
#     # print(movie)
#     movies.append(movie[0])
#
# sorted_movies = sort_items(movies)
# # print(sorted_movies)
# sorted_tets = sort_items_prefix(tet, "m")
#
# yesshu = parallelize_on_rows(partial(dist_func, sorted_tets=sorted_tets, spec=spec2))


def func_get_movie(element):
    # print(element)
    item = [x for x, y in element.graph.nodes(data=True) if y.get('root')]
    # if len(item) >= 1:
    myes = item[0]
    # else:
    #     myes = item
    # print(myes)
    return myes
if __name__ == '__main__':
    file = open("movie_tets.obj", 'rb')
    tet = pickle.load(file)
    file.close()
    movies = []

    for t in tet:
        movie = [x for x, y in t.graph.nodes(data=True) if y.get('root')]
        # print(movie)
        movies.append(movie[0])
    # for t in test_tet:
    #     movie = [x for x, y in t.graph.nodes(data=True) if y.get('root')]
    #     # print(movie)
    #     movies.append(movie[0])

    sorted_movies = sort_items(movies)
    # print(sorted_movies)
    sorted_tets = sort_items_prefix(tet, "m")

    item_item = pd.DataFrame(index=sorted_movies, columns=sorted_movies).fillna(0.0)
    splits = split_df(item_item, mp.cpu_count())
    tet_dict = {}
    for i in sorted_tets:
        yes = func_get_movie(i)
        tet_dict[yes] = i
    # num_workers = mp.cpu_count()
    # num_workers = 2
    # pool = mp.Pool(num_workers)
    # data = []

    for split in splits:
        # for column in df:
        # yes = map(partial(dist_func, sorted_tets=sorted_tets, spec=spec2), df)
        series_rows = pd.Series(split.index)
        series_cols = pd.Series(split.columns)
        # noinspection PyTypeChecker
        # ny = mpp.mapply(item_item, series_rows.apply(lambda x: series_cols.apply(lambda y: dist_func(x, y, tet_dict=tet_dict, spec=spec2))))
        res = pd.DataFrame(series_rows.apply(lambda x: series_cols.apply(lambda y: dist_func(x, y, tet_dict=tet_dict, spec=spec2))))
        res.index = series_rows
        res.columns = series_cols
        print("yes")
    # df['parcels'] = pool.map(func,df0['parcels'].values) # specify the function and arguments to map
    # pool.close()
    # pool.join()
    item_item.to_csv("TEST_TEST_TEST_TEST_TEST.csv", sep='\t')


    # data_split = np.array_split(data, num_cpu)
    # pool = Pool(num_cpu)
    # data = pd.concat(pool.map(func, data_split))
    # pool.close()
    # pool.join()
    # return data
def user_user_sim(tet, spec):
    # num_users = len(user_list)
    # print(num_users)
    # spec2 = [["user,", "movie", "genre", "director", "rating", "award"],
    #          [("user", "movie"), ("movie", "director"), ("movie", "rating"), ("director", "award")],
    #          ["movie", "user", "director"]]
    #
    # genres = get_genres()
    # speci_test = tet_specification2(spec2[0], spec2[1], spec2[2], genres)
    # print(speci_test)
    values = []
    users = []
    # tets = [x for x in tet]
    # sorted_tets = sort_tets(tet)
    # for x in sorted_tets:
    #     print(x.get_histogram())
    sorted_tets = []
    for x in tet:
        user = [x for x, y in x.graph.nodes(data=True) if y.get('root')]
        users.append(user[0])

    sorted_users = sort_users(users)
    print(sorted_users)
    # for x in sorted_users:
    #     sorted_tets.append(get_tet_user(tet, x))
    sorted_tets = sort_tets(tet)


    user_user = pd.DataFrame(index=sorted_users, columns=sorted_users).fillna(0.0)
    print(user_user.head(5))
    # print(users)
    # print(users)

    for i, y in enumerate(sorted_tets):
        # print(i.get_histogram())
        # print(i.graph.nodes(data=True).)
        # print(i.graph.nodes(data=True))
        # user1 = [x for x, y in i.graph.nodes(data=True) if y.get('root')]
        # print(values)
        for j, z in enumerate(sorted_tets):
            if sorted_users[i] == sorted_users[j]:
                # values.append(1)
                user_user[sorted_users[i]][sorted_users[j]] = 1
                print(sorted_users[i], sorted_users[j])
                break
            else:
                myes = y.get_histogram()
                myes2 = z.get_histogram()
                dist = calc_distance(myes, myes2, spec, 'user')
                user_user[sorted_users[i]][sorted_users[j]] = dist
                # values.append(dist)

    user_user.to_csv("user_user_matrix.csv", sep='\t')
    # return user_user

def func(element):
    # print(element)
    return int(element.split("m")[1])

def func_2(element):
    return int(element.split("u")[1])

def func_get_user(element):
        user = [x for x, y in element.graph.nodes(data=True) if y.get('root')]
        myes = user[0]
        # print(myes)
        return int(myes.split("u")[1])

def sort_items_prefix(items, prefixx):
    # print(items)
    sortlist = sorted(items, key=partial(func_get_root, prefix=prefixx))
    return sortlist

def func_get_root(element, prefix):
    # print(element)
    item = [x for x, y in element.graph.nodes(data=True) if y.get('root')]
    # if len(item) >= 1:
    myes = item[0]
    # else:
    #     myes = item
    # print(myes)
    return int(myes.split(prefix)[1])

def sort_items(items):
    sortlist = sorted(items, key=func)
    return sortlist

def sort_users(users):
    sortlist = sorted(users, key=func_2)
    return sortlist

def sort_tets(tets):
    sortlist = sorted(tets, key=func_get_user)
    return sortlist

def item_item_sim(tet, spec):
    movies = []
    for t in tet:
        movie = [x for x, y in t.graph.nodes(data=True) if y.get('root')]
        # print(movie)
        movies.append(movie[0])
    # for t in test_tet:
    #     movie = [x for x, y in t.graph.nodes(data=True) if y.get('root')]
    #     # print(movie)
    #     movies.append(movie[0])

    sorted_movies = sort_items(movies)
    # print(sorted_movies)
    sorted_tets = sort_items_prefix(tet, "m")

    item_item = pd.DataFrame(index=sorted_movies, columns=sorted_movies).fillna(0.0)
    print(item_item.shape)
    # exit(0)
    for i, y in enumerate(sorted_tets):
        # print(i)
        # if i == 10:
        #     item_item.to_csv("item_item_matrixxxx.csv", sep='\t')
        #     break
        for j, z in enumerate(sorted_tets):
            if sorted_movies[i] == sorted_movies[j]:
                # hist1 = y.get_histogram()
                # hist2 = z.get_histogram()
                # dist = calc_distance(hist1, hist2, spec, 'movie')
                item_item[sorted_movies[i]][sorted_movies[j]] = 0.0  #They have distance 0 i.e they are the same
                break
            else:
                hist11 = y.get_histogram()
                hist22 = z.get_histogram()
                dist = calc_distance(hist11, hist22, spec, 'movie')
                item_item[sorted_movies[i]][sorted_movies[j]] = dist

    item_item.to_csv("item_item_matrix_peter_correct.csv", sep='\t')
    # return item_item
    #
    #     for i, y in enumerate(sorted_tets):
    #         # print(i.get_histogram())
    #         # print(i.graph.nodes(data=True).)
    #         # print(i.graph.nodes(data=True))
    #         # user1 = [x for x, y in i.graph.nodes(data=True) if y.get('root')]
    #         # print(values)
    #         for j, z in enumerate(sorted_tets):
    #             if sorted_users[i] == sorted_users[j]:
    #                 # values.append(1)
    #                 user_user[sorted_users[i]][sorted_users[j]] = 1
    #                 print(sorted_users[i], sorted_users[j])
    #                 break
    #             else:
    #                 myes = y.get_histogram()
    #                 myes2 = z.get_histogram()
    #                 dist = calc_distance(myes, myes2, spec, 'user')
    #                 user_user[sorted_users[i]][sorted_users[j]] = dist
    #                 # values.append(dist)



    # print(hejDaniel)
    # for t in tet:
    #     for x, y in t.graph.nodes(data=True):
    #         print(x,y)
            # print(t.ht.nodes(data=True))
    # tet[i].ht.nodes(data=True)) for i in range(top)]
        # print("yes")
        # print(t.get_histogram())

    # attributes = {}
    # for t in tet:
    #     for x, y in t.graph.nodes(data=True):
    #         attributes[x] = y
    #
    # for t in tet:
    #     for x, y in t.graph.nodes(data=True):
    #         # print(x, y)
    #         if y.get('type') == 'has_rated':
    #             # print(f'has genres {y["count"]}, node {x}')
    #             print(x)
    #             # for lul in t.graph.neighbors(x):
    #             for n in descendants(t.get_graph(),x):
    #                  print("--" + n, attributes[n])



        # print(attributes)
                    # for k in n:
                    #     print(k)
        # print(x.get_histogram())
        # nodesssss = [n for n in edge_dfs(x.graph.nodes(data=True), source=root)]
        # for j in x.graph.nodes(data=True):
            # if type(j[0]) is str and j[0][0] == 'm':
            #     print(j)
                # print(j)
            #     print(j[0][0])
            # print(j)
            # if type(j) is str and j[0] == 'm':
            #     print("matwawwmatasfddsffds")
        # for x, y in user.graph.nodes(data=True):
        #     if type(x) is str and x[0] == 'm':
        # print(x.graph.nodes(data=True))
    # total_movies = []
    # for x in tet:
    #     movies = get_movies_in_user(x)
    #     for i in movies:
    #         total_movies.append(i)
    # # print(total_movies)

    # no_duplicates = list(set(total_movies))
    # no_duplicates = [list(v) for v in dict(total_movies).items()]

    # no_duplicates = list(set(total_movies))
    # sorted_items = sort_items(no_duplicates)
    # # print(sorted_items)
    # item_item = pd.DataFrame(index=sorted_items, columns=sorted_items).fillna(0.0)

    # print(no_duplicates)
    # item_item.to_csv("item_item_matrix.csv", sep=',')
    # for i, y in enumerate(tet):
    #     # print(i.get_histogram())
    #     # print(i.graph.nodes(data=True).)
    #     # print(i.graph.nodes(data=True))
    #     # user1 = [x for x, y in i.graph.nodes(data=True) if y.get('root')]
    #     # print(values)
    #     for j, z in enumerate(tet):
    #         if sorted_items[i] == sorted_items[j]:
    #             # values.append(1)
    #             item_item[sorted_items[i]][sorted_items[j]] = 1
    #             print(sorted_items[i], sorted_items[j])
    #             break
    #         else:
    #             myes = y.get_histogram()
    #             myes2 = z.get_histogram()
    #             dist = calc_distance(myes, myes2, spec, 'movie')
    #             item_item[sorted_items[i]][sorted_items[j]] = dist
    #             # values.append(dist)
    #
    # item_item.to_csv("item_item_matrix.csv", sep='\t')

def interaction_matrix(tet):
    users = []
    for x in tet:
        user = [x for x, y in x.graph.nodes(data=True) if y.get('root')]
        users.append(user[0])
    print(users)

    total_movies = []
    for x in tet:
        movies = get_movies_in_user(x)
        for i in movies:
            total_movies.append(i)
    # print(total_movies)

    # no_duplicates = list(set(total_movies))
    # no_duplicates = [list(v) for v in dict(total_movies).items()]
    no_duplicates = list(set(total_movies))
    sorted_items = sort_items(no_duplicates)
    sorted_users = sort_users(users)
    sorted_tets = sort_tets(tet)

    user_item = pd.DataFrame(index=sorted_users, columns=sorted_items).fillna(0)
    # print(user_item.head(5))

    for i, y in enumerate(sorted_tets):
        movies = get_movies_in_user(y)
        user = [x for x, z in y.graph.nodes(data=True) if z.get('root')]
        print(user[0], movies)
        for j in movies:
            user_item[j][sorted_users[i]] = 1
            # print(users[i], j)

    trans = user_item.T
    trans.to_csv("user_item_matrix_peterr_ratings.csv", sep='\t')
    # user_item.to_csv("user_item_matrix.csv", sep='\t')
    # return user_item
    # movies = []
    # for u in other_users_hist:
    #     temp_movies = get_movies_user(u)
    #     for i in temp_movies:
    #         movies.append(i)
    # no_duplicates = [list(v) for v in dict(movies).items()]
    # mu = get_movies_in_user(user_hist)
    # no_mas = [(x,y) for x,y in no_duplicates if x not in mu]
    # tmp_val = len(no_duplicates) - len(no_mas)

def user_item_rating_matrix(tet):
    users = []
    for x in tet:
        user = [x for x, y in x.graph.nodes(data=True) if y.get('root')]
        users.append(user[0])

    total_movies = []
    for x in tet:
        movies = get_movies_in_user(x)
        for i in movies:
            total_movies.append(i)
    # print(total_movies)

    # no_duplicates = list(set(total_movies))
    # no_duplicates = [list(v) for v in dict(total_movies).items()]
    no_duplicates = list(set(total_movies))
    print(users)
    sorted_users = sort_users(users)
    sorted_items = sort_items(no_duplicates)

    user_item = pd.DataFrame(index=sorted_users, columns=sorted_items).fillna(0)
    # print(user_item.head(5))

    for i, y in enumerate(tet):
        movies = get_movies_in_user(y)
        for j in movies:
            user_item[j][sorted_users[i]] = get_rating(y, j)
            # print(users[i], j)

    # return user_item
    trans = user_item.T
    trans.to_csv("user_item_matrix_peterr_ratings.csv", sep='\t')

def non_neg_matrix_fac(matrix):
    df = pd.read_csv(matrix, sep='\t', index_col=0)
    # print(df.head())
    ranks = len(df.index)
    model = NMF(n_components=ranks, init='random', max_iter=100)
    W = model.fit_transform(df)
    H = model.components_
    print(H)
    print(W)
    # new = model.transform(df)
    ndf = np.dot(W,H)
    for i in ndf:
        print(i)

    # print(ndf)
    # "requires=['cvxopt', 'numpy', 'scipy']"
    # ranks = len(matrix.index)
    # nnmf = pymf.NMF(matrix, num_bases=ranks)
    # nnmf.factorize()

# non_neg_matrix_fac("user_item_rating_matrix.csv")
# myesss = sort_items(["m1000", "m900", "m714", "m1300"])
# print(myesss)


            # user2 = [x for x, y in j.graph.nodes(data=True) if y.get('root')]
            # if user1[0] == user2[0]:
                # dist = __distance(i, j, speci_test, "user")
            #     values.append(1)
            #     continue
            # else:
            #     myes = i.get_histogram()
            #     myes2 = j.get_histogram()
            #     values.append(0)
                # print(myes)
                # dist = calc_distance(myes, myes2, speci_test, 'user')
                # values.append(dist)
        # print(username)

    # for i in users_list:
    #     for j in users_list:
    #         if i == j:
    #             __distance(i, j, speci_test, "user")
    #             continue
    #         else:
    #             __distance(i, j, speci_test, "user")

# calc_similarity(users_list)
#    for t in tet:
#         username = [x for x, y in t.graph.nodes(data=True) if y.get('root')]
#         if username[0] == user:
#             return t



# df_user_user = pd.DataFrame
# def user_user_sim():

def item_feature_matrix():
    items = np.unique(data.movieId)
    tmp_dat = data.drop(['genres', 'director', 'title'], axis=1)
    gen = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                         'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance',
                         'Sci-Fi', 'Thriller', 'War', 'Western']
    tmp_dat = tmp_dat.reindex(columns=['movieId', 'rating', 'votes', 'budget', 'gross', 'Action',
                                       'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                                       'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance',
                                        'Sci-Fi', 'Thriller', 'War', 'Western'])
    res_dat = tmp_dat #.set_index('movieId')
    for idx, item in enumerate(data.iterrows()):
        genres = item[1][2]
        for genre in gen:
            if genre in genres:
                res_dat.at[idx, genre] = 1
            else:
                res_dat.at[idx, genre] = 0
    res_dat = res_dat.set_index('movieId')
    print(res_dat)
    res_dat.to_csv('item_feature_matrix.csv', sep='\t')