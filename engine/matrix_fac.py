from engine.multiset import *
from engine.data_setup import *
from collections import defaultdict
import itertools
import pandas as pd

from engine.recommender import *
from engine.recommender import __distance

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
    for x in tet:
        user = [x for x, y in x.graph.nodes(data=True) if y.get('root')]
        users.append(user[0])

    user_user = pd.DataFrame(index=users, columns=users).fillna(0.0)
    print(user_user.head(5))
    # print(users)
    # print(users)

    for i, y in enumerate(tet):
        # print(i.get_histogram())
        # print(i.graph.nodes(data=True).)
        # print(i.graph.nodes(data=True))
        # user1 = [x for x, y in i.graph.nodes(data=True) if y.get('root')]
        # print(values)
        for j, z in enumerate(tet):
            if users[i] == users[j]:
                # values.append(1)
                user_user[users[i]][users[j]] = 1
                print(users[i], users[j])
                break
            else:
                myes = y.get_histogram()
                myes2 = z.get_histogram()
                dist = calc_distance(myes, myes2, spec, 'user')
                user_user[users[i]][users[j]] = dist
                # values.append(dist)

    user_user.to_csv("user_user_matrix.csv", sep='\t')


def func(element):
    return int(element.split("m")[1])


def sort_items(items):
    sortlist = sorted(items, key=func)
    return sortlist

def item_item_sim(tet, spec):
    for x in tet:
        print(x.get_histogram())
        # for j in x.graph.nodes(data=True):
        #     print(j)
            # if type(j) is str and j[0] == 'm':
            #     print(j)
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

    user_item = pd.DataFrame(index=users, columns=sorted_items).fillna(0)
    print(user_item.head(5))

    for i, y in enumerate(tet):
        movies = get_movies_in_user(y)
        for j in movies:
            user_item[j][users[i]] = 1
            # print(users[i], j)

    user_item.to_csv("user_item_matrix.csv", sep='\t')
    # movies = []
    # for u in other_users_hist:
    #     temp_movies = get_movies_user(u)
    #     for i in temp_movies:
    #         movies.append(i)
    # no_duplicates = [list(v) for v in dict(movies).items()]
    # mu = get_movies_in_user(user_hist)
    # no_mas = [(x,y) for x,y in no_duplicates if x not in mu]
    # tmp_val = len(no_duplicates) - len(no_mas)
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
