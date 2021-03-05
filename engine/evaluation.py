from engine.metric_tree import *
from collections import defaultdict

# returns a sorted list of a users movies based on rating
def get_movies_user(user):
    movies = {}
    for x, y in user.graph.nodes(data=True):
        if type(x) is str and x[0] == 'm':
            for n in user.graph.neighbors(x):
                if type(n) is str and n[:2] == 'ur': #user.graph[n].get['type'] == 'has_user_rating':
                    rat = user.graph.nodes(data=True)[n]['mult']
                    movies[x] = rat
    sort_movies = sorted(movies.items(), key=lambda k: k[1], reverse=True)
    res = sort_movies #[:top_k_movies]
    return res


# returns a list of all movies for a user
def get_movies_in_user(user):
    tmp_list = []
    for x, y in user.graph.nodes(data=True):
        if type(x) is str and x[0] == 'm':
            tmp_list.append(x)
    return tmp_list


# returns a list of movies from other similar users, sorted based on rating
def get_movies(user_hist, other_users_hist):
    movies = []
    for u in other_users_hist:
        temp_movies = get_movies_user(u)
        for i in temp_movies:
            movies.append(i)
    no_duplicates = [list(v) for v in dict(movies).items()]
    mu = get_movies_in_user(user_hist)
    no_mas = [(x,y) for x,y in no_duplicates if x not in mu]
    sort_movies = sorted(no_mas, key=lambda k: k[1], reverse=True)
    res = sort_movies #[:top_k_movies]
    return res


def average(lst):
    return sum(lst) / len(lst)


def get_similarity(user_hist, other_users_hist):
    usersims = []
    mu = get_movies_in_user(user_hist)
    for u in other_users_hist:
        temp_movies = get_movies_user(u)
        movies = []
        for i in temp_movies:
            movies.append(i)
        # no_duplicates = [list(v) for v in dict(movies).items()]
        no_mas = [(x, y) for x, y in movies if x not in mu]
        tmp_val = len(movies) - len(no_mas)
        if tmp_val == 0:
            sim_tmp = 1
        elif len(movies) == 0:
            sim_tmp = 0
        else:
            sim_tmp = tmp_val / len(movies)
        # print(sim_tmp)
        usersims.append(sim_tmp)

    saverage = average(usersims)
    return saverage

# returns a TET for an user
def get_tet_user(tet,user):
    for t in tet:
        username = [x for x, y in t.graph.nodes(data=True) if y.get('root')]
        if username[0] == user:
            return t


# return the rating for a movie
def get_rating(user, movieid):
    for x, y in user.graph.nodes(data=True):
        if type(x) is str and x == movieid:
            for n in user.graph.neighbors(x):
                if type(n) is str and n[:2] == 'ur': #user.graph[n].get['type'] == 'has_user_rating':
                    rat = user.graph.nodes(data=True)[n]['mult']
                    return rat
    return 0


def create_movie_rec_dict(tet_train, tet_test, metric_tree, mt_search_k, spec):
    user_est_true = defaultdict(list)
    for tet in tet_train:
        username = [x for x,y in tet.graph.nodes(data=True) if y.get('root')]
        user_leftout = get_tet_user(tet_test, username[0])
        similar_users = mt_search(metric_tree, tet, mt_search_k, spec)
        predicted_movies = get_movies(tet, similar_users)
        for mid, est in predicted_movies:
            true_r = get_rating(user_leftout, mid)
            user_est_true[username[0]].append((est, true_r))
    return user_est_true


def precision_recall_at_k(user_est_true, k, threshold=2):
    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
    return precisions, recalls


# # evaluation function, returns precision and recall for 1 user
# def __recall(predictions, user_leftout, k, threshold=4):
#     n_rel = sum((get_rating(user_leftout, mov) >= threshold) for (mov, _) in predictions)
#     n_rec_k = sum((est >= threshold) for (_, est) in predictions[:k])
#     n_rel_and_rec_k = sum(((get_rating(user_leftout,mov) >= threshold) and (est >= threshold))
#                           for (mov, est) in predictions[:k])
#     precisions = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
#     recalls = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
#     # print(f'n_rel: {n_rel}, n_rec_k: {n_rec_k}, n_rel_and_rec_k: {n_rel_and_rec_k}')
#     # print(f'PRECISION :: {precisions}')
#     # print(f' RECALL ::  {recalls}')
#     return precisions, recalls
#
#
# # evaluation call function, iterates through all users and returns average precision and recall
# def recall(tet_train, tet_test, metric_tree, mt_search_k, spec, k_movies):
#     tmp = []
#     for user in tet_train:
#         username = [x for x,y in user.graph.nodes(data=True) if y.get('root')]
#         user_leftout = get_tet_user(tet_test,username[0])
#         similar_users = mt_search(metric_tree, user, mt_search_k, spec)
#         predicted_movies = get_movies(user, similar_users)
#         tmp.append(__recall(predicted_movies, user_leftout, k_movies))
#     precision = 0.0
#     precision_count = 0
#     rec = 0.0
#     rec_count = 0
#     for x,y in tmp:
#         precision = precision + x
#         rec = rec + y
#         precision_count = precision_count + 1
#         rec_count = rec_count + 1
#     precision_res = precision / precision_count
#     recall_res = rec / rec_count
#     # print(tmp)
#     return precision_res, recall_res
