from engine.metric_tree import *
from collections import defaultdict
from typing import List
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
def get_tet_user(tet, user):
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
    sim_score = 0.0
    for tet in tet_train:
        username = [x for x,y in tet.graph.nodes(data=True) if y.get('root')]
        user_leftout = get_tet_user(tet_test, username[0])
        similar_users = mt_search(metric_tree, tet, mt_search_k, spec)
        predicted_movies = get_movies(tet, similar_users)
        for mid, est in predicted_movies:
            true_r = get_rating(user_leftout, mid)
            user_est_true[username[0]].append((est, true_r))
        sim_score = sim_score + get_similarity(tet, similar_users)
    sim_score = sim_score / len(tet_train)
    return user_est_true, sim_score


def get_movie_actual_and_pred(tet_train, tet_test, metric_tree, mt_search_k, spec):
    user = []
    for tet in tet_train:
        act_est = defaultdict(list)
        username = [x for x,y in tet.graph.nodes(data=True) if y.get('root')]
        user_leftout = get_tet_user(tet_test, username[0])
        similar_users = mt_search(metric_tree, tet, mt_search_k, spec)
        predicted_movies = get_movies(tet, similar_users)
        actual = get_movies_in_user(user_leftout)
        act_est['actual'].append(actual)
        yo = foo(predicted_movies)
        act_est['pred'].append(yo)
        user.append(act_est)

    return user

def format_model_third(tet_train, tet_test, metric_tree, mt_search_k, spec):
    predicted = []
    actual = []
    for tet in tet_train:
        username = [x for x, y in tet.graph.nodes(data=True) if y.get('root')]
        user_leftout = get_tet_user(tet_test, username[0])
        similar_users = mt_search(metric_tree, tet, mt_search_k, spec)
        predicted_movies = get_movies(tet, similar_users)
        act = get_movies_in_user(user_leftout)
        actual.append(act)
        yo = foo(predicted_movies)
        predicted.append(yo)
    return predicted, actual

def foo(predicted):
    l = []
    for (x,y) in predicted:
        l.append(x)
    return l


def precision_recall_at_k(user_est_true, k, threshold=4):
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


def yallah(habi, k):
    r_sum = 0
    for x, y in habi.items():
        r = apk(y['actual'][0], y['prediction'][0], k)
        # print(f'precision {r}')
        r_sum += r
    return r_sum / len(habi) if len(habi) != 0 else 0

def yallah2(habi, k):
    r_sum = 0
    for x in habi:
        r = apk(x['actual'], x['prediction'], k)
        # print(f'precision {r}')
        r_sum += r
    return r_sum / len(habi) if len(habi) != 0 else 0


def apk(actual, predicted, k):
    pred = predicted
    act = actual
    if len(pred) > k:
        pred = pred[:k]
    score = 0.0
    num_hits = 0.0
    for i,p in enumerate(pred):
        if p in act and p not in pred[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
    if not act:
        print('NOT ACTUAL')
        return 0.0
    return score / min(len(act), k)


def cholo(train, test):
    lst = []
    for n in train:
        n_username = [x for x,y in n.graph.nodes(data=True) if y.get('root')]
        for m in test:
            m_username = [x for x, y in m.graph.nodes(data=True) if y.get('root')]
            if n_username == m_username:
                lst.append(n)
    return lst

def recommender_precision(predicted: List[list], actual: List[list]) -> int:
    """
    Computes the precision of each user's list of recommendations, and averages precision over all users.
    ----------
    actual : a list of lists
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        precision: int
    """
    def calc_precision(predicted, actual):
        prec = [value for value in predicted if value in actual]
        prec = np.round(float(len(prec)) / float(len(predicted)), 4)
        return prec

    precision = np.mean(list(map(calc_precision, predicted, actual)))
    return precision


def recommender_recall(predicted: List[list], actual: List[list]) -> int:
    """
    Computes the recall of each user's list of recommendations, and averages precision over all users.
    ----------
    actual : a list of lists
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        recall: int
    """
    def calc_recall(predicted, actual):
        reca = [value for value in predicted if value in actual]
        reca = np.round(float(len(reca)) / float(len(actual)), 4)
        return reca

    recall = np.mean(list(map(calc_recall, predicted, actual)))
    return recall