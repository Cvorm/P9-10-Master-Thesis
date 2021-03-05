import time
import sys

from engine.recommender import *
from engine.evaluation import *

# tet specification settings: [[nodes],[edges]]
specification_movie = [["user", "has_rated", "has_genres", "has_imdb_rating", "has_user_rating", "has_votes", "has_director", "has_awards", "has_nominations",
                        'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                        'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
                       [("user", "has_rated"), ("has_rated", "has_genres"), ("has_rated", "has_imdb_rating"), ("has_rated", "has_user_rating"),
                        ("has_rated", "has_votes"), ("has_rated", "has_director"),
                        ("has_director", "has_awards"), ("has_director", "has_nominations"),
                        ("has_genres", "Action"),  ("has_genres", "Adventure"), ("has_genres", "Animation"),  ("has_genres", "Children"),  ("has_genres", "Comedy"),
                        ("has_genres", "Crime"),  ("has_genres", "Documentary"), ("has_genres", "Drama"),  ("has_genres", "Fantasy"),  ("has_genres", "Film-Noir"),
                        ("has_genres", "Horror"),  ("has_genres", "IMAX"), ("has_genres", "Musical"),  ("has_genres", "Mystery"),  ("has_genres", "Romance"),
                        ("has_genres", "Sci-Fi"),  ("has_genres", "Thriller"), ("has_genres", "War"),  ("has_genres", "Western")]]

# SETTINGS
inp = sys.argv
# logistic evaluation function settings
log_bias = 0
log_weight = 1
# histogram settings
bin_size = 10
bin_amount = 10
# metric tree settings

mt_depth = 12 # int(inp[3])
bucket_max_mt = 30
mt_search_k = 1 # int(inp[1])
k_movies = 10 # int(inp[2])

# print settings
top = 5
# seed
np.random.seed(1)


# overall run function, where we run our 'pipeline'
def run():
    f = open("output.txt", "a")
    print('Running...', file=f)
    start_time_total = time.time()

    print('Formatting data...', file=f)
    x_train, x_test = run_data()
    print('Building TET specification...')
    print('Building TET specification...', file=f)
    start_time = time.time()
    spec = tet_specification(specification_movie[0], specification_movie[1])
    print("--- %s seconds ---" % (time.time() - start_time), file=f)

    print('Generating TETs according to specification...', file=f)
    print('Generating TETs according to specification...')
    start_time = time.time()
    tet = create_user_movie_tet(spec, x_train)
    test_tet = create_user_movie_tet(spec, x_test)
    print("--- %s seconds ---" % (time.time() - start_time), file=f)

    # print('Counting TETs...')
    # print('Counting TETs...', file=f)
    # start_time = time.time()
    # [g.count_tree() for g in tet]
    # [g.count_tree() for g in test_tet]
    # print("--- %s seconds ---" % (time.time() - start_time), file=f)

    print('Performing Logistic Evaluation on TETs...')
    print('Performing Logistic Evaluation on TETs...', file=f)
    start_time = time.time()
    [g.logistic_eval(log_bias, log_weight) for g in tet]
    [g.logistic_eval(log_bias, log_weight) for g in test_tet]
    print("--- %s seconds ---" % (time.time() - start_time), file=f)

    # [print(tet[i].graph.nodes(data=True)) for i in range(top)]

    print('Generating histograms and building histogram trees...')
    print('Generating histograms...', file=f)
    start_time = time.time()

    [g.histogram(spec, 'user') for g in tet]
    [g.histogram(spec, 'user') for g in test_tet]
    print("--- %s seconds ---" % (time.time() - start_time), file=f)
    # [print(tet[i].ht.nodes(data=True)) for i in range(top)]

    print('Building Metric Tree...')
    print('Building Metric Tree...', file=f)
    start_time = time.time()
    mts = mt_build(tet, mt_depth, bucket_max_mt, spec)
    print(f' MT nodes: {mts.nodes}', file=f)
    print(f' MT edges: {mts.edges}', file=f)
    print("--- %s seconds ---" % (time.time() - start_time), file=f)

    print('Searching Metric Tree', file=f)
    # start_time = time.time()
    # target_user = tet[3]    # test_tet[0]
    # mts_res = mt_search(mts, target_user, mt_search_k, spec)
    # predicted_movies = get_movies(target_user, mts_res)
    # seen_movies = get_movies_in_user(target_user)
    # sim_test = get_similarity(target_user, mts_res)
    # print(sim_test)

    movie_dict = create_movie_rec_dict(tet, test_tet, mts, mt_search_k, spec)
    precisions, recalls = precision_recall_at_k(movie_dict, k_movies)
    print(sum(prec for prec in precisions.values()) / len(precisions))
    print(sum(rec for rec in recalls.values()) / len(recalls))


    # print('SEEN',file=f)
    # [print(get_movies_from_id(m),file=f) for m in seen_movies]
    # print('PREDICTION',file=f)
    # [print(get_movies_from_id(m[0]),file=f) for m in predicted_movies[:k_movies]]
    # print("--- %s seconds ---\n" % (time.time() - start_time), file=f)
    # print(f'SETTINGS: num of sim neighbors: {mt_search_k}, num of movies: {k_movies}', file=f)
    # print(f'RESULT: {recall(tet,test_tet, mts, mt_search_k, spec, k_movies)}', file=f)

    print('|| ------ COMPLETE ------ ||', file=f)
    print('Total run time: %s seconds.' % (time.time() - start_time_total), file=f)
    print('Amount of users: %s.' % len(tet), file=f)
    # print(f'Evaluation: {recall(tet, test_tet, mts, mt_search_k, spec, k_movies)}')
    print('|| ---------------------- ||\n', file=f)
    print(f'Top {top} users histogram:')
    [print(tet[i].ht.nodes(data=True)) for i in range(top)]
    [print(tet[i].graph.nodes(data=True)) for i in range(top)]
    f.close()


run()
