import time
import sys
# from engine.matrix_fac import *
# from engine.recommender import *
# from engine.evaluation import *
from engine.crossing import *
from experiments.baselines import *

# tet specification settings: [[nodes],[edges]]
specification_movie = [["user", "has_rated", "has_genres", "has_imdb_rating", "has_user_rating", "has_votes", "has_director", "has_awards", "has_nominations",
                        'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                        'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western',
                        ],
                       [("user", "has_rated"), ("has_rated", "has_genres"), ("has_rated", "has_imdb_rating"), ("has_rated", "has_user_rating"),
                        ("has_rated", "has_votes"), ("has_rated", "has_director"),
                        ("has_director", "has_awards"), ("has_director", "has_nominations"),
                        ("has_genres", "Action"),  ("has_genres", "Adventure"), ("has_genres", "Animation"),  ("has_genres", "Children"),  ("has_genres", "Comedy"),
                        ("has_genres", "Crime"),  ("has_genres", "Documentary"), ("has_genres", "Drama"),  ("has_genres", "Fantasy"),  ("has_genres", "Film-Noir"),
                        ("has_genres", "Horror"),  ("has_genres", "IMAX"), ("has_genres", "Musical"),  ("has_genres", "Mystery"),  ("has_genres", "Romance"),
                        ("has_genres", "Sci-Fi"),  ("has_genres", "Thriller"), ("has_genres", "War"),  ("has_genres", "Western")]
                       ]
# "has_budget", "has_gross" ("has_rated", "has_budget"), ("has_rated", "has_gross"),
# SETTINGS
inp = sys.argv
# logistic evaluation function settings
log_bias = -5
log_weight = 2

# histogram settings
bin_size = 10
bin_amount = 10
# metric tree settings


mt_depth = 12 # int(inp[3])
bucket_max_mt = 30
mt_search_k = 30 # int(inp[1])
k_movies = 10 # int(inp[2])


# print settings
top = 5
# seed
np.random.seed(1)
def run_baselines():
    print('Running baselines...')
    x_train, x_test = run_data()
    print('Running SVD...')
    run_SVD(x_train, x_test, k_movies)
    print('Running KNN...')
    run_KNN(x_train, x_test, k_movies)
    print('Running Normal Predictor...')
    run_NORMPRED(x_train, x_test, k_movies)


# overall run function, where we run our 'pipeline'
def run():
    format_data()
    f = open("output.txt", "a")
    print('Running...', file=f)
    start_time_total = time.time()

    print('Formatting data...', file=f)
    # x_train, x_test = run_data()
    b_train, b_test = run_book_data()
    print('ass')
    print('Building TET specification...')
    print('Building TET specification...', file=f)
    start_time = time.time()
    # spec = tet_specification(specification_movie[0], specification_movie[1])
    print("--- %s seconds ---" % (time.time() - start_time), file=f)

    print('Generating TETs according to specification...', file=f)
    print('Generating TETs according to specification...')
    start_time = time.time()
    # tet = create_user_movie_tet(spec, x_train)
    # test_tet = create_user_movie_tet(spec, x_test)
    book_tet = create_user_book_tet(book_spec, b_train)
    book_test_tet = create_user_book_tet(book_spec, b_test)
    print(f'Training length: {len(book_tet)}, Test length: {len(book_test_tet)}')
    book_tet = cholo(book_tet, book_test_tet)
    print(f'Length new TET {len(book_tet)}')


    print("--- %s seconds ---" % (time.time() - start_time), file=f)

    print('Performing Logistic Evaluation on TETs...')
    print('Performing Logistic Evaluation on TETs...', file=f)
    start_time = time.time()
    # [g.logistic_eval(log_bias, log_weight) for g in tet]
    # [g.logistic_eval(log_bias, log_weight) for g in test_tet]
    [g.logistic_eval(log_bias, log_weight) for g in book_tet]
    [g.logistic_eval(log_bias, log_weight) for g in book_test_tet]
    print("--- %s seconds ---" % (time.time() - start_time), file=f)

    print('Generating histograms and building histogram trees...')
    print('Generating histograms...', file=f)
    start_time = time.time()

    # [g.histogram(spec, 'user') for g in tet]
    # [g.histogram(spec, 'user') for g in test_tet]
    [g.histogram(book_spec, 'user') for g in book_tet]
    [g.histogram(book_spec, 'user') for g in book_test_tet]
    print("--- %s seconds ---" % (time.time() - start_time), file=f)
    [print(book_tet[i].ht.nodes(data=True)) for i in range(top)]
    [print(book_tet[i].graph.nodes(data=True)) for i in range(top)]

    print('Building Metric Tree...')
    print('Building Metric Tree...', file=f)
    start_time = time.time()
    # mts = mt_build(tet, mt_depth, bucket_max_mt, spec)
    mts_book = mt_build(book_tet, mt_depth, bucket_max_mt, book_spec)
    # print(f' MT nodes: {mts.nodes}', file=f)
    # print(f' MT edges: {mts.edges}', file=f)
    # print("--- %s seconds ---" % (time.time() - start_time), file=f)


    # start_time = time.time()
    # target_user = tet[3]    # test_tet[0]
    # mts_res = mt_search(mts, target_user, mt_search_k, spec)
    # predicted_movies = get_movies(target_user, mts_res)
    # seen_movies = get_movies_in_user(target_user)
    # sim_test = get_similarity(target_user, mts_res)
    # print(sim_test)
    print('Evaluating model...')
    # movie_dict, sim_score = create_movie_rec_dict(tet, test_tet, mts, mt_search_k, spec)
    # precisions, recalls = precision_recall_at_k(movie_dict, k_movies)
    # print(f' PRECISION COUNT: {sum(prec for prec in precisions.values()) / len(precisions)}')
    # print(f' RECALL COUNT: {sum(rec for rec in recalls.values()) / len(recalls)}')

    print('eabuelaube book bmodles')
    print(f'{len(book_tet), len(book_test_tet)}')

    book_dict, book_sim_score = create_book_rec_dict(book_tet, book_test_tet, mts_book, mt_search_k, book_spec)
    precisions, recalls = precision_recall_at_k(book_dict, k_movies)
    print(f' PRECISION COUNT: {sum(prec for prec in precisions.values()) / len(precisions)}')
    print(f' RECALL COUNT: {sum(rec for rec in recalls.values()) / len(recalls)}')


    # print('SEEN',file=f)
    # [print(get_movies_from_id(m),file=f) for m in seen_movies]
    # print('PREDICTION',file=f)
    # [print(get_movies_from_id(m[0]),file=f) for m in predicted_movies[:k_movies]]
    # print("--- %s seconds ---\n" % (time.time() - start_time), file=f)
    # print(f'SETTINGS: num of sim neighbors: {mt_search_k}, num of movies: {k_movies}', file=f)
    # print(f'RESULT: {recall(tet,test_tet, mts, mt_search_k, spec, k_movies)}', file=f)

    print('|| ------ COMPLETE ------ ||', file=f)
    print('Total run time: %s seconds.' % (time.time() - start_time_total), file=f)
    # print('Amount of users: %s.' % len(tet), file=f)
    # print(f'Overall user similarity: {sim_score}')
    # print(f'Evaluation: {recall(tet, test_tet, mts, mt_search_k, spec, k_movies)}')
    print('|| ---------------------- ||\n', file=f)
    print(f'Top {top} users histogram:')
    [print(book_tet[i].ht.nodes(data=True)) for i in range(top)]
    [print(book_tet[i].graph.nodes(data=True)) for i in range(top)]
    f.close()


run()
run_baselines()