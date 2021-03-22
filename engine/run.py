import time
import sys
from experiments.baselines import *
from engine.matrix_fac import *
from engine.crossing import *

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
                        ("has_genres", "Sci-Fi"),  ("has_genres", "Thriller"), ("has_genres", "War"),  ("has_genres", "Western")]]

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
bucket_max_mt = 25
mt_search_k = 1 # int(inp[1])
k_movies = 5 # int(inp[2])


# print settings
top = 5
# seed
np.random.seed(1)


def run_baselines():
    print('Running baselines...')
    x_train, x_test = run_data() #data_test()
    # print('Running SVD...')
    # run_SVD(x_train, x_test, k_movies)
    print('Running KNN...')
    run_KNN(x_train, x_test, k_movies)
    # print('Running Normal Predictor...')
    # run_NORMPRED(x_train, x_test, k_movies)


# overall run function, where we run our 'pipeline'
def run():
    # format_data()
    f = open("output.txt", "a")
    print('Running...', file=f)
    start_time_total = time.time()

    print('Formatting data...', file=f)

    x_train, x_test = run_data()
    b_train, b_test = run_book_data()
    print("------------------------------")
    print(x_train)
    print(x_test)
    print("------------------------------")

    print('Building TET specification...')
    print('Building TET specification...', file=f)
    start_time = time.time()
    spec = tet_specification(specification_movie[0], specification_movie[1])
    spec2 = tet_specification(specification_moviessss[0], specification_moviessss[1])
    print("--- %s seconds ---" % (time.time() - start_time), file=f)


    print('Generating TETs according to specification...', file=f)
    print('Generating TETs according to specification...')
    start_time = time.time()
    tet = create_user_movie_tet(spec, x_train)
    test_tet = create_user_movie_tet(spec, x_test)
    movie_tet = create_movie_tet(spec2, x_train, "movie")
    print("--- %s seconds ---" % (time.time() - start_time), file=f)

    # print(f'Training length: {len(tet)}, Test length: {len(test_tet)}')
    # tet = cholo(tet,test_tet)
    # print(tet)
    # print(f'Length new TET {len(tet)}')
    book_tet = create_user_book_tet(book_spec, b_train)
    book_test_tet = create_user_book_tet(book_spec, b_test)
    print(f'Training length: {len(book_tet)}, Test length: {len(book_test_tet)}')
    book_tet = cholo(book_tet, book_test_tet)
    print(f'Length new TET {len(book_tet)}')

    print("--- %s seconds ---" % (time.time() - start_time), file=f)

    print('Performing Logistic Evaluation on TETs...')
    print('Performing Logistic Evaluation on TETs...', file=f)
    start_time = time.time()
    [g.logistic_eval(log_bias, log_weight) for g in tet]
    [g.logistic_eval(log_bias, log_weight) for g in test_tet]
    [g.logistic_eval(log_bias, log_weight) for g in movie_tet]
    [g.logistic_eval(log_bias, log_weight) for g in book_tet]
    [g.logistic_eval(log_bias, log_weight) for g in book_test_tet]
    print("--- %s seconds ---" % (time.time() - start_time), file=f)

    print('Generating histograms and building histogram trees...')
    print('Generating histograms...', file=f)
    start_time = time.time()

    [g.histogram(spec, 'user') for g in tet]
    [g.histogram(spec, 'user') for g in test_tet]
    [g.histogram(spec2, 'movie') for g in movie_tet]
    print("--- %s seconds ---" % (time.time() - start_time), file=f)
    [print(tet[i].ht.nodes(data=True)) for i in range(top)]
    user_item_rating_matrix(tet)
    [g.histogram(book_spec, 'user') for g in book_tet]
    [g.histogram(book_spec, 'user') for g in book_test_tet]
    print("--- %s seconds ---" % (time.time() - start_time), file=f)
    [print(tet[i].ht.nodes(data=True)) for i in range(top)]
    [print(tet[i].graph.nodes(data=True)) for i in range(top)]

    [print(book_tet[i].ht.nodes(data=True)) for i in range(top)]
    [print(book_tet[i].graph.nodes(data=True)) for i in range(top)]

    print('Building Metric Tree...')
    print('Building Metric Tree...', file=f)
    start_time = time.time()
    mts = mt_build(tet, mt_depth, bucket_max_mt, spec)
    mts_book = mt_build(book_tet, mt_depth, bucket_max_mt, book_spec)
    print("--- %s seconds ---" % (time.time() - start_time), file=f)


    print('Evaluating model...')
    movie_true = get_movie_actual_and_pred(tet, test_tet, mts, mt_search_k, spec)
    predicted, actual = format_model_third(tet, test_tet, mts, mt_search_k, spec)
    print(f'Alternative Precision {recommender_precision(predicted, actual)}')
    print(f'Alternative Recall {recommender_recall(predicted, actual)}')
    print(f'APK {yallah2(movie_true, k_movies)}')
    movie_dict, sim_score = create_movie_rec_dict(tet, test_tet, mts, mt_search_k, spec)
    precisions, recalls = precision_recall_at_k(movie_dict, k_movies)

    print('eabuelaube book bmodles')
    print(f'{len(book_tet), len(book_test_tet)}')

    book_dict, book_sim_score = create_book_rec_dict(book_tet, book_test_tet, mts_book, mt_search_k, book_spec)
    precisions, recalls = precision_recall_at_k(book_dict, k_movies)
    print(f' PRECISION COUNT: {sum(prec for prec in precisions.values()) / len(precisions)}')
    print(f' RECALL COUNT: {sum(rec for rec in recalls.values()) / len(recalls)}')

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