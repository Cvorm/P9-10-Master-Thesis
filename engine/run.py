import time
import sys

from engine.matrix_fac import *
from engine.recommender import *

# tet specification settings: [nodes],[edges], [free variables]
specification_movie = [["user", "has_rated", "has_genres", "has_votes", "has_imdb_rating", "has_user_rating", "has_director", "has_awards", "has_nominations",
                        'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                        'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
                       [("user", "has_rated"), ("has_rated", "has_genres"), ("has_rated", "has_votes"), ("has_rated", "has_imdb_rating"), ("has_rated", "has_user_rating"),
                        ("has_rated", "has_director"),
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


specification = [["user", "rated_high", "rated_low", "genre_h", "genre_l"],
         [("user", "rated_high"), ("user", "rated_low"), ("rated_high", "genre_h"), ("rated_low", "genre_l")]]

spec4 = [["user", "rated_high", "rated_low", "genre_h", "genre_l"],
         [("user", "rated_high"), ("user", "rated_low")]]

# SETTINGS
inp = sys.argv
# logistic evaluation function settings
log_bias = -16
log_weight = 1.5
# histogram settings
bin_size = 10
bin_amount = 10
# metric tree settings

mt_depth = 12 # int(inp[3])
bucket_max_mt = 30
mt_search_k = 3 # int(inp[1])
k_movies = 25 # int(inp[2])

# print settings
top = 5
# seed
np.random.seed(1)


# overall run function, where we run our 'pipeline'
def run_imdb_stuff():
    run_data()
    #foo = [x for x in updated_actor["actorId"]]
    update_actor_data("yo")


def run():
    f = open("output.txt", "a")
    print('Running...', file=f)
    start_time_total = time.time()

    print('Formatting data...', file=f)
    x_train, x_test = run_data()
    print("------------------------------")
    print(x_train)
    print(x_test)
    print("------------------------------")
    genres = get_genres()

    # print('Generating graph...', file=f)
    # start_time = time.time()
    # training_graph = generate_bipartite_graph(x_train)
    # test_graph = generate_bipartite_graph(x_test)
    # print("--- %s seconds ---" % (time.time() - start_time), file=f)

    print('Building TET specification...')
    print('Building TET specification...', file=f)
    start_time = time.time()
    # spec = tet_specification(specification[0], specification[1])
    spec = tet_specification(specification_movie[0], specification_movie[1])
    spec2 = tet_specification(specification_moviessss[0], specification_moviessss[1])
    print("--- %s seconds ---" % (time.time() - start_time), file=f)

    print('Generating TET according to graph and specification...', file=f)
    print('Generating TET according to graph and specification...')
    start_time = time.time()
    tet = create_user_tet(spec, x_train, "user")
    # tet = create_tet(spec, x_train)
    test_tet = create_user_tet(spec, x_test, "user")
    movie_tet = create_movie_tet(spec2, x_train, "movie")
    # movies_tet = create_movie_tet(spec2, x_train, "movie")
    print("--- %s seconds ---" % (time.time() - start_time), file=f)

    print('Counting TETs...')
    print('Counting TETs...', file=f)
    start_time = time.time()
    [g.count_tree() for g in tet]
    [g.count_tree() for g in test_tet]
    [g.count_tree() for g in movie_tet]
    print("--- %s seconds ---" % (time.time() - start_time), file=f)

    print('Performing Logistic Evaluation on TETs...')
    print('Performing Logistic Evaluation on TETs...', file=f)
    start_time = time.time()
    [g.logistic_eval(log_bias, log_weight) for g in tet]
    [g.logistic_eval(log_bias, log_weight) for g in test_tet]
    [g.logistic_eval(log_bias, log_weight) for g in movie_tet]
    print("--- %s seconds ---" % (time.time() - start_time), file=f)

    [print(tet[i].graph.nodes(data=True)) for i in range(top)]


    print('Generating histograms...')
    print('Generating histograms...', file=f)
    start_time = time.time()

    # spec_hist = tet_specification2(spec4[0], spec4[1], genres)
    [g.histogram(spec) for g in tet]
    [g.histogram(spec) for g in test_tet]
    [g.histogram2(spec2) for g in movie_tet]
    print("--- %s seconds ---" % (time.time() - start_time), file=f)
    [print(tet[i].ht.nodes(data=True)) for i in range(top)]
    # user_user_sim(tet, spec)
    item_item_sim(movie_tet, spec2)
    # print('Building Metric Tree')
    # print('Building Metric Tree', file=f)
    # start_time = time.time()
    # mts = mt_build(tet, mt_depth, bucket_max_mt, spec)
    # print(f' MT nodes: {mts.nodes}', file=f)
    # print(f' MT edges: {mts.edges}', file=f)
    # print("--- %s seconds ---" % (time.time() - start_time), file=f)
    #
    #
    # print('Searching Metric Tree', file=f)
    # start_time = time.time()
    # target_user = tet[3]    # test_tet[0]
    # mts_res = mt_search(mts, target_user, mt_search_k, spec)
    # predicted_movies, sim_test = get_movies(target_user, mts_res)
    # seen_movies = get_movies_juujiro(target_user)
    #
    #
    # print('SEEN',file=f)
    # [print(get_movies_from_id(m),file=f) for m in seen_movies]
    # print('PREDICTION',file=f)
    # [print(get_movies_from_id(m[0]),file=f) for m in predicted_movies[:k_movies]]
    # print("--- %s seconds ---\n" % (time.time() - start_time), file=f)
    # print(f'SETTINGS: num of sim neighbors: {mt_search_k}, num of movies: {k_movies}', file=f)
    # print(f'RESULT: {recall(tet,test_tet, mts, mt_search_k, spec, k_movies)}', file=f)
    # print('|| ------ COMPLETE ------ ||', file=f)
    # print('Total run time: %s seconds.' % (time.time() - start_time_total), file=f)
    # print('Amount of users: %s.' % len(tet), file=f)
    # print('|| ---------------------- ||\n', file=f)
    # f.close()
    # print(f'Top {top} users histogram:')
    # [print(tet[i].ht.nodes(data=True)) for i in range(top)]
    # [print(tet[i].graph.nodes(data=True)) for i in range(top)]


#run_imdb_stuff()
run()
