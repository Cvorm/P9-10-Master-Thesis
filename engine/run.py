import time
import sys

from engine.matrix_fac import *
from engine.recommender import *

# SETTINGS

# tet specification settings: [nodes],[edges], [free variables]
spec = [["user,", "movie", "genre", "director", ],
        [("user", "movie"), ("movie", "genre"), ("movie", "director")],
        ["movie","user"]]
spec2 = [["user,", "movie", "genre", "director", "rating", "award"],
        [("user", "movie"), ("movie", "director"), ("movie", "rating"), ("director", "award")],
        ["movie", "user", "director"]]

inp = sys.argv
# logistic evaluation function settings
log_bias = -7
log_weight = 1
# histogram settings
bin_size = 10
bin_amount = 10
# metric tree settings
mt_depth = 7
bucket_max_mt = 30
mt_search_k = 1
k_movies = 25
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

    print('Generating graph...', file=f)
    start_time = time.time()
    training_graph = generate_bipartite_graph(x_train)
    test_graph = generate_bipartite_graph(x_test)
    print("--- %s seconds ---" % (time.time() - start_time), file=f)

    print('Building TET specification...')
    print('Building TET specification...', file=f)
    start_time = time.time()
    speci = tet_specification(spec[0],spec[1],spec[2])
    print("--- %s seconds ---" % (time.time() - start_time), file=f)

    print('Generating TET according to graph and specification...', file=f)
    print('Generating TET according to graph and specification...')
    start_time = time.time()
    tet = generate_tet(training_graph, 'user', speci)
    test_tet = generate_tet(test_graph, 'user', speci)
    print('Adding rating and award information to graph...')
    print('Adding rating and award information to graph...', file=f)
    update_tet(tet,x_train)
    update_tet(test_tet,x_test)
    print("--- %s seconds ---" % (time.time() - start_time), file=f)

    print('Counting TETs...')
    print('Counting TETs...', file=f)
    start_time = time.time()
    [g.count_tree() for g in tet]
    [g.count_tree() for g in test_tet]
    print("--- %s seconds ---" % (time.time() - start_time), file=f)

    print('Performing Logistic Evaluation on TETs...')
    print('Performing Logistic Evaluation on TETs...', file=f)
    start_time = time.time()
    [g.logistic_eval(log_bias, log_weight) for g in tet]
    [g.logistic_eval(log_bias, log_weight) for g in test_tet]
    print("--- %s seconds ---" % (time.time() - start_time), file=f)

    print('Generating histograms...')
    print('Generating histograms...', file=f)
    start_time = time.time()
    speci_test = tet_specification2(spec2[0], spec2[1], spec2[2], genres)
    [g.histogram(speci_test) for g in tet]
    [g.histogram(speci_test) for g in test_tet]

    # calc_similarity(tet, speci_test)
    # item_item_sim(tet, speci_test)
    interaction_matrix(tet)

    print("--- %s seconds ---" % (time.time() - start_time), file=f)
    print("++++++++++++++++++++++++++++++++++++done++++++++++++++++++++++++++++++++++++")
    # print('Building Metric Tree')
    # print('Building Metric Tree', file=f)
    # start_time = time.time()
    # mts = mt_build(tet, mt_depth, bucket_max_mt, speci_test)
    # print(f' MT nodes: {mts.nodes}', file=f)
    # print(f' MT edges: {mts.edges}', file=f)
    # # [print(mts[i].graph.nodes(data=True)) for i in range(5)]
    # print("--- %s seconds ---" % (time.time() - start_time), file=f)
    # #
    # print('Searching Metric Tree', file=f)
    # start_time = time.time()
    # target_user = tet[3]    # test_tet[0]
    # mts_res = mt_search(mts, target_user, mt_search_k, speci_test)
    # predicted_movies, sim_test = get_movies(target_user, mts_res)
    # seen_movies = get_movies_juujiro(target_user)
    # print('SEEN',file=f)
    # [print(get_movies_from_id(m),file=f) for m in seen_movies]
    # print('PREDICTION',file=f)
    # [print(get_movies_from_id(m[0]),file=f) for m in predicted_movies[:k_movies]]
    #
    # # mts_res2 = mt_search(tet, mts, n2, mt_search_k, speci_test)
    # # username = [x for x,y in target_user.graph.nodes(data=True) if y.get('root')]
    # print("--- %s seconds ---\n" % (time.time() - start_time), file=f)
    # print(f'SETTINGS: num of sim neighbors: {mt_search_k}, num of movies: {k_movies}', file=f)
    # print(f'RESULT: {recall(tet,test_tet, mts, mt_search_k, speci_test, k_movies)}', file=f)
    # #
    # # print(f'Amount of similar users found for user {username[0]}: {len(mts_res)}')
    # # print(f'User {username[0]}\'s histogram')
    # # print(f'HISTOGRAM: {target_user.ht.nodes(data=True)}')
    # # print('----------------------------')
    # # for res in mts_res:
    # #     _res_id = [x for x, y in res.graph.nodes(data=True) if y.get('root')]
    # #     print(f'USER ID: {_res_id[0]}, HISTOGRAM: {res.ht.nodes(data=True)}')
    # # print('----------------------------')
    # # for res in mts_res2:
    # #     _res_id = [x for x, y in res.graph.nodes(data=True) if y.get('root')]
    # #     print(f'USER ID: {_res_id[0]}, HISTOGRAM: {res.ht.nodes(data=True)}')
    # # print('|| ---------------------- ||\n')
    # #get_movies_from_id()
    # print('|| ------ COMPLETE ------ ||', file=f)
    # print('Total run time: %s seconds.' % (time.time() - start_time_total), file=f)
    # print('Amount of users: %s.' % len(tet), file=f)
    # print('|| ---------------------- ||\n', file=f)
    # f.close()
    # #username = [x for x, y in target_user.graph.nodes(data=True) if y.get('root')]
    # # print(f'Top {top} users:')
    # # [print(tet[i].graph.nodes(data=True)) for i in range(top)]
    # # print(f'Top {top} users:')
    # # [print(test_tet[i].graph.nodes(data=True)) for i in range(top)]
    # print(f'Top {top} users histogram:')
    # [print(tet[i].ht.nodes(data=True)) for i in range(top)]
    # [print(tet[i].graph.nodes(data=True)) for i in range(top)]


#run_imdb_stuff()
run()
# def movie_search(hist_tree1, hist_tree2):
