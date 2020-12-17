import time
from engine.recommender import *

# SETTINGS

# tet specification settings: [nodes],[edges], [free variables]
spec = [["user,", "movie", "genre", "director", ],
        [("user", "movie"), ("movie", "genre"), ("movie", "director")],
        ["movie","user"]]
spec2 = [["user,", "movie", "genre", "director", "rating", "award"],
        [("user", "movie"), ("movie", "director"), ("movie", "rating"), ("director", "award")],
        ["movie", "user", "director"]]

# logistic evaluation function settings
log_bias = -5
log_weight = 1
# histogram settings
bin_size = 10
bin_amount = 10
# metric tree settings
mt_depth = 7
bucket_max_mt = 25
mt_search_k = 3
# print settings
top = 5
# seed
np.random.seed(1)


# overall run function, where we run our 'pipeline'
def run_imdb_stuff():
    run_data()
    foo = [x for x in updated_actor["actorId"]]
    update_actor_data(foo)


def run():
    print('Running...')
    start_time_total = time.time()

    print('Formatting data...')
    x_train, x_test = run_data()
    genres = get_genres()

    print('Generating graph...')
    start_time = time.time()
    training_graph = generate_bipartite_graph(x_train)
    test_graph = generate_bipartite_graph(x_test)
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Building TET specification...')
    start_time = time.time()
    speci = tet_specification(spec[0],spec[1],spec[2])
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Generating TET according to graph and specification...')
    start_time = time.time()
    tet = generate_tet(training_graph, 'user', speci)
    test_tet = generate_tet(test_graph, 'user', speci)
    print('Adding rating and award information to graph...')
    update_tet(tet,x_train)
    update_tet(test_tet,x_test)
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Counting TETs...')
    start_time = time.time()
    [g.count_tree() for g in tet]
    [g.count_tree() for g in test_tet]
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Performing Logistic Evaluation on TETs...')
    start_time = time.time()
    [g.logistic_eval(log_bias, log_weight) for g in tet]
    [g.logistic_eval(log_bias, log_weight) for g in test_tet]
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Generating histograms...')
    start_time = time.time()
    speci_test = tet_specification2(spec2[0], spec2[1], spec2[2], genres)
    [g.histogram(speci_test) for g in tet]
    [g.histogram(speci_test) for g in test_tet]
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Building Metric Tree')
    start_time = time.time()
    mts = mt_build(tet, mt_depth, bucket_max_mt, speci_test)
    print(f' MT nodes: {mts.nodes}')
    print(f' MT edges: {mts.edges}')
    # [print(mts[i].graph.nodes(data=True)) for i in range(5)]
    print("--- %s seconds ---" % (time.time() - start_time))
    #
    print('Searching Metric Tree')
    start_time = time.time()
    # for t1 in test_tet:
    #     username = [x for x,y in t1.graph.nodes(data=True) if y.get('root')]
    #     if username[0] == 'u100':
    #         n1 = t1
    #
    # for t2 in tet:
    #     username = [x for x,y in t2.graph.nodes(data=True) if y.get('root')]
    #     if username[0] == 'u100':
    #         n2 = t2
    tmpp = []
    hitrates = []
    for t in tet:
        username = [x for x,y in t.graph.nodes(data=True) if y.get('root')]
        tmpp.append(username[0])
        t2 = get_tet_user(test_tet,username[0])
        mts_res = mt_search(tet, mts, t, mt_search_k, speci_test)
        myessss = get_movies(t, mts_res, 0.8, 1, 10)
        ta = get_movies_juujiro(t2, 20)
        #print(f'ALL MOVIES: {ta}')
        hitrates.append(hitrate(myessss,ta))
        #print(hitrate(myessss, ta))
    print(f'tmppp {tmpp}')
    print(f'hitrates: {hitrates}')
    tmp_val = 0.0
    tmp_val_len = 0
    for hitz in hitrates:
        tmp_val = tmp_val + hitz
        tmp_val_len = tmp_val_len + 1
    tmp_hit = tmp_val / tmp_val_len
    print(tmp_val)
    print(tmp_val_len)
    print(f'over all hitrate: {tmp_hit}')

    # target_user = n1    # test_tet[0]
    # mts_res = mt_search(tet, mts, target_user, mt_search_k, speci_test)
    # mts_res2 = mt_search(tet, mts, n2, mt_search_k, speci_test)
    # username = [x for x,y in target_user.graph.nodes(data=True) if y.get('root')]
    print("--- %s seconds ---\n" % (time.time() - start_time))
    #
    # print(f'Amount of similar users found for user {username[0]}: {len(mts_res)}')
    # print(f'User {username[0]}\'s histogram')
    # print(f'HISTOGRAM: {target_user.ht.nodes(data=True)}')
    # print('----------------------------')
    # for res in mts_res:
    #     _res_id = [x for x, y in res.graph.nodes(data=True) if y.get('root')]
    #     print(f'USER ID: {_res_id[0]}, HISTOGRAM: {res.ht.nodes(data=True)}')
    # print('----------------------------')
    # for res in mts_res2:
    #     _res_id = [x for x, y in res.graph.nodes(data=True) if y.get('root')]
    #     print(f'USER ID: {_res_id[0]}, HISTOGRAM: {res.ht.nodes(data=True)}')
    # print('|| ---------------------- ||\n')

    print('|| ------ COMPLETE ------ ||')
    print('Total run time: %s seconds.' % (time.time() - start_time_total))
    print('Amount of users: %s.' % len(tet))
    print('|| ---------------------- ||\n')
    #username = [x for x, y in target_user.graph.nodes(data=True) if y.get('root')]
    # print(f'Top {top} users:')
    # [print(tet[i].graph.nodes(data=True)) for i in range(top)]
    # print(f'Top {top} users:')
    # [print(test_tet[i].graph.nodes(data=True)) for i in range(top)]
    # print(f'Top {top} users histogram:')
    # [print(tet[i].ht.nodes(data=True)) for i in range(top)]

    # myessss = get_movies(target_user, mts_res, 0.8, 1, 50)
    #test_movies = get_movies(n2, mts_res2, 0.8, 1, 10)
    #print(f'myess {myessss}')
    #print(len(myessss))
    # ids = [x[0] for x in myessss]
    #ids2 = [x[0] for x in test_movies]
    #print(f'ID TRAIN {ids}')
    #print(f'ID TEST {ids2}')
    # movies = get_movies_from_id(ids)
    #movies2 = get_movies_from_id(ids2)
    #print(len(movies))
    # for title, genres in movies.items():
    #     print(f'title: {title} genres: {genres}')
    # for title, genres in movies2.items():
    #     print(f'title: {title} genres: {genres}')
    # print(myessss)
    # print(len(myessss))
    # ta = get_movies_juujiro(n2,50)
    # print(f'ALL MOVIES: {ta}')
    # print(hitrate(myessss, ta))


run()
# def movie_search(hist_tree1, hist_tree2):
