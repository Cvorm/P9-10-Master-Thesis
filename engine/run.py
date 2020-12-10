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
mt_search_k = 5
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
    run_data()
    genres = get_genres()

    print('Generating graph...')
    start_time = time.time()
    graph = generate_bipartite_graph(genres)
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Building TET specification...')
    start_time = time.time()
    speci = tet_specification(spec[0],spec[1],spec[2])
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Generating TET according to graph and specification...')
    start_time = time.time()
    tet = generate_tet(graph, 'user', speci)
    print('Adding rating and award information to graph...')
    update_tet(tet)
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Counting TETs...')
    start_time = time.time()
    [g.count_tree() for g in tet]
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Performing Logistic Evaluation on TETs...')
    start_time = time.time()
    [g.logistic_eval(log_bias, log_weight) for g in tet]
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Generating histograms...')
    start_time = time.time()
    speci_test = tet_specification2(spec2[0], spec2[1], spec2[2], genres)
    [g.histogram(speci_test) for g in tet]
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
    target_user = tet[2]
    mts_res = mt_search(tet, mts, target_user, mt_search_k, speci_test)
    username = [x for x,y in target_user.graph.nodes(data=True) if y.get('root')]
    print("--- %s seconds ---\n" % (time.time() - start_time))

    print(f'Amount of similar users found for user {username[0]}: {len(mts_res)}')
    print(f'User {username[0]}\'s histogram')
    print(target_user.ht.nodes(data=True))
    print('----------------------------')
    for res in mts_res:
        _res_id = [x for x, y in res.graph.nodes(data=True) if y.get('root')]
        print(f'USER ID: {_res_id[0]}, HISTOGRAM: {res.ht.nodes(data=True)}')
    print('|| ---------------------- ||\n')

    print('|| ------ COMPLETE ------ ||')
    print('Total run time: %s seconds.' % (time.time() - start_time_total))
    print('Amount of users: %s.' % len(tet))
    print('|| ---------------------- ||\n')
    print(f'Top {top} users:')
    [print(tet[i].graph.nodes(data=True)) for i in range(top)]
    print(f'Top {top} users histogram:')
    [print(tet[i].ht.nodes(data=True)) for i in range(top)]

    # res = 10000000.0
    # movies = ["",""]
    # temp_dist = 0.0
    # for x,y in tet[0].graph.nodes(data=True):
    #     if type(x) is str and x[0] == 'm':
    #         for i,j in tet[1].graph.nodes(data=True):
    #             # print(i)
    #             if type(i) is str and i[0] == 'm':
    #                 temp_dist = abs(y['value'] - j['value'])
    #                 print(temp_dist)
    #                 if temp_dist < res:
    #                     # print(temp_dist)
    #                     res = temp_dist
    #                     movies[0] = x
    #                     movies[1] = i
    #         # print(x)
    # print("mov1:", movies[0], "mov2:", movies[1], "dist:", res)
    #             # print(y['value'])
#run_imdb_stuff()
    myessss = get_movies(target_user, mts_res, 0.8, 1, 20)
    print(len(myessss))
    ids = [x[0] for x in myessss]
    print(ids)
    movies = get_movies_from_id(ids)
    print(len(movies))
    for title, genres in movies.items():
        print(f'title: {title} genres: {genres}')
    # print(myessss)
    # print(len(myessss))
run()
# def movie_search(hist_tree1, hist_tree2):
