import time
from engine.recommender import *

# SETTINGS
# tet specification settings: [nodes],[edges], [free variables]
spec = [["user,", "movie", "genre", "director", ],
        [("user", "movie"), ("movie", "genre"), ("movie", "director")],
        ["movie","user"]]
spec2 = [["user,", "movie", "genre", "director", ],
        [("user", "movie"), ("movie", "director")],
        ["movie","user"]]
#spec2 = [['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']]
# logistic evaluation function settings
log_bias = -5
log_weight = 1
# histogram settings
bin_size = 10
bin_amount = 10
# metric tree settings
mt_depth = 7
bucket_max_mt = 10
mt_search_k = 5
# print settings
top = 10
# seed
np.random.seed(1)

# overall run function, where we run our 'pipeline'
def run():
    print('Running...')
    start_time_total = time.time()
    print('Formatting data...')
    run_data()
    genres = get_genres()
    print(genres)
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
    print("--- %s seconds ---" % (time.time() - start_time))

    #TILFØJ RATINGS

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
    mts_res = mt_search(tet,mts,mt_search_k, speci_test)
    print(f'Amount of similar users found: {len(mts_res)}')
    print(mts_res[0].graph.nodes(data=True))
    # [print(x.graph.nodes(data=True)) for x in mts_res]
    print("--- %s seconds ---\n" % (time.time() - start_time))

    print('|| ------ COMPLETE ------ ||')
    print('Total run time: %s seconds.' % (time.time() - start_time_total))
    print('Amount of users: %s.' % len(tet))
    print('|| ---------------------- ||\n')
    print(f'Top {top} users:')
    [print(tet[i].graph.nodes(data=True)) for i in range(top)]
    print(f'Top {top} users histogram:')
    [print(tet[i].ht.nodes(data=True)) for i in range(top)]


run()