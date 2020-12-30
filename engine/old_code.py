# def get_data():
#     with open('Data/csvfile.csv', 'w',encoding='utf-8') as f:
#         writer = csv.writer(f, delimiter=',')
#         writer.writerow(('id', 'title', 'year', 'rating', 'genres', 'director', 'country'))
#         for x in range(0,25): # iteratates through each movie
#             id = data['movieId'][x]
#             movie = moviesDB.get_movie(id)
#             # TITLE
#             try: title = movie['title']
#             except: title = 'null'
#             # YEAR
#             try: year = movie['year']
#             except: year = 'null'
#             # RATING
#             try: rating = movie['rating']
#             except: rating = 'null'
#             # DIRECTORS
#             try: director = movie['directors']
#             except: director = 'null'
#             # COUNTRIES
#             try: country = movie['countries']
#             except: country = 'null'
#             # GENRES
#             try: genres = movie['genres']
#             except: genres = 'null'
#             # votes = movie['votes'],kind = movie['kind'],plot = movie['plot'],aka = movie['akas'],
#             # casting = movie['cast']
#             print(f'{x+1}/25\r', end="")
#             writer.writerow((id,title,year,rating,genres,director,country))
#     print()

# def combine_data():
#     all_filenames = ['Data\has_genre.csv', 'Data\has_rating.csv', 'Data\directed_by.csv']
#     combined_csv = pd.concat(
#         [pd.read_csv(f, encoding="ISO-8859-1", engine='python', delimiter='\t') for f in all_filenames])
#     combined_and_shuffled_csv = combined_csv.sample(frac=1)
#     combined_and_shuffled_csv.to_csv("Data\combined.csv", index=False, sep='\t')
def transform_data():
    relations = ['has_genre', 'directed_by', 'rated','country'] #'acted_by',
    with open('../Data/knowledge-tree.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(('head', 'relation', 'tail'))
        for x in range(500):  # len(data)
            movieID = data['title'][x]
            id = data['movieId'][x]
            try:
                movie = moviesDB.get_movie(moviesDB.search_movie(movieID)[0].movieID)
            except:
                continue
            try:
                title = movie['title']
            except:
                title = 'null'
            print(f'.:{x + 1}/{len(data)}:. - {title}')
            for r in relations:
                if r == 'has_genre':
                    try:
                        genres = movie['genres']
                        for g in genres:
                            writer.writerow(('m' + str(id), 'has_genre', g))
                    except:
                        g = 'null'
                        writer.writerow(('m' + str(id), 'has_genre', g))
                if r == 'directed_by':
                    try:
                        director = movie['directors']
                        for d in director:
                            writer.writerow(('m' + str(id), 'has_director', d))
                    except:
                        d = 'null'
                        writer.writerow(('m' + str(id), 'has_director', d))
                if r == 'acted_by':
                    try:
                        cast = movie['cast']
                        for a in cast[:5]:
                            writer.writerow(('m' + str(id), 'has_actors', a))
                    except:
                        cast = 'null'
                        writer.writerow(('m' + str(id), 'has_actors', cast))
                if r == 'rated':
                    try:
                        rating = ratings['rating']
                        _movie = [x for x in ratings if x['movieID'] == id]
                        for mm in _movie:
                            #writer.writerow(('u' + str(int(rating['userId'][id])), 'has_rated', 'm' + str(id)))
                            writer.writerow(('u' + str(int(mm['userId'][id])), 'has_rated', 'm' + str(id)))
                    except:
                        writer.writerow(('u' + str(int(rating[id])), 'has_rated', 'm' + str(id)))
                if r == 'country':
                    try:
                        country = movie['countries']
                        for c in country:
                            writer.writerow(('m' + str(id), 'has_countries', c))
                    except:
                        c = 'null'
                        writer.writerow(('m' + str(id), 'has_countries', c))


def split_data():
    ds_init = pd.read_csv('Data\knowledge-tree.csv', delimiter='\t', encoding='utf-8')  # engine='python'
    ds = ds_init.sample(frac=1)  # shuffles the data
    bookmark = 0  # len(ds)
    for i in ['movie-train', 'movie-valid', 'movie-test']:
        with open('Data/%s.txt' % i, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            for j in range(round(len(ds) / 3) - 1):
                writer.writerow(ds.iloc[bookmark + j])
        bookmark = bookmark + round(len(ds) / 3 - 1)


def foo(n,g):
    e = g.edges(n)
    g.nodes[n]['count'] += len(e)
    for (n1,n2) in e:
        g.nodes[n2]['count'] += len(e)
        #print(n2)
        foo(n2,g)

    # def __count_helper(self,node):
    #     curr_node = self.graph.nodes[node]
    #     for n in self.graph.neighbors(curr_node):
    #
    # def count_of_counts(self):
    #     root = [x for x, y in self.graph.nodes(data=True) if y.get('root')]
    #     self.graph.nodes(data=True)[root]['count'] = len(self.graph.edges(root))
    #     for n in self.graph.neighbors(root):
    #         self.__count_helper(n)

    def __count_tree(self, pred):
        predecessors = [list(self.graph.predecessors(node)) for node in pred]
        final = list(set(self.__flat_list(predecessors)))
        if len(pred) > 0:
            for x in pred:
                print(f'pred: {x}')
                for p in final:
                    print(f'final: {final}')
                    if self.graph.nodes(data=True)[p].get('free'):
                        combined = [(self.graph.nodes(data=True)[x]['count'], self.graph.nodes(data=True)[x]['mult'])]
                        self.graph.nodes(data=True)[p]['mult'] += combined
                    else:
                        self.graph.nodes(data=True)[p]['mult'] += self.graph.nodes(data=True)[x]['count']
            self.__count_tree(final)
        else:
            for x in pred:
                print('YOOOO')

    # def count_tree(self):
    #     leaf_nodes = [node for node in self.graph.nodes if
    #                   (self.graph.in_degree(node) != 0 and self.graph.out_degree(node) == 0)]
    #     predecessors = [list(self.graph.predecessors(node)) for node in leaf_nodes]
    #     final = list(set(self.__flat_list(predecessors)))
    #     self.__count_tree(leaf_nodes)
    #     freevar = [x for x, y in self.graph.nodes(data=True) if y.get('free')]
    #     print(freevar)
    #     root = [x for x, y in self.graph.nodes(data=True) if y.get('root')]
    #     for z in freevar:
    #         ree = list(self.graph.nodes(data=True)[z].get('mult'))
    #         # if len(ree) <= 1: continue
    #         # else:
    #         #     print(ree)
    #             # ree = list(self.graph.nodes(data=True)[z]['mult'])
    #         for ind, (n, m) in enumerate(ree):
    #             for ind2, (nn, mm) in enumerate(ree):
    #                 if n == nn:
    #                     v = n + 1
    #                     ree[ind] = (nn, v)
    #                     ree[ind2] = (nn, v)
    #         self.graph.nodes(data=True)[z]['mult'] = ree
    #     # ree = list(self.graph.nodes(data=True)[root[0]]['mult'])
    #     # for ind, (n, m) in enumerate(ree):
    #     #     for ind2, (nn, mm) in enumerate(ree):
    #     #         if n == nn:
    #     #             v = n + 1
    #     #             ree[ind] = (nn, v)
    #     #             ree[ind2] = (nn, v)
    #     #self.graph.nodes(data=True)[root[0]]['mult'] = ree

    # nei = list(self.graph.neighbors(node))
    # rn = 0
    # for n in nei:
    #     print(n)
    #     m = self.graph.nodes(data=True)[n]['mult']
    #     if type(m) == int:
    #         rn += 1
    #         print('SINGLE INT')
    #         print(m)
    #     elif any(isinstance(i, tuple) for i in m):
    #         rn += 1
    #         print('LIST OF LIST')
    #         print(m)
    #     else:
    #         rn += 1
    #         print('ELSE')
    #         print(m)

    true_pos = 0
    for mid, val in predictions:
        for movieId, predictedRating in leftout:
            if movieId == mid:
                rat = get_rating(user_leftout, mid)
                print(rat)
                if rat >= 4:
                    true_pos = true_pos + 1
    false_pos = len(predictions) - true_pos
    false_neg = len(leftout) - true_pos
    print(f' TRUE POSITIVE : {true_pos} FALSE POSITIVE : {false_pos} FALSE NEGATIVE {false_neg}')

    if true_pos == 0:
        return 0, 0
    else:
        precision = true_pos / (true_pos+false_pos)
        recalll = true_pos / (true_pos+false_neg)
        return precision, recalll


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

    # for t1 in test_tet:
    #     username = [x for x,y in t1.graph.nodes(data=True) if y.get('root')]
    #     if username[0] == 'u100':
    #         n1 = t1
    #
    # for t2 in tet:
    #     username = [x for x,y in t2.graph.nodes(data=True) if y.get('root')]
    #     if username[0] == 'u100':
    #         n2 = t2
    # tmpp = []
    # hitrates = []
    # for t in tet:
    #     username = [x for x,y in t.graph.nodes(data=True) if y.get('root')]
    #     tmpp.append(username[0])
    #     t2 = get_tet_user(test_tet,username[0])
    #     mts_res = mt_search(tet, mts, t, mt_search_k, speci_test)
    #     myessss = get_movies(t, mts_res, 0.8, 1, 100)
    #     ta = get_movies_juujiro(t2, 20)
    #     #print(f'ALL MOVIES: {ta}')
    #     hitrates.append(hitrate(myessss,ta))
    #     #print(hitrate(myessss, ta))
    # print(f'tmppp {tmpp}')
    # print(f'hitrates: {hitrates}')
    # tmp_val = 0.0
    # tmp_val_len = 0
    # for hitz in hitrates:
    #     tmp_val = tmp_val + hitz
    #     tmp_val_len = tmp_val_len + 1
    # tmp_hit = tmp_val / tmp_val_len
    # print(tmp_val)
    # print(tmp_val_len)
    # print(f'over all hitrate: {tmp_hit}')