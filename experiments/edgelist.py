import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import SVDpp
from surprise import Dataset
from surprise.model_selection import cross_validate
from node2vec import Node2Vec
from engine.recommender import *
from surprise.model_selection import PredefinedKFold
from surprise import accuracy
from collections import defaultdict
import csv

movies_data = pd.read_csv("../engine/movie.csv")
rating_data = pd.read_csv("../Data/ratings.csv")
print(movies_data)

users = rating_data['userId']

def transform_data():
    relations = ['has_genre', 'directed_by', 'rated'] #'acted_by',
    with open('../Data/Cvorm/edges.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(('head', 'relation', 'tail'))
        for movie in range(len(movies_data)):
            # movieID = movies_data['title'][movie]
            id = movies_data['movieId'][movie]
            id1 = id.strip('m')
            id2 = int(id1)
            # print(type(id2))
            # print(movies_data['genres'][movie])

    # with open('../Data/knowledge-tree.csv', 'w', encoding='utf-8') as f:
    #     writer = csv.writer(f, delimiter='\t')
    #     writer.writerow(('head', 'relation', 'tail'))
    #     for x in range(500):  # len(data)
    #         movieID = data['title'][x]
    #         id = data['movieId'][x]
    #         try:
    #             movie = moviesDB.get_movie(moviesDB.search_movie(movieID)[0].movieID)
    #         except:
    #             continue
    #         try:
    #             title = movie['title']
    #         except:
    #             title = 'null'
    #         print(f'.:{x + 1}/{len(data)}:. - {title}')
            for r in relations:
                if r == 'has_genre':
                    try:
                        genres = movies_data['genres'][movie]
                        genres = genres.strip('[]')
                        genres = genres.split(',')
                        # print(genres)
                        for g in genres:
                            writer.writerow((id, 'has_genre', g))
                        #     print(g)
                    except:
                        g = 'null'
                        writer.writerow((id, 'has_genre', g))
                if r == 'directed_by':
                    try:
                        director = movies_data['director'][movie]
                        director = director.strip('[]')
                        director = director.split(',')
                        for d in director:
                            writer.writerow((id, 'has_director', d))
                    except:
                        d = 'null'
                        writer.writerow((id, 'has_director', d))
                if r == 'acted_by':
                    try:
                        cast = movies_data['cast'][movie]
                        for a in cast[:5]:
                            writer.writerow((id, 'has_actors', a))
                    except:
                        cast = 'null'
                        writer.writerow((id, 'has_actors', cast))

                # if type(id) is not int:
                #     id1 = id.strip('m')
                #     id2 = int(id1)

                # print(id2)

                # if r == 'rated':
                #     try:
                #         rating = rating_data['rating']
                #         _movie = [x for x in rating_data if x['movieID'] == int(id)]
                #         print(_movie)
                if r == 'rated':
                    try:
                        rating = rating_data['rating']
                        # print(rating)
                        # _movie = [x for x in rating_data['movieId'] if x == id2]
                        # _movie = [x for x in rating_data if x['movieId'] == id2]
                        _movie = rating_data.loc[rating_data['movieId'] == id2]
                        # print(_movie)
                        for index, mm in _movie.iterrows():
                            # print(mm['userId'])
                            # print(mm['userId'])
                            # writer.writerow(('u' + str(int(rating['userId'][id])), 'has_rated', 'm' + str(id)))
                            writer.writerow(('u' + str(int(mm['userId'])), 'has_rated', 'm' + str(id2)))

                            # writer.writerow(('u' + str(int(mm['userId'][id2])), 'has_rated', 'm' + str(id2)))
                            # print("yes")
                    except:
                        writer.writerow(('u' + str(int(rating[id2])), 'has_rated', 'm' + str(id2)))
                        # print("ytesmysyemmysesyeydrhhfghfgt")
                #         for mm in _movie:
                #             #writer.writerow(('u' + str(int(rating['userId'][id])), 'has_rated', 'm' + str(id)))
                #             writer.writerow(('u' + mm['userId'][int(id.replace('m',''))], 'has_rated', id))
                    # except:
                    #     writer.writerow(('u' + str(int(rating_data[1])), 'has_rated', id))
                # if r == 'country':
                #     try:
                #         country = movie['countries']
                #         for c in country:
                #             writer.writerow(('m' + str(id), 'has_countries', c))
                #     except:
                #         c = 'null'
                #         writer.writerow(('m' + str(id), 'has_countries', c))
transform_data()