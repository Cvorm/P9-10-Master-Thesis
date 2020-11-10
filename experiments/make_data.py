import pandas as pd
import imdb
import csv
import psycopg2
import sqlite3
from imdb import IMDb
import re


from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT # <-- ADD THIS LINE
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import prediction_algorithms
import surprise as sur
from surprise import KNNBasic
from surprise.model_selection import train_test_split, cross_validate, KFold


 # conn = sqlite3.connect('TestDB.db')  # You can create a new database by changing the name within the quotes
 # c = conn.cursor()  # The database will be saved in the location where your 'py' file is saved

# res = pd.DataFrame(columns=['title', 'genre', 'directors', 'cast'])
# res2 = pd.DataFrame(columns=['userId', 'movieId', 'rating', ])
# # res['directors'] = res['directors'].fillna('')
# moviesDB = imdb.IMDb()


#yes


# print(res.head(5))
# print(res["genre"])
# # kg = pd.DataFrame(columns=['head', 'relation', 'tail'])
# # print(data)
#
def get_data(num_ratings, file_name):
    moviesDB = IMDb('s3', 'sqlite:///C:\\Users\\Darkmaster\\PycharmProjects\\Recommender\\TestDB.db')

    # results = DB.search_movie('the matrix')
    # for result in results:
    #     print(result.movieID, result)
    link_data = pd.read_csv('../Data/links.csv')
    movies_data = pd.read_csv('../Data/movies.csv')
    ratings = pd.read_csv('../Data/ratings.csv')
    res = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp', 'title', 'genre', 'directors', 'cast'])
    res['userId'] = ratings['userId']
    res['movieId'] = ratings['movieId']
    res['rating'] = ratings['rating']
    res['timestamp'] = ratings['timestamp']

    for x in range(num_ratings):
        # for x in range(5):
        ID = res.iloc[x]['movieId']
        res.loc[x, 'genre'] = movies_data.loc[movies_data['movieId'] == ID, 'genres'].iloc[0]
    # movie_titles = pd.DataFrame(columns=['title'])
    # movie_titles['title'] = movies_data['title']
    # print(movie_titles.head(10))
    # for x in range(len(movie_titles)):

    for x in range(num_ratings):
        # title_at = movie_titles['title'][x]
        ID = res.iloc[x]['movieId']
        movie_ID = link_data.loc[link_data['movieId'] == ID, 'imdbId'].iloc[0]
        print(movie_ID)

        # movie_title = movies_data.loc[movies_data['movieId'] == ID, 'title'].iloc[0]
        # newtitle = re.sub(r" ?\([^)]+\)", "", movie_title)


        # print(newtitle)
        try:
            # movie = moviesDB.get_movie(moviesDB.search_movie(movie_ID)[0].movieID)
            movie = moviesDB.get_movie(movie_ID)
            # movie = moviesDB.search_movie(newtitle)[0]
        except:
            continue
        # movie = moviesDB.search_movie(movie_title)
        # for result in movie:
        #     print(result.movieID, result)

        # print(movie['title'])
        # print(movie.keys())
        # print(movie['genres'])
        # print(movie['director'])

        # results = moviesDB.search_movie('Toy Story (1995)')
        # for result in results:
        #     print(result.movieID, result)
        # # res.loc[x, res.columns.get_loc('title')] = movie['title']
        # # print(movie['directors'])
        #
        try:
            res.loc[x, 'title'] = movie['title']
        except:
            res.loc[x, 'title'] = 'null'
        #res.loc[x, 'genre'] = movie['genre']


        #
        if movie.has_key('director') and len(movie['director']) > 0:
            try:
                # res.loc[x, 'directors'] = ''
                director_list = []
                for director in movie['director']:
                   director_list.append(director['name'])
                res.at[x, 'directors'] = director_list
            except:
                res.at[x, 'directors'] = 'null'
        else:
            res.at[x, 'directors'] = 'null'
        #
        if movie.has_key('cast') and len(movie['cast']) > 0:
            try:
                # res.loc[x, 'cast'] = ''
                actor_list = []
                for actor in movie['cast']:
                   actor_list.append(actor['name'])
                res.at[x, 'cast'] = actor_list
            except:
                res.at[x, 'cast'] = 'null'
        else:
            res.at[x, 'cast'] = 'null'

    res.to_csv(file_name, index=False)

get_data(500, 'testtest.csv')

# get_data(5)
# get_data(3)
# print(res.head())
# print(res['cast'])
# yes.to_csv('testtest.csv', index=False)
#

# stingerBELL = "Heat (1995)"
# newtitle = re.sub(r" ?\([^)]+\)", "", stingerBELL)
# movie = moviesDB.search_movie(newtitle)
# for result in movie:
#    print(result.movieID, result)

# get_data(2)
# print(res.head())
#
# def get_movie_data(movie_title):
#     movie = moviesDB.get_movie(moviesDB.search_movie(movie_title)[0].movieID)







        # for director in movie['directors']:
        #     res.loc[x, 'directors'] += director['name']
        # for actor in movie['cast']:
        #     res.loc[x, 'cast'] += actor['name']


            # print(director['name'])
        # res.loc[x, 'directors'] = movie['directors']
        # res.loc[x, 'cast'] = movie['cast']

# def get_data(num_movies):
#     movie_titles = pd.DataFrame(columns=['title'])
#     movie_titles['title'] = movies_data['title']
#     print(movie_titles.head(10))
#     # for x in range(len(movie_titles)):
#     for x in range(num_movies):
#         title_at = movie_titles['title'][x]
#         movie = moviesDB.get_movie(moviesDB.search_movie(title_at)[0].movieID)
#         # res.loc[x, res.columns.get_loc('title')] = movie['title']
#         # print(movie['directors'])
#
#         res.loc[x, 'title'] = movie['title']
#         res.loc[x, 'genre'] = movie['genre']
#         res.loc[x, 'directors'] = ''
#         for director in movie['directors']:
#             res.loc[x, 'directors'] += director['name']
#             # print(director['name'])
#         # res.loc[x, 'directors'] = movie['directors']
#         # res.loc[x, 'cast'] = movie['cast']


    # with open('Data/baseline_data', 'w', encoding='utf-8') as f:
    #
    # for x in range(25):
    #     with



# print(res['directors'])
# print(res['cast'])

# the_matrix = moviesDB.get_movie('0133093')
# print(sorted(the_matrix.keys()))

# print(moviesDB.get_movie_infoset())

# print('Using ALS')
# bsl_options = {'method': 'als',
#                'n_epochs': 5,
#                'reg_u': 12,
#                'reg_i': 5
#                }
# algo = sur.BaselineOnly(bsl_options=bsl_options)
# algo = KNNBasic(bsl_options=bsl_options)
#
# print('Using SGD')
# bsl_options = {'method': 'sgd',
#                'learning_rate': .00005,
#                }
# algo = sur.BaselineOnly(bsl_options=bsl_options)

# kf = KFold(n_splits=3)
#
# algo = SVD()
#
# for trainset, testset in kf.split(data_test):
#
#     algo.fit(trainset)
#     predictions = algo.test(testset)
#
#     accuracy.rmse(predictions, verbose=True)