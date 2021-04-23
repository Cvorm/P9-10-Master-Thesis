import pandas as pd
import numpy as np
import imdb
import re
from collections import defaultdict
import itertools
from functools import partial
from sklearn.model_selection import train_test_split, GroupShuffleSplit

moviesDB = imdb.IMDb()
data = pd.read_csv('../Data/movie_new.csv', converters={'cast': eval}, thousands=',')
movieratings = pd.read_csv('../Data/ratings_50k.csv', converters={'cast': eval})
links = pd.read_csv('../Data/links.csv')
rdata = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
adata = pd.DataFrame(columns=['actorId','awards'])
books = pd.read_csv('../Data/BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ["ISBN", "BookTitle","BookAuthor", "YearOfPublication", "Publisher", "ImageURLS", "ImageURLM", "ImageURLL"]
users = pd.read_csv('../Data/BX-Users - Kopi.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ["UserID","Location","Age"]
bookratings = pd.read_csv('../Data/BX-Book-Ratings2.csv', sep=';', error_bad_lines=False, encoding="latin-1")
bookratings.columns = ["UserID", "ISBN", "BookRating"]

# updated_data = pd.read_csv('../Data/movie_new.csv', converters={'cast': eval})

updated_actor = pd.read_csv('../Data/actor_data_new.csv', converters={'cast': eval}) # 'awards': eval, 'nominations': eval

data['rating'] = data['rating'].fillna(0).astype(float)
data['votes'] = data['votes'].fillna(0).astype(float)
data['budget'] = data['budget'].fillna(0).astype(float)
data['gross'] = data['gross'].fillna(0).astype(float)

# function used for updating the movies in movielens dataset by adding data from IMDb


def data_test():
    format_data()
    normalize_all_data()
    x_train, x_test = train_test_split(rdata, test_size=0.2)
    return x_train, x_test


def update_movie_data():
    actor_id_l = []
    for index, movie in data.iterrows():
        print(f'{index} / {len(data)}')
        # if index == 100: break
        s = str(movie['movieId'])
        sx = s[1:]
        temp = links[links['movieId'] == int(sx)]
        try:
            imovie = moviesDB.get_movie(temp['imdbId'])
            print(imovie)
            print(imovie['rating'])
            print(imovie['votes'])

        except:
            print("fail to get movie")
            continue
        try:
            rating = imovie['rating']
        except:
            rating = 0
            print('no rating')
        data.at[index, 'rating'] = str(rating)
        try:
            director = ""
            for d in imovie['directors']:
                # print(d['name'])
                director = 'a' + str(d.personID)
            # print(data.at[index, 'director'])
        except:
            print('except')
        data.at[index, 'director'] = str(director)
        try:
            votes = imovie['votes']
        except:
            votes = 0
        data.at[index, 'votes'] = str(votes)
        try: box = imovie.get('box office')
        except: print('except')
        try:
            budget = str(box['Budget'])
            budget_match = re.search("[0-9]{1,3}(,[0-9]{3}){1,4}", budget)
            budget_clean = budget_match.group()
        except:
            budget_clean = 0
            print('no budget')
        data.at[index, 'budget'] = str(budget_clean)
        try:
            gross = str(box['Cumulative Worldwide Gross'])
            gross_match = re.search("[0-9]{1,3}(,[0-9]{3}){1,4}", gross)
            gross_clean = gross_match.group()
        except:
            gross_clean = 0
            print('no gross')
        data.at[index, 'gross'] = str(gross_clean)


        # try:
        #     temp_list = []
        #     for x,y in imovie.get('box office').items():
        #         temp_list.append([x,y])
        #     box_l = temp_list
        # except:
        #     print('fail box office')
        # data.at[index, 'box'] = str(box_l)

    data.to_csv('movie_new.csv', index=False)
    return actor_id_l


# function used for finding the actors in movielens dataset by adding data from IMDb
def update_actor_data(actor_list):
    #testd = actor_list
    testd = data['director'].astype(str)
    s = set(testd)
    ss = list(s)
    adata['actorId'] = ss
    for index, actor in adata.iterrows():
        print(f'{index} \ {len(adata)}')
        awards = []
        j = str(actor['actorId'])
        jx = j[1:]
        temp_dict = {"Winner": 0, "Nominee": 0}
        try:
            award = moviesDB.get_person_awards(jx)
            for x, y in award['data'].items():
                # print(n,m)
                for a in y:
                    res = a.get('result')
                    if res == 'Winner':
                        temp_dict["Winner"] += 1
                    if res == 'Nominee':
                        temp_dict['Nominee'] += 1
                    awards.append(res)
            print(temp_dict)
            adata.at[index, 'awards'] = temp_dict["Winner"]
            adata.at[index, 'nominations'] = temp_dict["Nominee"]
        except:
            print('fail')
            adata.at[index, 'awards'] = temp_dict["Winner"]
            adata.at[index, 'nominations'] = temp_dict["Nominee"]
    adata.to_csv('actor_data_new.csv', index=False)


# helper function for running the updating functions
def update_data(movie, actor):
    if movie:
        actor_list = update_movie_data()
        if actor:
            update_actor_data(actor_list)


# returns split dataset, training and test
def split_data():
    df = rdata
    ranks = df.groupby('userId')['timestamp'].rank(method='first')
    counts = df['userId'].map(df.groupby('userId')['timestamp'].apply(len))
    # myes = (ranks / counts) > 0.8
    df['new_col'] = (ranks / counts) > 0.70  # percentage
    # print(myes)
    # print(df.head())
    train = df.loc[df['new_col'] == False]
    test = df.loc[df['new_col'] == True]

    train = train.drop(['new_col'], axis=1)
    test = test.drop(['new_col'], axis=1)

    return train, test


# formats data
def format_data():
    rdata['userId'] = 'u' + movieratings['userId'].astype(str)
    rdata['movieId'] = 'm' + movieratings['movieId'].astype(str)
    rdata['rating'] = movieratings['rating']
    rdata['timestamp'] = movieratings['timestamp']
    bookratings['UserID'] = 'u' + bookratings['UserID'].astype(str)
    users['UserID'] = 'u' + users['UserID'].astype(str)
    # data['genres'] = [str(m).split("|") for m in data.genres] # dont uncomment this PLEASE, unless for testing purposes

    # data['movieId'] = 'm' + data['movieId'].astype(str)


# function for normalizing data, returns a normalized dataframe
def __normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# function that normalizes some MovieLens features
def normalize_all_data():
    data['votes'] = __normalize_data(data['votes'])
    data['rating'] = __normalize_data(data['rating'])
    data['gross'] = __normalize_data(data['gross'])
    data['budget'] = __normalize_data(data['budget'])
    updated_actor['awards'] = __normalize_data(updated_actor['awards'])
    updated_actor['nominations'] = __normalize_data(updated_actor['nominations'])


def normalize_book_data():
    users['Age'] = __normalize_data(users['Age'])
    bookratings['BookRating'] = __normalize_data(bookratings['BookRating'])


# runs the necessary functions from data_setup for recommender.py to function
def run_data(normalize=True):
    format_data()
    update_data(False, False)
    print(f'Coverage of data: {coverage(data)}')
    if normalize == True:
        normalize_all_data()
    x_train, x_test = split_data()
    return x_train, x_test


def cholo(train, test):
    lst = []
    for n in train:
        n_username = [x for x, y in n.graph.nodes(data=True) if y.get('root')]
        for m in test:
            m_username = [x for x, y in m.graph.nodes(data=True) if y.get('root')]
            if n_username == m_username:
                lst.append(n)
    return lst

def run_book_data():
    X = bookratings
    # print(y)
    x_train, x_test = train_test_split(X, test_size=0.2)
    # next(GroupShuffleSplit(test_size=.20, n_splits=2, random_state=7).split(bookratings, groups=bookratings['UserID']))
    # train_inds, test_inds

    return x_train, x_test

# returns the coverage of each feature in the data set
def coverage(dat):
    tmp = dict()  # pd.DataFrame(columns=dat.columns)
    for column in dat:
        tmp_count = 0
        for entry in data[column]:
            if entry == 0 or pd.isna(entry):
                tmp_count = tmp_count + 1
        tmp[column] = (len(dat[column]) - tmp_count) / len(dat[column]) if tmp_count != 0 else 1
    return tmp

def format_data_matrix():
    format_data()
    movies = np.unique(data.movieId)
    users = list(set(rdata['userId'].tolist()))
    sorted_movies = sort_items(movies)
    sorted_users = sort_users(users)
    user_item = pd.DataFrame(index=movies, columns=sorted_users).fillna(0)

    for index, row in rdata.iterrows():
        user_item[row['userId']][row['movieId']] = row['rating']
    # print(user_item)
    user_item.to_csv("user_item_ny.csv", sep='\t')
    # list(set(total_movies))

def func(element):
    # print(element)
    return int(element.split("m")[1])

def func_2(element):
    return int(element.split("u")[1])

def func_get_user(element):
        user = [x for x, y in element.graph.nodes(data=True) if y.get('root')]
        myes = user[0]
        # print(myes)
        return int(myes.split("u")[1])

def sort_items_prefix(items, prefixx):
    # print(items)
    sortlist = sorted(items, key=partial(func_get_root, prefix=prefixx))
    return sortlist

def func_get_root(element, prefix):
    # print(element)
    item = [x for x, y in element.graph.nodes(data=True) if y.get('root')]
    # if len(item) >= 1:
    myes = item[0]
    # else:
    #     myes = item
    # print(myes)
    return int(myes.split(prefix)[1])

def sort_items(items):
    sortlist = sorted(items, key=func)
    return sortlist

def sort_users(users):
    sortlist = sorted(users, key=func_2)
    return sortlist

def sort_tets(tets):
    sortlist = sorted(tets, key=func_get_user)
    return sortlist

# format_data_matrix()


