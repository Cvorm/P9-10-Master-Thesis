import pandas as pd
import numpy as np
import imdb
import re

moviesDB = imdb.IMDb()
# data = pd.read_csv('../Data/movies.csv')
# ratings = pd.read_csv('../Data/ratings.csv')
data = pd.read_csv('../Data/movie_new.csv', converters={'cast': eval})
movieratings = pd.read_csv('../Data/ratings25.csv', converters={'cast': eval})
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
# ratings = pd.read_csv('../Data2/ratings.dat', sep='::', names=['userId', 'movieId', 'rating','timestamp'], converters={'cast': eval})
# data = pd.read_csv('../Data2/movies.dat', sep='::', names=['movieId', 'title', 'genres'], converters={'cast': eval})
# ratings.to_csv('ratings1.csv',index=False)
# data.to_csv('movie1.csv', index=False)

# ass = pd.read_csv('../Data/BX-Books.csv', converters={'cast': eval})
# ass1 = pd.read_csv('../Data/BX-Users.csv', converters={'cast': eval})
# ass2 = pd.read_csv('../Data/BX-Book-Ratings.csv', converters={'cast': eval})


# function used for updating the movies in movielens dataset by adding data from IMDb
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


def split_data():
    df = rdata
    ranks = df.groupby('userId')['timestamp'].rank(method='first')
    counts = df['userId'].map(df.groupby('userId')['timestamp'].apply(len))
    # myes = (ranks / counts) > 0.8
    df['new_col'] = (ranks / counts) > 0.70
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
    data['genres'] = [str(m).split("|") for m in data.genres]
    # data['movieId'] = 'm' + data['movieId'].astype(str)


# function for normalizing data, returns a normalized dataframe
def __normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# function that normalizes some MovieLens features
def normalize_all_data():
    data['votes'] = __normalize_data(data['votes'])
    data['rating'] = __normalize_data(data['rating'])
    updated_actor['awards'] = __normalize_data(updated_actor['awards'])
    updated_actor['nominations'] = __normalize_data(updated_actor['nominations'])


def normalize_book_data():
    users['Age'] = __normalize_data(users['Age'])
    bookratings['BookRating'] = __normalize_data(bookratings['BookRating'])

# runs the necessary functions from data_setup for recommender.py to function
def run_data():
    format_data()
    update_data(False, False)
    x_train, x_test = split_data()
    normalize_all_data()
    #x_train, x_test = train_test_split(rdata, test_size=0.3)
    return x_train, x_test



    # train.to_csv(r'C:\Users\Darkmaster\PycharmProjects\Recommender\Data\Cvorm\training.csv', header=False, index=False)
    # test.to_csv(r'C:\Users\Darkmaster\PycharmProjects\Recommender\Data\Cvorm\testing.csv', header=False, index=False)


print(np.min(data.votes))
print(np.max(data.votes))
print(np.min(data.rating))
print(np.max(data.rating))

print(np.min(updated_actor.awards))
print(np.min(updated_actor.nominations))
print(np.max(updated_actor.awards))
print(np.max(updated_actor.nominations))