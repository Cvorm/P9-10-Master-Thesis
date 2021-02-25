import pandas as pd
import imdb

moviesDB = imdb.IMDb()
# data = pd.read_csv('../Data/movies.csv')
# ratings = pd.read_csv('../Data/ratings.csv')
data = pd.read_csv('movies.csv', converters={'cast': eval})
ratings = pd.read_csv('ratings1.csv', converters={'cast': eval})
links = pd.read_csv('../Data/links.csv')
rdata = pd.DataFrame(columns=['userId', 'movieId', 'rating'])

adata = pd.DataFrame(columns=['actorId','awards'])
xdata = pd.DataFrame(columns=['movieId','actors','directors','budget'])

updated_data = pd.read_csv('movie.csv', converters={'cast': eval})
updated_actor = pd.read_csv('actor_data.csv', converters={'awards': eval})

# ratings = pd.read_csv('../Data2/ratings.dat', sep='::', names=['userId', 'movieId', 'rating','timestamp'], converters={'cast': eval})
# data = pd.read_csv('../Data2/movies.dat', sep='::', names=['movieId', 'title', 'genres'], converters={'cast': eval})
# ratings.to_csv('ratings1.csv',index=False)
# data.to_csv('movie1.csv', index=False)

# function used for updating the movies in movielens dataset by adding data from IMDb
def update_movie_data():
    actor_id_l = []
    for index, movie in data.iterrows():
        print(f'{index} / {len(data)}')
        # if index == 5: break
        s = str(movie['movieId'])
        sx = s[1:]
        temp = links[links['movieId'] == int(sx)]
        try:
            imovie = moviesDB.get_movie(temp['imdbId'])
            print(imovie)
        except:
            print("fail to get movie")
            continue
        try:
            director = ""
            for d in imovie['directors']:
                # print(d['name'])
                director = 'a' + str(d.personID)
            # print(data.at[index, 'director'])
        except:
            print('except')
        data.at[index, 'director'] = str(director)

        # cast_l = []
        # try:
        #     for actor in imovie['cast']:
        #         cast_l.append('a' + str(actor.personID))
        #         actor_id_l.append(actor.personID)
        # except:
        #     print('fail cast')
        # #print(cast_l)
        # data.at[index, 'cast'] = str(cast_l)
        # box_l = []
        # try:
        #     temp_list = []
        #     for x,y in imovie.get('box office').items():
        #         temp_list.append([x,y])
        #     box_l = temp_list
        # except:
        #     print('fail box office')
        # data.at[index, 'box'] = str(box_l)

    data.to_csv('movie.csv', index=False)
    return actor_id_l


# function used for finding the actors in movielens dataset by adding data from IMDb
def update_actor_data(actor_list):
    testd = updated_data['director'].astype(str)
    s = set(testd)
    ss = list(s)
    adata['actorId'] = ss
    for index, actor in adata.iterrows():
        print(f'{index} \ {len(adata)}')
        awards = []
        j = str(actor['actorId'])
        print(j)
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
        except:
            print('fail')
            adata.at[index, 'awards'] = temp_dict["Winner"]
    print(adata)
    adata.to_csv('actor_data.csv', index=False)


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
    rdata['userId'] = 'u' + ratings['userId'].astype(str)
    rdata['movieId'] = 'm' + ratings['movieId'].astype(str)
    rdata['rating'] = ratings['rating']
    rdata['timestamp'] = ratings['timestamp']
    data['genres'] = [str(m).split("|") for m in data.genres]
    data['movieId'] = 'm' + data['movieId'].astype(str)



# runs the necesarry functions from data_setup for recommender.py to function
def run_data():
    format_data()
    update_data(False, False)
    x_train, x_test = split_data()
    #x_train, x_test = train_test_split(rdata, test_size=0.3)
    return x_train, x_test



    # train.to_csv(r'C:\Users\Darkmaster\PycharmProjects\Recommender\Data\Cvorm\training.csv', header=False, index=False)
    # test.to_csv(r'C:\Users\Darkmaster\PycharmProjects\Recommender\Data\Cvorm\testing.csv', header=False, index=False)
