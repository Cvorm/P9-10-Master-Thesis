import pandas as pd
import imdb

moviesDB = imdb.IMDb()
data = pd.read_csv('Data/movies.csv')


def get_data():
    for x in range(0,len(data)):
        id = data['movieId'][x]
        movie = moviesDB.get_movie(id)
        title = movie['title']
        year = movie['year']
        rating = movie['rating']
        director = movie['directors']
        country = movie['countries']
        genres = movie['genres']
        votes = movie['votes']
        kind = movie['kind']
        plot = movie['plot']
        aka = movie['akas']
        #casting = movie['cast']
        print(movie)
    
get_data()