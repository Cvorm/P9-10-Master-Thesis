import pandas as pd
import imdb

moviesDB = imdb.IMDb()
data = pd.read_csv('Data/movies.csv')
def set_header():
    f = open('Data/csvfile.csv', 'w') #overwrites the file, leaving only a header
    f.write('title, year, rating\n')
    f.close()
def get_data():
    set_header()
    for x in range(0,2): #len(data) iteratates through each movie
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
        text = str(movie) + ',' + str(year)
        print(movie)
        f = open('Data/csvfile.csv', 'a')
        f.write(title + ',' + str(year) + ',' + str(rating) + '\n') #appends chosen features to file
        f.close()


get_data()