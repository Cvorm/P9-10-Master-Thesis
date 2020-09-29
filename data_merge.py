import pandas as pd
import imdb

moviesDB = imdb.IMDb()
print(dir(moviesDB))

data = pd.read_csv('Data/movies.csv')
print(data)

id = data['movieId'][2]
print(id)
movie = moviesDB.get_movie(id)
print(movie.data)


title = movie['title']
year = movie['year']
rating = movie['rating']
director = movie['directors']
country = movie['countries']
genres = movie['genres']
votes = movie['votes']
kind = movie['kind']
plot = movie['plot']
#casting = movie['cast']

print(title)
print(rating)
print(genres)