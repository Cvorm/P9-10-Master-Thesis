import pandas as pd
import imdb
import csv

moviesDB = imdb.IMDb()
data = pd.read_csv('Data/movies.csv')
ratings = pd.read_csv('Data/ratings.csv')
kg = pd.DataFrame(columns=['head', 'relation', 'tail'])
print(data)


def transform_data():
    relations = ['has_genre', 'directed_by', 'rated','country'] #'acted_by',
    with open('Data/knowledge-tree.csv', 'w', encoding='utf-8') as f:
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


transform_data()
split_data()

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
