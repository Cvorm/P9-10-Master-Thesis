import pandas as pd
import imdb
import csv
moviesDB = imdb.IMDb()
data = pd.read_csv('Data/movies.csv')


def get_data():
    with open('Data/csvfile.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(('id', 'title', 'year', 'rating', 'genres', 'director', 'country'))
        for x in range(0,25): # iteratates through each movie
            id = data['movieId'][x]
            movie = moviesDB.get_movie(id)
            # TITLE
            try: title = movie['title']
            except: title = 'null'
            # YEAR
            try: year = movie['year']
            except: year = 'null'
            # RATING
            try: rating = movie['rating']
            except: rating = 'null'
            # DIRECTORS
            try: director = movie['directors']
            except: director = 'null'
            # COUNTRIES
            try: country = movie['countries']
            except: country = 'null'
            # GENRES
            try: genres = movie['genres']
            except: genres = 'null'
            # votes = movie['votes'],kind = movie['kind'],plot = movie['plot'],aka = movie['akas'],
            # casting = movie['cast']
            print(movie)
            writer.writerow((str(id),str(title),str(year),str(rating),str(genres),str(director),str(country)))


def transform_data():
    updated_data = pd.read_csv('Data/csvfile.csv',encoding = "ISO-8859-1", engine='python')
    relations = ['has_genre', 'has_rating', 'directed_by']
    for r in relations:
        if r == 'has_genre':
            with open('Data/has_genre.csv', 'w') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(('head','relation','tail'))
                for index, row in updated_data.iterrows():
                    writer.writerow((row['id'],'has_genre',row['genres']))
        if r == 'has_rating':
            with open('Data/has_rating.csv', 'w') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(('head','relation','tail'))
                for index, row in updated_data.iterrows():
                    writer.writerow((row['id'],'has_rating',row['rating']))
        if r == 'directed_by':
            with open('Data/directed_by.csv', 'w') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(('head','relation','tail'))
                for index, row in updated_data.iterrows():
                    writer.writerow((row['id'],'directed_by',row['director']))


def combine_data():
    all_filenames = ['Data\has_genre.csv','Data\has_rating.csv','Data\directed_by.csv']
    combined_csv = pd.concat([pd.read_csv(f,encoding = "ISO-8859-1", engine='python', delimiter='\t') for f in all_filenames])
    combined_and_shuffled_csv = combined_csv.sample(frac=1)
    combined_and_shuffled_csv.to_csv("Data\combined.csv",index=False,sep='\t')


def split_data():
    ds = pd.read_csv('Data\combined.csv', encoding = "utf-8", engine='python', delimiter='\t')
    bookmark =  0 #len(ds)
    print(len(ds))
    for i in ['movie-training.txt','movie-valid.txt','movie-test.txt']:
        with open('Data/%s.csv' % i, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            for j in range(round(len(ds)/3)):
                writer.writerow(ds.iloc[bookmark+j])
        bookmark = bookmark + round(len(ds)/3)
        print(bookmark)


get_data()
transform_data()
combine_data()
split_data()