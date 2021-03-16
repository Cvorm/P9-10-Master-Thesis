import pandas as pd
import numpy as np
from engine.data_setup import *
from recommender import *

# print(books.shape)
# print(users.shape)
# print(ratings.shape)

books.drop(["ImageURLS", "ImageURLM", "ImageURLL"],axis=1,inplace=True)
# print(books.head(5))
#
# print(books.dtypes)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 800)

# print(books.YearOfPublication.unique())

# print(books.loc[books.YearOfPublication == 'DK Publishing Inc',:])
books.loc[books.ISBN == '078946697X', 'YearOfPublication'] = 2000
books.loc[books.ISBN == '078946697X', 'BookAuthor'] = "Michael Teitelbaum"
books.loc[books.ISBN == '078946697X', 'Publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '078946697X', 'BookTitle'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"

books.loc[books.ISBN == '0789466953', 'YearOfPublication'] = 2000
books.loc[books.ISBN == '0789466953', 'BookAuthor'] = "James Buckley"
books.loc[books.ISBN == '0789466953', 'Publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '0789466953', 'BookTitle'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"

# print(books.loc[books.YearOfPublication == 'Gallimard',:])
books.loc[books.ISBN == '2070426769', 'YearOfPublication'] = 2003
books.loc[books.ISBN == '2070426769', 'BookAuthor'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
books.loc[books.ISBN == '2070426769', 'Publisher'] = "Gallimard"
books.loc[books.ISBN == '2070426769', 'BookTitle'] = "Peuple du ciel, suivi de 'Les Bergers"

books.YearOfPublication = pd.to_numeric(books.YearOfPublication, errors='coerce')
# print(sorted(books.YearOfPublication.unique()))

books.loc[(books.YearOfPublication > 2006) | (books.YearOfPublication == 0),'YearOfPublication'] = np.nan
books.YearOfPublication.fillna(round(books.YearOfPublication.mean()), inplace=True)
books.YearOfPublication = books.YearOfPublication.astype(np.int32)

# print(sorted(books.YearOfPublication.unique()))
#
# print(books.loc[books.Publisher.isnull(),:])
books.loc[books.ISBN == '193169656X', 'Publisher'] = "other"
books.loc[books.ISBN == '1931696993', 'Publisher'] = "other"

# print(sorted(users.Age.unique()))

users.loc[(users.Age > 90) | (users.Age < 5), 'Age'] = np.nan
users.Age = users.Age.fillna(users.Age.mean())
users.Age = users.Age.astype(np.int32)

# print(sorted(users.Age.unique()))

location = users.Location.str.split(', ', n=2, expand=True)
location.columns=['city', 'state', 'country']
users['city'] = location['city']
users['state'] = location['state']
users['country'] = location['country']

users.drop(["Location", "city", "state"],axis=1,inplace=True)
# users.loc[(users.country == None)] = 'usa'
users["country"].fillna(method ='ffill', inplace = True)

# users['country'] = users['country'].astype('|S80')
# [s.strip('.') for s in users.country]
# books.country = pd.books.country.to_string(books.country, errors='coerce')
#
# print(users.dtypes)
#
print(users.country.nunique())
# print(users)
ratings_new = bookratings[bookratings.ISBN.isin(books.ISBN)]
ratings_new = ratings_new[ratings_new.UserID.isin(users.UserID)]

normalize_book_data()
print(bookratings.BookRating.unique())
# print(ratings.shape)
# print(ratings_new.shape)

tetnodes =[]
tetnodes.append('user')
tetnodes.append('location')
tetnodes.append('age')
tetnodes.append('has_rated')
# tetnodes.append('publisher')
# tetnodes.append('author')
tetnodes.append('rating')

tetedges = []
tetedges.append(('user', 'has_rated'))
tetedges.append(('user', 'location'))
tetedges.append(('user', 'age'))
# tetedges.append(('has_rated', 'publisher'))
# tetedges.append(('has_rated', 'author'))
tetedges.append(('has_rated', 'rating'))

for c in users.country.unique():
    tetnodes.append(c)
    tetedges.append(('location', c))

print(books.Publisher.nunique())
# uniquelist.append(users.country.unique())

tetspecification = []
# print(uniquelist)
tetspecification.append(tetnodes)
tetspecification.append(tetedges)
print(tetspecification)
book_spec = tet_specification(tetspecification[0], tetspecification[1])
# for c in uniquelist:

