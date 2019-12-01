import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def getMovieIndex(title):
    return df[df['title']==title]['index'].values[0]

def getMovieTitle(index):
    return df[df['index']==index]['title'].values[0]

##Step 1: Read CSV File
df = pd.read_csv("../../../dataset/movie_dataset.csv")
#print(df.head)

##Step 2: Select Features
features = ['keywords','cast','genres','director']
for feature in features:
    df[feature] = df[feature].fillna('')
#print(df.head)

##Step 3: Create a column in DF which combines all selected features
def combine_faeture(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']

df['combined_feature'] = df.apply(combine_faeture,axis=1)
#print(df['combined_feature'])

##Step 4: Create count matrix from this new combined column
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined_feature'])

##Step 5: Compute the Cosine Similarity based on the count_matrix
feature_similarity = cosine_similarity(count_matrix)

movie_user_likes = "Matrix"

## Step 6: Get index of this movie from its title
movie_user_index = getMovieIndex(movie_user_likes)
#print(movie_user_index)

## Step 7: store movie index and  and feature_similarity togather
similar_movies = list(enumerate(feature_similarity[movie_user_index]))
#print(similar_movies)

## Step 8: Get a list of similar movies in descending order of similarity score
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)
#print(sorted_similar_movies)

## Step 9: Print titles of first 50 movies
i =0
for movie in sorted_similar_movies:
    print(getMovieTitle(movie[0]))
    i = i + 1
    if i == 50 :
        break

