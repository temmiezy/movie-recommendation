#########################################################################################################
# Notes

#########################################################################################################

import pandas as pd

pd.set_option('display.max_columns', None)

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import sys
import os
from pathlib import Path


def load_movies_data(m_cols=None):
    my_file = Path('files/movie_titles2.csv')
    if my_file.is_file():
        movie_titles_data2 = pd.read_csv('files/movie_titles2.csv', sep=',', names=m_cols, encoding='latin-1')
        return movie_titles_data2
    else:
        movie_titles_data = pd.read_csv('files/movie_titles.csv', sep=',', names=m_cols, encoding='latin-1')
        movie_titles_data.columns = ['movieId', 'title', 'genres']
        movie_titles_data.to_csv('files/movie_titles2.csv', index=False)  # save to new csv file
        movie_titles_data2 = pd.read_csv('files/movie_titles2.csv', sep=',', names=m_cols, encoding='latin-1')
        return movie_titles_data2


def load_ratings_data(m_cols=None):
    my_file = Path('files/ratings2.csv')
    if my_file.is_file():
        ratings_data2 = pd.read_csv('files/ratings2.csv', sep=',', names=m_cols, encoding='latin-1')
        return ratings_data2
    else:
        ratings_data = pd.read_csv('files/ratings.csv', sep=',', names=m_cols, encoding='latin-1')
        ratings_data.columns = ['movieId', 'userId', 'rating', 'timestamp']
        ratings_data.to_csv('files/ratings2.csv', index=False)  # save to new csv file
        ratings_data2 = pd.read_csv('files/ratings2.csv', sep=',', names=m_cols, encoding='latin-1')
        return ratings_data2


def get_crs_data(final_dataset):
    csr_data = csr_matrix(final_dataset.values)
    return csr_data


def get_movie_model(csr_data):
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    knn.fit(csr_data)
    return knn


def index_in_list(a_list, index):
    return index < len(a_list)


def get_movie_recommendation(movie_name, knn, movies, final_dataset, csr_data):
    n_movies_to_reccomend = 10
    movie_list = movies[movies['title'].str.contains(movie_name.strip())]
    # print(movie_list)
    # sys.exit()
    if len(movie_list):
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_reccomend + 1)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
                                   key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        recs = {}
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            # print(len(list(movies.iloc[idx]['title'])))
            # sys.exit()
            if len(list(movies.iloc[idx]['title'])) > 0:
                recs[val[0]] = list(movies.iloc[idx]['title'])[0]
                # print(list(movies.iloc[idx]['title'])[0])
        # sys.exit()
        #recommend_frame.append({'Title': movies.iloc[idx]['title'].values[0], 'Distance': val[1]})
        # df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_reccomend + 1))
        # print(recs)
        # sys.exit()
        return recs
    else:
        return "No movies found. Please check your input"


def get_recommended_movies(movie):
    # print(movie)

    movies_titles_data = load_movies_data()

    ratings_data = load_ratings_data()

    final_dataset = ratings_data.pivot(index='movieId', columns='userId', values='rating')
    final_dataset.fillna(0, inplace=True)

    no_user_voted = ratings_data.groupby('movieId')['rating'].agg('count')
    no_movies_voted = ratings_data.groupby('userId')['rating'].agg('count')

    final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index, :]
    final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]

    csr_data = get_crs_data(final_dataset)
    final_dataset.reset_index(inplace=True)

    knn = get_movie_model(csr_data)
    df = get_movie_recommendation(movie, knn, movies_titles_data, final_dataset, csr_data)

    # print(df)
    # sys.exit()
    return df
