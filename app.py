import streamlit as st
import pickle
import joblib
import pandas as pd
from tmdbv3api import TMDb, Movie
import requests
from sklearn.metrics.pairwise import cosine_similarity


tmdb=TMDb()
tmdb.api_key="4dae242264c1766dd6c3a5bde7515a07"
base_url="https://image.tmdb.org/t/p/w500/"

movies_dict=pickle.load(open("movies_dictionary.pkl","rb"))
#similarity=pickle.load(open("similarity.pkl","rb"))
vectors = joblib.load("vectors.pkl")
movies=pd.DataFrame(movies_dict)

#function to fetch posters of movies
def fetch_poster(movie_id):
    try:
        movie=Movie()
        movie_details=movie.details(movie_id)
        path=movie_details.poster_path
        poster_url=base_url+path
        if movie_details and path:
            return poster_url
        else:
            return None
    except requests.exceptions.ConnectTimeout:
        print("TMDB timeout for movie id:", movie_id)
        return None
    except Exception as e:
        print("Error fetching poster:", e)
        return None

#function to recommend 5 movies along with their posters
def recommend(movie):
    movie_index=movies[movies['title']==movie].index[0]
    dist = cosine_similarity(vectors[movie_index].reshape(1, -1), vectors).flatten()
    recommend_movies=sorted(enumerate(dist.tolist()),reverse=True,key=lambda x:x[1])[1:6]
    movie_list=[]
    posters=[]
    for i in recommend_movies:
        id=movies.iloc[i[0]].movie_id
        #posters will be fetched using api
        movie_list.append(movies.iloc[i[0]].title)
        posters.append(fetch_poster(id))
    return (movie_list,posters)

def main():
    st.title("Movie Recommender")
    option=st.selectbox("Enter a Movie",movies['title'].values)
    if st.button("Find Recommendations"):
        movie,poster=recommend(option)
        st.write("Top 5 Movie Recommendations")
        col=st.columns(5)
        for i in range(len(movie)):
            col[i].write(movie[i])
            col[i].image(poster[i],width="stretch")


if __name__=='__main__':
    main()