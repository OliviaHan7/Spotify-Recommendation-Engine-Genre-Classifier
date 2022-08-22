# import Flask and jsonify
from flask import Flask, jsonify, request
# import Resource, Api and reqparser
from flask_restful import Resource, Api, reqparse
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from bs4 import BeautifulSoup
import requests
import spotipy
import streamlit as st 
import os 
from spotipy.oauth2 import SpotifyClientCredentials
from PIL import Image
from lyricsgenius import Genius
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler,LabelEncoder,OneHotEncoder
from PIL import Image
from imblearn.pipeline import Pipeline as imbPipeline


st.set_page_config(layout="wide")
st.title('Spotify Song Recommendation and Genre Prediction')
st.markdown('by Olivia Han')
                        
                           
SPOTIPY_CLIENT_ID =SPOTIPY_CLIENT_ID
SPOTIPY_CLIENT_SECRET =SPOTIPY_CLIENT_SECRET 

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials (
        client_id = SPOTIPY_CLIENT_ID, 
        client_secret = SPOTIPY_CLIENT_SECRET
    )
)

def ohe_prep(df, column, new_name): 
    
    tf_df = pd.get_dummies(df[column])
    feature_names = tf_df.columns
    tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
    tf_df.reset_index(drop = True, inplace = True)    
    return tf_df

#function to build entire feature set
def create_feature_set(df):
    float_cols=['acousticness', 'danceability', 'energy',
   'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo',
   'valence', 'duration_ms','dominant_sentiment_score']

    #tfidf genre lists
    tfidf_gerne = TfidfVectorizer()
    tfidf_matrix =  tfidf_gerne.fit_transform(df['artist_genre'].apply(lambda x: "".join(x)))

    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + "|" + i for i in tfidf_gerne.get_feature_names()]
    genre_df.reset_index(drop = True, inplace=True)


    year_ohe = ohe_prep(df, 'release_year','year') * 0.5
    popularity_ohe = ohe_prep(df, 'popularity','pop') * 0.15

    #scale float columns
    floats = df[float_cols].reset_index(drop = True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * 0.2

    #concanenate all features
    final = pd.concat([genre_df, floats_scaled, popularity_ohe, year_ohe], axis = 1)
    
    #add song id
    # final['track_id']=df['track_id'].values
    
    return final

def song_feature(df_song):
    float_cols=['acousticness', 'danceability', 'energy',
       'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo',
       'valence', 'duration_ms','dominant_sentiment_score']
    feature=create_feature_set(df_song)
    song_vector=feature.sum(axis=0)
    all_cols=list(complete_feature_set.columns)
    song_col=list(feature.columns)
    df_em=pd.DataFrame(columns=all_cols)
    
    df_em[song_col]=feature
    
    song_vector=df_em.sum(axis=0)
    return song_vector

def recommend(df, song_vector, complete_features):
    # complete_features=complete_features.drop('track_id',axis=1)

    df['sim'] = cosine_similarity(complete_features.values, song_vector.values.reshape(1, -1))[:,0]
    reco = df[df['popularity']==1].sort_values('sim',ascending = False).head(20)
    recommendation=reco[['track_name','artist','sim']]
    track_name=list(recommendation['track_name'])
    artist=list(recommendation['artist'])


    return track_name,artist,df
def load_pkl(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj


genre_model=load_pkl('pipeline.pkl')
def predict_genre(df):
    df.key=df.key.astype('object')
    df['mode']=df['mode'].astype('object')

    df_test=df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness','acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo','duration_ms']]
    
    # t=pd.DataFrame(df_test.sum(axis=0)).T

    average_genre=genre_model.predict(df_test)
    return average_genre
 

#---------------------------------------End of Function------------------------------#




tracks = pickle.load(open('song_df.pkl','rb'))
complete_feature_set=pickle.load(open('complete_feature_set.pkl','rb'))

# song_vec = pickle.load(open('song_vector.pkl','rb'))

def load_pkl(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

genre_model=load_pkl('pipeline.pkl')

song = st.text_input('Track Name', "")
st.write('The current track name is', song)

artist = st.text_input('Artist', '')
st.write('The current artist name is', artist)

if st.button('Show Track'):
    df_track=tracks[(tracks['artist']==artist)&(tracks['track_name']==song)]
    # st.write(df_track)
    p = sp.search(q='artist:'+artist+ ' track:'+song, type='track')
    st.sidebar.image(p['tracks']['items'][0]['album']['images'][0]['url'])
if st.button('Recommendation'):
    st.subheader(f"Recommendations based off {song} by {artist}")
    df_track=tracks[(tracks['artist']==artist)&(tracks['track_name']==song)]

    song_vec=song_feature(df_track)

    re_track_name,re_artist,df_reco=recommend(tracks,song_vec,complete_feature_set)

    col1, col2, col3, col4 = st.columns(4)
    p=[]
    for i in range(0,20):
        try:
            q=sp.search(q='artist:'+re_artist[i]+ ' track:'+re_track_name[i], type='track')
            p.append(q['tracks']['items'][0]['album']['images'][0]['url'])
        except IndexError:
            p.append(Image.open('image_not.png'))    

        
    with col1:
        
        
        st.write(re_artist[0], "--", re_track_name[0])
        st.image(p[0])

        st.write(re_artist[1], "--", re_track_name[1])
        st.image(p[1])        

        st.write(re_artist[2], "--", re_track_name[2])
        st.image(p[2]) 

        st.write(re_artist[3], "--", re_track_name[3])
        st.image(p[3])  

        st.write(re_artist[4], "--", re_track_name[4])
        st.image(p[4])

    with col2:


        st.write(re_artist[5], "--", re_track_name[5])
        st.image(p[5])

        st.write(re_artist[6], "--", re_track_name[6])
        st.image(p[6])

        st.write(re_artist[7], "--", re_track_name[7])
        st.image(p[7])
        
        st.write(re_artist[8], "--", re_track_name[8])

        st.image(p[8])

        st.write(re_artist[9], "--", re_track_name[9])

        st.image(p[9])
    
        
        
        
    with col3:

        st.write(re_artist[10], "--", re_track_name[10])
        # p = sp.search(q='artist:'+re_artist[10]+ ' track:'+re_track_name[10], type='track')
        # i = p['tracks']['items'][0]['album']['images'][0]['url']
        st.image(p[10])

        st.write(re_artist[11], "--", re_track_name[11])
        # p = sp.search(q='artist:'+re_artist[11]+ ' track:'+re_track_name[11], type='track')
        # i = p['tracks']['items'][0]['album']['images'][0]['url']
        st.image(p[11])
        
        st.write(re_artist[12], "--", re_track_name[12])
        # p = sp.search(q='artist:'+re_artist[12]+ ' track:'+re_track_name[12], type='track')
        # i = p['tracks']['items'][0]['album']['images'][0]['url']
        st.image(p[12])

        st.write(re_artist[13], "--", re_track_name[13])
        # p = sp.search(q='artist:'+re_artist[13]+ ' track:'+re_track_name[13], type='track')
        # i = p['tracks']['items'][0]['album']['images'][0]['url']
        st.image(p[13])

        st.write(re_artist[14], "--", re_track_name[14])
        # p = sp.search(q='artist:'+re_artist[14]+ ' track:'+re_track_name[14], type='track')
        # i = p['tracks']['items'][0]['album']['images'][0]['url']
        st.image(p[14])
   
        
    with col4:

        st.write(re_artist[15], "--", re_track_name[15])
        # p = sp.search(q='artist:'+re_artist[15]+ ' track:'+re_track_name[15], type='track')
        # i = p['tracks']['items'][0]['album']['images'][0]['url']
        st.image(p[15])

        st.write(re_artist[16], "--", re_track_name[16])
        # p = sp.search(q='artist:'+re_artist[16]+ ' track:'+re_track_name[16], type='track')
        # i = p['tracks']['items'][0]['album']['images'][0]['url']
        st.image(p[16])

        st.write(re_artist[17], "--", re_track_name[17])
        # p = sp.search(q='artist:'+re_artist[17]+ ' track:'+re_track_name[17], type='track')
        # i = p['tracks']['items'][0]['album']['images'][0]['url']
        st.image(p[17])

        st.write(re_artist[18], "--", re_track_name[18])
        # p = sp.search(q='artist:'+re_artist[18]+ ' track:'+re_track_name[18], type='track')
        # i = p['tracks']['items'][0]['album']['images'][0]['url']
        st.image(p[18])

        st.write(re_artist[19], "--", re_track_name[19])
        # p = sp.search(q='artist:'+re_artist[15]+ ' track:'+re_track_name[15], type='track')
        # i = p['tracks']['items'][0]['album']['images'][0]['url']
        st.image(p[19])

if st.button('Predict Playlist Genre'):
    
    
    df_track=tracks[(tracks['artist']==artist)&(tracks['track_name']==song)]
    song_vec=song_feature(df_track)
    re_track_name,re_artist,df_reco=recommend(tracks,song_vec,complete_feature_set)
    prediction=round(predict_genre(df_reco).mean(),0)

    if prediction == 0:
        st.text('Emo')
    if prediction == 1:
        st.text('Rap')

    if prediction == 2:
        st.text("Pop")

    if prediction == 3:
        st.text("Rnb")
    if prediction == 4:
        st.text("Trap Metal")

    if prediction == 5:
        st.text("dnb")
        
    if prediction == 6:
        st.text("hardstyle")

    if prediction == 7:
        st.text("techhouse")
      
    if prediction == 8:
        st.text("techno")
    if prediction == 9:
        st.text("trance")   
    if prediction == 10:
        st.text("trap")   
    st.sidebar.markdown("11 Unique Genres:")
    st.sidebar.markdown("'Emo': 0, 'Rap': 1, 'Pop': 2, 'RnB': 3,")
    st.sidebar.markdown("'Trap Metal': 4, 'dnb': 5, 'hardstyle': 6,")
    st.sidebar.markdown("'techhouse': 7, 'techno': 8, 'trance': 9, 'trap': 10")
