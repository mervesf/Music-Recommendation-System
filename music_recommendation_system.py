# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 12:25:42 2023

@author: user
"""
import numpy as np
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import base64
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
import string
from collections import Counter
import random


# Define your Spotify API credentials here
client_id = ''
client_secret = ''
redirect_uri = 'http://localhost:3001/callback'  #A URI you set in the Spotify Developer Dashboard
data=pd.read_csv('data.csv')

#  Start the authorization process and get tokens
scope ='user-library-read user-read-recently-played'  #Permission to access the API you want (e.g. read the user library)
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope=scope))

# An example API request: Get songs from user's library
saved_tracks = sp.current_user_saved_tracks()
if saved_tracks:
    print("Songs from the user's library:")
    for idx, item in enumerate(saved_tracks['items']):
        track = item['track']
        print(f"{idx + 1}: {track['name']} - {', '.join([artist['name'] for artist in track['artists']])}")
else:
    print("No song found in the user's library.")
    
recently_played = sp.current_user_recently_played(limit=10)
if recently_played:
    print('The last songs the user listened to:')
    for idx, item in enumerate(recently_played['items']):
        track = item['track']
        print(f"{idx + 1}: {track['name']} - {', '.join([artist['name'] for artist in track['artists']])}")
else:
    print("The user's last track was not found.")
print(type(recently_played))

user_playlist=sp.playlist('0cuVLxcXnoeq1L8sf5ZYDu')#The parameter here is the parameter of the playlist:

data_song=[]
for item in user_playlist['tracks']['items']:
    item_=item['track']
    data_song.append([item_['name'],item_['id'],item_['artists'][0]['name'],item_['popularity']])

data_song_=[]
for item in recently_played['items']:
    item_=item['track']
    data_song_.append([item_['name'],item_['id'],item_['artists'][0]['name'],item_['popularity']])




#Creating dataframes according to the characteristics of the listened songs
for i in range(len(data_song)):
    print(i)
    for j in sp.audio_features([data_song[i][1]])[0].values():
            
            print(j)
            data_song[i].append(j)
    for j in sp.audio_features([data_song[i][1]])[0].keys():
            
            print(j)
data_=pd.DataFrame(data_song,columns=('name','id','artists','popularity',
        'danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness',
        'liveness','valence','tempo','type','id','uri','track_href','analysis_url','duration_ms','time_signature'))
data_.drop(['id','type','id','uri','track_href','analysis_url','time_signature',],axis=1,inplace=True)


#Music Recommendation
numerical_column=np.arange(2,15)
text_columns=np.arange(0,2)
stopWords = set(stopwords.words('english'))
points_feature=[]
def normalizasyon(data):
    scaler=StandardScaler()
    data['loudness']=scaler.fit_transform(np.array(data['loudness']).reshape(-1,1))
    col_list=[i for i in  data.columns if i not in ['name', 'artists','loudness']]
    data_1=np.array(data.loc[:,col_list])
    data.loc[:,col_list]=(data_1-np.min(data_1,axis=0))/(np.max(data_1,axis=0)-np.min(data_1,axis=0))
    def text_columns_(text):
        text_=word_tokenize(text)
        for j in text_:
            if j.lower() in  stopWords or j.lower() in string.punctuation:
                text_.remove(j)
        return text_
    data['name']=[text_columns_(i) for i in data['name']]
    data['artists']=[text_columns_(i) for i in data['artists']]
    return data
data_=normalizasyon(data_)
var_list=[np.var(data_[data_.columns[i]]) for i in numerical_column]
min_varyans=var_list.index(min(var_list))
corr_list=[data_.iloc[:,numerical_column[min_varyans]].corr(data_.iloc[:,numerical_column[j]]) for j in range(len(numerical_column))]
corr_dict=dict(zip(data_.columns[numerical_column],corr_list))

#Segmenting the categoric column using the distribution of our column with the lowest variance
summary = sorted(data_.iloc[:,numerical_column[min_varyans]].describe())
summary_=summary[0:1]+summary[2:4]+summary[5:7]
categorical_level_=[]
categorical_level_.append([data_.loc[(data_.iloc[:,numerical_column[min_varyans]]>=summary_[j])&(data_.iloc[:,numerical_column[min_varyans]]<=summary_[j+1]),data_.columns[text_columns[0]]] for j in range(len(summary_)-1)])
categorical_level_.append([data_.loc[(data_.iloc[:,numerical_column[min_varyans]]>=summary_[j])&(data_.iloc[:,numerical_column[min_varyans]]<=summary_[j+1]),data_.columns[text_columns[1]]] for j in range(len(summary_)-1)])

# feature scoring based on the column with the lowest variance:
sorted_keys =sorted(corr_dict, key=lambda x: corr_dict[x])
sorted_keys_value=[0.1+(0.2)*i for i in range(len(sorted_keys))]
sorted_keys_=dict(zip(sorted_keys,sorted_keys_value))
def make_playlist(count,corr_list,categorical_level_,data,numerical_column,text_columns,min_varyans,songdata):
    col_=[songdata.iloc[:,j] for i in range(len(data.columns)) for j in range(len(songdata.columns)) if (data.columns[i]==songdata.columns[j])]
    song_data=pd.DataFrame(col_).T
    error_steps=[]
    residual_index=[]
    #normalization
    song_data=normalizasyon(song_data)
    while True:
        error=0
        index_=np.random.randint(songdata.index[0],songdata.index[-1]+1,count)
        residual_index.append(index_)
        temp_data=song_data.loc[index_,:]
        for j in range(len(temp_data.columns)):
            if j in numerical_column:
                corr_=temp_data.iloc[:,j].corr(temp_data.iloc[:,numerical_column[min_varyans]])
                error+=(np.abs(corr_dict[data.columns[j]]-corr_)*sorted_keys_[data.columns[j]])
            elif j==numerical_column[min_varyans]:
                vector=np.random.normal(np.mean(data_.iloc[:,j]),np.sqrt(np.var(data_.iloc[:,j])),count).reshape(-1,1)
                error+=((np.square(vector-np.array(temp_data.iloc[:,j].reshape(-1,1)))).sum()*sorted_keys_[data.columns[j]])
            else:
                point=0
                summary_new= sorted(temp_data.iloc[:,numerical_column[min_varyans]].describe())
                summary_new_=summary_new[0:1]+summary_new[2:4]+summary_new[5:7]
                categorical_level_new=[]
                categorical_level_new.append([temp_data.loc[(temp_data.iloc[:,numerical_column[min_varyans]]>=summary_new_[i])&(temp_data.iloc[:,numerical_column[min_varyans]]<=summary_new_[i+1]),temp_data.columns[text_columns[0]]] for i in range(len(summary_new_)-1)])
                categorical_level_new.append([temp_data.loc[(temp_data.iloc[:,numerical_column[min_varyans]]>=summary_new_[i])&(temp_data.iloc[:,numerical_column[min_varyans]]<=summary_new_[i+1]),temp_data.columns[text_columns[1]]] for i in range(len(summary_new_)-1)])
                for i in range(len(categorical_level_new)):
                    if i==0:
                        for k in range(len(categorical_level_[i])):
                            point+=((categorical_level_new[i][k].isin(categorical_level_[i][k]).sum())*0.2)
                    else:
                        for k in range(len(categorical_level_[i])):
                            point+=((categorical_level_new[i][k].isin(categorical_level_[i][k]).sum())*0.2)

        error_steps.append(error-point)
        residual_flatten=[residual_index[i][j] for i in range(len(residual_index)) for j in range(len(residual_index[i]))]
        all_use=Counter(residual_flatten)
        if error<=2.3:
            min_index=sorted(enumerate(error_steps),key=lambda x: x[1])[0][0]
            playlist=songdata.loc[residual_index[min_index],:]
            return playlist
            break

index_1=np.random.randint(data.index[0],data.index[-1]+1,350)
data_temp=data.loc[index_1,:].reset_index(drop=True)
playlist_=make_playlist(20,corr_list,categorical_level_,data_,numerical_column,text_columns,min_varyans,data_temp)
    




    
    
    
