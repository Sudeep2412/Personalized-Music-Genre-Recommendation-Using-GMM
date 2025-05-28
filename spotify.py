import spotipy
from spotipy.oauth2 import SpotifyOAuth
import numpy as np

# Set up Spotify OAuth (make sure you’ve set these env variables or hardcode)
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id='6261b7fc74824676990a458c6245997a',
    client_secret='4c1d7d203fb34783b0f0853bd1bf99a4',
    redirect_uri='http://localhost:8888/callback',
    scope='user-top-read user-read-recently-played'
))

# Step 1: Get user's top 20 tracks
def get_user_top_tracks_features(limit=20):
    top_tracks = sp.current_user_top_tracks(limit=limit, time_range='medium_term')
    track_ids = [track['id'] for track in top_tracks['items']]
    audio_features = sp.audio_features(track_ids)
    
    # Extract relevant features
    features = []
    for f in audio_features:
        features.append([
            f['tempo'],
            f['loudness'],
            f['key'],
            f['mode'],
            f['danceability'],
            f['energy'],
            f['valence'],
            f['acousticness'],
            f['instrumentalness']
        ])
    return np.array(features)

# Step 2: Create Spotify-based profile for GMM
def get_spotify_user_profile_vector(gmm_model, scaler):
    user_features = get_user_top_tracks_features()
    user_scaled = scaler.transform(user_features)
    user_clusters = gmm_model.predict(user_scaled)
    cluster_profile = np.bincount(user_clusters, minlength=gmm_model.n_components) / len(user_clusters)
    return cluster_profile
