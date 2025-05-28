import os
import h5py
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from joblib import dump, load
import random
from time import sleep

# Paths
DATASET_PATH = "./MillionSongSubset/"
MODEL_PATH = "gmm_model.joblib"

# Step 1: Helper - Get all song paths
def get_all_song_files(base_path):
    song_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".h5"):
                song_files.append(os.path.join(root, file))
    return song_files

# Step 2: Extract advanced features with librosa (from raw audio if available)
def extract_features_with_librosa(audio_path):
    try:
        y, sr = librosa.load(audio_path, duration=30)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

        feature_vector = [
            tempo,
            np.mean(spectral_centroid),
            np.std(spectral_centroid),
            *np.mean(mfcc, axis=1),
            *np.std(mfcc, axis=1),
            *np.mean(chroma, axis=1),
            *np.std(chroma, axis=1)
        ]
        return np.array(feature_vector)
    except Exception as e:
        return None

# Step 3: Extract features from h5 files (tempo, loudness, etc.)
def extract_features_from_h5(h5_file):
    try:
        with h5py.File(h5_file, 'r') as f:
            tempo = f['analysis']['songs'][:]['tempo'][0]
            loudness = f['analysis']['songs'][:]['loudness'][0]
            duration = f['analysis']['songs'][:]['duration'][0]
            key = f['analysis']['songs'][:]['key'][0]
            mode = f['analysis']['songs'][:]['mode'][0]
            time_signature = f['analysis']['songs'][:]['time_signature'][0]

            return np.array([tempo, loudness, duration, key, mode, time_signature])
    except:
        return None

def get_track_metadata(h5_path):
    with h5py.File(h5_path, 'r') as f:
        artist = f['metadata']['songs'][:]['artist_name'][0].decode()
        title = f['metadata']['songs'][:]['title'][0].decode()
        return artist, title

# Step 4: Load and combine features
print("Extracting features...")
all_files = get_all_song_files(DATASET_PATH)
features = []
file_paths = []

for file in tqdm(all_files[:5000]):  # For full dataset, remove slicing
    basic_feat = extract_features_from_h5(file)
    if basic_feat is not None:
        features.append(basic_feat)
        file_paths.append(file)

features = np.array(features)

# Step 5: Scale + Reduce Dimensionality
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 6: Train GMM
if os.path.exists(MODEL_PATH):
    print("Loading existing GMM model...")
    gmm = load(MODEL_PATH)
else:
    print("Training GMM model...")
    gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=42)
    gmm.fit(features_scaled)
    dump(gmm, MODEL_PATH)
    print("Model saved.")

# Predict cluster labels
cluster_labels = gmm.predict(features_scaled)

# Optional: Load metadata (if available)
def load_genre_metadata(metadata_path):
    if os.path.exists(metadata_path):
        return pd.read_csv(metadata_path)
    else:
        return None

# Step 7: Simulate a user profile
def simulate_user_profile(cluster_labels, num_liked=10):
    indices = random.sample(range(len(cluster_labels)), num_liked)
    liked_clusters = cluster_labels[indices]
    cluster_probs = np.bincount(liked_clusters, minlength=gmm.n_components) / num_liked
    return cluster_probs

# Step 8: Recommend songs
def recommend_songs(user_profile, cluster_labels, file_paths, top_n=10):
    song_scores = []

    for i, cluster in enumerate(cluster_labels):
        prob = user_profile[cluster]
        song_scores.append((i, prob))

    song_scores.sort(key=lambda x: x[1], reverse=True)
    recommended_indices = [i for i, _ in song_scores[:top_n]]
    return [file_paths[i] for i in recommended_indices]

# ---- MAIN FLOW ----
print("\n🎧 Simulating user profile and recommending songs...\n")
user_profile = simulate_user_profile(cluster_labels, num_liked=10)
recommended_songs = recommend_songs(user_profile, cluster_labels, file_paths, top_n=10)

print("Top Recommendations:")
for song in recommended_songs:
    artist, title = get_track_metadata(song)
    print(f"🎵 {title} by {artist}")
