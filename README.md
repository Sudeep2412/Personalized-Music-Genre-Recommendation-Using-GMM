```markdown

# Personalized Music Genre Recommendation Using GMM 🎶

This work implements **Gaussian Mixture Models (GMM)** to recommend a user’s most preferred music genre. It analyzes a **Million Song Dataset** and clusters songs into genres, suggesting similar ones based on audio features.

---

## 🚀 Features

✅ Loads song features (tempo, valence, danceability, etc.)  

✅ Trains a GMM to cluster songs into genre  

✅ Recommendations for the Spotify playlist of the user.  

✅ Lightweight code as dataset is excluded to keep repo size small.

---

## ⚙️ Technologies Used

- **Python** (core code)

- **scikit-learn** (GMM implementation)

- **joblib** (model persistence)

- **Spotify API** (for user playlists)

- **h5py** (reading `.h5` song files – *not included*)

---

## 📁 Dataset

**Dataset:** [Million Song Dataset](https://labrosa.ee.columbia.edu/millionsong/)  

⚠️ Note: The dataset is not provided in the repo due to its size (~10GB). You should download it separately and place it in the `millionsongsubset/` directory.

---

## 🛠️ Project Structure

```

.

├── model.py           

├── spotify.py           

├── .gitignore           

├── requirements.txt

```


