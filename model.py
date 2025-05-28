import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import argparse
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import warnings
from typing import Optional, Tuple, List, Dict, Union
import pickle
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class MusicRecommender:
    """
    A music recommendation system using Gaussian Mixture Models to cluster similar songs
    and provide personalized recommendations based on user preferences.
    
    Attributes:
        dataset_dir (str): Path to the directory containing the dataset.
        data (pd.DataFrame): DataFrame containing song data.
        gmm_model (GaussianMixture): Trained GMM model.
        scaler (StandardScaler): Feature scaler.
        feature_columns (list): List of feature columns used for modeling.
    """
    
    def __init__(self, dataset_dir: Optional[str] = None):
        """
        Initialize the music recommender.
        
        Args:
            dataset_dir: Path to the directory containing the dataset.
        """
        self.dataset_dir = dataset_dir
        self.data = None
        self.gmm_model = None
        self.scaler = None
        self.feature_columns = ['tempo', 'loudness', 'duration', 'artist_encoded', 'title_encoded']
        self._default_feature_columns = self.feature_columns.copy()
        
    def find_h5_files(self, root_dir: str) -> List[str]:
        """
        Find all .h5 files in the dataset directory.
        
        Args:
            root_dir: Path to the root directory to search.
            
        Returns:
            List of paths to .h5 files.
        """
        logger.info(f"Searching for h5 files in {root_dir}...")
        h5_files = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".h5"):
                    h5_files.append(os.path.join(subdir, file))
        logger.info(f"Found {len(h5_files)} h5 files.")
        return h5_files

    def load_data(self, dataset_dir=None):
        if dataset_dir:
            self.dataset_dir = dataset_dir
        
        if not self.dataset_dir:
            raise ValueError("Dataset directory not specified.")
            
        h5_files = self.find_h5_files(self.dataset_dir)
        if not h5_files:
            raise FileNotFoundError(f"No h5 files found in {self.dataset_dir}")
            
        all_data = []
        skipped_files = 0
        
        logger.info("Loading song data from h5 files...")
        for file_path in tqdm(h5_files):
            try:
                with h5py.File(file_path, 'r') as f:
                    # Extract metadata from MSD subset structure
                    metadata = f['metadata']['songs'][0]
                    analysis = f['analysis']['songs'][0]
                    
                    song_data = {
                        'artist': metadata['artist_name'].decode('utf-8'),
                        'title': metadata['title'].decode('utf-8'),
                        'tempo': analysis['tempo'],
                        'loudness': analysis['loudness'],
                        'duration': analysis['duration'],
                        'year': metadata['year'],
                        'song_id': metadata['song_id'].decode('utf-8')  # Useful for debugging
                    }
                    
                    # Add additional features if available
                    if 'energy' in analysis.dtype.names:
                        song_data['energy'] = analysis['energy']
                    if 'danceability' in analysis.dtype.names:
                        song_data['danceability'] = analysis['danceability']
                    
                    all_data.append(song_data)
            except Exception as e:
                skipped_files += 1
                logger.debug(f"Error in {file_path}: {str(e)}")
                continue
        
        if skipped_files > 0:
            logger.warning(f"Skipped {skipped_files} files due to errors.")
        
        if not all_data:
            raise ValueError("No valid data extracted. Check file structure.")
        
        df = pd.DataFrame(all_data)
        df = self._clean_data(df)
        self.data = df
        logger.info(f"Loaded {len(df)} songs.")
        return df

    def _safe_decode(self, h5_file: h5py.File, path: str) -> str:
        """Safely decode a string from an h5 file."""
        try:
            data = h5_file[path]
            if len(data) > 0:
                return data[0].decode('utf-8').strip()
            return "Unknown"
        except (KeyError, AttributeError):
            return "Unknown"

    def _safe_get(self, h5_file: h5py.File, path: str) -> Optional[float]:
        """Safely get a numeric value from an h5 file."""
        try:
            parts = path.split('/')
            group = h5_file
            for part in parts[:-1]:
                group = group[part]
            return group[parts[-1]][0]
        except (KeyError, IndexError):
            return None

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the raw data."""
        # Drop rows with missing essential values
        df = df.dropna(subset=['title', 'artist']).copy()
        
        # Convert titles to lowercase for easier matching
        df['title'] = df['title'].str.strip().str.lower()
        df['artist'] = df['artist'].str.strip()
        
        # Fill missing numerical values with median
        for col in ['tempo', 'loudness', 'duration']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        return df

    def preprocess_features(self) -> Tuple[np.ndarray, StandardScaler]:
        """
        Encode categorical features and standardize numerical data.
        
        Returns:
            Tuple containing:
                - processed features array
                - scaler object
                
        Raises:
            ValueError: If data is not loaded.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        logger.info("Preprocessing features...")
        
        # Encode categorical features
        le_artist = LabelEncoder()
        le_title = LabelEncoder()
        self.data['artist_encoded'] = le_artist.fit_transform(self.data['artist'])
        self.data['title_encoded'] = le_title.fit_transform(self.data['title'])
        
        # Determine which features to use
        self.feature_columns = self._determine_feature_columns()
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.data[self.feature_columns])
        
        logger.info(f"Preprocessed {len(self.feature_columns)} features.")
        return X_scaled, self.scaler

    def _determine_feature_columns(self) -> List[str]:
        """Determine which feature columns to use based on data availability."""
        # Start with default features that must be present
        features = [col for col in self._default_feature_columns 
                   if col in self.data.columns]
        
        # Add additional features if available and have sufficient data
        additional_features = ['year', 'energy', 'danceability']
        for feat in additional_features:
            if (feat in self.data.columns and 
                self.data[feat].notna().sum() > len(self.data) * 0.8):  # At least 80% coverage
                features.append(feat)
        
        return features

    def train_gmm_model(self, 
                       X: np.ndarray, 
                       n_components_range: range = range(2, 11), 
                       max_iter: int = 100) -> GaussianMixture:
        """
        Train GMM model and select the best number of components.
        
        Args:
            X: Preprocessed feature matrix.
            n_components_range: Range of cluster numbers to try.
            max_iter: Maximum number of iterations for GMM training.
            
        Returns:
            Trained GMM model.
        """
        logger.info("Training GMM models to find optimal number of clusters...")
        best_model, best_score = None, -1
        results = []
        
        for n in n_components_range:
            logger.info(f"Training GMM with {n} components...")
            
            gmm = GaussianMixture(
                n_components=n,
                random_state=42,
                max_iter=max_iter,
                n_init=3,
                verbose=0
            )
            gmm.fit(X)
            
            # Evaluate model
            if n > 1:
                labels = gmm.predict(X)
                score = silhouette_score(X, labels)
                results.append((n, score))
                logger.info(f"Silhouette Score for {n} components: {score:.4f}")
                
                if score > best_score:
                    best_model, best_score = gmm, score
        
        # Fallback if no model was selected
        if best_model is None and n_components_range:
            n = max(n_components_range)
            best_model = GaussianMixture(n_components=n, random_state=42, max_iter=max_iter)
            best_model.fit(X)
        
        # Visualize results
        self._plot_silhouette_scores(results)
        
        self.gmm_model = best_model
        best_n = best_model.n_components if best_model else "unknown"
        logger.info(f"Selected optimal number of clusters: {best_n}")
        return best_model

    def _plot_silhouette_scores(self, results: List[Tuple[int, float]]) -> None:
        """Plot silhouette scores for different numbers of clusters."""
        if not results:
            return
            
        components, scores = zip(*results)
        plt.figure(figsize=(10, 6))
        plt.plot(components, scores, 'o-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Scores for Different Numbers of Clusters')
        plt.grid(True)
        
        plot_path = 'silhouette_scores.png'
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved silhouette score plot to {plot_path}")

    def get_song_recommendations(self, 
                                song_title: str, 
                                artist: Optional[str] = None, 
                                n_recommendations: int = 5) -> pd.DataFrame:
        """
        Get song recommendations based on a reference song.
        
        Args:
            song_title: Title of the reference song.
            artist: Artist name to disambiguate songs with the same title.
            n_recommendations: Number of recommendations to return.
            
        Returns:
            DataFrame of recommended songs.
            
        Raises:
            ValueError: If model is not trained.
        """
        if self.data is None or self.gmm_model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Normalize input
        song_title = song_title.lower().strip()
        
        # Find the reference song
        song_row = self._find_song(song_title, artist)
        if song_row.empty:
            return pd.DataFrame()
            
        # Get recommendations
        recommendations = self._generate_recommendations(song_row, song_title, artist, n_recommendations)
        
        logger.info(f"Found {len(recommendations)} recommendations for '{song_title}'.")
        return recommendations[['title', 'artist', 'tempo', 'loudness', 'duration']]

    def _find_song(self, song_title: str, artist: Optional[str]) -> pd.DataFrame:
        """Find a song in the dataset."""
        if artist:
            artist = artist.strip()
            mask = ((self.data['title'] == song_title) & 
                    (self.data['artist'].str.lower() == artist.lower()))
            song_row = self.data[mask]
        else:
            song_row = self.data[self.data['title'] == song_title]
        
        if song_row.empty:
            self._suggest_similar_songs(song_title)
            return pd.DataFrame()
            
        if len(song_row) > 1 and artist is None:
            self._handle_ambiguous_song(song_row, song_title)
            return pd.DataFrame()
            
        return song_row

    def _suggest_similar_songs(self, song_title: str) -> None:
        """Suggest similar songs when the exact match isn't found."""
        similar_titles = self.data[self.data['title'].str.contains(song_title, case=False)]
        if not similar_titles.empty:
            logger.info("Did you mean one of these?")
            for i, (_, row) in enumerate(similar_titles.head(5).iterrows()):
                logger.info(f"  {i+1}. '{row['title']}' by {row['artist']}")

    def _handle_ambiguous_song(self, song_row: pd.DataFrame, song_title: str) -> None:
        """Handle cases where multiple songs have the same title."""
        logger.info(f"Multiple songs with title '{song_title}' found. Please specify the artist:")
        for i, (_, row) in enumerate(song_row.head(5).iterrows()):
            logger.info(f"  {i+1}. '{row['title']}' by {row['artist']}")

    def _generate_recommendations(self, 
                                song_row: pd.DataFrame, 
                                song_title: str, 
                                artist: Optional[str], 
                                n_recommendations: int) -> pd.DataFrame:
        """Generate recommendations for a given song."""
        # Get features for the reference song
        song_features = song_row[self.feature_columns].iloc[0:1]
        song_scaled = self.scaler.transform(song_features)
        
        # Find the cluster of the reference song
        song_cluster = self.gmm_model.predict(song_scaled)[0]
        
        # Get all songs in the same cluster
        all_features = self.scaler.transform(self.data[self.feature_columns])
        all_clusters = self.gmm_model.predict(all_features)
        cluster_songs = self.data[all_clusters == song_cluster]
        
        # Exclude the reference song from recommendations
        if artist:
            mask = ~((cluster_songs['title'] == song_title) & 
                    (cluster_songs['artist'].str.lower() == artist.lower()))
        else:
            mask = (cluster_songs['title'] != song_title)
        
        recommendations = cluster_songs[mask]
        
        # Sample recommendations if there are too many
        if len(recommendations) > n_recommendations:
            recommendations = recommendations.sample(n=n_recommendations, random_state=42)
        
        return recommendations

    def visualize_clusters(self, 
                         X: Optional[np.ndarray] = None, 
                         labels: Optional[np.ndarray] = None, 
                         save_path: Optional[str] = None) -> None:
        """
        Visualize GMM clustering results using t-SNE.
        
        Args:
            X: Feature matrix. If None, uses the loaded data.
            labels: Cluster labels. If None, predicts using the trained model.
            save_path: Path to save the visualization image.
            
        Raises:
            ValueError: If data or model is not available when needed.
        """
        if X is None:
            if self.data is None or self.scaler is None:
                raise ValueError("Data not loaded or preprocessed.")
            X = self.scaler.transform(self.data[self.feature_columns])
        
        if labels is None:
            if self.gmm_model is None:
                raise ValueError("Model not trained.")
            labels = self.gmm_model.predict(X)
        
        logger.info("Generating t-SNE visualization of clusters (this may take a while)...")
        tsne = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=min(30, len(X)-1),  # Reduced perplexity for better performance
            n_iter=500
        )
        X_tsne = tsne.fit_transform(X)
        
        plt.figure(figsize=(12, 8))
        scatter = sns.scatterplot(
            x=X_tsne[:, 0], 
            y=X_tsne[:, 1], 
            hue=labels, 
            palette='viridis', 
            alpha=0.7,
            s=50  # Smaller point size for better visibility with many points
        )
        plt.title('GMM Music Clustering Visualization')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        
        # Adjust legend based on number of clusters
        self._adjust_cluster_legend(scatter, labels)
        
        output_path = save_path or 'cluster_visualization.png'
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Visualization saved to {output_path}")

    def _adjust_cluster_legend(self, scatter, labels: np.ndarray) -> None:
        """Adjust the cluster legend for better readability."""
        handles, legend_labels = scatter.get_legend_handles_labels()
        num_clusters = len(set(labels))
        
        if num_clusters > 15:
            # For large numbers of clusters, just show the first 10 and indicate there are more
            plt.legend(
                handles[:15], 
                [f"Cluster {i}" for i in range(15)] + ["..."],
                title=f"Top 15 of {num_clusters} Clusters",
                bbox_to_anchor=(1.05, 1),
                loc='upper left'
            )
        else:
            plt.legend(
                title="Clusters",
                bbox_to_anchor=(1.05, 1),
                loc='upper left'
            )

    def get_dataset_stats(self) -> Dict[str, Union[int, Tuple[float, float]]]:
        """
        Get and display statistics about the loaded dataset.
        
        Returns:
            Dictionary of statistics.
            
        Raises:
            ValueError: If data is not loaded.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        stats = {
            'total_songs': len(self.data),
            'unique_artists': self.data['artist'].nunique(),
            'tempo_range': (self.data['tempo'].min(), self.data['tempo'].max()),
            'loudness_range': (self.data['loudness'].min(), self.data['loudness'].max()),
            'duration_range': (self.data['duration'].min(), self.data['duration'].max()),
        }
        
        logger.info("\nDataset Statistics:")
        logger.info(f"  Total songs: {stats['total_songs']}")
        logger.info(f"  Unique artists: {stats['unique_artists']}")
        logger.info(f"  Tempo range: {stats['tempo_range'][0]:.2f} to {stats['tempo_range'][1]:.2f} BPM")
        logger.info(f"  Loudness range: {stats['loudness_range'][0]:.2f} to {stats['loudness_range'][1]:.2f} dB")
        logger.info(f"  Duration range: {stats['duration_range'][0]:.2f} to {stats['duration_range'][1]:.2f} seconds")
        
        return stats

    def train(self, 
             n_components_range: range = range(2, 11), 
             max_iter: int = 100) -> None:
        """
        Complete training pipeline: load data, preprocess, and train the model.
        
        Args:
            n_components_range: Range of cluster numbers to try.
            max_iter: Maximum number of iterations for GMM training.
            
        Raises:
            ValueError: If dataset directory is not specified.
        """
        if self.data is None:
            if self.dataset_dir:
                self.load_data()
            else:
                raise ValueError("Dataset directory not specified.")
                
        self.get_dataset_stats()
        X, self.scaler = self.preprocess_features()
        self.gmm_model = self.train_gmm_model(X, n_components_range, max_iter)
        self.visualize_clusters(X, self.gmm_model.predict(X))
        
        logger.info("Model training complete!")
        
    def save_model(self, model_path: str = "music_recommender_model.pkl") -> None:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model file.
            
        Raises:
            ValueError: If model is not trained.
        """
        if self.gmm_model is None or self.scaler is None:
            raise ValueError("Model not trained. Nothing to save.")
            
        model_data = {
            'gmm_model': self.gmm_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'data_columns': list(self.data.columns) if self.data is not None else []
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Model saved to {model_path}")
        
    def load_model(self, model_path: str = "music_recommender_model.pkl") -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model file.
            
        Raises:
            FileNotFoundError: If model file not found.
            ValueError: If error occurs during loading.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
            
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.gmm_model = model_data['gmm_model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")

def create_interactive_app():
    """Create an interactive command-line application for the music recommender."""
    parser = argparse.ArgumentParser(
        description='Music Recommendation System using Gaussian Mixture Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset and model parameters
    parser.add_argument('--dataset', '-d', 
                       type=str, 
                       help='Path to the dataset directory containing h5 files')
    parser.add_argument('--load-model', '-l', 
                       type=str, 
                       help='Load a trained model from file')
    parser.add_argument('--save-model', '-s', 
                       type=str, 
                       help='Save the trained model to file')
    
    # Training parameters
    parser.add_argument('--max-clusters', '-c', 
                       type=int, 
                       default=10, 
                       help='Maximum number of clusters to try during training')
    parser.add_argument('--max-iter', '-i', 
                       type=int, 
                       default=100, 
                       help='Maximum iterations for GMM training')
    
    # Recommendation parameters
    parser.add_argument('--recommend', '-r', 
                       type=str, 
                       help='Get recommendations for a song')
    parser.add_argument('--artist', '-a', 
                       type=str, 
                       help='Artist name (optional, to disambiguate songs)')
    parser.add_argument('--num-recs', '-n', 
                       type=int, 
                       default=5, 
                       help='Number of recommendations to return')
    
    # Visualization
    parser.add_argument('--visualize', '-v', 
                       action='store_true', 
                       help='Generate cluster visualization')
    
    # Logging
    parser.add_argument('--log-level', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='INFO',
                       help='Set the logging level')
    
    args = parser.parse_args()
    
    # Configure logging
    logger.setLevel(args.log_level)
    
    try:
        recommender = MusicRecommender(dataset_dir=args.dataset)
        
        # Load or train model
        if args.load_model:
            try:
                recommender.load_model(args.load_model)
                logger.info("Successfully loaded pre-trained model.")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                if args.dataset:
                    logger.info("Falling back to training a new model...")
                    recommender.train(
                        n_components_range=range(2, args.max_clusters + 1), 
                        max_iter=args.max_iter
                    )
        elif args.dataset:
            recommender.train(
                n_components_range=range(2, args.max_clusters + 1), 
                max_iter=args.max_iter
            )
        else:
            logger.warning("No dataset or model specified. System will only work with pre-loaded models.")
        
        # Save model if requested
        if args.save_model and recommender.gmm_model is not None:
            recommender.save_model(args.save_model)
        
        # Generate visualization if requested
        if args.visualize and recommender.gmm_model is not None:
            recommender.visualize_clusters()
        
        # Get recommendations if requested
        if args.recommend:
            recommendations = recommender.get_song_recommendations(
                args.recommend, 
                artist=args.artist, 
                n_recommendations=args.num_recs
            )
            if not recommendations.empty:
                print("\nRecommended Songs:")
                for i, (_, row) in enumerate(recommendations.iterrows()):
                    print(f"{i+1}. '{row['title']}' by {row['artist']}")
        
        # Interactive mode if no specific actions were requested
        if not any([args.recommend, args.visualize, args.save_model]) and recommender.gmm_model is not None:
            run_interactive_mode(recommender)
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

def run_interactive_mode(recommender: MusicRecommender) -> None:
    """
    Run an interactive mode for getting recommendations.
    
    Args:
        recommender: Trained recommender instance.
    """
    print("\n=== Music Recommendation Interactive Mode ===")
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            song_title = input("Enter a song title: ").strip()
            if song_title.lower() == 'exit':
                break
            
            artist = input("Enter artist name (optional): ").strip()
            if not artist:
                artist = None
            
            try:
                n_recommendations = int(input("Number of recommendations [5]: ") or "5")
            except ValueError:
                n_recommendations = 5
                print("Invalid input. Using default of 5 recommendations.")
            
            recommendations = recommender.get_song_recommendations(
                song_title, 
                artist=artist, 
                n_recommendations=n_recommendations
            )
            
            if not recommendations.empty:
                print("\nRecommended Songs:")
                for i, (_, row) in enumerate(recommendations.iterrows()):
                    print(f"{i+1}. '{row['title']}' by {row['artist']}")
            else:
                print("\nNo recommendations found.")
            
            print("\n" + "-" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            continue

def main():
    """Main entry point for the application."""
    try:
        create_interactive_app()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()