import os
import h5py
import pandas as pd

def load_dataset_from_folder(root_folder):
    """
    Recursively loads all HDF5 (.h5) files from the given folder structure into a single DataFrame.
    
    Args:
        root_folder (str): The path to the dataset folder (e.g., "millionsongsubset")

    Returns:
        pd.DataFrame: Combined DataFrame with all songs.
    """
    all_data = []
    
    # Walk through all directories and subdirectories
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith('.h5'):  # Load only HDF5 files
                file_path = os.path.join(dirpath, file)
                print(f"Loading {file_path}...")
                
                try:
                    with h5py.File(file_path, 'r') as h5_file:
                        # Example: Extract artist name and song title
                        artist_name = h5_file['metadata/songs'][0]['artist_name'].decode()
                        song_title = h5_file['metadata/songs'][0]['title'].decode()

                        # Append extracted data to list
                        all_data.append({'artist': artist_name, 'title': song_title})
                
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")

    # Convert collected data into a DataFrame
    if all_data:
        full_dataset = pd.DataFrame(all_data)
        print(f"Loaded {len(full_dataset)} records from {len(all_data)} files.")
        return full_dataset
    else:
        print("No HDF5 files found.")
        return pd.DataFrame()

# ✅ Use WSL path format if needed
dataset_path = "/mnt/c/Users/sudee/OneDrive/Desktop/Personalized-Music-Genre-Recommendation-Using-GMM/millionsongsubset"
full_data = load_dataset_from_folder(dataset_path)

# Display some info about the loaded dataset
print(full_data.head())
print(full_data.info())
