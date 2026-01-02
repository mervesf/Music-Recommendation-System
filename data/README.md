# Data Directory

## Required Dataset

Download `data.csv` from Kaggle and place it here:
**[Spotify Dataset by Vatsal Mavani](https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset)**

### Download Steps:
1. Go to the Kaggle link above
2. Click on "data.csv" (not the other files)
3. Download and place in this folder

### Expected Structure:
```
data/
└── data.csv    # ~170,000 tracks with audio features
```

### Dataset Columns:
The `data.csv` file contains these columns:
- `name` - Track name
- `artists` - Artist name(s)
- `id` - Spotify track ID
- `danceability` - 0.0 to 1.0
- `energy` - 0.0 to 1.0
- `key` - 0 to 11
- `loudness` - dB value
- `mode` - 0 (minor) or 1 (major)
- `speechiness` - 0.0 to 1.0
- `acousticness` - 0.0 to 1.0
- `instrumentalness` - 0.0 to 1.0
- `liveness` - 0.0 to 1.0
- `valence` - 0.0 to 1.0
- `tempo` - BPM
- `duration_ms` - Duration in milliseconds
- `popularity` - 0 to 100

> **Note:** The `data.csv` file is not included in this repository due to size (~25MB) and licensing. Please download directly from Kaggle.
