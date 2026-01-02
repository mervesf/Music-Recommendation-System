<h1 align="center">
  🎵 Spotify Music Recommendation System
</h1>

<p align="center">
  <strong>A personalized music recommendation engine powered by Spotify's API and audio feature analysis</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Spotify%20API-v1-green.svg" alt="Spotify API">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
</p>

---

## 🎯 Overview

This system analyzes your Spotify listening history to understand your music preferences, then generates personalized playlist recommendations based on audio features like danceability, energy, valence, and more.

Unlike basic recommendation systems that rely solely on collaborative filtering, this engine performs **deep audio feature analysis** to understand *why* you like certain songs and finds tracks with similar sonic characteristics.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Your Listening History                       │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│    │  Saved   │  │  Recent  │  │ Playlist │  │  Liked   │       │
│    │  Tracks  │  │  Played  │  │  Tracks  │  │  Songs   │       │
│    └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│         │             │             │             │             │
│         └─────────────┴─────────────┴─────────────┘             │
│                           │                                     │
│                    ┌──────▼──────┐                              │
│                    │   Feature   │                              │
│                    │  Extraction │                              │
│                    └──────┬──────┘                              │
│                           │                                     │
│    ┌──────────────────────▼──────────────────────┐              │
│    │           Audio Feature Analysis            │              │
│    │  ┌─────────────┐  ┌─────────────────────┐   │              │
│    │  │ Correlation │  │ Variance Analysis   │   │              │
│    │  │   Matrix    │  │ (Find Preferences)  │   │              │
│    │  └─────────────┘  └─────────────────────┘   │              │
│    └──────────────────────┬──────────────────────┘              │
│                           │                                     │
│                    ┌──────▼──────┐                              │
│                    │ Weighted    │                              │
│                    │ Matching    │                              │
│                    └──────┬──────┘                              │
│                           │                                     │
│              ┌────────────▼────────────┐                        │
│               🎧 Personalized Playlist                         |
│              └─────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

## ✨ Features

- **🔐 Secure OAuth Authentication** - Safe connection to your Spotify account
- **📊 Audio Feature Analysis** - Analyzes 13+ audio features per track
- **🎯 Preference Detection** - Identifies your most consistent music preferences
- **🔄 Correlation-Based Matching** - Finds songs with similar sonic signatures
- **📝 Playlist Generation** - Creates and saves playlists directly to Spotify
- **🧩 Modular Architecture** - Easy to extend and customize

## 🔬 How It Works

### Audio Features Analyzed

| Feature | Description | Range |
|---------|-------------|-------|
| **Danceability** | How suitable for dancing | 0.0 - 1.0 |
| **Energy** | Intensity and activity | 0.0 - 1.0 |
| **Valence** | Musical positiveness | 0.0 - 1.0 |
| **Acousticness** | Acoustic vs electronic | 0.0 - 1.0 |
| **Instrumentalness** | Vocal vs instrumental | 0.0 - 1.0 |
| **Speechiness** | Presence of spoken words | 0.0 - 1.0 |
| **Liveness** | Presence of audience | 0.0 - 1.0 |
| **Tempo** | Beats per minute | 50 - 200 |
| **Loudness** | Overall loudness (dB) | -60 - 0 |
| **Key** | Musical key | 0 - 11 |
| **Mode** | Major (1) or Minor (0) | 0 - 1 |

### Algorithm Overview

1. **Data Collection**: Fetches your saved tracks, recently played, and playlist tracks
2. **Feature Extraction**: Gets audio features for each track via Spotify API
3. **Preference Analysis**: Identifies the feature with minimum variance (your most consistent preference)
4. **Correlation Mapping**: Builds a correlation matrix between all features
5. **Weight Assignment**: Assigns importance weights based on correlation strength
6. **Candidate Scoring**: Scores candidate tracks against your preference profile
7. **Optimization**: Iteratively selects the best matching tracks

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Spotify Developer Account ([Create one here](https://developer.spotify.com/dashboard))

### Step 1: Clone the Repository

```bash
git clone https://github.com/mervesf/Music-Recommendation-System.git
cd Music-Recommendation-System
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download the Dataset

This project uses the **Spotify Dataset** from Kaggle as the candidate pool for recommendations.

1. Go to [Kaggle - Spotify Dataset](https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset)
2. Download only `data.csv` (contains ~170,000 tracks with audio features)
3. Place it in the `data/` folder:

```
data/
└── data.csv    # ~170K tracks with audio features
```

> **Note:** The dataset contains tracks with pre-extracted audio features including danceability, energy, valence, tempo, and more.

### Step 4: Configure Spotify Credentials

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new application
3. Add `http://localhost:3001/callback` as a Redirect URI
4. Copy your credentials:

```bash
cp config.example.py config.py
# Edit config.py with your credentials
```

Or use environment variables:

```bash
export SPOTIFY_CLIENT_ID="your_client_id"
export SPOTIFY_CLIENT_SECRET="your_client_secret"
```

## 🚀 Usage

### Quick Start

```python
from src import MusicRecommendationSystem

# Initialize
recommender = MusicRecommendationSystem(
    client_id="your_client_id",
    client_secret="your_client_secret"
)

# Connect to Spotify
recommender.connect()

# Build your music profile
user_profile = recommender.build_user_profile(
    use_saved_tracks=True,
    use_recent_tracks=True
)

# Get recommendations
recommendations = recommender.get_recommendations(
    candidate_source='csv',
    csv_path='data/songs.csv',
    n_recommendations=20
)

# Create a playlist on Spotify
playlist_id = recommender.create_playlist(
    recommendations,
    playlist_name="My AI Playlist 🎵"
)
```

### Command Line

```bash
python main.py
```

### Using Individual Components

```python
from src import (
    SpotifyAuthenticator,
    SpotifyDataFetcher,
    DataProcessor,
    RecommendationEngine
)

# Custom authentication
auth = SpotifyAuthenticator(client_id, client_secret)
client = auth.authenticate()

# Fetch data
fetcher = SpotifyDataFetcher(client)
tracks = fetcher.get_saved_tracks(limit=100)
track_df = fetcher.build_track_dataframe(tracks)

# Process and recommend
processor = DataProcessor()
engine = RecommendationEngine(processor)
engine.fit(track_df)

recommendations = engine.recommend(candidate_df, n_recommendations=20)
```

## 📁 Project Structure

```
Music-Recommendation-System/
├── src/
│   ├── __init__.py          # Package exports
│   └── recommender.py       # Core recommendation engine
├── data/
│   └── .gitkeep             # Data directory
├── main.py                  # Entry point
├── config.example.py        # Configuration template
├── requirements.txt         # Dependencies
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT License
└── README.md               # Documentation
```

## 📚 API Reference

### MusicRecommendationSystem

Main class that orchestrates all components.

| Method | Description |
|--------|-------------|
| `connect()` | Authenticate with Spotify |
| `build_user_profile()` | Analyze listening history |
| `get_recommendations()` | Generate recommendations |
| `create_playlist()` | Save playlist to Spotify |

### RecommendationEngine

Core recommendation logic.

| Method | Description |
|--------|-------------|
| `fit(df)` | Train on user's tracks |
| `recommend(candidates, n)` | Generate n recommendations |

### DataProcessor

Data normalization and preprocessing.

| Method | Description |
|--------|-------------|
| `normalize_dataframe(df)` | Normalize audio features |

## 🛠️ Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `saved_limit` | 100 | Max saved tracks to analyze |
| `recent_limit` | 50 | Max recent tracks to analyze |
| `n_recommendations` | 20 | Tracks per recommendation |
| `error_threshold` | 2.3 | Matching precision threshold |
| `max_iterations` | 1000 | Max optimization iterations |

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Ideas for Contribution

- [ ] Add genre-based filtering
- [ ] Implement mood detection
- [ ] Create a web interface
- [ ] Add playlist diversity controls
- [ ] Support for Apple Music / YouTube Music

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Spotify Web API](https://developer.spotify.com/documentation/web-api/) for providing audio features
- [Spotipy](https://spotipy.readthedocs.io/) for the excellent Python wrapper
- [Vatsal Mavani](https://www.kaggle.com/vatsalmavani) for the [Spotify Dataset](https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset) on Kaggle
- The open-source community for inspiration

---

<p align="center">
  Built by <a href="https://github.com/mervesf">Merve</a> · ⭐ Star if you found this useful!
</p>

<p align="center">
  <a href="https://github.com/mervesf/Music-Recommendation-System/issues">Report Bug</a> •
  <a href="https://github.com/mervesf/Music-Recommendation-System/issues">Request Feature</a>
</p>
