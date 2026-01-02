# -*- coding: utf-8 -*-
"""
Spotify Music Recommendation System
A personalized music recommendation engine using Spotify's API and audio feature analysis.

Author: Merve SF
License: MIT
"""

import numpy as np
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrackInfo:
    """Data class for storing track information."""
    name: str
    track_id: str
    artist: str
    popularity: int


class SpotifyAuthenticator:
    """Handles Spotify API authentication."""
    
    DEFAULT_SCOPE = 'user-library-read user-read-recently-played playlist-modify-public playlist-modify-private'
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str = 'http://127.0.0.1:3001/callback'
    ):
        """
        Initialize Spotify authenticator.
        
        Args:
            client_id: Spotify API client ID
            client_secret: Spotify API client secret
            redirect_uri: OAuth redirect URI
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self._spotify_client = None
    
    def authenticate(self, scope: Optional[str] = None) -> spotipy.Spotify:
        """
        Authenticate with Spotify API.
        
        Args:
            scope: Permission scope for API access
            
        Returns:
            Authenticated Spotify client
        """
        scope = scope or self.DEFAULT_SCOPE
        
        auth_manager = SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope=scope
        )
        
        self._spotify_client = spotipy.Spotify(auth_manager=auth_manager)
        logger.info("Successfully authenticated with Spotify API")
        
        return self._spotify_client
    
    @property
    def client(self) -> spotipy.Spotify:
        """Get the authenticated Spotify client."""
        if self._spotify_client is None:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        return self._spotify_client


class DataProcessor:
    """Handles data preprocessing and normalization."""
    
    AUDIO_FEATURES = [
        'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'duration_ms', 'popularity'
    ]
    
    def __init__(self):
        """Initialize data processor."""
        self.scaler = StandardScaler()
        self.stop_words = set(stopwords.words('english'))
    
    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numerical and text columns in the dataframe.
        
        Args:
            df: Input dataframe with audio features
            
        Returns:
            Normalized dataframe
        """
        df = df.copy()
        
        # Standardize loudness (has different scale)
        if 'loudness' in df.columns:
            df['loudness'] = self.scaler.fit_transform(
                df['loudness'].values.reshape(-1, 1)
            )
        
        # Min-max normalization for other numerical columns
        numerical_cols = [col for col in self.AUDIO_FEATURES 
                        if col in df.columns and col != 'loudness']
        
        for col in numerical_cols:
            col_data = df[col].values
            min_val, max_val = col_data.min(), col_data.max()
            if max_val > min_val:
                df[col] = (col_data - min_val) / (max_val - min_val)
        
        # Process text columns
        for col in ['name', 'artists']:
            if col in df.columns:
                df[col] = df[col].apply(self._tokenize_text)
        
        return df
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize and clean text.
        
        Args:
            text: Input text string
            
        Returns:
            List of cleaned tokens
        """
        if not isinstance(text, str):
            return []
        
        tokens = word_tokenize(text.lower())
        return [
            token for token in tokens
            if token not in self.stop_words and token not in string.punctuation
        ]


class SpotifyDataFetcher:
    """Fetches and processes data from Spotify API."""
    
    def __init__(self, spotify_client: spotipy.Spotify):
        """
        Initialize data fetcher.
        
        Args:
            spotify_client: Authenticated Spotify client
        """
        self.sp = spotify_client
    
    def get_saved_tracks(self, limit: int = 50) -> List[TrackInfo]:
        """
        Fetch user's saved tracks.
        
        Args:
            limit: Maximum number of tracks to fetch
            
        Returns:
            List of TrackInfo objects
        """
        tracks = []
        results = self.sp.current_user_saved_tracks(limit=min(limit, 50))
        
        while results and len(tracks) < limit:
            for item in results['items']:
                track = item['track']
                tracks.append(TrackInfo(
                    name=track['name'],
                    track_id=track['id'],
                    artist=track['artists'][0]['name'],
                    popularity=track['popularity']
                ))
            
            if results['next'] and len(tracks) < limit:
                results = self.sp.next(results)
            else:
                break
        
        logger.info(f"Fetched {len(tracks)} saved tracks")
        return tracks[:limit]
    
    def get_recently_played(self, limit: int = 50) -> List[TrackInfo]:
        """
        Fetch user's recently played tracks.
        
        Args:
            limit: Maximum number of tracks to fetch
            
        Returns:
            List of TrackInfo objects
        """
        results = self.sp.current_user_recently_played(limit=min(limit, 50))
        
        tracks = [
            TrackInfo(
                name=item['track']['name'],
                track_id=item['track']['id'],
                artist=item['track']['artists'][0]['name'],
                popularity=item['track']['popularity']
            )
            for item in results['items']
        ]
        
        logger.info(f"Fetched {len(tracks)} recently played tracks")
        return tracks
    
    def get_playlist_tracks(self, playlist_id: str) -> List[TrackInfo]:
        """
        Fetch tracks from a specific playlist.
        
        Args:
            playlist_id: Spotify playlist ID
            
        Returns:
            List of TrackInfo objects
        """
        playlist = self.sp.playlist(playlist_id)
        
        tracks = [
            TrackInfo(
                name=item['track']['name'],
                track_id=item['track']['id'],
                artist=item['track']['artists'][0]['name'],
                popularity=item['track']['popularity']
            )
            for item in playlist['tracks']['items']
            if item['track'] is not None
        ]
        
        logger.info(f"Fetched {len(tracks)} tracks from playlist")
        return tracks
    
    def get_audio_features(self, track_ids: List[str]) -> pd.DataFrame:
        """
        Fetch audio features for a list of tracks.
        
        Args:
            track_ids: List of Spotify track IDs
            
        Returns:
            DataFrame with audio features
        """
        features_list = []
        
        # Spotify API allows max 100 tracks per request
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i+100]
            features = self.sp.audio_features(batch)
            features_list.extend([f for f in features if f is not None])
        
        return pd.DataFrame(features_list)
    
    def build_track_dataframe(self, tracks: List[TrackInfo]) -> pd.DataFrame:
        """
        Build a complete dataframe with track info and audio features.
        
        Args:
            tracks: List of TrackInfo objects
            
        Returns:
            DataFrame with track information and audio features
        """
        # Get basic track info
        track_data = pd.DataFrame([
            {
                'name': t.name,
                'track_id': t.track_id,
                'artists': t.artist,
                'popularity': t.popularity
            }
            for t in tracks
        ])
        
        # Get audio features
        features_df = self.get_audio_features([t.track_id for t in tracks])
        
        if features_df.empty:
            return track_data
        
        # Merge and clean
        features_df = features_df.rename(columns={'id': 'track_id'})
        merged = track_data.merge(features_df, on='track_id', how='left')
        
        # Drop unnecessary columns
        cols_to_drop = ['type', 'uri', 'track_href', 'analysis_url', 'time_signature']
        merged = merged.drop(columns=[c for c in cols_to_drop if c in merged.columns])
        
        return merged


class RecommendationEngine:
    """Core recommendation engine using audio feature analysis."""
    
    FEATURE_COLUMNS = [
        'popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'duration_ms'
    ]
    
    def __init__(self, processor: DataProcessor):
        """
        Initialize recommendation engine.
        
        Args:
            processor: DataProcessor instance for normalization
        """
        self.processor = processor
        self.feature_weights = {}
        self.correlation_dict = {}
        self.reference_column = None
    
    def fit(self, user_tracks_df: pd.DataFrame) -> 'RecommendationEngine':
        """
        Fit the recommendation model on user's listening history.
        
        Args:
            user_tracks_df: DataFrame of user's tracks with audio features
            
        Returns:
            Self for method chaining
        """
        # Normalize data
        normalized_df = self.processor.normalize_dataframe(user_tracks_df)
        
        # Find column with minimum variance (most consistent preference)
        numerical_cols = [c for c in self.FEATURE_COLUMNS if c in normalized_df.columns]
        variances = {col: normalized_df[col].var() for col in numerical_cols}
        self.reference_column = min(variances, key=variances.get)
        
        logger.info(f"Reference column (min variance): {self.reference_column}")
        
        # Calculate correlations with reference column
        ref_values = normalized_df[self.reference_column]
        self.correlation_dict = {
            col: ref_values.corr(normalized_df[col])
            for col in numerical_cols
        }
        
        # Assign weights based on correlation ranking
        sorted_cols = sorted(self.correlation_dict.keys(), 
                           key=lambda x: self.correlation_dict[x])
        self.feature_weights = {
            col: 0.1 + 0.2 * i 
            for i, col in enumerate(sorted_cols)
        }
        
        self._user_profile = normalized_df
        
        logger.info("Model fitted successfully")
        return self
    
    def recommend(
        self,
        candidate_pool: pd.DataFrame,
        n_recommendations: int = 20,
        max_iterations: int = 1000,
        error_threshold: float = 2.3
    ) -> pd.DataFrame:
        """
        Generate playlist recommendations.
        
        Args:
            candidate_pool: DataFrame of candidate tracks
            n_recommendations: Number of tracks to recommend
            max_iterations: Maximum optimization iterations
            error_threshold: Error threshold for accepting recommendations
            
        Returns:
            DataFrame of recommended tracks
        """
        if not hasattr(self, '_user_profile'):
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Normalize candidate pool
        normalized_candidates = self.processor.normalize_dataframe(candidate_pool.copy())
        
        best_playlist = None
        best_error = float('inf')
        
        for iteration in range(max_iterations):
            # Random sample from candidates
            sample_indices = np.random.choice(
                len(normalized_candidates), 
                size=min(n_recommendations, len(normalized_candidates)),
                replace=False
            )
            sample = normalized_candidates.iloc[sample_indices]
            
            # Calculate error score
            error = self._calculate_error(sample)
            
            if error < best_error:
                best_error = error
                best_playlist = candidate_pool.iloc[sample_indices].copy()
            
            if error <= error_threshold:
                logger.info(f"Found optimal playlist at iteration {iteration + 1}")
                break
        
        logger.info(f"Best error score: {best_error:.4f}")
        return best_playlist if best_playlist is not None else candidate_pool.head(n_recommendations)
    
    def _calculate_error(self, sample: pd.DataFrame) -> float:
        """
        Calculate error score for a sample playlist.
        
        Args:
            sample: Sampled tracks DataFrame
            
        Returns:
            Error score (lower is better)
        """
        error = 0.0
        ref_values = sample[self.reference_column]
        
        for col in self.feature_weights:
            if col not in sample.columns:
                continue
            
            if col == self.reference_column:
                # Compare distribution with user profile
                target_mean = self._user_profile[col].mean()
                target_std = self._user_profile[col].std()
                sample_values = sample[col].values
                
                expected = np.random.normal(target_mean, target_std, len(sample))
                error += np.sum((expected - sample_values) ** 2) * self.feature_weights[col]
            else:
                # Compare correlation structure
                sample_corr = ref_values.corr(sample[col])
                target_corr = self.correlation_dict.get(col, 0)
                error += abs(target_corr - sample_corr) * self.feature_weights[col]
        
        return error


class MusicRecommendationSystem:
    """
    Main class for the Music Recommendation System.
    Combines all components for easy usage.
    """
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str = 'http://127.0.0.1:3001/callback'
    ):
        """
        Initialize the Music Recommendation System.
        
        Args:
            client_id: Spotify API client ID
            client_secret: Spotify API client secret
            redirect_uri: OAuth redirect URI
        """
        self.authenticator = SpotifyAuthenticator(client_id, client_secret, redirect_uri)
        self.processor = DataProcessor()
        self.engine = RecommendationEngine(self.processor)
        
        self._fetcher = None
        self._is_authenticated = False
    
    def connect(self) -> 'MusicRecommendationSystem':
        """
        Connect to Spotify API.
        
        Returns:
            Self for method chaining
        """
        client = self.authenticator.authenticate()
        self._fetcher = SpotifyDataFetcher(client)
        self._is_authenticated = True
        
        logger.info("Connected to Spotify")
        return self
    
    def build_user_profile(
        self,
        use_saved_tracks: bool = True,
        use_recent_tracks: bool = True,
        playlist_id: Optional[str] = None,
        saved_limit: int = 50,
        recent_limit: int = 50
    ) -> pd.DataFrame:
        """
        Build user profile from listening history.
        
        Args:
            use_saved_tracks: Include saved tracks
            use_recent_tracks: Include recently played tracks
            playlist_id: Optional playlist ID to include
            saved_limit: Max saved tracks to fetch
            recent_limit: Max recent tracks to fetch
            
        Returns:
            DataFrame with user's track profile
        """
        self._check_connection()
        
        all_tracks = []
        
        if use_saved_tracks:
            all_tracks.extend(self._fetcher.get_saved_tracks(saved_limit))
        
        if use_recent_tracks:
            all_tracks.extend(self._fetcher.get_recently_played(recent_limit))
        
        if playlist_id:
            all_tracks.extend(self._fetcher.get_playlist_tracks(playlist_id))
        
        # Remove duplicates by track_id
        seen_ids = set()
        unique_tracks = []
        for track in all_tracks:
            if track.track_id not in seen_ids:
                seen_ids.add(track.track_id)
                unique_tracks.append(track)
        
        user_df = self._fetcher.build_track_dataframe(unique_tracks)
        
        # Fit the recommendation engine
        self.engine.fit(user_df)
        
        logger.info(f"Built user profile with {len(user_df)} unique tracks")
        return user_df
    
    def get_recommendations(
        self,
        candidate_source: str = 'csv',
        csv_path: Optional[str] = None,
        playlist_id: Optional[str] = None,
        n_recommendations: int = 20,
        sample_size: int = 350
    ) -> pd.DataFrame:
        """
        Get personalized track recommendations.
        
        Args:
            candidate_source: Source of candidates ('csv' or 'playlist')
            csv_path: Path to CSV file with candidate tracks
            playlist_id: Playlist ID for candidate tracks
            n_recommendations: Number of tracks to recommend
            sample_size: Size of candidate pool to sample
            
        Returns:
            DataFrame of recommended tracks
        """
        self._check_connection()
        
        if candidate_source == 'csv' and csv_path:
            candidates = pd.read_csv(csv_path)
            # Sample from large datasets
            if len(candidates) > sample_size:
                candidates = candidates.sample(n=sample_size, random_state=42)
        elif candidate_source == 'playlist' and playlist_id:
            tracks = self._fetcher.get_playlist_tracks(playlist_id)
            candidates = self._fetcher.build_track_dataframe(tracks)
        else:
            raise ValueError("Invalid candidate source. Provide csv_path or playlist_id.")
        
        recommendations = self.engine.recommend(
            candidates,
            n_recommendations=n_recommendations
        )
        
        return recommendations
    
    def create_playlist(
        self,
        recommendations: pd.DataFrame,
        playlist_name: str = "AI Recommendations",
        description: str = "Generated by Music Recommendation System"
    ) -> str:
        """
        Create a Spotify playlist from recommendations.
        
        Args:
            recommendations: DataFrame of recommended tracks
            playlist_name: Name for the new playlist
            description: Playlist description
            
        Returns:
            Playlist ID
        """
        self._check_connection()
        
        user_id = self.authenticator.client.current_user()['id']
        
        playlist = self.authenticator.client.user_playlist_create(
            user_id,
            playlist_name,
            public=True,
            description=description
        )
        
        track_uris = [f"spotify:track:{tid}" for tid in recommendations['track_id']]
        self.authenticator.client.playlist_add_items(playlist['id'], track_uris)
        
        logger.info(f"Created playlist: {playlist_name} with {len(track_uris)} tracks")
        return playlist['id']
    
    def _check_connection(self):
        """Check if connected to Spotify."""
        if not self._is_authenticated:
            raise RuntimeError("Not connected. Call connect() first.")


# Download NLTK data
def setup_nltk():
    """Download required NLTK data."""
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
