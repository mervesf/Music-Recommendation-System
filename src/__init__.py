# -*- coding: utf-8 -*-
"""
Music Recommendation System

A personalized music recommendation engine using Spotify's API.
"""

from .recommender import (
    MusicRecommendationSystem,
    SpotifyAuthenticator,
    SpotifyDataFetcher,
    DataProcessor,
    RecommendationEngine,
    TrackInfo,
    setup_nltk
)

__version__ = "1.0.0"
__author__ = "Merve SF"

__all__ = [
    "MusicRecommendationSystem",
    "SpotifyAuthenticator",
    "SpotifyDataFetcher",
    "DataProcessor",
    "RecommendationEngine",
    "TrackInfo",
    "setup_nltk"
]
