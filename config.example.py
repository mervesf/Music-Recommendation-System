# -*- coding: utf-8 -*-
"""
Configuration file for Spotify Music Recommendation System.

IMPORTANT: Never commit this file with real credentials!
Add config.py to your .gitignore file.

Instructions:
1. Go to https://developer.spotify.com/dashboard
2. Create a new app
3. Copy your Client ID and Client Secret
4. Set the Redirect URI to http://localhost:3001/callback
"""

# Spotify API Credentials
SPOTIFY_CLIENT_ID = "your_client_id_here"
SPOTIFY_CLIENT_SECRET = "your_client_secret_here"
SPOTIFY_REDIRECT_URI = "http://localhost:3001/callback"

# Default Settings
DEFAULT_SAVED_TRACKS_LIMIT = 100
DEFAULT_RECENT_TRACKS_LIMIT = 50
DEFAULT_RECOMMENDATIONS = 20
DEFAULT_CANDIDATE_SAMPLE_SIZE = 350

# Recommendation Engine Settings
ERROR_THRESHOLD = 2.3
MAX_ITERATIONS = 1000
