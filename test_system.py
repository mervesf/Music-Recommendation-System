#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script for Music Recommendation System

Run this script to verify all components are working correctly.
Usage: python test_system.py
"""

import sys

def print_status(message, success=True):
    """Print status with emoji."""
    emoji = "✅" if success else "❌"
    print(f"{emoji} {message}")

def print_header(title):
    """Print section header."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}\n")

# ============================================
# TEST 1: Import Check
# ============================================
print_header("TEST 1: Importing Modules")

try:
    import numpy as np
    print_status("numpy imported")
except ImportError as e:
    print_status(f"numpy import failed: {e}", False)
    print("   Run: pip install numpy")

try:
    import pandas as pd
    print_status("pandas imported")
except ImportError as e:
    print_status(f"pandas import failed: {e}", False)
    print("   Run: pip install pandas")

try:
    import spotipy
    print_status("spotipy imported")
except ImportError as e:
    print_status(f"spotipy import failed: {e}", False)
    print("   Run: pip install spotipy")

try:
    from sklearn.preprocessing import StandardScaler
    print_status("scikit-learn imported")
except ImportError as e:
    print_status(f"scikit-learn import failed: {e}", False)
    print("   Run: pip install scikit-learn")

try:
    import nltk
    print_status("nltk imported")
except ImportError as e:
    print_status(f"nltk import failed: {e}", False)
    print("   Run: pip install nltk")

# ============================================
# TEST 2: NLTK Data Check
# ============================================
print_header("TEST 2: NLTK Data")

try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    
    # Test tokenization
    tokens = word_tokenize("Hello world test")
    print_status("NLTK punkt tokenizer working")
    
    # Test stopwords
    stops = stopwords.words('english')
    print_status(f"NLTK stopwords loaded ({len(stops)} words)")
    
except LookupError as e:
    print_status("NLTK data missing", False)
    print("   Run these commands in Python:")
    print("   >>> import nltk")
    print("   >>> nltk.download('punkt')")
    print("   >>> nltk.download('stopwords')")
    print("   >>> nltk.download('punkt_tab')")

# ============================================
# TEST 3: Project Module Import
# ============================================
print_header("TEST 3: Project Modules")

try:
    from src.recommender import (
        MusicRecommendationSystem,
        SpotifyAuthenticator,
        SpotifyDataFetcher,
        DataProcessor,
        RecommendationEngine,
        TrackInfo
    )
    print_status("All project modules imported successfully")
except ImportError as e:
    print_status(f"Project module import failed: {e}", False)
    print("   Make sure you're in the project root directory")

# ============================================
# TEST 4: Dataset Check
# ============================================
print_header("TEST 4: Kaggle Dataset")

import os
dataset_path = 'data/data.csv'

if os.path.exists(dataset_path):
    try:
        df = pd.read_csv(dataset_path)
        print_status(f"Dataset loaded: {len(df):,} tracks")
        
        # Check required columns
        required_cols = ['name', 'artists', 'danceability', 'energy', 'valence']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print_status(f"Missing columns: {missing_cols}", False)
        else:
            print_status("All required columns present")
        
        # Show sample
        print(f"\n   Sample tracks:")
        for i, row in df.head(3).iterrows():
            print(f"   - {row['name'][:40]} by {row['artists'][:20]}")
            
    except Exception as e:
        print_status(f"Dataset read error: {e}", False)
else:
    print_status("Dataset not found", False)
    print(f"   Expected location: {dataset_path}")
    print("   Download from: https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset")

# ============================================
# TEST 5: DataProcessor Test
# ============================================
print_header("TEST 5: DataProcessor")

try:
    from src.recommender import DataProcessor
    
    processor = DataProcessor()
    print_status("DataProcessor initialized")
    
    # Create test dataframe
    test_df = pd.DataFrame({
        'name': ['Test Song 1', 'Test Song 2'],
        'artists': ['Artist A', 'Artist B'],
        'popularity': [50, 75],
        'danceability': [0.5, 0.8],
        'energy': [0.6, 0.9],
        'loudness': [-5.0, -3.0],
        'valence': [0.4, 0.7],
        'tempo': [120, 140],
        'acousticness': [0.2, 0.1],
        'instrumentalness': [0.0, 0.1],
        'liveness': [0.1, 0.2],
        'speechiness': [0.05, 0.1],
        'key': [5, 7],
        'mode': [1, 0],
        'duration_ms': [200000, 180000]
    })
    
    normalized = processor.normalize_dataframe(test_df)
    print_status("Normalization working")
    
except Exception as e:
    print_status(f"DataProcessor test failed: {e}", False)

# ============================================
# TEST 6: Spotify Credentials Check
# ============================================
print_header("TEST 6: Spotify Credentials")

config_exists = os.path.exists('config.py')
env_id = os.getenv('SPOTIFY_CLIENT_ID')
env_secret = os.getenv('SPOTIFY_CLIENT_SECRET')

if config_exists:
    print_status("config.py found")
    try:
        from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
        if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_ID != "your_client_id_here":
            print_status("Client ID configured")
        else:
            print_status("Client ID not set in config.py", False)
        
        if SPOTIFY_CLIENT_SECRET and SPOTIFY_CLIENT_SECRET != "your_client_secret_here":
            print_status("Client Secret configured")
        else:
            print_status("Client Secret not set in config.py", False)
    except ImportError:
        print_status("Could not import from config.py", False)
        
elif env_id and env_secret:
    print_status("Environment variables configured")
else:
    print_status("No credentials found", False)
    print("   Option 1: Copy config.example.py to config.py and edit it")
    print("   Option 2: Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET env vars")

# ============================================
# SUMMARY
# ============================================
print_header("TEST SUMMARY")

print("""
If all tests passed (✅), you're ready to run:

    python main.py

If any test failed (❌), follow the instructions above to fix it.

Need help? Check the README.md for detailed setup instructions.
""")

# Optional: Test Spotify connection
print("-" * 50)
test_spotify = input("Test Spotify connection? (y/n): ").lower().strip()

if test_spotify == 'y':
    print_header("TEST 7: Spotify Connection")
    
    try:
        # Get credentials
        client_id = os.getenv('SPOTIFY_CLIENT_ID')
        client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            try:
                from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
                client_id = SPOTIFY_CLIENT_ID
                client_secret = SPOTIFY_CLIENT_SECRET
            except:
                pass
        
        if not client_id or not client_secret:
            print_status("No credentials available for connection test", False)
        else:
            from src.recommender import MusicRecommendationSystem
            
            print("🔐 Connecting to Spotify (browser will open for auth)...")
            recommender = MusicRecommendationSystem(client_id, client_secret)
            recommender.connect()
            print_status("Successfully connected to Spotify!")
            
            # Get user info
            user = recommender.authenticator.client.current_user()
            print_status(f"Logged in as: {user['display_name']}")
            
    except Exception as e:
        print_status(f"Connection failed: {e}", False)
