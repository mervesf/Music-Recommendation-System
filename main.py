#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Music Recommendation System - Main Entry Point

This script demonstrates how to use the Music Recommendation System
to generate personalized playlist recommendations.

Usage:
    python main.py

Make sure to set your Spotify API credentials in config.py
"""

import os
import pandas as pd
from src.recommender import MusicRecommendationSystem, setup_nltk, DataProcessor, RecommendationEngine


def main():
    """Main function to run the recommendation system."""
    
    # Setup NLTK data
    setup_nltk()
    
    # Check if dataset exists
    dataset_path = 'data/data.csv'
    if not os.path.exists(dataset_path):
        print("Dataset not found!")
        print(f"Please download data.csv from:")
        print("https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset")
        print(f"And place it in: {dataset_path}")
        return
    
    print("Loading Kaggle dataset...")
    full_dataset = pd.read_csv(dataset_path)
    print(f"Loaded {len(full_dataset):,} tracks\n")
    
    # Get credentials from config.py
    client_id = None
    client_secret = None
    
    try:
        from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
        client_id = SPOTIFY_CLIENT_ID
        client_secret = SPOTIFY_CLIENT_SECRET
    except ImportError:
        pass
    
    if not client_id or client_id == "your_client_id_here":
        client_id = os.getenv('SPOTIFY_CLIENT_ID', '')
    if not client_secret or client_secret == "your_client_secret_here":
        client_secret = os.getenv('SPOTIFY_CLIENT_SECRET', '')
    
    # Option 1: Use Spotify to get user's playlist names (not audio features)
    # Option 2: Manually input favorite artists/genres
    
    print("=" * 60)
    print("  MUSIC RECOMMENDATION SYSTEM")
    print("=" * 60)
    print("\nHow would you like to build your profile?")
    print("1. Enter your favorite artists")
    print("2. Enter your favorite genres")
    print("3. Random recommendations (explore new music)")
    
    choice = input("\nYour choice (1/2/3): ").strip()
    
    if choice == '1':
        # Artist-based recommendation
        print("\nEnter your favorite artists (comma separated):")
        print("Example: Taylor Swift, The Weeknd, Drake")
        artists_input = input("> ").strip()
        
        if not artists_input:
            print("No artists entered. Using random selection.")
            user_tracks = full_dataset.sample(n=50, random_state=42)
        else:
            artists = [a.strip().lower() for a in artists_input.split(',')]
            
            # Find tracks by these artists
            mask = full_dataset['artists'].str.lower().str.contains('|'.join(artists), na=False)
            user_tracks = full_dataset[mask]
            
            if len(user_tracks) == 0:
                print(f"No tracks found for these artists. Using similar search...")
                # Fuzzy search
                for artist in artists:
                    mask = full_dataset['artists'].str.lower().str.contains(artist[:4], na=False)
                    user_tracks = pd.concat([user_tracks, full_dataset[mask]])
            
            if len(user_tracks) == 0:
                print("Still no matches. Using random selection.")
                user_tracks = full_dataset.sample(n=50, random_state=42)
            else:
                user_tracks = user_tracks.head(100)  # Limit to 100 tracks
                print(f"Found {len(user_tracks)} tracks from your favorite artists!")
    
    elif choice == '2':
        # Genre-based (using audio features as proxy)
        print("\nSelect your preferred music style:")
        print("(Based on audio features, not genre tags)\n")
        print("1. Energetic & Danceable - Fast, rhythmic, party vibes")
        print("   [High energy + High danceability]")
        print()
        print("2. Chill & Acoustic - Soft, calm, unplugged")
        print("   [Low energy + High acousticness]")
        print()
        print("3. Happy & Upbeat - Positive, cheerful mood")
        print("   [High valence]")
        print()
        print("4. Instrumental - No vocals, background music")
        print("   [High instrumentalness + Low speechiness]")
        print()
        print("5. Sad & Emotional - Melancholic, slow, deep")
        print("   [Low valence + Low energy]")
        print()
        print("6. Loud & Intense - Raw, powerful, electric")
        print("   [High energy + Low acousticness]")
        print()
        print("7. Mixed / Balanced - A bit of everything")
        
        style = input("\nYour style (1-7): ").strip()
        
        if style == '1':
            user_tracks = full_dataset[
                (full_dataset['energy'] > 0.7) & 
                (full_dataset['danceability'] > 0.7)
            ].head(100)
        elif style == '2':
            user_tracks = full_dataset[
                (full_dataset['energy'] < 0.5) & 
                (full_dataset['acousticness'] > 0.5)
            ].head(100)
        elif style == '3':
            user_tracks = full_dataset[
                full_dataset['valence'] > 0.7
            ].head(100)
        elif style == '4':
            user_tracks = full_dataset[
                (full_dataset['instrumentalness'] > 0.5) & 
                (full_dataset['speechiness'] < 0.3)
            ].head(100)
        elif style == '5':
            user_tracks = full_dataset[
                (full_dataset['valence'] < 0.3) & 
                (full_dataset['energy'] < 0.5)
            ].head(100)
        elif style == '6':
            user_tracks = full_dataset[
                (full_dataset['energy'] > 0.8) & 
                (full_dataset['acousticness'] < 0.3)
            ].head(100)
        else:
            user_tracks = full_dataset.sample(n=100, random_state=42)
        
        print(f"Selected {len(user_tracks)} tracks matching your style!")
    
    else:
        # Random exploration
        print("\nSelecting random tracks for exploration...")
        user_tracks = full_dataset.sample(n=50, random_state=42)
        print(f"Selected {len(user_tracks)} random tracks!")
    
    # Build recommendation engine
    print("\nAnalyzing music profile...")
    processor = DataProcessor()
    engine = RecommendationEngine(processor)
    
    # Fit on user's selected tracks
    engine.fit(user_tracks)
    
    # Display user's profile
    print("\nYour Music Profile:")
    print("-" * 40)
    feature_cols = ['danceability', 'energy', 'valence', 'acousticness']
    for col in feature_cols:
        if col in user_tracks.columns:
            avg = user_tracks[col].mean()
            bar = '#' * int(avg * 20)
            print(f"   {col.capitalize():15} : {bar:<20} {avg:.2f}")
    print()
    
    # Generate recommendations from different part of dataset
    print("Generating recommendations...")
    
    # Use tracks NOT in user's selection as candidates
    candidate_pool = full_dataset[~full_dataset.index.isin(user_tracks.index)]
    candidate_sample = candidate_pool.sample(n=min(500, len(candidate_pool)), random_state=123)
    
    recommendations = engine.recommend(
        candidate_sample,
        n_recommendations=20
    )
    
    print(f"Generated {len(recommendations)} recommendations!\n")
    
    # Display recommendations
    print("Your Personalized Recommendations:")
    print("=" * 70)
    for idx, (_, track) in enumerate(recommendations.iterrows(), 1):
        name = str(track.get('name', 'Unknown'))[:35]
        artist = str(track.get('artists', 'Unknown'))[:25]
        print(f"   {idx:2}. {name:<35} - {artist}")
    print("=" * 70)
    
    # Save to CSV option
    save_option = input("\nSave recommendations to CSV? (y/n): ").lower().strip()
    if save_option == 'y':
        output_file = 'my_recommendations.csv'
        recommendations[['name', 'artists', 'popularity']].to_csv(output_file, index=False)
        print(f"Saved to {output_file}")
    
    # If Spotify credentials available, offer to create playlist
    if client_id and client_secret and client_id != "your_client_id_here":
        create_playlist = input("\nCreate playlist on Spotify? (y/n): ").lower().strip()
        if create_playlist == 'y':
            try:
                recommender = MusicRecommendationSystem(client_id, client_secret)
                recommender.connect()
                
                # Need to search for tracks on Spotify since we only have names
                print("Searching tracks on Spotify...")
                sp = recommender.authenticator.client
                
                track_uris = []
                for _, track in recommendations.iterrows():
                    query = f"{track['name']} {track['artists']}"
                    results = sp.search(q=query, type='track', limit=1)
                    if results['tracks']['items']:
                        track_uris.append(results['tracks']['items'][0]['uri'])
                
                if track_uris:
                    print("\nEnter playlist name (or press Enter for 'AI Recommendations'):")
                    playlist_name = input("> ").strip()
                    # Clean the playlist name - remove any control characters
                    playlist_name = ''.join(char for char in playlist_name if char.isprintable())
                    if not playlist_name:
                        playlist_name = "AI Recommendations"
                    
                    user_id = sp.current_user()['id']
                    playlist = sp.user_playlist_create(user_id, playlist_name, public=True)
                    sp.playlist_add_items(playlist['id'], track_uris)
                    print(f"\nPlaylist '{playlist_name}' created with {len(track_uris)} tracks!")
                    print(f"https://open.spotify.com/playlist/{playlist['id']}")
                else:
                    print("Could not find tracks on Spotify.")
            except Exception as e:
                print(f"Error creating playlist: {e}")
    
    print("\nThank you for using Music Recommendation System!")


if __name__ == "__main__":
    main()