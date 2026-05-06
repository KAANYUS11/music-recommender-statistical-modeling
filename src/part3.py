"""
Part 3: Recommender Design
"""

import pandas as pd
from typing import List, Dict, Any
from part1 import load_and_merge_data


def _ensure_track_name(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize track name column to 'track_name' if suffixed after merge."""
    if 'track_name' in df.columns:
        return df
    if 'track_name_y' in df.columns:
        df = df.rename(columns={'track_name_y': 'track_name'})
    elif 'track_name_x' in df.columns:
        df = df.rename(columns={'track_name_x': 'track_name'})
    return df


# Cached global stats for reuse across recommenders
_SONG_STATS_CACHE = None

def _get_song_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a cached DataFrame of song statistics (count, average rating, etc.).
    Computes it once and stores it in _SONG_STATS_CACHE.
    """
    global _SONG_STATS_CACHE
    if _SONG_STATS_CACHE is not None:
        return _SONG_STATS_CACHE

    df = _ensure_track_name(df)
    # Group by song and keep artist/track for display
    stats = df.groupby(['song_id', 'track_name', 'primary_artist_name']).agg(
        rating_count=('rating', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index()
    
    # Precompute popularity score
    stats['pop_score'] = stats['rating_count'] * stats['avg_rating']
    
    _SONG_STATS_CACHE = stats
    return stats

def recommend_popular(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Recommends songs using the track metadata popularity (track_popularity from tracks.csv).
    Fallback to rating-derived pop_score if metadata is missing.
    """
    df = _ensure_track_name(df)
    if 'track_popularity' in df.columns:
        # Use track-level popularity directly from tracks.csv
        base = df[['song_id', 'track_name', 'primary_artist_name', 'track_popularity']].drop_duplicates('song_id')
        return base.sort_values('track_popularity', ascending=False).head(n)
    # Fallback: use rating-derived popularity if track_popularity not available
    song_stats = _get_song_stats(df)
    return song_stats.sort_values('pop_score', ascending=False).head(n)


def recommend_personalized(df: pd.DataFrame, user_ratings: List[Dict[str, Any]], n: int = 5, candidates_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Personalized recommender: blend popularity, feature fit, and artist bias.
    """
    if not user_ratings:
        return recommend_popular(df, n)

    df = _ensure_track_name(df)
    user_df = pd.DataFrame(user_ratings)
    if user_df.empty:
        return recommend_popular(df, n)

    liked = user_df[user_df['rating'] >= 4]
    if liked.empty:
        liked = user_df

    total_count = len(user_df)
    confidence = min(1.0, total_count / 10.0)
    success_count = len(user_df[user_df['rating'] >= 4])
    p_hat = success_count / total_count if total_count else 0.1
    is_picky = p_hat < 0.2

    feature_cols = [
        'ab_genre_rosamerica_value', 'ab_danceability_value',
        'ab_mood_happy_value', 'ab_mood_party_value',
        'ab_mood_relaxed_value', 'ab_timbre_value'
    ]

    if candidates_df is not None:
        candidates = candidates_df.copy()
    else:
        track_features = df[['song_id', 'track_name', 'primary_artist_name', *feature_cols]].drop_duplicates('song_id')
        stats = _get_song_stats(df)
        candidates = stats.merge(track_features, on=['song_id', 'track_name', 'primary_artist_name'], how='left')

    rated_ids = set(user_df['song_id'])
    candidates = candidates[~candidates['song_id'].isin(rated_ids)]
    if candidates.empty:
        return recommend_popular(df, n)

    user_mean = user_df['rating'].mean()

    # Artist bias: relative to user's mean rating
    history_tracks = df[df['song_id'].isin(user_df['song_id'])].drop(columns=['rating'], errors='ignore')
    history_tracks = history_tracks.merge(user_df[['song_id', 'rating']], on='song_id', how='inner')
    
    artist_means = history_tracks.groupby('primary_artist_name')['rating'].mean()
    artist_bias_map = artist_means - user_mean
    candidates['artist_bias'] = candidates['primary_artist_name'].map(artist_bias_map).fillna(0.0)

    # Feature preference scores (weighted by how discriminative each feature is)
    candidates['feat_score_sum'] = 0.0
    candidates['feat_weight_sum'] = 0.0

    for col in feature_cols:
        if col not in candidates.columns:
            continue
            
        if col not in history_tracks.columns:
             continue
             
        pref_series = history_tracks.groupby(col)['rating'].mean() - user_mean
        
        if len(pref_series) > 1:
            span = pref_series.max() - pref_series.min()
            w = min(1.0, max(0.0, span / 2.0)) * confidence
        else:
            w = 0.2 * confidence

        if w <= 0:
            continue
            
        mapped_pref = candidates[col].map(pref_series).fillna(0.0)
        candidates['feat_score_sum'] += mapped_pref * w
        candidates['feat_weight_sum'] += w

    candidates['feat_score'] = candidates['feat_score_sum'] / candidates['feat_weight_sum'].replace(0, 1)

    pop_min = candidates['pop_score'].min()
    pop_max = candidates['pop_score'].max()
    pop_range = pop_max - pop_min if pop_max != pop_min else 1.0
    candidates['pop_norm'] = (candidates['pop_score'] - pop_min) / pop_range

    pop_w = 0.6 if is_picky else 0.4
    feat_w = 0.3 if is_picky else 0.5
    artist_w = 0.1

    candidates['personal_score'] = (
        pop_w * candidates['pop_norm'] +
        feat_w * candidates['feat_score'] +
        artist_w * candidates['artist_bias']
    )

    # Light diversity: limit to top 2 per artist
    candidates['artist_rank'] = candidates.groupby('primary_artist_name')['personal_score'].rank(ascending=False, method='first')
    trimmed = candidates[candidates['artist_rank'] <= 2]

    recs = trimmed.sort_values(['personal_score', 'avg_rating'], ascending=False).head(n)
    return recs

def main():
    print("Part 3: Recommender Design Implementation")
    
    # Load Data
    full_df = _ensure_track_name(load_and_merge_data('data/user_ratings.csv', 'data/tracks.csv'))

    # Test Popularity Recommender
    recs = recommend_popular(full_df, n=5)
    print("\n--- Popularity Recommender ---")
    print(recs.head())

    # Test Personalized Recommender
    random_user = full_df['user_id'].sample(n=1).iloc[0]
    print(f"\nSelecting random user: {random_user}")
    
    user_ratings_df = full_df[full_df['user_id'] == random_user]
    user_history = user_ratings_df.to_dict('records')
    
    recs = recommend_personalized(full_df, user_history, n=5)
    print("\n--- Personalized Recommender ---")
    print(recs[['song_id', 'track_name', 'primary_artist_name', 'personal_score']])
    
if __name__ == "__main__":
    main()
