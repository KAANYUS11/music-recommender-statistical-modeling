# Part 4: Monte Carlo Evaluation (Popular vs Personalized)
"""
Offline replay using real user ratings to compare Popular and Personalized recommenders.
"""

import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple

from part1 import load_and_merge_data
from part3 import _ensure_track_name, _get_song_stats, recommend_popular, recommend_personalized


def _prepare_user_sequences(
    df: pd.DataFrame,
    min_ratings: int = 5,
    require_five_star: bool = False
) -> Dict[str, pd.DataFrame]:
    """Collect per-user sequences sorted by round_idx."""
    users: Dict[str, pd.DataFrame] = {}
    for uid, grp in df.groupby('user_id'):
        if len(grp) < min_ratings:
            continue
        if require_five_star and not (grp['rating'] == 5).any():
            continue
        users[uid] = grp.sort_values('round_idx')
    return users


def _build_history(df_user: pd.DataFrame, k: int = 3) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Build the seed history (prefer non-5★) and return remaining pool."""
    df_user = df_user.reset_index(drop=True)

    non_fives = df_user[df_user['rating'] != 5].index.tolist()
    seed_idxs = non_fives[:k]
    if len(seed_idxs) < k:
        extras = [i for i in range(len(df_user)) if i not in seed_idxs]
        seed_idxs.extend(extras[:k - len(seed_idxs)])

    remaining_idxs = [i for i in range(len(df_user)) if i not in seed_idxs]
    seed = df_user.iloc[seed_idxs]
    remaining = df_user.iloc[remaining_idxs]

    history = [
        {'song_id': row.song_id, 'rating': row.rating, 'track_name': row.track_name}
        for _, row in seed.iterrows()
    ]
    return history, remaining


def _recommend(
    model: str,
    df: pd.DataFrame,
    history: List[Dict[str, Any]],
    n: int = 1,
    candidates_df: pd.DataFrame = None
) -> pd.DataFrame:
    """Dispatch to the correct recommender."""
    if model == 'Popular':
        return recommend_popular(df, n=n)
    if model == 'Personalized':
        subset = None
        if candidates_df is not None:
            subset = candidates_df[candidates_df['song_id'].isin(df['song_id'])]
        return recommend_personalized(df, history, n=n, candidates_df=subset)
    return pd.DataFrame()


def _evaluate_user(
    uid: str,
    users_dict: Dict[str, pd.DataFrame],
    full_df: pd.DataFrame,
    model: str,
    topk: int = 1,
    max_batches: int = 200,
    candidates_df: pd.DataFrame = None
) -> Dict[str, float]:
    """
    Replay a single user's ratings. Recommendations are filtered to songs this user rated
    (so we have ground-truth scores).
    """
    df_user = users_dict[uid].copy()
    history, remaining = _build_history(df_user, k=3)
    if remaining.empty:
        return {'hit_at_k': 0.0, 'avg_rating': 0.0, 'time_to_5': 0.0}

    unseen_ids = set(remaining['song_id'])
    rating_map = remaining.set_index('song_id')['rating'].to_dict()

    total_ratings: List[int] = []
    songs_seen = 0
    time_to_5 = None
    hit_at_k = False

    batches = 0
    while unseen_ids and batches < max_batches:
        candidate_df = full_df[full_df['song_id'].isin(unseen_ids)]
        recs = _recommend(model, candidate_df, history, n=topk, candidates_df=candidates_df).head(topk)
        if recs.empty:
            break

        batch_ratings: List[int] = []
        for _, row in recs.iterrows():
            sid = row['song_id']
            r = rating_map.get(sid)
            if r is None:
                continue
            batch_ratings.append(r)
            history.append({'song_id': sid, 'rating': r, 'track_name': row['track_name']})
            unseen_ids.discard(sid)
            songs_seen += 1
            if r == 5:
                if time_to_5 is None:
                    time_to_5 = songs_seen
                if songs_seen <= 3:
                    hit_at_k = True
                break

        total_ratings.extend(batch_ratings)
        if time_to_5 is not None:
            break
        batches += 1

    avg_rating = float(np.mean(total_ratings)) if total_ratings else 0.0
    return {
        'hit_at_k': float(hit_at_k),
        'avg_rating': avg_rating,
        'time_to_5': float(time_to_5) if time_to_5 is not None else 0.0,
    }


def evaluate_models(
    topk: int = 1,
    n_trials: int = 3000,
    ratings_path: str = 'data/user_ratings.csv',
    require_five_star: bool = False
) -> None:
    print(f"Loading data from {ratings_path}...")
    full_df = _ensure_track_name(load_and_merge_data(ratings_path, 'data/tracks.csv'))
    _get_song_stats(full_df)  # precompute cache for recommenders

    users = _prepare_user_sequences(full_df, min_ratings=5, require_five_star=require_five_star)
    real_user_ids = list(users.keys())
    print(f"Total real users available: {len(real_user_ids)}")

    # Build a global candidate table for personalized scoring
    feature_cols = [
        'ab_genre_rosamerica_value', 'ab_danceability_value',
        'ab_mood_happy_value', 'ab_mood_party_value',
        'ab_mood_relaxed_value', 'ab_timbre_value'
    ]
    feat_cols_present = [c for c in feature_cols if c in full_df.columns]
    song_stats = _get_song_stats(full_df)
    cols_to_select = ['song_id', 'track_name', 'primary_artist_name'] + feat_cols_present
    track_features = full_df[cols_to_select].drop_duplicates('song_id')
    candidates_df = song_stats.merge(
        track_features,
        on=['song_id', 'track_name', 'primary_artist_name'],
        how='left'
    )

    rng = np.random.default_rng(42)
    sampled_user_ids = rng.choice(real_user_ids, size=n_trials, replace=True)

    models = ['Popular', 'Personalized']
    results = {m: [] for m in models}

    print(f"Starting sequential evaluation for {n_trials} trials...")
    for m in models:
        print(f"Evaluating model: {m}")
        model_results = []
        for i, uid in enumerate(sampled_user_ids):
            if n_trials >= 10 and i % (n_trials // 10) == 0:
                print(f"  Progress: {i}/{n_trials}")
            res = _evaluate_user(uid, users, full_df, m, topk, 20, candidates_df=candidates_df)
            model_results.append(res)
        results[m] = model_results

    metric_names = ['hit_at_k', 'avg_rating', 'time_to_5']
    print(f"\nMonte Carlo Evaluation Results (N={n_trials} trials)")
    header = f"{'Metric':<12} | " + " | ".join([f"{m:<18}" for m in models])
    print(header)
    print("-" * len(header))

    for metric in metric_names:
        row = [f"{metric:<12}"]
        for m in models:
            vals = [r[metric] for r in results[m]]
            mu = np.mean(vals)
            sem = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            ci = 1.96 * sem
            row.append(f"{mu:.3f} ± {ci:.3f}")
        print(" | ".join(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline Monte Carlo recommender evaluation.")
    parser.add_argument("--topk", type=int, default=1, help="Number of recommendations evaluated per batch.")
    parser.add_argument("--n-trials", type=int, default=2804, help="Number of sampled user replay trials.")
    parser.add_argument(
        "--ratings-path",
        default="data/enlarged_user_ratings.csv",
        help="Ratings CSV used for the evaluation replay.",
    )
    parser.add_argument(
        "--include-users-without-five-star",
        action="store_true",
        help="Include users who never gave a 5-star rating.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_models(
        topk=args.topk,
        ratings_path=args.ratings_path,
        require_five_star=not args.include_users_without_five_star,
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    main()
