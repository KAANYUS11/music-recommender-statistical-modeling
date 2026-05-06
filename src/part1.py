# Part 1: Conditional Probability Modeling

"""
Part 1: Conditional Probability Modeling
Modularized for flexible queries (e.g., P(Year | 3*), P(Artist | 5*), etc.)
"""

import pandas as pd

# ---------------------------------------------------------
# Core Math Helpers
# ---------------------------------------------------------

def laplace_smooth(hits: pd.Series, total_counts: pd.Series, alpha: int, k: int) -> pd.Series:
    """
    Helper function to calculate the smoothed probability.
    Formula: (hits + alpha) / (total_counts + alpha * k)
    """
    return (hits + alpha) / (total_counts + (alpha * k))

def calculate_conditional_prob(df: pd.DataFrame, feature_col: str, target_col: str = 'rating', target_val: int = 5, alpha: int = 1, k: int = 2) -> pd.DataFrame:
    """
    Calculates P(Target=target_val | Feature) with Laplace smoothing.
    Returns a DataFrame containing counts, priors, and conditional probs.
    """
    # 1. Group by Feature and Calculate Counts (Denominator: Count(F))
    feature_counts = df[feature_col].value_counts()
    
    # 2. Count occurrences where Target is met (Numerator: Count(Target and F))
    target_hits = df[df[target_col] == target_val][feature_col].value_counts()
    
    # Reindex to ensure indices match, filling missing intersections with 0
    target_hits = target_hits.reindex(feature_counts.index, fill_value=0)
    
    # 3. Apply Laplace Smoothing
    probs = laplace_smooth(target_hits, feature_counts, alpha, k)
    
    # 4. Calculate Prior P(Feature) for Bayes Rule
    p_feature = feature_counts / len(df)
    
    # Create result DataFrame
    results = pd.DataFrame({
        'feature_value': feature_counts.index,
        'count': feature_counts.values,
        'hits': target_hits.values,
        f'P({target_val}*|Feature)': probs.values,
        'P(Feature)': p_feature.values
    })
    
    return results

# ---------------------------------------------------------
# Analysis Functions
# ---------------------------------------------------------

def load_and_merge_data(ratings_path: str, tracks_path: str) -> pd.DataFrame:
    """Loads CSVs and merges them into a single DataFrame."""
    print("\nLoading and merging data...")
    ratings_df = pd.read_csv(ratings_path)
    tracks_df = pd.read_csv(tracks_path)
    
    full_df = pd.merge(
        ratings_df, tracks_df, 
        left_on='song_id', right_on='track_id', 
        how='inner'
    )
    print(f"Data Loaded. Combined Rows: {len(full_df)}")
    return full_df

def analyze_feature_prob(df: pd.DataFrame, feature_col: str, target_rating: int = 5, top_n: int = 5) -> None:
    """
    Wrapper to calculate and print top probabilities for a single feature.
    Example: P(5* | Artist) or P(3* | Year)
    """
    print(f"\n--- Analyzing P({target_rating}* | {feature_col}) ---")
    
    if feature_col not in df.columns:
        print(f"Warning: Column '{feature_col}' not found.")
        return None

    probs_df = calculate_conditional_prob(df, feature_col, target_val=target_rating)
    
    col_name = f'P({target_rating}*|Feature)'
    print(probs_df.sort_values(col_name, ascending=False).head(top_n))
    return probs_df

def analyze_interaction(df: pd.DataFrame, col1: str, col2: str, target_rating: int = 5, top_n: int = 5) -> None:
    """
    Combines two columns into an interaction feature and calculates probability.
    Example: P(5* | Mood=Happy + Timbre=Bright)
    """
    print(f"\n--- Analyzing Interaction: {col1} + {col2} ---")
    
    if col1 not in df.columns or col2 not in df.columns:
        print("Warning: One or both interaction columns missing.")
        return None

    # Create temporary interaction column
    interaction_col = f"{col1}_x_{col2}"
    # Ensure values are strings for concatenation
    df[interaction_col] = df[col1].astype(str) + " + " + df[col2].astype(str)
    
    probs_df = calculate_conditional_prob(df, interaction_col, target_val=target_rating)
    
    col_name = f'P({target_rating}*|Feature)'
    print(probs_df.sort_values(col_name, ascending=False).head(top_n))
    return probs_df

def calculate_bayes_inverse(prob_df: pd.DataFrame, df: pd.DataFrame, target_rating: int = 5) -> pd.DataFrame:
    """
    Applies Bayes' Theorem to invert probabilities.
    Calculates P(Feature | Target) = P(Target | Feature) * P(Feature) / P(Target)
    """
    if prob_df is None: return None
    
    # Calculate Global P(Target)
    total_target_hits = len(df[df['rating'] == target_rating])
    p_target_global = total_target_hits / len(df)
    
    if p_target_global == 0:
        print("Global probability of target is 0, cannot invert.")
        return prob_df

    # Copy to avoid SettingWithCopy warnings
    bayes_df = prob_df.copy()
    
    # P(F | T) = (P(T | F) * P(F)) / P(T)
    cond_col = f'P({target_rating}*|Feature)'
    bayes_col = f'P(Feature|{target_rating}*)'
    
    bayes_df[bayes_col] = (bayes_df[cond_col] * bayes_df['P(Feature)']) / p_target_global
    
    print(f"\n--- Bayesian Inversion: P(Feature | {target_rating}*) ---")
    print(bayes_df.sort_values(bayes_col, ascending=False).head(5))
    return bayes_df

def analyze_personal_prob(df: pd.DataFrame, user_id: int, feature_col: str, target_rating: int = 5) -> pd.DataFrame:
    """
    Filters data for a specific user and runs the feature analysis.
    """
    print(f"\n--- Personal Analysis for User: {user_id} ---")
    user_df = df[df['user_id'] == user_id]
    
    if user_df.empty:
        print("User not found or has no history.")
        return None
        
    return analyze_feature_prob(user_df, feature_col, target_rating)


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    # 1. Load Data
    full_df = load_and_merge_data('data/user_ratings.csv', 'data/tracks.csv')
    
    # -----------------------------------------------------
    # Step 2: Global Probabilities 
    # -----------------------------------------------------

    # Standard Task: P(5* | Artist)
    artist_probs = analyze_feature_prob(full_df, 'primary_artist_name', target_rating=5)
    
    # Standard Task: P(5* | Year)
    analyze_feature_prob(full_df, 'album_release_year', target_rating=5)
    
    # Standard Task: P(5* | Explicit)
    analyze_feature_prob(full_df, 'explicit', target_rating=5)
    

    # -----------------------------------------------------
    # Step 3: Feature Interactions
    # -----------------------------------------------------

    # Standard Task: Mood + Timbre
    analyze_interaction(full_df, 'ab_mood_happy_value', 'ab_timbre_value', target_rating=5)

    # -----------------------------------------------------
    # Step 4: Bayesian Interpretation
    # -----------------------------------------------------

    # Standard Task: Invert P(5* | Artist) to get P(Artist | 5*)
    calculate_bayes_inverse(artist_probs, full_df, target_rating=5)
    
   
    # -----------------------------------------------------
    # Step 5: Personal Analysis
    # -----------------------------------------------------
    
    # Grab a sample user
    if 'user_id' in full_df.columns:
        target_user = full_df['user_id'].unique()[0]
        # Analyze P(5* | Artist) for this user
        analyze_personal_prob(full_df, target_user, 'primary_artist_name', target_rating=5)

if __name__ == "__main__":
    main()
