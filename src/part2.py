# Part 2: User Variability Modeling

"""
Part 2: User Variability Modeling
Functional implementation for analyzing user recommendation needs.
"""

import pandas as pd
import numpy as np
import math

from part1 import load_and_merge_data

# ---------------------------------------------------------
# Core Math Helpers 
# ---------------------------------------------------------

def lgamma_vec(x):
    """Vectorized log-gamma function."""
    return np.array([math.lgamma(val) for val in np.atleast_1d(x)])

def minimize(fun, x0, args=(), bounds=None, tol=1e-5, max_iter=1000):
    """
    Minimization function for 2 parameters (alpha, beta).
    Uses an adaptive step coordinate descent approach.
    """
    current_params = np.array(x0, dtype=float)
    current_score = fun(current_params, *args)
    
    step_sizes = np.array([0.1, 0.1])
    
    for _ in range(max_iter):
        improved = False
        for i in range(len(current_params)):
            # Try increasing
            test_params = current_params.copy()
            test_params[i] += step_sizes[i]
            
            # Check bounds
            if bounds:
                lower, upper = bounds[i]
                if (lower is not None and test_params[i] < lower) or \
                   (upper is not None and test_params[i] > upper):
                    pass # Skip out of bounds
                else:
                    score = fun(test_params, *args)
                    if score < current_score:
                        current_score = score
                        current_params = test_params
                        improved = True
                        step_sizes[i] *= 1.2 # Grow step
                        continue

            # Try decreasing
            test_params = current_params.copy()
            test_params[i] -= step_sizes[i]
             # Check bounds
            if bounds:
                lower, upper = bounds[i]
                if (lower is not None and test_params[i] < lower) or \
                   (upper is not None and test_params[i] > upper):
                    pass 
                else:
                    score = fun(test_params, *args)
                    if score < current_score:
                        current_score = score
                        current_params = test_params
                        improved = True
                        step_sizes[i] *= 1.2
                        continue
            
            # If no improvement in this direction, shrink step
            step_sizes[i] *= 0.5
            
        if np.max(step_sizes) < tol:
            break
            
    return type('OptimizeResult', (object,), {'x': current_params, 'success': True, 'message': 'Converged'})

def std_normal_cdf(x):
    """CDF of standard normal distribution."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def t_test_ind(a, b):
    """
    Welch's t-test implementation.
    Returns (t_stat, p_value).
    """
    a = np.array(a)
    b = np.array(b)
    n1, n2 = len(a), len(b)
    m1, m2 = np.mean(a), np.mean(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    
    # t-statistic
    denom = np.sqrt(v1/n1 + v2/n2)
    if denom == 0:
        return 0.0, 1.0
    t_stat = (m1 - m2) / denom
    
    # Degrees of freedom (Welch-Satterthwaite)
    num_df = (v1/n1 + v2/n2)**2
    den_df = (v1/n1)**2 / (n1 - 1) + (v2/n2)**2 / (n2 - 1)
    df = num_df / den_df
    
    # p-value using Normal approximation (as fallback for complex t-dist integration)
    # For df > 30, T distribution is very close to Normal.
    # If df is small, this is an approximation. Given dataset size (~50), it is acceptable.
    # Using Normal CDF for 2-sided test:
    p_val = 2 * (1 - std_normal_cdf(abs(t_stat)))
    
    return t_stat, p_val

def mannwhitneyu(x, y, use_continuity=True):
    """
    Mann-Whitney U test implementation.
    Returns (u_stat, p_value).
    """
    x = np.array(x)
    y = np.array(y)
    n1, n2 = len(x), len(y)
    
    # Rank data
    combined = np.concatenate([x, y])
    ranks = pd.Series(combined).rank().values
    r1 = ranks[:n1]
    
    u1 = np.sum(r1) - n1*(n1+1)/2
    u2 = n1*n2 - u1
    
    u_stat = u1 
    
    # Normal approximation parameters
    mu_u = n1 * n2 / 2
    sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    
    if sigma_u == 0:
        z = 0
    else:
        # Continuity correction
        numerator = u_stat - mu_u
        if use_continuity:
            if numerator > 0:
                numerator -= 0.5
            elif numerator < 0:
                numerator += 0.5
                
        z = numerator / sigma_u
        
    p_val = 2 * (1 - std_normal_cdf(abs(z))) # 2-sided
    
    return u_stat, p_val

def dagostino_k2_test(x):
    """
    D'Agostino's K^2 test for normality.
    Combines skewness and kurtosis tests.
    Returns (statistic, p_value).
    """
    x = np.array(x)
    n = len(x)
    if n < 8:
        # D'Agostino test requires n >= 8 (approx) for validity
        return 0.0, 1.0 # Fail to reject
        
    # S = Skewness, K = Kurtosis (Excess)
    # We need raw moments for formulas
    m = np.mean(x)
    s = np.std(x, ddof=0) # use population std for Pearson skew/kurt definitions used in test
    
    if s == 0:
        return 0.0, 1.0
        
    # Skewness (b1^(1/2) in literature)
    g1 = np.mean(((x - m) / s)**3)
    
    # Kurtosis (b2 in literature, regular kurtosis, normal=3)
    b2 = np.mean(((x - m) / s)**4)
    
    # --- Transform Skewness (Z1) ---
    y = g1 * math.sqrt(((n + 1) * (n + 3)) / (6 * (n - 2)))
    beta2_g1 = (3 * (n**2 + 27*n - 70) * (n + 1) * (n + 3)) / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
    w2 = -1 + math.sqrt(2 * (beta2_g1 - 1))
    
    delta = 1 / math.sqrt(math.log(math.sqrt(w2)))
    alpha = math.sqrt(2 / (w2 - 1))
    
    # Z1 = delta * ln(Y/alpha + sqrt((Y/alpha)^2 + 1))
    term = y / alpha
    z1 = delta * math.log(term + math.sqrt(term**2 + 1))
    
    # --- Transform Kurtosis (Z2) ---
    # Mean and Var of b2
    e_b2 = (3 * (n - 1)) / (n + 1)
    var_b2 = (24 * n * (n - 2) * (n - 3)) / ((n + 1)**2 * (n + 3) * (n + 5))
    
    x_val = (b2 - e_b2) / math.sqrt(var_b2)
    
    root_beta1_b2 = ((6 * (n**2 - 5*n + 2)) / ((n + 7) * (n + 9))) * math.sqrt((6 * (n + 3) * (n + 5)) / (n * (n - 2) * (n - 3)))
    
    a = 6 + (8 / root_beta1_b2) * ((2 / root_beta1_b2) + math.sqrt(1 + 4 / (root_beta1_b2**2)))
    
    val_for_cbrt = (1 - 2/a) / (1 + x_val * math.sqrt(2 / (a - 4)))
    # Cube root handling for neg numbers if needed (though typically correct domain)
    if val_for_cbrt < 0:
        cbrt_term = -((-val_for_cbrt)**(1/3))
    else:
        cbrt_term = val_for_cbrt**(1/3)
        
    z2 = (1 - 2/(9*a) - cbrt_term) / math.sqrt(2/(9*a))
    
    # --- K^2 Statistic ---
    k2 = z1**2 + z2**2
    
    # p-value (Chi-squared with 2 dof)
    p_val = math.exp(-k2 / 2)
    
    return type('TestResult', (object,), {'statistic': k2, 'pvalue': p_val})

def bg_neg_log_likelihood(params: tuple[float, float], tu_values: np.ndarray) -> float:
    """
    Calculates the negative log-likelihood for the Beta-Geometric distribution.
    """
    alpha, beta = params
    if alpha <= 0 or beta <= 0:
        return np.inf
    
    n = len(tu_values)
    
    # Log Likelihood Calculation using lgamma_vec
    # log P(T=t) = ln B(alpha+1, beta+t-1) - ln B(alpha, beta)
    
    # Note: loggamma(alpha) + loggamma(beta) - loggamma(alpha+beta) is ln B(alpha, beta)
    
    # Vectorized calculation
    term1 = lgamma_vec(alpha + 1)
    term2 = lgamma_vec(beta + tu_values - 1)
    term3 = lgamma_vec(alpha + beta + tu_values)
    
    ll_sum = np.sum(term1 + term2 - term3)
    
    ll_denom = n * (lgamma_vec(alpha)[0] + lgamma_vec(beta)[0] - lgamma_vec(alpha + beta)[0])
    
    log_likelihood = ll_sum - ll_denom
    return -log_likelihood

# ---------------------------------------------------------
# Analysis Functions
# ---------------------------------------------------------

def calculate_time_to_favorite(df: pd.DataFrame) -> pd.Series:
    """
    Calculates Tu: The number of trials (recommendations) until the first 5* rating.
    Users who never give a 5* rating are excluded.
    """
    print("\n--- Calculating Time-to-Favorite (Tu) ---")
    
    # Sort by user and round to ensure correct order
    df_sorted = df.sort_values(['user_id', 'round_idx'])
    
    # Filter for 5* ratings
    hits = df_sorted[df_sorted['rating'] == 5].groupby('user_id').first()
    
    # Tu is the number of trials, so it should be round_idx + 1 (assuming round_idx starts at 0)
    tu_series = hits['round_idx'] + 1
    
    print(f"Found {len(tu_series)} users with at least one 5* rating.")
    return tu_series

def fit_geometric_model(tu_series: pd.Series) -> float:
    """
    Estimates parameter p using the method of moments (1 / mean(Tu)).
    Prints the estimated p.
    """
    print("\n--- Fitting Geometric Model ---")
    mean_tu = tu_series.mean()
    p_hat = 1 / mean_tu
    print(f"Mean Tu: {mean_tu:.4f}")
    print(f"Estimated p (Geometric): {p_hat:.4f}")
    return p_hat

def fit_beta_geometric_model(tu_series: pd.Series) -> tuple[float, float]:
    """
    Estimates alpha and beta using Maximum Likelihood Estimation.
    Prints the estimated parameters.
    """
    print("\n--- Fitting Beta-Geometric Model ---")
    
    initial_params = [1.0, 1.0]
    
    # Use minimize to find the optimal parameters
    result = minimize(
        bg_neg_log_likelihood, 
        initial_params, 
        args=(tu_series.values,),
        bounds=[(0.001, None), (0.001, None)], # alpha, beta > 0
    )
    
    if result.success:
        alpha, beta = result.x
        print(f"Estimated alpha: {alpha:.4f}")
        print(f"Estimated beta: {beta:.4f}")
        return alpha, beta
    else:
        print("Optimization failed:", result.message)
        return None, None

def perform_hypothesis_test(group1: pd.Series, group2: pd.Series, group1_name: str = "Group A", group2_name: str = "Group B") -> None:
    """
    Compares Time-to-Favorite between two groups.
    Decides between t-test and Mann-Whitney U test based on normality.
    """
    print(f"\n--- Hypothesis Testing: {group1_name} vs {group2_name} ---")
    
    # 1. Clean Data (Remove NaNs)
    g1 = group1.dropna()
    g2 = group2.dropna()

    if len(g1) < 2 or len(g2) < 2:
        print("One or both groups have insufficient data. Cannot perform test.")
        return

    # Basic Statistics
    print(f"Mean {group1_name}: {g1.mean():.4f} (n={len(g1)})")
    print(f"Mean {group2_name}: {g2.mean():.4f} (n={len(g2)})")

    # 2. Check Normality Assumption (D'Agostino's K^2)
    k2_g1 = dagostino_k2_test(g1)
    k2_g2 = dagostino_k2_test(g2)
    
    # If p-value > 0.05, we fail to reject null hypothesis -> Assume Normal
    is_normal = (k2_g1.pvalue > 0.05) and (k2_g2.pvalue > 0.05)
    
    # 3. Apply Appropriate Test
    if is_normal:
        print("\n>> Assumption Check: Data is approximately Normal. Using t-Test.")
        stat, p_val = t_test_ind(g1, g2)
        test_name = "t-test"
    else:
        print("\n>> Assumption Check: Data is Skewed (Non-Normal). Using Mann-Whitney U.")
        stat, p_val = mannwhitneyu(g1, g2)
        test_name = "Mann-Whitney U"

    # 4. Results & Interpretation
    print(f"{test_name} Result: Stat={stat:.4f}, p-value={p_val:.4e}")
    
    alpha = 0.05
    if p_val < alpha:
        print(f">> Result: Significant difference found (p <= 0.05)")
    else:
        print(f">> Result: No significant difference found (p > 0.05)")

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def main():
    # 1. Load Data
    full_df = load_and_merge_data('data/user_ratings.csv', 'data/tracks.csv')
    
    # -----------------------------------------------------
    # Step 2: Calculate Tu
    # -----------------------------------------------------
    tu_series = calculate_time_to_favorite(full_df)
    
    if tu_series.empty:
        print("No users found with 5* ratings. Exiting.")
        return

    # -----------------------------------------------------
    # Step 3: Geometric Model
    # -----------------------------------------------------
    fit_geometric_model(tu_series)
    
    # -----------------------------------------------------
    # Step 4: Beta-Geometric Model
    # -----------------------------------------------------
    fit_beta_geometric_model(tu_series)
    
    # -----------------------------------------------------
    # Step 5: Hypothesis Testing
    # -----------------------------------------------------
    
    # Standart Task: Define groups by preferred release year (Recent (>= 2010) vs Vintage (< 2010) Songs)
    
    # Get 5* ratings to determine user preference
    five_star_ratings = full_df[full_df['rating'] == 5]
    
    # Calc avg release year per user for their favorites
    user_avg_year = five_star_ratings.groupby('user_id')['album_release_year'].mean()
    
    # Split users using median year
    median_year = user_avg_year.median()
    print(f"\nDefining Groups by Preferred Release Year (Median split: {median_year})")
    
    recent_lovers_ids = user_avg_year[user_avg_year >= median_year].index
    vintage_lovers_ids = user_avg_year[user_avg_year < median_year].index
    
    group_recent = tu_series[tu_series.index.isin(recent_lovers_ids)]
    group_vintage = tu_series[tu_series.index.isin(vintage_lovers_ids)]
    
    perform_hypothesis_test(group_recent, group_vintage, "Recent Lovers", "Vintage Lovers")

if __name__ == "__main__":
    main()