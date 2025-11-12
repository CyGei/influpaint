"""
Functions for analyzing spatial correlation in weekly incidence data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

from influpaint.utils import SeasonAxis
from .data_utils import normalize_samples_shape, get_real_weeks


def compute_spatial_correlation_per_week(data: np.ndarray) -> list[float]:
    """Compute spatial correlation for each week.

    For each week, computes the mean pairwise correlation across all locations.

    Args:
        data: Array of shape (weeks, places) or (samples, weeks, places)

    Returns:
        List of correlation coefficients, one per week
    """
    if data.ndim == 2:
        # Single sample: (weeks, places)
        weeks, places = data.shape
        correlations = []

        for w in range(weeks):
            week_data = data[w, :]
            # Skip if all NaN or not enough variation
            if np.all(np.isnan(week_data)) or np.nanstd(week_data) < 1e-6:
                continue

            # For spatial correlation, we look at correlation across locations for this week
            # Since we have a single time point, we can't compute correlation in the traditional sense
            # Instead, we'll use the approach of computing correlation with the mean
            # or we can collect pairs of locations

            # Alternative: for a proper spatial correlation, we should compute
            # correlation between location time series up to this point
            # But the user wants "weekly incidence correlation" which suggests
            # correlation of incidence values across space for each week

            # Let's use coefficient of variation or correlation with mean pattern
            # Actually, for spatial correlation, we want correlation between locations
            # across time. Let me rethink this.
            pass

    elif data.ndim == 3:
        # Multiple samples: (samples, weeks, places)
        pass

    return correlations


def compute_weekly_incidence_correlation(inv_samples: np.ndarray,
                                         season_axis: SeasonAxis) -> list[float]:
    """Compute weekly incidence correlation across states.

    For each pair of states, compute the temporal correlation of their time series.
    This measures spatial synchrony - how correlated different locations are with each other.

    Args:
        inv_samples: (N, 1, weeks, places) or (N, weeks, places)
        season_axis: SeasonAxis object

    Returns:
        List of correlation coefficients (one per state pair per sample)
    """
    arr = normalize_samples_shape(inv_samples)
    n, c, w, p = arr.shape
    real_weeks = get_real_weeks(arr)

    correlations = []

    # For each sample
    for sample_idx in range(n):
        sample_data = arr[sample_idx, 0, :real_weeks, :]  # (weeks, places)

        # Compute pairwise state correlations (temporal correlation between states)
        for state1 in range(p):
            for state2 in range(state1 + 1, p):
                ts1 = sample_data[:, state1]
                ts2 = sample_data[:, state2]

                # Remove NaN values
                valid = ~(np.isnan(ts1) | np.isnan(ts2))
                if np.sum(valid) < 3:  # Need at least 3 time points
                    continue

                t1 = ts1[valid]
                t2 = ts2[valid]

                # Check for sufficient variation
                if np.std(t1) < 1e-6 or np.std(t2) < 1e-6:
                    continue

                # Compute correlation
                corr = np.corrcoef(t1, t2)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

    return correlations


def compute_random_correlation(inv_samples: np.ndarray,
                               season_axis: SeasonAxis,
                               n_permutations: int = 100) -> list[float]:
    """Compute expected correlation at random by permuting time series.

    Randomly shuffles each state's time series independently to break temporal structure
    while preserving marginal distributions.

    Args:
        inv_samples: (N, 1, weeks, places) or (N, weeks, places)
        season_axis: SeasonAxis object
        n_permutations: Number of random permutations

    Returns:
        List of correlation coefficients from permuted data
    """
    arr = normalize_samples_shape(inv_samples)
    n, c, w, p = arr.shape
    real_weeks = get_real_weeks(arr)

    correlations = []

    for perm_idx in range(n_permutations):
        # Randomly select a sample
        sample_idx = np.random.randint(0, n)
        sample_data = arr[sample_idx, 0, :real_weeks, :].copy()

        # Permute each state's time series independently
        for state_idx in range(p):
            perm = np.random.permutation(real_weeks)
            sample_data[:, state_idx] = sample_data[perm, state_idx]

        # Compute pairwise state correlations on permuted data
        for state1 in range(p):
            for state2 in range(state1 + 1, p):
                ts1 = sample_data[:, state1]
                ts2 = sample_data[:, state2]

                valid = ~(np.isnan(ts1) | np.isnan(ts2))
                if np.sum(valid) < 3:
                    continue

                t1 = ts1[valid]
                t2 = ts2[valid]

                if np.std(t1) < 1e-6 or np.std(t2) < 1e-6:
                    continue

                corr = np.corrcoef(t1, t2)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

    return correlations


def compute_observed_correlation(season_axis: SeasonAxis,
                                 n_seasons: int = 3) -> list[float]:
    """Compute correlation from historical observed data.

    For each pair of states, compute the temporal correlation of their time series
    across the most recent seasons.

    Args:
        season_axis: SeasonAxis object
        n_seasons: Number of historical seasons to analyze

    Returns:
        List of correlation coefficients from observed data
    """
    gt_df = pd.read_csv('influpaint/data/nhsn_flusight_past.csv')

    correlations = []

    # Get the most recent n_seasons (excluding 2021-2022 season due to incomplete data)
    all_seasons = sorted(gt_df['fluseason'].unique())
    all_seasons = [s for s in all_seasons if s != 2021]  # Exclude 2021-2022 season
    seasons = all_seasons[-n_seasons:]

    for season_year in seasons:
        season_data = gt_df[gt_df['fluseason'] == season_year]
        season_pivot = season_data.pivot(columns='location_code', values='value', index='season_week')

        # Get list of locations
        locations = season_pivot.columns.tolist()

        # Compute pairwise state correlations
        for i, loc1 in enumerate(locations):
            for j in range(i + 1, len(locations)):
                loc2 = locations[j]

                ts1 = season_pivot[loc1].values
                ts2 = season_pivot[loc2].values

                valid = ~(np.isnan(ts1) | np.isnan(ts2))
                if np.sum(valid) < 3:
                    continue

                t1 = ts1[valid]
                t2 = ts2[valid]

                if np.std(t1) < 1e-6 or np.std(t2) < 1e-6:
                    continue

                corr = np.corrcoef(t1, t2)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

    return correlations


def plot_weekly_incidence_correlation(inv_samples: np.ndarray,
                                      season_axis: SeasonAxis,
                                      save_path: Optional[str] = None,
                                      n_permutations: int = 100) -> plt.Figure:
    """Plot weekly incidence correlation comparison.

    Creates box plots comparing:
    1. Expected correlation at random (permuted data)
    2. Correlation with influpaint (generated samples)
    3. Observed correlation in historical seasons

    Args:
        inv_samples: (N, 1, weeks, places) or (N, weeks, places)
        season_axis: SeasonAxis object
        save_path: Optional path to save the figure
        n_permutations: Number of random permutations for null distribution

    Returns:
        matplotlib Figure object
    """
    print("Computing random correlations...")
    random_corr = compute_random_correlation(inv_samples, season_axis, n_permutations)

    print("Computing influpaint correlations...")
    influpaint_corr = compute_weekly_incidence_correlation(inv_samples, season_axis)

    print("Computing observed correlations...")
    observed_corr = compute_observed_correlation(season_axis)

    # Prepare data for box plot
    data = []
    for corr in random_corr:
        data.append({'Category': 'Expected at random', 'Correlation': corr})
    for corr in influpaint_corr:
        data.append({'Category': 'Influpaint', 'Correlation': corr})
    for corr in observed_corr:
        data.append({'Category': 'Observed', 'Correlation': corr})

    df = pd.DataFrame(data)

    # Create figure with vertical box plots
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

    # Create vertical box plot
    sns.boxplot(data=df, x='Category', y='Correlation', ax=ax,
                order=['Expected at random', 'Influpaint', 'Observed'],
                palette=['lightgray', 'skyblue', 'salmon'])

    # Add individual points with high alpha (very transparent)
    sns.stripplot(data=df, x='Category', y='Correlation', ax=ax,
                  order=['Expected at random', 'Influpaint', 'Observed'],
                  color='black', alpha=0.05, size=2, jitter=True)

    ax.set_xlabel('', fontsize=13)
    ax.set_ylabel('Weekly incidence correlation (across states)', fontsize=13)
    ax.set_title('Spatial correlation of influenza incidence', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    sns.despine(ax=ax, trim=True)

    # Add statistics
    for i, category in enumerate(['Expected at random', 'Influpaint', 'Observed']):
        cat_data = df[df['Category'] == category]['Correlation']
        median = cat_data.median()
        mean = cat_data.mean()
        ax.text(i, ax.get_ylim()[1] * 0.95, f'Med: {median:.3f}\nMean: {mean:.3f}',
                ha='center', va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', pad=2))

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved correlation figure to {save_path}")

    return fig
