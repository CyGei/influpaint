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
                                         season_axis: SeasonAxis,
                                         method: str = 'temporal') -> list[float]:
    """Compute weekly incidence correlation across space.

    For each pair of weeks, compute the spatial correlation (correlation across locations).
    This measures how similar the spatial pattern is between different weeks.

    Args:
        inv_samples: (N, 1, weeks, places) or (N, weeks, places)
        season_axis: SeasonAxis object
        method: 'temporal' for temporal correlation, 'spatial' for spatial

    Returns:
        List of correlation coefficients
    """
    arr = normalize_samples_shape(inv_samples)
    n, c, w, p = arr.shape
    real_weeks = get_real_weeks(arr)

    correlations = []

    # For each sample
    for sample_idx in range(n):
        sample_data = arr[sample_idx, 0, :real_weeks, :]  # (weeks, places)

        # Compute pairwise weekly correlations (spatial correlation between weeks)
        for week1 in range(real_weeks):
            for week2 in range(week1 + 1, real_weeks):
                pattern1 = sample_data[week1, :]
                pattern2 = sample_data[week2, :]

                # Remove NaN values
                valid = ~(np.isnan(pattern1) | np.isnan(pattern2))
                if np.sum(valid) < 3:  # Need at least 3 points
                    continue

                p1 = pattern1[valid]
                p2 = pattern2[valid]

                # Check for sufficient variation
                if np.std(p1) < 1e-6 or np.std(p2) < 1e-6:
                    continue

                # Compute correlation
                corr = np.corrcoef(p1, p2)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

    return correlations


def compute_random_correlation(inv_samples: np.ndarray,
                               season_axis: SeasonAxis,
                               n_permutations: int = 100) -> list[float]:
    """Compute expected correlation at random by permuting locations.

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

    # Take a subset of samples to permute
    n_samples = min(n, 50)  # Limit to avoid too many computations

    for perm_idx in range(n_permutations):
        # Randomly select a sample
        sample_idx = np.random.randint(0, n)
        sample_data = arr[sample_idx, 0, :real_weeks, :].copy()

        # Permute locations independently for each week
        for week in range(real_weeks):
            perm = np.random.permutation(p)
            sample_data[week, :] = sample_data[week, perm]

        # Compute correlations on permuted data
        for week1 in range(real_weeks):
            for week2 in range(week1 + 1, real_weeks):
                pattern1 = sample_data[week1, :]
                pattern2 = sample_data[week2, :]

                valid = ~(np.isnan(pattern1) | np.isnan(pattern2))
                if np.sum(valid) < 3:
                    continue

                p1 = pattern1[valid]
                p2 = pattern2[valid]

                if np.std(p1) < 1e-6 or np.std(p2) < 1e-6:
                    continue

                corr = np.corrcoef(p1, p2)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

    return correlations


def compute_observed_correlation(season_axis: SeasonAxis,
                                 n_seasons: int = 3) -> list[float]:
    """Compute correlation from historical observed data.

    Args:
        season_axis: SeasonAxis object
        n_seasons: Number of historical seasons to analyze

    Returns:
        List of correlation coefficients from observed data
    """
    gt_df = pd.read_csv('influpaint/data/nhsn_flusight_past.csv')

    correlations = []

    # Get the most recent n_seasons
    seasons = sorted(gt_df['fluseason'].unique())[-n_seasons:]

    for season_year in seasons:
        season_data = gt_df[gt_df['fluseason'] == season_year]
        season_pivot = season_data.pivot(columns='location_code', values='value', index='season_week')

        weeks = season_pivot.index.values
        n_weeks = len(weeks)

        # Compute pairwise weekly correlations
        for i, week1 in enumerate(weeks):
            for j in range(i + 1, n_weeks):
                week2 = weeks[j]

                pattern1 = season_pivot.loc[week1, :].values
                pattern2 = season_pivot.loc[week2, :].values

                valid = ~(np.isnan(pattern1) | np.isnan(pattern2))
                if np.sum(valid) < 3:
                    continue

                p1 = pattern1[valid]
                p2 = pattern2[valid]

                if np.std(p1) < 1e-6 or np.std(p2) < 1e-6:
                    continue

                corr = np.corrcoef(p1, p2)[0, 1]
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
        data.append({'Category': 'Random', 'Correlation': corr})
    for corr in influpaint_corr:
        data.append({'Category': 'Influpaint', 'Correlation': corr})
    for corr in observed_corr:
        data.append({'Category': 'Observed', 'Correlation': corr})

    df = pd.DataFrame(data)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

    # Create box plot
    sns.boxplot(data=df, x='Category', y='Correlation', ax=ax,
                order=['Random', 'Influpaint', 'Observed'],
                palette=['lightgray', 'skyblue', 'salmon'])

    # Add individual points
    sns.stripplot(data=df, x='Category', y='Correlation', ax=ax,
                  order=['Random', 'Influpaint', 'Observed'],
                  color='black', alpha=0.3, size=2, jitter=True)

    ax.set_xlabel('Data source', fontsize=13)
    ax.set_ylabel('Weekly incidence correlation (spatial)', fontsize=13)
    ax.set_title('Spatial correlation of weekly incidence patterns', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    sns.despine(ax=ax, trim=True)

    # Add statistics
    for i, category in enumerate(['Random', 'Influpaint', 'Observed']):
        cat_data = df[df['Category'] == category]['Correlation']
        median = cat_data.median()
        mean = cat_data.mean()
        ax.text(i, ax.get_ylim()[1] * 0.95, f'Med: {median:.3f}\nMean: {mean:.3f}',
                ha='center', va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', pad=2))

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved correlation figure to {save_path}")

    return fig
