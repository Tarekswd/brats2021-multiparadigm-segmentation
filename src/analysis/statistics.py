"""
Rigorous statistical analysis for paired model comparisons (Section 42).
Computes:
  - Paired t-test
  - Wilcoxon signed-rank test
  - Cohen's d effect size
  - 95% Confidence Intervals for mean difference
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy import stats


def paired_statistical_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_name: str = "Dice"
) -> Dict[str, Any]:
    """
    Perform paired subject-level statistical comparison between two architectures.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    assert len(a) == len(b), "Score arrays must have equal length"

    diff = a - b
    n = len(diff)
    mean_diff = float(np.mean(diff))
    median_diff = float(np.median(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0

    # 95% Confidence Interval for mean difference
    sem = std_diff / np.sqrt(n) if n > 0 else 0.0
    ci_low, ci_high = stats.t.interval(0.95, df=max(1, n - 1), loc=mean_diff, scale=sem) if n > 1 else (mean_diff, mean_diff)

    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(a, b) if n > 1 else (0.0, 1.0)

    # Wilcoxon signed-rank test (non-parametric)
    try:
        w_stat, w_pval = stats.wilcoxon(a, b)
    except Exception:
        w_stat, w_pval = 0.0, 1.0

    # Cohen's d effect size
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0) if n > 1 else 1.0
    cohens_d = float(mean_diff / pooled_std) if pooled_std > 1e-7 else 0.0

    return {
        "metric": metric_name,
        "n_subjects": n,
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "ci_95": (float(ci_low), float(ci_high)),
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_pval),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_pval),
        "cohens_d": cohens_d,
        "statistically_significant_p05": bool(w_pval < 0.05)
    }
