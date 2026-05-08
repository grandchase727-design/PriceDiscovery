###############################################################################
# Purged K-Fold Cross-Validation with Embargo (Lopez de Prado, AFML Ch.7)
# ============================================================================
# Standard K-Fold leaks when labels span multiple observations (our forward-21d
# targets overlap). Purged K-Fold:
#   1. For each test fold, remove from train any sample whose forward label
#      window overlaps the test set (purge).
#   2. Additionally drop `embargo` samples immediately after the test block
#      (embargo) — protects against serial correlation across the boundary.
###############################################################################

import numpy as np
from typing import Iterator, Tuple, List


class PurgedKFold:
    """
    K-Fold CV with forward-label purging and embargo for time-series.

    Parameters
    ----------
    n_splits : int
        Number of folds (sequential, non-shuffled).
    label_horizon : int
        Number of steps each label looks forward (e.g. 1 for monthly forward-1M).
        In units of rows (not days) — assumes rows are ordered in time.
    embargo : int
        Additional rows to drop after each test fold.
    """

    def __init__(self, n_splits: int = 5, label_horizon: int = 1, embargo: int = 1):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.label_horizon = label_horizon
        self.embargo = embargo

    def split(self, X) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        indices = np.arange(n)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1

        current = 0
        test_blocks: List[Tuple[int, int]] = []
        for fs in fold_sizes:
            test_blocks.append((current, current + fs))
            current += fs

        for start, stop in test_blocks:
            test_idx = indices[start:stop]

            # Purge: remove train rows whose forward label window touches the test block
            purge_left = max(0, start - self.label_horizon)
            purge_right = stop  # test block itself not in train

            # Embargo: skip `embargo` rows after the test block
            embargo_right = min(n, stop + self.embargo)

            train_mask = np.ones(n, dtype=bool)
            train_mask[purge_left:purge_right] = False
            train_mask[stop:embargo_right] = False
            train_idx = indices[train_mask]

            if len(train_idx) == 0:
                continue
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class PurgedWalkForward:
    """
    Expanding-window walk-forward with purge + embargo at the train/test boundary.

    Uses ONLY past data for each test fold — closer to real-time deployment
    simulation than PurgedKFold. Preferred for TAA backtests.

    For each fold:
      train = rows [0, fold_start - label_horizon - embargo)
      test  = rows [fold_start, fold_start + fold_size)
    """

    def __init__(self, n_splits: int = 5, label_horizon: int = 1,
                 embargo: int = 1, min_train: int = 24):
        self.n_splits = n_splits
        self.label_horizon = label_horizon
        self.embargo = embargo
        self.min_train = min_train

    def split(self, X) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        indices = np.arange(n)
        # Reserve the tail for n_splits roughly-equal test blocks
        total_test = n - self.min_train
        fold_size = max(1, total_test // self.n_splits)

        start = self.min_train
        for k in range(self.n_splits):
            stop = start + fold_size if k < self.n_splits - 1 else n
            if start >= n:
                break
            test_idx = indices[start:stop]
            train_end = max(0, start - self.label_horizon - self.embargo)
            train_idx = indices[:train_end]
            if len(train_idx) >= self.min_train and len(test_idx) > 0:
                yield train_idx, test_idx
            start = stop

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


if __name__ == "__main__":
    import pandas as pd

    dates = pd.date_range("2007-07-31", periods=225, freq="ME")

    print("PurgedKFold (n=225, n_splits=5, horizon=1, embargo=1) — for validation:")
    cv = PurgedKFold(n_splits=5, label_horizon=1, embargo=1)
    for i, (tr, te) in enumerate(cv.split(dates), 1):
        tr_range = (dates[tr].min().date(), dates[tr].max().date()) if len(tr) else ("-", "-")
        te_range = (dates[te].min().date(), dates[te].max().date())
        print(f"  Fold {i}: train {len(tr):>3} {tr_range}  |  test {len(te):>3} {te_range}")

    print("\nPurgedWalkForward (n=225, n_splits=5, horizon=1, embargo=1) — for backtest:")
    wf = PurgedWalkForward(n_splits=5, label_horizon=1, embargo=1, min_train=36)
    for i, (tr, te) in enumerate(wf.split(dates), 1):
        tr_range = (dates[tr].min().date(), dates[tr].max().date()) if len(tr) else ("-", "-")
        te_range = (dates[te].min().date(), dates[te].max().date())
        print(f"  Fold {i}: train {len(tr):>3} {tr_range}  |  test {len(te):>3} {te_range}")
