"""
data_fetcher.py — NAV Data Fetching, Preprocessing & Quality Control
=====================================================================

Responsibilities:
  1. Fetch historical NAV from mftool (per-fund independent histories)
  2. Data quality filtering (weekdays, dedup, gap handling)
  3. Returns DataFrame construction (NaN-aware, outer join)
  4. Walk-forward data slicing for backtest (no lookahead)
  5. Synthetic fallback when mftool unavailable

Key design decisions:
  - Each fund's NAV is fetched independently and sorted ascending.
  - Outer join preserves each fund's full date range (no forced common start).
  - Covariance in optimizer.py uses pairwise overlapping periods (pandas default).
  - MIN_TRADING_DAYS = 756 (~3 years): funds below this are excluded.
  - MAX_LOOKBACK_DAYS = 1260 (~5 years): caps recency to avoid old-regime bias.
  - Funds failing CAGR/vol quality checks are flagged (not silently excluded)
    so optimizer.py can cap their estimates rather than drop them.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


# ── FUND UNIVERSE ─────────────────────────────────────────────────────────────

SELECTED_SCHEMES: Dict[str, str] = {
    "Parag Parikh":  "122639",
    "Quant":         "120843",
    "ICICI":         "134799",
    "Kotak":         "145552",
    "Motilal Oswal": "129046",
    "HDFC":          "118955",
    "Aditya Birla":  "120564",
    "ITI":           "151379",
    "Tata":          "144546",
    "Invesco":       "149763",
    "Bank of India": "148404",
    "HSBC":          "120046",
    "Edelweiss":     "140353",
    "WhiteOak":      "150346",
}

# ── FETCH CONFIGURATION ───────────────────────────────────────────────────────

# Minimum trading days to include a fund (~3 years).
# Funds with shorter history are excluded to avoid bull-market window bias.
MIN_TRADING_DAYS: int = 756

# Maximum lookback in trading days (~5 years).
# Caps to keep data relevant; avoids stale pre-2020 regime data dominating.
# Set to None to use all available history.
MAX_LOOKBACK_DAYS: Optional[int] = 1260

# Maximum forward-fill gap for missing NAV submissions (public holidays).
# 3 days = safe upper bound for Indian market closures (Diwali, etc.).
# Do NOT increase: filling longer gaps manufactures artificial zero-return periods.
MAX_FFILL_DAYS: int = 3


# ── NAV FETCH ────────────────────────────────────────────────────────────────

def _fetch_single_nav(mf, name: str, code: str) -> Optional[pd.Series]:
    """
    Fetch one fund's NAV history from mftool.

    Returns pd.Series (DatetimeIndex → float NAV), sorted ascending,
    weekdays only, deduplicated. Returns None on any failure.
    """
    try:
        data = mf.get_scheme_historical_nav(code)
        if data is None or "data" not in data or not data["data"]:
            logger.warning("No data for %s (code %s)", name, code)
            return None

        df = pd.DataFrame(data["data"])

        if "date" not in df.columns or "nav" not in df.columns:
            logger.warning("Unexpected mftool columns for %s: %s", name, df.columns.tolist())
            return None

        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df["nav"]  = pd.to_numeric(df["nav"], errors="coerce")

        df = df.dropna(subset=["date", "nav"])
        if df.empty:
            logger.warning("Empty DataFrame after cleaning for %s", name)
            return None

        # Sort ascending (mftool returns newest-first)
        df = df.sort_values("date").reset_index(drop=True)

        # Weekdays only — removes stray weekend entries sometimes present in mftool
        df = df[df["date"].dt.dayofweek < 5]

        # Drop duplicate dates: keep last entry for that day (most recent submission)
        df = df.drop_duplicates(subset="date", keep="last")

        if df.empty:
            return None

        s = df.set_index("date")["nav"].rename(name)
        return s

    except Exception as exc:
        logger.warning("Exception fetching %s (code %s): %s", name, code, exc)
        return None


def fetch_nav_data(schemes: Optional[Dict[str, str]] = None) -> Optional[pd.DataFrame]:
    """
    Fetch NAV for all schemes and return a DataFrame of DAILY RETURNS.

    Each fund uses its own independent date range (outer join).
    Returns DataFrame has NaN where a fund has no data; optimizer handles
    NaN via pairwise covariance computation.

    Returns None if fewer than 3 funds can be loaded.

    Quality filtering:
      - Funds with < MIN_TRADING_DAYS excluded.
      - Funds with > MAX_LOOKBACK_DAYS trimmed to most recent window.
      - Quality flags (CAGR > 30%, vol < 5%) handled in optimizer.py.
    """
    if schemes is None:
        schemes = SELECTED_SCHEMES

    try:
        from mftool import Mftool
        mf = Mftool()
    except ImportError:
        logger.warning("mftool not installed — using synthetic data")
        return None

    nav_map:        Dict[str, pd.Series] = {}
    failed:         list = []
    excluded_short: list = []

    for name, code in schemes.items():
        s = _fetch_single_nav(mf, name, code)
        if s is None:
            failed.append(name)
            continue

        # Trim to max lookback (most recent N trading days)
        if MAX_LOOKBACK_DAYS is not None and len(s) > MAX_LOOKBACK_DAYS:
            s = s.iloc[-MAX_LOOKBACK_DAYS:]

        # Enforce minimum history
        if len(s) < MIN_TRADING_DAYS:
            excluded_short.append((name, len(s)))
            logger.warning(
                "Excluding %s: only %d trading days (need >= %d)",
                name, len(s), MIN_TRADING_DAYS,
            )
            continue

        nav_map[name] = s
        logger.info(
            "Loaded %-18s  %4d days  %s → %s  NAV %7.3f → %8.3f",
            name, len(s),
            s.index[0].date(), s.index[-1].date(),
            float(s.iloc[0]), float(s.iloc[-1]),
        )

    if failed:
        logger.info("Fetch failed (%d funds): %s", len(failed), failed)
    if excluded_short:
        logger.info("Excluded — short history: %s", excluded_short)

    if len(nav_map) < 3:
        logger.error("Only %d funds loaded — insufficient for optimization", len(nav_map))
        return None

    # Combine: outer join preserves each fund's full date range
    all_nav = pd.concat(nav_map.values(), axis=1, join="outer")
    all_nav.sort_index(inplace=True)
    all_nav.index.name = "date"

    # Forward-fill short gaps (public holidays, missed submissions).
    # Limit prevents manufacturing multi-day zero-return stretches.
    all_nav = all_nav.ffill(limit=MAX_FFILL_DAYS)

    # Daily returns
    returns = all_nav.pct_change().iloc[1:]   # drop first all-NaN row
    returns = returns.dropna(how="all")        # drop rows entirely NaN

    # Remove extreme daily return outliers (|r| > 15% in single day = data error)
    # Replace with NaN rather than drop row (preserves other funds' data)
    returns = returns.where(returns.abs() < 0.15, other=np.nan)

    logger.info(
        "Returns DataFrame: %d rows x %d funds  (%s → %s)",
        len(returns), len(returns.columns),
        returns.index[0].date(), returns.index[-1].date(),
    )
    return returns


def get_returns(schemes: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, str]:
    """
    Returns (returns_df, source) where source is 'live' or 'synthetic'.
    Called by app.py on startup.
    """
    returns = fetch_nav_data(schemes)
    if returns is not None and len(returns.columns) >= 3:
        return returns, "live"

    logger.info("Falling back to synthetic return data")
    return _synthetic_returns(), "synthetic"


# ── WALK-FORWARD DATA SLICER ──────────────────────────────────────────────────

def get_training_slice(returns_df: pd.DataFrame, eval_year: int,
                        train_years: int = 3) -> pd.DataFrame:
    """
    Return the training data window for a given evaluation year.

    Training window: [eval_year - train_years, eval_year - 1] (inclusive).
    This strictly excludes the evaluation year — no lookahead bias.

    Used by run_walk_forward_backtest() in optimizer.py.

    Args:
        returns_df: Full returns DataFrame (all years)
        eval_year:  The year being evaluated (out-of-sample)
        train_years: Number of prior years to use for estimation

    Returns:
        Sliced returns DataFrame for the training window.
        Empty DataFrame if insufficient data.
    """
    start = f"{eval_year - train_years}-01-01"
    end   = f"{eval_year - 1}-12-31"

    try:
        sliced = returns_df.loc[start:end].copy()
    except KeyError:
        return pd.DataFrame()

    # Drop funds with no data in this training window
    valid_cols = [c for c in sliced.columns if sliced[c].notna().sum() >= 60]
    sliced     = sliced[valid_cols]

    logger.debug(
        "Walk-forward training slice for %d: %s → %s  (%d rows, %d funds)",
        eval_year, start, end, len(sliced), len(sliced.columns),
    )
    return sliced


def get_available_eval_years(returns_df: pd.DataFrame,
                              history: Optional[dict] = None,
                              train_years: int = 3) -> list:
    """
    Return the list of evaluation years for which sufficient training
    data exists in returns_df.

    A year is evaluable if the training window [year-train_years, year-1]
    has at least 60 rows (roughly 3 months) for at least 3 funds.
    """
    if history is None:
        from optimizer import NIFTY_HISTORY
        history = NIFTY_HISTORY

    available = []
    for year in sorted(history.keys()):
        sliced = get_training_slice(returns_df, year, train_years)
        if len(sliced) >= 60 and len(sliced.columns) >= 3:
            available.append(year)

    return available


# ── SYNTHETIC FALLBACK ─────────────────────────────────────────────────────────

def _synthetic_returns() -> pd.DataFrame:
    """
    Generate synthetic daily returns matching realistic Indian flexi-cap profile:
    ~13% geometric CAGR, 17-21% annualized volatility, 0.80+ intra-equity correlation.

    Used only when mftool is unavailable or returns insufficient data.
    The Cholesky decomposition ensures correct cross-fund correlation structure.
    """
    from optimizer import FUNDS_DEFAULT, _SYNTHETIC_RETURNS, _SYNTHETIC_COV

    n_days = 1260   # ~5 years of trading days
    np.random.seed(99)

    # Cholesky for correlated multivariate normal daily returns
    cov_daily = _SYNTHETIC_COV / 252
    try:
        L = np.linalg.cholesky(cov_daily)
    except np.linalg.LinAlgError:
        # Add jitter if matrix is not strictly PD
        cov_daily += 1e-8 * np.eye(len(FUNDS_DEFAULT))
        L = np.linalg.cholesky(cov_daily)

    Z      = np.random.randn(n_days, len(FUNDS_DEFAULT))
    daily  = Z @ L.T + _SYNTHETIC_RETURNS / 252

    dates   = pd.bdate_range(end="2024-12-31", periods=n_days)
    returns = pd.DataFrame(daily, index=dates, columns=FUNDS_DEFAULT)
    return returns
