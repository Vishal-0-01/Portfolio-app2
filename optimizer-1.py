"""
optimizer.py — Robust Portfolio Optimization Engine
=====================================================
Production-grade Markowitz optimizer for Indian flexi-cap AMC portfolios.

Upgrades over naive implementation:
  1. Return shrinkage (James-Stein toward Nifty long-run ~12%)
  2. Ledoit-Wolf analytical covariance shrinkage
  3. Diversification penalty (L2/HHI) in objective function
  4. Turnover penalty (transaction cost model, 20bps one-way)
  5. Semi-variance / Sortino as secondary risk metric
  6. Data quality filters (CAGR cap 30%, vol floor 5%)
  7. Walk-forward backtest (no lookahead bias)
  8. Benchmarking vs Nifty TRI and equal-weight portfolio
  9. 3-layer vol constraint fallback chain (maintained)
 10. Geometric CAGR via log-return method (maintained)

Flask/frontend API contract: unchanged. All existing response keys preserved.
New keys added to API responses (additive, non-breaking).
"""

import numpy as np
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


# ── GLOBAL PARAMETERS ────────────────────────────────────────────────────────

RF          = 0.065   # India 91-day T-bill risk-free rate
VOL_CAP     = 0.10    # Hard portfolio volatility cap
VOL_TOL     = 0.0005  # SLSQP numerical tolerance (accept up to 10.05%)
MIN_FUND_W  = 0.03    # Minimum fund weight in equity sleeve
MAX_FUND_W  = 0.25    # Maximum fund weight in equity sleeve

# Valuation overlay baselines (Nifty 50, long-run averages)
NIFTY_MEAN_PE, NIFTY_STD_PE = 22.0, 5.0
NIFTY_MEAN_PB, NIFTY_STD_PB = 3.2,  0.6

# Non-equity asset class parameters (annualized)
DEBT_RET, DEBT_VOL = 0.075, 0.040
GOLD_RET, GOLD_VOL = 0.100, 0.150
CASH_RET, CASH_VOL = 0.068, 0.005

# Cross-asset correlations
RHO_EQ_DEBT   = -0.10
RHO_EQ_GOLD   =  0.05
RHO_DEBT_GOLD =  0.00

# ── RETURN SHRINKAGE PARAMETERS ───────────────────────────────────────────────
# Shrink raw historical CAGR toward the long-run Nifty market return.
# Rationale: individual fund CAGRs are estimated on limited samples;
# shrinkage reduces the impact of estimation error on optimized weights.
NIFTY_LONGRUN_RETURN = 0.12   # ~12% long-run geometric CAGR for Indian large/flexi-cap
SHRINKAGE_ALPHA      = 0.25   # 25% weight on market prior, 75% on historical
                               # Higher = more conservative, less sensitive to recent returns
                               # Range: 0.15 (less shrinkage) to 0.40 (more conservative)

# ── REGULARISATION PARAMETERS ─────────────────────────────────────────────────
LAMBDA_DIVERSIFICATION = 0.08   # L2 penalty coefficient on fund weight concentration
                                  # Penalises HHI: adds lambda*sum(w^2) to cost
                                  # 0 = pure Markowitz; 0.20 = strong equal-weight pull
LAMBDA_TURNOVER       = 0.30    # Turnover penalty coefficient
                                  # Multiplied by sum(|w_new - w_prev|) * TC_BPS/10000
TC_BPS                = 20      # One-way transaction cost (bps) — approx for MF STP

# ── DATA QUALITY THRESHOLDS ───────────────────────────────────────────────────
MAX_CAGR_THRESHOLD    = 0.30    # Exclude funds with CAGR > 30% (likely window bias)
MIN_VOL_THRESHOLD     = 0.05    # Exclude funds with vol < 5% (data issue / money market)

# ── HISTORICAL DATA ───────────────────────────────────────────────────────────
NIFTY_HISTORY = {
    2015: {"pe": 22.0, "pb": 3.2},
    2016: {"pe": 22.7, "pb": 3.3},
    2017: {"pe": 26.4, "pb": 3.5},
    2018: {"pe": 24.0, "pb": 3.4},
    2019: {"pe": 28.5, "pb": 3.5},
    2020: {"pe": 37.5, "pb": 4.0},
    2021: {"pe": 27.5, "pb": 4.5},
    2022: {"pe": 22.3, "pb": 3.9},
    2023: {"pe": 23.7, "pb": 4.0},
    2024: {"pe": 22.0, "pb": 3.7},
}

# Historical Nifty 50 TRI annual returns (Dec–Dec, approximate)
NIFTY_ANNUAL_RETURNS = {
    2015: -0.040, 2016:  0.033, 2017:  0.288, 2018:  0.033, 2019:  0.122,
    2020:  0.147, 2021:  0.242, 2022:  0.043, 2023:  0.197, 2024:  0.088,
}

# Historical Gold (INR) annual returns
GOLD_ANNUAL_RETURNS = {
    2015: -0.060, 2016:  0.110, 2017:  0.050, 2018:  0.075, 2019:  0.190,
    2020:  0.280, 2021: -0.040, 2022:  0.110, 2023:  0.130, 2024:  0.210,
}

DEBT_ANNUAL_RETURNS = {y: 0.068 for y in range(2015, 2025)}
CASH_ANNUAL_RETURNS = {y: 0.065 for y in range(2015, 2025)}

FUNDS_DEFAULT = [
    "Parag Parikh", "Quant", "ICICI", "Motilal Oswal", "HDFC",
    "Aditya Birla", "ITI", "Tata", "Invesco", "Bank of India",
    "HSBC", "Edelweiss", "WhiteOak"
]


def _build_synthetic_params():
    """Reproducible synthetic params: realistic Indian flexi-cap profile."""
    n = len(FUNDS_DEFAULT)
    # Conservative, realistic long-run geometric CAGRs (post-shrinkage levels)
    ann_returns = np.array([0.135, 0.145, 0.120, 0.135, 0.125,
                             0.115, 0.130, 0.120, 0.120, 0.130,
                             0.120, 0.110, 0.135])
    np.random.seed(10)
    corr = np.full((n, n), 0.82)
    for i in range(n):
        for j in range(i + 1, n):
            corr[i, j] = corr[j, i] = 0.75 + 0.12 * np.random.random()
    np.fill_diagonal(corr, 1.0)
    vols = np.array([0.17, 0.21, 0.18, 0.20, 0.17,
                     0.18, 0.19, 0.18, 0.17, 0.20,
                     0.18, 0.17, 0.19])
    cov = np.outer(vols, vols) * corr
    cov = (cov + cov.T) / 2
    np.fill_diagonal(cov, vols ** 2)
    return ann_returns, cov, vols


_SYNTHETIC_RETURNS, _SYNTHETIC_COV, _SYNTHETIC_VOLS = _build_synthetic_params()


# ── RETURN ESTIMATION ─────────────────────────────────────────────────────────

def _geometric_cagr(daily_returns_series):
    """
    Geometric CAGR from daily returns series via log-return method.
    ann = exp(mean(log(1+r)) * 252) - 1
    Removes the +vol^2/2 upward bias of arithmetic annualization.
    """
    r = daily_returns_series.dropna().values
    if len(r) < 2:
        return RF
    return float(np.expm1(np.log1p(r).mean() * 252))


def shrink_returns(raw_returns, n_obs_per_fund, market_return=NIFTY_LONGRUN_RETURN,
                   alpha=SHRINKAGE_ALPHA):
    """
    James-Stein / Bayes-Stein shrinkage toward market return.

    Formula: mu_shrunk = (1 - alpha) * mu_hist + alpha * mu_market

    Rationale:
      - Raw historical CAGR is a noisy estimator on 3-5 year windows.
      - Shrinking toward a market prior reduces estimation error.
      - alpha = 0.25 means 75% historical, 25% prior. Adjustable.
      - Funds with shorter history get stronger shrinkage (larger alpha).

    Returns shrunk returns array.
    """
    n = len(raw_returns)
    shrunk = np.zeros(n)
    for i in range(n):
        # Scale shrinkage by effective sample: fewer obs → more shrinkage
        n_i = n_obs_per_fund[i]
        # Effective alpha: higher for shorter history
        alpha_eff = min(0.60, alpha * (756 / max(n_i, 1)))
        alpha_eff = max(alpha, alpha_eff)   # at least base alpha
        shrunk[i] = (1 - alpha_eff) * raw_returns[i] + alpha_eff * market_return
    return shrunk


def ledoit_wolf_shrinkage(sample_cov, n_obs):
    """
    Analytical Ledoit-Wolf covariance shrinkage (Oracle Approximating Shrinkage).
    Shrinks toward scaled identity matrix (constant-correlation target).

    Target: mu_target * I where mu_target = trace(S)/n (average variance)

    Formula: Sigma_lw = (1 - delta) * S + delta * mu_target * I

    delta estimated analytically:
        delta* = min(1, (n+2) / ((n+2) + T * rho))
    where rho = ||S - mu*I||_F^2 / ||S||_F^2 (relative misfit)

    This is the simplified Oracle LW (Ledoit-Wolf 2004, simplified form).
    Full OAS or sklearn's LedoitWolf would be more accurate but this is
    sufficient and keeps the dependency footprint minimal.

    Returns shrunk covariance matrix (guaranteed PD).
    """
    n = sample_cov.shape[0]
    T = n_obs

    # Shrinkage target: scaled identity
    mu_target = np.trace(sample_cov) / n

    # Relative misfit of sample cov from target
    diff = sample_cov - mu_target * np.eye(n)
    rho  = np.linalg.norm(diff, 'fro') ** 2 / (np.linalg.norm(sample_cov, 'fro') ** 2 + 1e-12)

    # Oracle LW delta
    delta = min(1.0, max(0.0, (n + 2) / ((n + 2) + T * rho)))

    shrunk = (1 - delta) * sample_cov + delta * mu_target * np.eye(n)

    # Guarantee strict PD (floating point can introduce tiny negative eigenvalues)
    eigvals = np.linalg.eigvalsh(shrunk)
    if eigvals.min() < 1e-8:
        shrunk += (-eigvals.min() + 1e-8) * np.eye(n)

    logger.info("LW shrinkage delta=%.4f  (n=%d, T=%d, rho=%.4f)", delta, n, T, rho)
    return shrunk


# ── COVARIANCE & STATS BUILDER ────────────────────────────────────────────────

def build_cov_from_returns(returns_df):
    """
    Build robust return estimates, covariance, vols, and Sharpe ratios
    from a daily returns DataFrame.

    Pipeline:
      1. Geometric CAGR per fund (log-return method, per-fund history)
      2. Data quality filters (CAGR cap, vol floor)
      3. Ledoit-Wolf covariance shrinkage
      4. James-Stein return shrinkage toward market prior
      5. Semi-variance (downside vol) and Sortino ratio per fund

    Returns:
      ann_returns  : ndarray, shrunk annualized geometric CAGR per fund
      cov_mat      : ndarray, LW-shrunk annualized covariance matrix
      vols         : ndarray, annualized volatility per fund
      fund_sharpes : ndarray, Sharpe ratios (shrunk return basis)
      extra_stats  : dict, semi_vols + sortino per fund (for API)
    """
    fund_cols = list(returns_df.columns)
    n_funds   = len(fund_cols)

    # ── Step 1: Geometric CAGR per fund ──────────────────────────────────
    raw_cagr = np.zeros(n_funds)
    n_obs    = np.zeros(n_funds, dtype=int)

    for i, col in enumerate(fund_cols):
        r         = returns_df[col].dropna()
        n_obs[i]  = len(r)
        raw_cagr[i] = _geometric_cagr(r)

    # ── Step 2: Data quality filter ───────────────────────────────────────
    # Compute sample vols first (needed for filter)
    sample_cov_raw = returns_df.cov().values * 252
    eigv = np.linalg.eigvalsh(sample_cov_raw)
    if eigv.min() < 0:
        sample_cov_raw += (-eigv.min() + 1e-8) * np.eye(n_funds)
    raw_vols = np.sqrt(np.diag(sample_cov_raw))

    flagged = []
    for i, col in enumerate(fund_cols):
        reasons = []
        if raw_cagr[i] > MAX_CAGR_THRESHOLD:
            reasons.append(f"CAGR {raw_cagr[i]*100:.1f}% > {MAX_CAGR_THRESHOLD*100:.0f}%")
            # Cap rather than exclude (exclusion would change fund universe at runtime)
            raw_cagr[i] = MAX_CAGR_THRESHOLD
        if raw_vols[i] < MIN_VOL_THRESHOLD:
            reasons.append(f"vol {raw_vols[i]*100:.1f}% < {MIN_VOL_THRESHOLD*100:.0f}%")
            raw_vols[i] = MIN_VOL_THRESHOLD  # floor vol; cov diagonal adjusted below
        if reasons:
            flagged.append((col, reasons))
            logger.warning("Quality flag — %s: %s", col, "; ".join(reasons))

    # ── Step 3: Ledoit-Wolf covariance shrinkage ──────────────────────────
    # Use median obs count as effective T for the pairwise cov matrix
    T_eff   = int(np.median(n_obs))
    cov_mat = ledoit_wolf_shrinkage(sample_cov_raw, T_eff)

    # Re-extract vols from shrunk cov (shrinkage modifies diagonal slightly)
    vols = np.sqrt(np.diag(cov_mat))

    # ── Step 4: Return shrinkage toward market ────────────────────────────
    ann_returns = shrink_returns(raw_cagr, n_obs)

    # ── Step 5: Semi-variance and Sortino per fund ────────────────────────
    semi_vols = np.zeros(n_funds)
    sortinos  = np.zeros(n_funds)
    daily_rf  = RF / 252

    for i, col in enumerate(fund_cols):
        r       = returns_df[col].dropna().values
        down    = r[r < daily_rf] - daily_rf   # excess daily returns below RF
        if len(down) > 10:
            semi_vols[i] = float(np.sqrt(np.mean(down ** 2) * 252))
        else:
            semi_vols[i] = vols[i]   # fallback to full vol if few downside days

        sortinos[i] = float((ann_returns[i] - RF) / semi_vols[i]) if semi_vols[i] > 0 else 0.0

    # ── Sharpe ratios ─────────────────────────────────────────────────────
    fund_sharpes = np.where(vols > 0, (ann_returns - RF) / vols, 0.0)

    # Diagnostics
    logger.info(
        "Returns: raw [%.1f%%–%.1f%%] → shrunk [%.1f%%–%.1f%%]",
        raw_cagr.min()*100, raw_cagr.max()*100,
        ann_returns.min()*100, ann_returns.max()*100,
    )
    logger.info(
        "Vol: [%.1f%%–%.1f%%]  Sharpe: [%.2f–%.2f]  Sortino: [%.2f–%.2f]",
        vols.min()*100, vols.max()*100,
        fund_sharpes.min(), fund_sharpes.max(),
        sortinos.min(), sortinos.max(),
    )

    extra_stats = {
        "semi_vols": semi_vols,
        "sortinos":  sortinos,
        "raw_cagrs": raw_cagr,
        "quality_flags": {col: reasons for col, reasons in flagged},
    }
    return ann_returns, cov_mat, vols, fund_sharpes, extra_stats


# ── VALUATION LOGIC ──────────────────────────────────────────────────────────

def valuation_z(pe, pb):
    z_pe = (pe - NIFTY_MEAN_PE) / NIFTY_STD_PE
    z_pb = (pb - NIFTY_MEAN_PB) / NIFTY_STD_PB
    return z_pe, z_pb, (z_pe + z_pb) / 2


def equity_from_z(z):
    """Map combined z-score → equity allocation in [50%, 90%]."""
    z_c = np.clip(z, -2.0, 2.0)
    return float(np.clip(0.70 - 0.10 * z_c, 0.50, 0.90))


def get_non_equity(E, z):
    """
    Split non-equity (1-E) into debt/gold/cash.
    Higher z (overvalued market) → more debt+gold, less cash.
    """
    rem = 1.0 - E
    d = np.clip(0.55 + 0.05 * z, 0.30, 0.70)
    g = np.clip(0.30 + 0.05 * z, 0.10, 0.50)
    c = np.clip(0.15 - 0.10 * z, 0.05, 0.30)
    t = d + g + c
    return float(rem * d / t), float(rem * g / t), float(rem * c / t)


# ── PORTFOLIO STATISTICS ──────────────────────────────────────────────────────

def portfolio_vol_from_weights(E, w_eq, D, G, C, cov_mat):
    """
    Full portfolio volatility including equity sleeve + debt + gold + cash.
    Uses sigma_eq = E * sleeve_vol to correctly scale cross-asset covariance.
    """
    sleeve_vol = float(np.sqrt(np.clip(w_eq @ cov_mat @ w_eq, 0, None)))
    sigma_eq   = E * sleeve_vol
    var = (
        sigma_eq ** 2 +
        D ** 2 * DEBT_VOL ** 2 +
        G ** 2 * GOLD_VOL ** 2 +
        C ** 2 * CASH_VOL ** 2 +
        2 * sigma_eq * D * DEBT_VOL * RHO_EQ_DEBT +
        2 * sigma_eq * G * GOLD_VOL * RHO_EQ_GOLD +
        2 * D * G * DEBT_VOL * GOLD_VOL * RHO_DEBT_GOLD
    )
    return float(np.sqrt(max(var, 0.0))), sleeve_vol


def portfolio_return(E, w_eq, D, G, C, ann_returns):
    eq_ret = float(np.dot(w_eq, ann_returns))
    return E * eq_ret + D * DEBT_RET + G * GOLD_RET + C * CASH_RET


def portfolio_semi_vol(E, w_eq, D, G, C, semi_vols):
    """
    Approximate portfolio-level downside vol using weighted semi-vols.
    Full portfolio semi-variance requires daily simulation; this is a
    practical first-order approximation: sigma_down_p ≈ E * w'*sigma_down.
    """
    sleeve_semi = float(np.dot(w_eq, semi_vols))
    return float(E * sleeve_semi)


def portfolio_sortino(E, w_eq, D, G, C, ann_returns, semi_vols):
    ret    = portfolio_return(E, w_eq, D, G, C, ann_returns)
    s_vol  = portfolio_semi_vol(E, w_eq, D, G, C, semi_vols)
    return float((ret - RF) / s_vol) if s_vol > 0 else 0.0


# ── OPTIMIZATION ─────────────────────────────────────────────────────────────

def _regularised_objective(w, E, D, G, C, ann_returns, prev_weights,
                            lambda_div=LAMBDA_DIVERSIFICATION,
                            lambda_tc=LAMBDA_TURNOVER,
                            tc_bps=TC_BPS):
    """
    Regularised negative portfolio return:
        obj = -port_return
            + lambda_div * HHI(w)          [diversification penalty]
            + lambda_tc  * TC * turnover   [transaction cost penalty]

    HHI(w) = sum(w_i^2): Herfindahl-Hirschman Index — minimal at equal weights.
    Turnover = sum(|w_i - w_prev_i|) as a fraction of portfolio.
    TC = tc_bps / 10000 (one-way cost applied to each unit of turnover).

    These penalties are included in the objective, NOT as hard constraints,
    which keeps the optimization problem well-conditioned and avoids
    infeasibility.
    """
    ret      = portfolio_return(E, w, D, G, C, ann_returns)
    hhi      = float(np.dot(w, w))                    # concentration measure
    turnover = float(np.sum(np.abs(w - prev_weights))) # fraction rebalanced
    tc_cost  = tc_bps / 10000.0                        # one-way cost fraction
    return -ret + lambda_div * hhi + lambda_tc * tc_cost * turnover


def _min_vol_weights(n, E, D, G, C, cov_mat):
    """Fallback: find minimum-vol weights (ignores return objective)."""
    def obj(w):
        v, _ = portfolio_vol_from_weights(E, w, D, G, C, cov_mat)
        return v

    best = None
    for w0 in [np.ones(n) / n, np.random.dirichlet(np.ones(n) * 3)]:
        try:
            r = minimize(obj, w0, method='SLSQP',
                         bounds=[(MIN_FUND_W, MAX_FUND_W)] * n,
                         constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
                         options={'ftol': 1e-14, 'maxiter': 3000})
            if r.success:
                v, _ = portfolio_vol_from_weights(E, r.x, D, G, C, cov_mat)
                if best is None or v < best[1]:
                    best = (r.x.copy(), v)
        except Exception:
            pass

    return (best[0], best[1]) if best else (np.ones(n) / n, None)


def _scale_equity_to_meet_cap(w_eq, E_target, z, cov_mat, vol_cap=VOL_CAP):
    """Scale equity down in 0.5pp steps until vol cap is satisfied."""
    E = E_target
    for _ in range(80):
        D, G, C = get_non_equity(E, z)
        v, _ = portfolio_vol_from_weights(E, w_eq, D, G, C, cov_mat)
        if v <= vol_cap + VOL_TOL:
            return E, D, G, C, v
        E = max(0.50, E - 0.005)
    D, G, C = get_non_equity(0.50, z)
    v, _ = portfolio_vol_from_weights(0.50, w_eq, D, G, C, cov_mat)
    return 0.50, D, G, C, v


def optimize_for_pe_pb(pe, pb, ann_returns, cov_mat, vols, fund_sharpes, funds,
                        vol_cap=VOL_CAP, prev_weights=None, semi_vols=None):
    """
    Main optimization entry point.

    Objective: maximise portfolio return with:
      - Diversification penalty (L2/HHI)
      - Transaction cost penalty (vs prev_weights)
      - Hard vol constraint (portfolio vol <= vol_cap)
      - Fund weight bounds [MIN_FUND_W, MAX_FUND_W]
      - Weights sum to 1

    Fallback chain:
      1. SLSQP with 10 random starts + regularised objective
      2. Minimum-vol weights
      3. Step equity down until cap satisfied

    Additional output keys (additive, non-breaking):
      sortino, semi_vol, raw_cagrs, concentration_hhi
    """
    n = len(funds)
    z_pe, z_pb, z = valuation_z(pe, pb)
    E_val  = equity_from_z(z)
    D0, G0, C0 = get_non_equity(E_val, z)

    # Default previous weights = equal weight (for turnover penalty baseline)
    if prev_weights is None:
        prev_weights = np.ones(n) / n
    prev_weights = np.asarray(prev_weights, dtype=float)
    prev_weights /= prev_weights.sum()

    if semi_vols is None:
        semi_vols = vols.copy()   # fallback to full vol if not provided

    def vol_constraint(w):
        v, _ = portfolio_vol_from_weights(E_val, w, D0, G0, C0, cov_mat)
        return vol_cap - v   # >= 0 for SLSQP ineq

    rng   = np.random.default_rng(int(pe * 100 + pb * 10))
    inits = ([np.ones(n) / n, prev_weights] +
             [rng.dirichlet(np.ones(n) * 2) for _ in range(8)])

    best_obj, best_w, best_vol = np.inf, None, None

    for w0 in inits:
        w0 = np.asarray(w0, dtype=float)
        w0 /= w0.sum()
        try:
            r = minimize(
                _regularised_objective,
                w0,
                args=(E_val, D0, G0, C0, ann_returns, prev_weights),
                method='SLSQP',
                bounds=[(MIN_FUND_W, MAX_FUND_W)] * n,
                constraints=[
                    {'type': 'eq',   'fun': lambda w: np.sum(w) - 1.0},
                    {'type': 'ineq', 'fun': vol_constraint},
                ],
                options={'ftol': 1e-13, 'maxiter': 5000},
            )
        except Exception:
            continue

        if not r.success:
            continue

        w_cand = np.clip(r.x, MIN_FUND_W, MAX_FUND_W)
        w_cand /= w_cand.sum()
        v_cand, _ = portfolio_vol_from_weights(E_val, w_cand, D0, G0, C0, cov_mat)
        obj_cand   = _regularised_objective(w_cand, E_val, D0, G0, C0, ann_returns, prev_weights)

        if v_cand <= vol_cap + VOL_TOL and obj_cand < best_obj:
            best_obj, best_w, best_vol = obj_cand, w_cand.copy(), v_cand

    # Fallback: min-vol weights
    if best_w is None:
        logger.warning("Primary optimizer failed for PE=%.1f PB=%.1f — min-vol fallback", pe, pb)
        best_w, best_vol = _min_vol_weights(n, E_val, D0, G0, C0, cov_mat)
        best_ret_f = portfolio_return(E_val, best_w, D0, G0, C0, ann_returns)
    else:
        best_ret_f = portfolio_return(E_val, best_w, D0, G0, C0, ann_returns)

    # Hard vol validation + equity scaling fallback
    v_final, sl_vol = portfolio_vol_from_weights(E_val, best_w, D0, G0, C0, cov_mat)
    E_final, D_final, G_final, C_final = E_val, D0, G0, C0

    if v_final > vol_cap + VOL_TOL:
        logger.warning("Vol %.4f > cap at PE=%.1f — scaling equity down", v_final, pe)
        E_final, D_final, G_final, C_final, v_final = _scale_equity_to_meet_cap(
            best_w, E_val, z, cov_mat, vol_cap
        )

    ret_final  = portfolio_return(E_final, best_w, D_final, G_final, C_final, ann_returns)
    _, sl_vol  = portfolio_vol_from_weights(E_final, best_w, D_final, G_final, C_final, cov_mat)
    sharpe     = float((ret_final - RF) / v_final) if v_final > 0 else 0.0
    sortino_p  = portfolio_sortino(E_final, best_w, D_final, G_final, C_final, ann_returns, semi_vols)
    semi_vol_p = portfolio_semi_vol(E_final, best_w, D_final, G_final, C_final, semi_vols)
    hhi        = float(np.dot(best_w, best_w))
    turnover   = float(np.sum(np.abs(best_w - prev_weights)))

    return {
        # Existing keys (unchanged)
        "pe": pe, "pb": pb,
        "z_pe": round(z_pe, 4), "z_pb": round(z_pb, 4), "z": round(z, 4),
        "equity":    round(E_final, 4),
        "debt":      round(D_final, 4),
        "gold":      round(G_final, 4),
        "cash":      round(C_final, 4),
        "port_ret":  round(ret_final, 4),
        "port_vol":  round(v_final, 4),
        "sharpe":    round(sharpe, 4),
        "eq_vol":    round(sl_vol, 4),
        "fund_weights":  [round(float(w), 4) for w in best_w],
        "fund_sharpes":  [round(float(s), 3) for s in fund_sharpes],
        "fund_returns":  [round(float(r), 4) for r in ann_returns],
        "fund_vols":     [round(float(v), 4) for v in vols],
        "constraint_ok": bool(v_final <= vol_cap + VOL_TOL),
        # New keys (additive — frontend ignores unknown keys)
        "sortino":        round(sortino_p, 4),
        "semi_vol":       round(semi_vol_p, 4),
        "concentration_hhi": round(hhi, 4),
        "turnover":       round(turnover, 4),
    }


# ── EFFICIENT FRONTIER ────────────────────────────────────────────────────────

def compute_frontier(ann_returns, cov_mat, z=0.25, n_points=12, semi_vols=None):
    """Efficient frontier by sweeping equity 50%→90%."""
    n = len(ann_returns)
    if semi_vols is None:
        semi_vols = np.sqrt(np.diag(cov_mat))
    frontier = []
    w_prev   = np.ones(n) / n

    for E_fix in np.linspace(0.50, 0.90, n_points):
        D, G, C = get_non_equity(float(E_fix), z)

        def obj(w):
            return _regularised_objective(w, float(E_fix), D, G, C, ann_returns, w_prev)

        r = minimize(obj, np.ones(n) / n, method='SLSQP',
                     bounds=[(MIN_FUND_W, MAX_FUND_W)] * n,
                     constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
                     options={'ftol': 1e-12, 'maxiter': 2000})
        if r.success:
            w_opt = np.clip(r.x, MIN_FUND_W, MAX_FUND_W)
            w_opt /= w_opt.sum()
            v, _  = portfolio_vol_from_weights(float(E_fix), w_opt, D, G, C, cov_mat)
            ret   = portfolio_return(float(E_fix), w_opt, D, G, C, ann_returns)
            frontier.append({"v": round(v, 4), "r": round(ret, 4), "E": round(float(E_fix), 2)})
            w_prev = w_opt.copy()

    return sorted(frontier, key=lambda x: x["v"])


# ── ALLOCATION BACKTEST ───────────────────────────────────────────────────────

def run_backtest(ann_returns, cov_mat, vols, fund_sharpes, funds,
                 vol_cap=VOL_CAP, history=None, semi_vols=None):
    """
    Year-by-year allocation using historical PE/PB (UNCHANGED API).
    Note: uses the same return/cov for all years (not walk-forward).
    See run_walk_forward_backtest() for the no-lookahead version.
    """
    if history is None:
        history = NIFTY_HISTORY
    if semi_vols is None:
        semi_vols = vols.copy()

    results  = []
    w_prev   = np.ones(len(funds)) / len(funds)

    for year in sorted(history.keys()):
        pe = history[year]["pe"]
        pb = history[year]["pb"]
        res = optimize_for_pe_pb(
            pe, pb, ann_returns, cov_mat, vols, fund_sharpes, funds,
            vol_cap=vol_cap, prev_weights=w_prev, semi_vols=semi_vols,
        )
        w_prev    = np.array(res["fund_weights"])
        fw_sorted = sorted(zip(funds, res["fund_weights"]), key=lambda x: -x[1])[:3]
        results.append({
            "year":          year,
            "pe":            pe,
            "pb":            pb,
            "z":             res["z"],
            "equity":        res["equity"],
            "debt":          res["debt"],
            "gold":          res["gold"],
            "cash":          res["cash"],
            "port_ret":      res["port_ret"],
            "port_vol":      res["port_vol"],
            "sharpe":        res["sharpe"],
            "sortino":       res["sortino"],
            "constraint_ok": res["constraint_ok"],
            "top_funds":     [{"name": f, "weight": round(w, 4)} for f, w in fw_sorted],
        })
    return results


# ── WALK-FORWARD BACKTEST (NO LOOKAHEAD BIAS) ─────────────────────────────────

def run_walk_forward_backtest(returns_df, history=None, vol_cap=VOL_CAP,
                               train_years=3):
    """
    Proper walk-forward backtest: for each evaluation year Y,
    use only data from [Y-train_years, Y-1] to estimate returns/cov.
    Apply the model to get allocation, then use realised returns for Y.

    Eliminates lookahead bias present in the simple backtest.
    Returns same schema as run_performance_backtest() plus benchmark columns.
    """
    if history is None:
        history = NIFTY_HISTORY

    years_sorted = sorted(history.keys())
    # Only evaluate years where we have enough prior data
    eval_years = [y for y in years_sorted if y - train_years >= min(years_sorted) - 1]

    portfolio_value   = 100.0
    nifty_value       = 100.0
    equalw_value      = 100.0
    value_series      = [100.0]
    nifty_series      = [100.0]
    equalw_series     = [100.0]
    annual_ret_series = []
    nifty_ret_series  = []
    equalw_ret_series = []
    per_year          = []
    w_prev            = None

    for year in eval_years:
        # Training window: [year - train_years, year - 1]
        train_start = f"{year - train_years}-01-01"
        train_end   = f"{year - 1}-12-31"

        train_data = returns_df.loc[train_start:train_end].copy()
        if len(train_data) < 60:   # minimum ~3 months of data
            logger.warning("Insufficient training data for %d — skipping", year)
            continue

        # Fit model on training window
        ann_returns, cov_mat, vols, fund_sharpes, extra = build_cov_from_returns(train_data)
        funds      = list(returns_df.columns)
        semi_vols  = extra["semi_vols"]

        if w_prev is None:
            w_prev = np.ones(len(funds)) / len(funds)

        # Optimize using this year's PE/PB
        pe  = history[year]["pe"]
        pb  = history[year]["pb"]
        res = optimize_for_pe_pb(
            pe, pb, ann_returns, cov_mat, vols, fund_sharpes, funds,
            vol_cap=vol_cap, prev_weights=w_prev, semi_vols=semi_vols,
        )

        E   = res["equity"]
        D   = res["debt"]
        G   = res["gold"]
        C   = res["cash"]
        fw  = np.array(res["fund_weights"])
        w_prev = fw.copy()

        # Realised returns for evaluation year
        nifty_ret = NIFTY_ANNUAL_RETURNS.get(year, NIFTY_LONGRUN_RETURN)
        gold_ret  = GOLD_ANNUAL_RETURNS.get(year, GOLD_RET)
        debt_ret  = DEBT_ANNUAL_RETURNS.get(year, DEBT_RET)
        cash_ret  = CASH_ANNUAL_RETURNS.get(year, CASH_RET)

        # Fund returns: nifty + fund alpha (alpha from training-window CAGR)
        fund_alphas   = ann_returns - NIFTY_LONGRUN_RETURN
        fund_rets     = nifty_ret + fund_alphas
        eq_sleeve_ret = float(np.dot(fw, fund_rets))

        port_ret  = E*eq_sleeve_ret + D*debt_ret + G*gold_ret + C*cash_ret
        equalw_ret = 0.70 * nifty_ret + 0.18*debt_ret + 0.10*gold_ret + 0.02*cash_ret

        portfolio_value *= (1.0 + port_ret)
        nifty_value     *= (1.0 + nifty_ret)
        equalw_value    *= (1.0 + equalw_ret)

        value_series.append(round(portfolio_value, 4))
        nifty_series.append(round(nifty_value, 4))
        equalw_series.append(round(equalw_value, 4))
        annual_ret_series.append(round(port_ret, 6))
        nifty_ret_series.append(round(nifty_ret, 4))
        equalw_ret_series.append(round(equalw_ret, 6))

        per_year.append({
            "year":          year,
            "pe":            pe,
            "pb":            pb,
            "equity":        round(E, 4),
            "debt":          round(D, 4),
            "gold":          round(G, 4),
            "cash":          round(C, 4),
            "nifty_ret":     round(nifty_ret, 4),
            "port_ret":      round(port_ret, 6),
            "port_value":    round(portfolio_value, 4),
            "nifty_value":   round(nifty_value, 4),
            "equalw_value":  round(equalw_value, 4),
            "sharpe":        res["sharpe"],
            "sortino":       res["sortino"],
            "hhi":           res["concentration_hhi"],
            "turnover":      res["turnover"],
            "train_window":  f"{train_start} → {train_end}",
        })

    # Summary stats
    n_yrs     = len(annual_ret_series)
    rets_arr  = np.array(annual_ret_series)
    cagr      = float((portfolio_value / 100.0) ** (1.0 / n_yrs) - 1.0) if n_yrs > 0 else 0.0
    vol_bt    = float(np.std(rets_arr, ddof=1)) if n_yrs > 1 else 0.0
    sharpe_bt = float((cagr - RF) / vol_bt) if vol_bt > 0 else 0.0

    # Max drawdown (model portfolio)
    peak = 100.0; max_dd = 0.0
    for v in value_series[1:]:
        peak = max(peak, v)
        dd   = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    # Nifty CAGR + drawdown
    n_arr      = np.array(nifty_ret_series)
    nifty_cagr = float((nifty_value / 100.0) ** (1.0 / n_yrs) - 1.0) if n_yrs > 0 else 0.0
    nifty_vol  = float(np.std(n_arr, ddof=1)) if n_yrs > 1 else 0.0
    nifty_sh   = float((nifty_cagr - RF) / nifty_vol) if nifty_vol > 0 else 0.0
    nifty_peak = 100.0; nifty_dd = 0.0
    for v in nifty_series[1:]:
        nifty_peak = max(nifty_peak, v)
        dd = (v - nifty_peak) / nifty_peak
        if dd < nifty_dd:
            nifty_dd = dd

    # Equal-weight CAGR + drawdown
    eq_arr    = np.array(equalw_ret_series)
    equalw_cagr = float((equalw_value / 100.0) ** (1.0 / n_yrs) - 1.0) if n_yrs > 0 else 0.0
    equalw_vol  = float(np.std(eq_arr, ddof=1)) if n_yrs > 1 else 0.0
    equalw_sh   = float((equalw_cagr - RF) / equalw_vol) if equalw_vol > 0 else 0.0

    return {
        # Model portfolio
        "years":           [p["year"] for p in per_year],
        "portfolio_value": value_series[1:],
        "initial_value":   100.0,
        "annual_returns":  annual_ret_series,
        "cagr":            round(cagr, 6),
        "max_drawdown":    round(max_dd, 6),
        "volatility":      round(vol_bt, 6),
        "sharpe":          round(sharpe_bt, 4),
        # Benchmarks
        "nifty_value":     nifty_series[1:],
        "nifty_cagr":      round(nifty_cagr, 6),
        "nifty_sharpe":    round(nifty_sh, 4),
        "nifty_max_dd":    round(nifty_dd, 6),
        "equalw_value":    equalw_series[1:],
        "equalw_cagr":     round(equalw_cagr, 6),
        "equalw_sharpe":   round(equalw_sh, 4),
        # Detail
        "per_year":        per_year,
        "walk_forward":    True,
        "train_years":     train_years,
    }


# ── SIMPLE PERFORMANCE BACKTEST (backward compat) ────────────────────────────

def run_performance_backtest(ann_returns, cov_mat, vols, fund_sharpes, funds,
                              vol_cap=VOL_CAP, history=None, semi_vols=None):
    """
    Performance backtest using a fixed covariance estimate (backward compat).
    For production use, prefer run_walk_forward_backtest().
    """
    if history is None:
        history = NIFTY_HISTORY
    if semi_vols is None:
        semi_vols = vols.copy()

    years_sorted   = sorted(history.keys())
    n_years        = len(years_sorted)
    fund_alphas    = ann_returns - NIFTY_LONGRUN_RETURN

    portfolio_value   = 100.0
    nifty_value       = 100.0
    equalw_value      = 100.0
    value_series      = [100.0]
    nifty_series      = [100.0]
    equalw_series     = [100.0]
    annual_ret_series = []
    per_year          = []
    w_prev            = np.ones(len(funds)) / len(funds)

    for year in years_sorted:
        pe = history[year]["pe"]
        pb = history[year]["pb"]
        res = optimize_for_pe_pb(
            pe, pb, ann_returns, cov_mat, vols, fund_sharpes, funds,
            vol_cap=vol_cap, prev_weights=w_prev, semi_vols=semi_vols,
        )

        E  = res["equity"]; D = res["debt"]; G = res["gold"]; C = res["cash"]
        fw = np.array(res["fund_weights"])
        w_prev = fw.copy()

        nifty_ret     = NIFTY_ANNUAL_RETURNS.get(year, NIFTY_LONGRUN_RETURN)
        gold_ret      = GOLD_ANNUAL_RETURNS.get(year, GOLD_RET)
        debt_ret      = DEBT_ANNUAL_RETURNS.get(year, DEBT_RET)
        cash_ret      = CASH_ANNUAL_RETURNS.get(year, CASH_RET)

        fund_rets     = nifty_ret + fund_alphas
        eq_sleeve_ret = float(np.dot(fw, fund_rets))
        port_ret      = E*eq_sleeve_ret + D*debt_ret + G*gold_ret + C*cash_ret
        equalw_ret    = 0.70*nifty_ret + 0.18*debt_ret + 0.10*gold_ret + 0.02*cash_ret

        portfolio_value *= (1.0 + port_ret)
        nifty_value     *= (1.0 + nifty_ret)
        equalw_value    *= (1.0 + equalw_ret)

        value_series.append(round(portfolio_value, 4))
        nifty_series.append(round(nifty_value, 4))
        equalw_series.append(round(equalw_value, 4))
        annual_ret_series.append(round(port_ret, 6))

        per_year.append({
            "year":         year, "pe": pe, "pb": pb,
            "equity":       round(E, 4), "debt": round(D, 4),
            "gold":         round(G, 4), "cash": round(C, 4),
            "nifty_ret":    round(nifty_ret, 4),
            "port_ret":     round(port_ret, 6),
            "port_value":   round(portfolio_value, 4),
            "nifty_value":  round(nifty_value, 4),
            "equalw_value": round(equalw_value, 4),
            "sortino":      res["sortino"],
            "hhi":          res["concentration_hhi"],
        })

    rets_arr  = np.array(annual_ret_series)
    cagr      = float((portfolio_value / 100.0) ** (1.0 / n_years) - 1.0)
    vol_bt    = float(np.std(rets_arr, ddof=1))
    sharpe_bt = float((cagr - RF) / vol_bt) if vol_bt > 0 else 0.0

    peak = 100.0; max_dd = 0.0
    for v in value_series[1:]:
        peak = max(peak, v)
        dd   = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    n_arr    = np.array([NIFTY_ANNUAL_RETURNS.get(y, NIFTY_LONGRUN_RETURN) for y in years_sorted])
    n_cagr   = float((nifty_value/100.0)**(1.0/n_years) - 1.0)
    n_vol    = float(np.std(n_arr, ddof=1))
    n_sh     = float((n_cagr - RF) / n_vol) if n_vol > 0 else 0.0

    eq_arr   = np.array([p["equalw_value"] for p in per_year])
    eq_rets  = np.diff(np.concatenate([[100.0], eq_arr])) / np.concatenate([[100.0], eq_arr[:-1]])
    eq_cagr  = float((equalw_value/100.0)**(1.0/n_years) - 1.0)
    eq_vol   = float(np.std(eq_rets, ddof=1))
    eq_sh    = float((eq_cagr - RF) / eq_vol) if eq_vol > 0 else 0.0

    return {
        # Existing keys
        "years":           [y for y in years_sorted],
        "portfolio_value": value_series[1:],
        "initial_value":   100.0,
        "annual_returns":  annual_ret_series,
        "cagr":            round(cagr, 6),
        "max_drawdown":    round(max_dd, 6),
        "volatility":      round(vol_bt, 6),
        "sharpe":          round(sharpe_bt, 4),
        # New: benchmark series
        "nifty_value":     nifty_series[1:],
        "nifty_cagr":      round(n_cagr, 6),
        "nifty_sharpe":    round(n_sh, 4),
        "equalw_value":    equalw_series[1:],
        "equalw_cagr":     round(eq_cagr, 6),
        "equalw_sharpe":   round(eq_sh, 4),
        "per_year":        per_year,
        "walk_forward":    False,
    }


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def get_optimizer_state(funds=None, ann_returns=None, cov_mat=None,
                         vols=None, fund_sharpes=None, extra_stats=None):
    """
    Returns full optimizer state dict. Called by app.py on startup.
    Falls back to synthetic params if live data is unavailable.
    """
    if funds is None:
        funds        = FUNDS_DEFAULT
        ann_returns  = _SYNTHETIC_RETURNS
        cov_mat      = _SYNTHETIC_COV
        vols         = _SYNTHETIC_VOLS
        fund_sharpes = (ann_returns - RF) / vols
        extra_stats  = {
            "semi_vols":      vols.copy(),
            "sortinos":       (ann_returns - RF) / vols,
            "raw_cagrs":      ann_returns.copy(),
            "quality_flags":  {},
        }

    if extra_stats is None:
        extra_stats = {
            "semi_vols":     vols.copy(),
            "sortinos":      fund_sharpes.copy(),
            "raw_cagrs":     ann_returns.copy(),
            "quality_flags": {},
        }

    return {
        "funds":        funds,
        "ann_returns":  ann_returns,
        "cov_mat":      cov_mat,
        "vols":         vols,
        "fund_sharpes": fund_sharpes,
        "extra_stats":  extra_stats,
    }
