"""
app.py — Flask API Server
==========================
Run:  python app.py
API:  http://localhost:5000

Endpoints:
  GET /api/optimize?pe=22&pb=3.5
  GET /api/backtest
  GET /api/backtest-performance
  GET /api/backtest-walkforward   (NEW: proper walk-forward, no lookahead)
  GET /api/frontier
  GET /api/funds
  GET /api/meta
  GET /health
"""

import os
import sys
import logging
import time

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR  = os.path.join(BASE_DIR, "backend")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
sys.path.insert(0, BACKEND_DIR)

from optimizer import (
    optimize_for_pe_pb,
    compute_frontier,
    run_backtest,
    run_performance_backtest,
    run_walk_forward_backtest,
    build_cov_from_returns,
    get_optimizer_state,
    VOL_CAP, RF,
    NIFTY_MEAN_PE, NIFTY_STD_PE,
    NIFTY_MEAN_PB, NIFTY_STD_PB,
    SHRINKAGE_ALPHA, LAMBDA_DIVERSIFICATION, LAMBDA_TURNOVER, TC_BPS,
    MAX_CAGR_THRESHOLD, MIN_VOL_THRESHOLD,
)
from data_fetcher import get_returns

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

# ── Global state (loaded once on startup) ─────────────────────────────────────
_STATE     = {}
_RETURNS_DF = None   # raw returns DataFrame, kept for walk-forward backtest


def _init_state():
    global _STATE, _RETURNS_DF
    logger.info("Initialising optimizer state...")
    t0 = time.time()

    returns_df, source = get_returns()
    _RETURNS_DF = returns_df
    logger.info("Data source: %s  |  shape: %s", source, returns_df.shape)

    funds = list(returns_df.columns)

    # Build robust estimates (shrinkage returns + LW covariance)
    ann_returns, cov_mat, vols, fund_sharpes, extra_stats = build_cov_from_returns(returns_df)

    semi_vols = extra_stats["semi_vols"]

    state = get_optimizer_state(funds, ann_returns, cov_mat, vols, fund_sharpes, extra_stats)
    state["source"]      = source
    state["returns_df"]  = returns_df   # kept for walk-forward

    # Pre-compute all backtest variants
    state["frontier"] = compute_frontier(ann_returns, cov_mat, semi_vols=semi_vols)
    state["backtest"] = run_backtest(
        ann_returns, cov_mat, vols, fund_sharpes, funds, semi_vols=semi_vols
    )
    state["backtest_performance"] = run_performance_backtest(
        ann_returns, cov_mat, vols, fund_sharpes, funds, semi_vols=semi_vols
    )

    # Walk-forward backtest (uses rolling windows from raw returns_df)
    try:
        state["backtest_walkforward"] = run_walk_forward_backtest(returns_df)
    except Exception as e:
        logger.warning("Walk-forward backtest failed (may need more data): %s", e)
        state["backtest_walkforward"] = {"error": str(e), "walk_forward": True}

    logger.info(
        "State ready in %.1fs — %d funds | source=%s",
        time.time() - t0, len(funds), source,
    )
    _STATE = state


# ── Response helpers ──────────────────────────────────────────────────────────

def _ok(data):
    return jsonify({"status": "ok", "data": data})


def _err(msg, code=400):
    return jsonify({"status": "error", "message": msg}), code


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return _ok({
        "ready":  bool(_STATE),
        "source": _STATE.get("source", "uninitialised"),
    })


@app.route("/api/funds")
def api_funds():
    """Return fund list with robust metrics (shrunk returns, sortino, etc.)."""
    funds       = _STATE["funds"]
    ann_r       = _STATE["ann_returns"].tolist()
    vols        = _STATE["vols"].tolist()
    sharpes     = _STATE["fund_sharpes"].tolist()
    extra       = _STATE.get("extra_stats", {})
    semi_vols   = extra.get("semi_vols", vols)
    sortinos    = extra.get("sortinos", sharpes)
    raw_cagrs   = extra.get("raw_cagrs", ann_r)
    flags       = extra.get("quality_flags", {})

    return _ok({
        "funds": [
            {
                "name":       funds[i],
                "return":     round(ann_r[i], 4),          # shrunk CAGR
                "raw_return": round(float(raw_cagrs[i]), 4) if hasattr(raw_cagrs, '__len__') else round(ann_r[i], 4),
                "vol":        round(vols[i], 4),
                "semi_vol":   round(float(semi_vols[i]), 4),
                "sharpe":     round(sharpes[i], 3),
                "sortino":    round(float(sortinos[i]), 3),
                "flags":      flags.get(funds[i], []),
            }
            for i in range(len(funds))
        ],
        "source": _STATE["source"],
        "model_params": {
            "shrinkage_alpha":       SHRINKAGE_ALPHA,
            "lambda_diversification": LAMBDA_DIVERSIFICATION,
            "lambda_turnover":       LAMBDA_TURNOVER,
            "tc_bps":                TC_BPS,
            "max_cagr_cap":          MAX_CAGR_THRESHOLD,
            "min_vol_floor":         MIN_VOL_THRESHOLD,
        },
    })


@app.route("/api/optimize")
def api_optimize():
    """
    GET /api/optimize?pe=22&pb=3.5

    Returns full portfolio allocation + metrics.
    Passes prev_weights from query param (optional) for turnover penalty.
    """
    try:
        pe = float(request.args.get("pe", 22))
        pb = float(request.args.get("pb", 3.5))
    except ValueError:
        return _err("pe and pb must be numeric")

    if not (10 <= pe <= 45):
        return _err("pe must be between 10 and 45")
    if not (1.0 <= pb <= 7.0):
        return _err("pb must be between 1.0 and 7.0")

    # Optional: client can pass previous weights for turnover penalty
    prev_weights = None
    prev_w_str   = request.args.get("prev_weights", None)
    if prev_w_str:
        try:
            import json as _json
            prev_weights = _json.loads(prev_w_str)
        except Exception:
            pass   # fallback to equal weight if malformed

    extra     = _STATE.get("extra_stats", {})
    semi_vols = extra.get("semi_vols", None)

    try:
        result = optimize_for_pe_pb(
            pe, pb,
            _STATE["ann_returns"],
            _STATE["cov_mat"],
            _STATE["vols"],
            _STATE["fund_sharpes"],
            _STATE["funds"],
            prev_weights=prev_weights,
            semi_vols=semi_vols,
        )
        result["fund_names"] = _STATE["funds"]
        return _ok(result)

    except Exception as e:
        logger.exception("Optimize failed for PE=%.1f PB=%.1f", pe, pb)
        return _err(f"Optimization error: {str(e)}", 500)


@app.route("/api/backtest")
def api_backtest():
    """GET /api/backtest — allocation history backtest."""
    return _ok(_STATE.get("backtest", []))


@app.route("/api/backtest-performance")
def api_backtest_performance():
    """GET /api/backtest-performance — realised performance backtest with benchmarks."""
    return _ok(_STATE.get("backtest_performance", {}))


@app.route("/api/backtest-walkforward")
def api_backtest_walkforward():
    """
    GET /api/backtest-walkforward — proper walk-forward backtest.
    No lookahead bias: each year estimated only from prior data.
    Includes benchmark comparison (Nifty TRI, equal-weight portfolio).
    """
    return _ok(_STATE.get("backtest_walkforward", {}))


@app.route("/api/frontier")
def api_frontier():
    """GET /api/frontier — efficient frontier points."""
    return _ok(_STATE.get("frontier", []))


@app.route("/api/meta")
def api_meta():
    """Return global model config."""
    return _ok({
        "vol_cap":                VOL_CAP,
        "rf":                     RF,
        "nifty_mean_pe":          NIFTY_MEAN_PE,
        "nifty_std_pe":           NIFTY_STD_PE,
        "nifty_mean_pb":          NIFTY_MEAN_PB,
        "nifty_std_pb":           NIFTY_STD_PB,
        "shrinkage_alpha":        SHRINKAGE_ALPHA,
        "lambda_diversification": LAMBDA_DIVERSIFICATION,
        "lambda_turnover":        LAMBDA_TURNOVER,
        "tc_bps":                 TC_BPS,
    })


# ── Serve frontend ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _init_state()
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info("Starting server on http://0.0.0.0:%d  (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
