"""
app.py — Flask API Server (FINAL + CACHING ENABLED)
"""

import os
import sys
import logging
import time
import pickle

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR  = os.path.join(BASE_DIR, "backend")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
CACHE_FILE   = os.path.join(BASE_DIR, "state_cache.pkl")

sys.path.insert(0, BACKEND_DIR)

from optimizer import (
    optimize_for_pe_pb,
    compute_frontier,
    run_backtest,
    run_performance_backtest,
    run_walk_forward_backtest,
    build_cov_from_returns,
    VOL_CAP, RF,
    NIFTY_MEAN_PE, NIFTY_STD_PE,
    NIFTY_MEAN_PB, NIFTY_STD_PB,
)
from data_fetcher import get_returns

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

# ── Global state ──────────────────────────────────────────────────────────────
_STATE = {}
_RETURNS_DF = None


def _init_state():
    global _STATE, _RETURNS_DF

    # ✅ STEP 1: Try loading cache
    if os.path.exists(CACHE_FILE):
        try:
            logger.info("⚡ Loading state from cache...")
            with open(CACHE_FILE, "rb") as f:
                _STATE = pickle.load(f)

            _RETURNS_DF = _STATE.get("returns_df", None)

            if _STATE.get("funds"):
                logger.info("✅ Cache loaded successfully")
                return
        except Exception as e:
            logger.warning("Cache load failed, rebuilding... %s", e)

    # ✅ STEP 2: Build fresh state
    try:
        logger.info("🚀 Initialising optimizer state...")
        t0 = time.time()

        returns_df, source = get_returns()

        if returns_df is None or returns_df.empty or len(returns_df.columns) < 3:
            raise ValueError("Insufficient data from data_fetcher")

        _RETURNS_DF = returns_df
        funds = list(returns_df.columns)

        ann_returns, cov_mat, vols, fund_sharpes, extra_stats = build_cov_from_returns(returns_df)

        state = {
            "funds": funds,
            "ann_returns": ann_returns,
            "cov_mat": cov_mat,
            "vols": vols,
            "fund_sharpes": fund_sharpes,
            "extra_stats": extra_stats,
            "source": source,
            "returns_df": returns_df,
        }

        # Safe computations
        try:
            state["frontier"] = compute_frontier(ann_returns, cov_mat)
        except Exception as e:
            logger.warning("Frontier failed: %s", e)
            state["frontier"] = []

        try:
            state["backtest"] = run_backtest(
                ann_returns, cov_mat, vols, fund_sharpes, funds
            )
        except Exception as e:
            logger.warning("Backtest failed: %s", e)
            state["backtest"] = []

        try:
            state["backtest_performance"] = run_performance_backtest(
                ann_returns, cov_mat, vols, fund_sharpes, funds
            )
        except Exception as e:
            logger.warning("Performance backtest failed: %s", e)
            state["backtest_performance"] = {}

        try:
            state["backtest_walkforward"] = run_walk_forward_backtest(returns_df)
        except Exception as e:
            logger.warning("Walk-forward failed: %s", e)
            state["backtest_walkforward"] = {"error": str(e)}

        _STATE = state

        logger.info(
            "✅ State ready in %.2fs | funds=%d | source=%s",
            time.time() - t0,
            len(funds),
            source,
        )

        # ✅ STEP 3: Save cache
        try:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(_STATE, f)
            logger.info("💾 Cache saved successfully")
        except Exception as e:
            logger.warning("Cache save failed: %s", e)

    except Exception as e:
        import traceback
        logger.error("🔥 INIT FAILED")
        traceback.print_exc()

        _STATE = {
            "funds": [],
            "ann_returns": [],
            "cov_mat": [],
            "vols": [],
            "fund_sharpes": [],
            "extra_stats": {},
            "source": "failed",
            "frontier": [],
            "backtest": [],
            "backtest_performance": {},
            "backtest_walkforward": {},
        }


# 🔴 CRITICAL: runs for gunicorn
_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _ok(data):
    return jsonify({"status": "ok", "data": data})


def _err(msg, code=400):
    return jsonify({"status": "error", "message": msg}), code


def _require_state():
    return bool(_STATE.get("funds"))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return _ok({
        "ready": _require_state(),
        "source": _STATE.get("source"),
    })


@app.route("/api/funds")
def api_funds():
    if not _require_state():
        return _err("State not initialized", 500)

    return _ok({
        "funds": [
            {
                "name": _STATE["funds"][i],
                "return": round(float(_STATE["ann_returns"][i]), 4),
                "vol": round(float(_STATE["vols"][i]), 4),
                "sharpe": round(float(_STATE["fund_sharpes"][i]), 3),
            }
            for i in range(len(_STATE["funds"]))
        ],
        "source": _STATE["source"],
    })


@app.route("/api/optimize")
def api_optimize():
    if not _require_state():
        return _err("State not initialized", 500)

    try:
        pe = float(request.args.get("pe", 22))
        pb = float(request.args.get("pb", 3.5))

        result = optimize_for_pe_pb(
            pe, pb,
            _STATE["ann_returns"],
            _STATE["cov_mat"],
            _STATE["vols"],
            _STATE["fund_sharpes"],
            _STATE["funds"],
        )

        result["fund_names"] = _STATE["funds"]

        # ── Buy/Sell signal computation ───────────────────────────────────
        target_weights  = result["fund_weights"]
        N               = len(target_weights)
        current_weights = [1.0 / N] * N

        delta_weights = [
            float(target_weights[i]) - float(current_weights[i])
            for i in range(N)
        ]

        turnover = sum(abs(dw) for dw in delta_weights)
        cost_estimate = turnover * 0.002

        rebalance_action = "SKIP" if turnover < 0.05 else "REBALANCE"

        result["delta_weights"]    = delta_weights
        result["turnover"]         = float(turnover)
        result["cost_estimate"]    = float(cost_estimate)
        result["rebalance_action"] = rebalance_action
        # ─────────────────────────────────────────────────────────────────

        return _ok(result)

    except Exception as e:
        logger.exception("Optimize failed")
        return _err(str(e), 500)


@app.route("/api/frontier")
def api_frontier():
    return _ok(_STATE.get("frontier", []))


@app.route("/api/backtest")
def api_backtest():
    return _ok(_STATE.get("backtest", []))


@app.route("/api/backtest-performance")
def api_backtest_performance():
    return _ok(_STATE.get("backtest_performance", {}))


@app.route("/api/backtest-walkforward")
def api_backtest_walkforward():
    return _ok(_STATE.get("backtest_walkforward", {}))


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
