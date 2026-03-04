"""Benchmark: scan-based vs convolution-based estimation for panel models.

Compares pytensor.scan-based generative models against convolution-based
(current pathmc compiler) estimation for three representative panel DGPs:

1. **Adstock only**: sales ~ adstock(spend, decay=theta)
2. **Adstock + AR**: sales ~ adstock(spend, decay=theta) + lag(sales)
3. **Multi-equation**: awareness ~ adstock(spend, decay=theta); sales ~ awareness + lag(sales)

Gate criteria (from scan_unification_plan.md):
- ≤3x slower → proceed to Phase 2
- 3–5x slower → proceed with note about future convolution fast-path
- >5x slower → abort; promote current scan engine as sole panel engine

Usage:
    python benchmarks/scan_vs_conv.py
    python benchmarks/scan_vs_conv.py --draws 200 --tune 200  # quick run
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt

import pathmc

N_UNITS = 5
T = 100
PANEL = {"unit": "region", "time": "week"}

SAMPLE_KWARGS: dict = dict(
    cores=1,
    target_accept=0.95,
)


# ---------------------------------------------------------------------------
# DGP generation  (spend scaled to ~[0, 1] so adstock stays moderate)
# ---------------------------------------------------------------------------

TRUE_PARAMS = {
    "adstock_only": {
        "intercept": 2.0,
        "beta_spend": 3.0,
        "decay": 0.7,
        "sigma": 0.5,
    },
    "adstock_ar": {
        "intercept": 1.0,
        "beta_spend": 2.0,
        "ar1": 0.3,
        "decay": 0.7,
        "sigma": 0.5,
    },
    "multi_eq": {
        "intercept_awareness": 0.0,
        "beta_spend": 3.0,
        "decay": 0.7,
        "sigma_awareness": 0.3,
        "intercept_sales": 1.0,
        "beta_awareness": 0.6,
        "ar1_sales": 0.3,
        "sigma_sales": 0.5,
    },
}


def _apply_adstock_numpy(x: np.ndarray, decay: float) -> np.ndarray:
    out = np.zeros_like(x)
    for t in range(len(x)):
        out[t] = x[t] + (decay * out[t - 1] if t > 0 else 0.0)
    return out


def generate_adstock_only(rng: np.random.Generator) -> pd.DataFrame:
    p = TRUE_PARAMS["adstock_only"]
    rows = []
    for uid in range(N_UNITS):
        spend = rng.uniform(0, 1, size=T)
        adstocked = _apply_adstock_numpy(spend, p["decay"])
        sales = (
            p["intercept"] + p["beta_spend"] * adstocked + rng.normal(0, p["sigma"], T)
        )
        for t in range(T):
            rows.append(
                {
                    "region": f"u{uid}",
                    "week": t + 1,
                    "spend": spend[t],
                    "sales": sales[t],
                }
            )
    return pd.DataFrame(rows)


def generate_adstock_ar(rng: np.random.Generator) -> pd.DataFrame:
    p = TRUE_PARAMS["adstock_ar"]
    rows = []
    for uid in range(N_UNITS):
        spend = rng.uniform(0, 1, size=T)
        adstocked = _apply_adstock_numpy(spend, p["decay"])
        sales = np.zeros(T)
        for t in range(T):
            prev = sales[t - 1] if t > 0 else p["intercept"] / (1 - p["ar1"])
            sales[t] = (
                p["intercept"]
                + p["beta_spend"] * adstocked[t]
                + p["ar1"] * prev
                + rng.normal(0, p["sigma"])
            )
        for t in range(T):
            rows.append(
                {
                    "region": f"u{uid}",
                    "week": t + 1,
                    "spend": spend[t],
                    "sales": sales[t],
                }
            )
    return pd.DataFrame(rows)


def generate_multi_eq(rng: np.random.Generator) -> pd.DataFrame:
    p = TRUE_PARAMS["multi_eq"]
    rows = []
    for uid in range(N_UNITS):
        spend = rng.uniform(0, 1, size=T)
        adstocked = _apply_adstock_numpy(spend, p["decay"])
        awareness = (
            p["intercept_awareness"]
            + p["beta_spend"] * adstocked
            + rng.normal(0, p["sigma_awareness"], T)
        )
        sales = np.zeros(T)
        for t in range(T):
            prev = (
                sales[t - 1] if t > 0 else p["intercept_sales"] / (1 - p["ar1_sales"])
            )
            sales[t] = (
                p["intercept_sales"]
                + p["beta_awareness"] * awareness[t]
                + p["ar1_sales"] * prev
                + rng.normal(0, p["sigma_sales"])
            )
        for t in range(T):
            rows.append(
                {
                    "region": f"u{uid}",
                    "week": t + 1,
                    "spend": spend[t],
                    "awareness": awareness[t],
                    "sales": sales[t],
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convolution-based models (current pathmc compiler, no random effects)
# ---------------------------------------------------------------------------


def fit_conv_adstock_only(
    df: pd.DataFrame,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
) -> az.InferenceData:
    model = pathmc.fit("sales ~ adstock(spend, decay=theta)", data=df, panel=PANEL)
    return model.sample(
        draws=draws, tune=tune, chains=chains, random_seed=seed, **SAMPLE_KWARGS
    )


def fit_conv_adstock_ar(
    df: pd.DataFrame,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
) -> az.InferenceData:
    model = pathmc.fit(
        "sales ~ adstock(spend, decay=theta) + lag(sales)", data=df, panel=PANEL
    )
    return model.sample(
        draws=draws, tune=tune, chains=chains, random_seed=seed, **SAMPLE_KWARGS
    )


def fit_conv_multi_eq(
    df: pd.DataFrame,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
) -> az.InferenceData:
    model = pathmc.fit(
        "awareness ~ adstock(spend, decay=theta); sales ~ awareness + lag(sales)",
        data=df,
        panel=PANEL,
    )
    return model.sample(
        draws=draws, tune=tune, chains=chains, random_seed=seed, **SAMPLE_KWARGS
    )


# ---------------------------------------------------------------------------
# Scan-based models (hand-built pytensor.scan, no random effects)
# ---------------------------------------------------------------------------


def _panel_sort_index(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Sort panel data by (unit, time) and return index mappings."""
    sorted_df = df.sort_values(["region", "week"])
    sort_idx = sorted_df.index.values
    reverse_idx = np.argsort(sort_idx)
    n_units = df["region"].nunique()
    n_time = len(df) // n_units
    return sort_idx, reverse_idx, n_units, n_time


def _sample(
    model: pm.Model, draws: int, tune: int, chains: int, seed: int
) -> az.InferenceData:
    kwargs = dict(
        draws=draws, tune=tune, chains=chains, random_seed=seed, **SAMPLE_KWARGS
    )
    if sys.platform == "darwin":
        kwargs["mp_ctx"] = "forkserver"
    with model:
        return pm.sample(**kwargs)


def fit_scan_adstock_only(
    df: pd.DataFrame,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
) -> az.InferenceData:
    sort_idx, _, n_units, n_time = _panel_sort_index(df)
    spend_sorted = df["spend"].values[sort_idx].reshape(n_units, n_time).T
    sales_sorted = df["sales"].values[sort_idx].reshape(n_units, n_time).T

    with pm.Model() as gen_model:
        spend_data = pm.Data("spend", spend_sorted)
        decay = pm.Beta("theta", alpha=2, beta=2)
        beta = pm.Normal("beta_sales", mu=0, sigma=10, shape=2)
        sigma = pm.HalfNormal("sigma_sales", sigma=1)

        def step(spend_t, prev_mu, prev_adstock, beta_, decay_):
            adstock_t = spend_t + decay_ * prev_adstock
            mu_t = beta_[0] + beta_[1] * adstock_t
            return mu_t, adstock_t

        (mu_all, _), _ = pytensor.scan(
            fn=step,
            sequences=[spend_data],
            outputs_info=[pt.zeros(n_units), pt.zeros(n_units)],
            non_sequences=[beta, decay],
        )
        pm.Deterministic("mu_sales", mu_all)
        pm.Normal("sales", mu=mu_all, sigma=sigma, shape=(n_time, n_units))

    est_model = pm.observe(gen_model, {"sales": sales_sorted})
    return _sample(est_model, draws, tune, chains, seed)


def fit_scan_adstock_ar(
    df: pd.DataFrame,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
) -> az.InferenceData:
    sort_idx, _, n_units, n_time = _panel_sort_index(df)

    spend_sorted = df["spend"].values[sort_idx].reshape(n_units, n_time).T
    sales_sorted = df["sales"].values[sort_idx].reshape(n_units, n_time).T

    with pm.Model() as gen_model:
        spend_data = pm.Data("spend", spend_sorted)
        decay = pm.Beta("theta", alpha=2, beta=2)
        beta = pm.Normal("beta_sales", mu=0, sigma=10, shape=3)
        sigma = pm.HalfNormal("sigma_sales", sigma=1)

        def step(spend_t, prev_mu, prev_adstock, beta_, decay_):
            adstock_t = spend_t + decay_ * prev_adstock
            mu_t = beta_[0] + beta_[1] * adstock_t + beta_[2] * prev_mu
            return mu_t, adstock_t

        (mu_all, _), _ = pytensor.scan(
            fn=step,
            sequences=[spend_data],
            outputs_info=[pt.zeros(n_units), pt.zeros(n_units)],
            non_sequences=[beta, decay],
        )
        pm.Deterministic("mu_sales", mu_all)
        pm.Normal("sales", mu=mu_all, sigma=sigma, shape=(n_time, n_units))

    est_model = pm.observe(gen_model, {"sales": sales_sorted})
    return _sample(est_model, draws, tune, chains, seed)


def fit_scan_multi_eq(
    df: pd.DataFrame,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
) -> az.InferenceData:
    sort_idx, _, n_units, n_time = _panel_sort_index(df)

    spend_sorted = df["spend"].values[sort_idx].reshape(n_units, n_time).T
    awareness_sorted = df["awareness"].values[sort_idx].reshape(n_units, n_time).T
    sales_sorted = df["sales"].values[sort_idx].reshape(n_units, n_time).T

    with pm.Model() as gen_model:
        spend_data = pm.Data("spend", spend_sorted)
        decay = pm.Beta("theta", alpha=2, beta=2)
        beta_aw = pm.Normal("beta_awareness", mu=0, sigma=10, shape=2)
        sigma_aw = pm.HalfNormal("sigma_awareness", sigma=1)
        beta_sl = pm.Normal("beta_sales", mu=0, sigma=10, shape=3)
        sigma_sl = pm.HalfNormal("sigma_sales", sigma=1)

        def step(spend_t, prev_sales_mu, prev_adstock, beta_aw_, beta_sl_, decay_):
            adstock_t = spend_t + decay_ * prev_adstock
            mu_awareness_t = beta_aw_[0] + beta_aw_[1] * adstock_t
            mu_sales_t = (
                beta_sl_[0] + beta_sl_[1] * mu_awareness_t + beta_sl_[2] * prev_sales_mu
            )
            return mu_sales_t, adstock_t, mu_awareness_t

        (mu_sales_all, _, mu_aw_all), _ = pytensor.scan(
            fn=step,
            sequences=[spend_data],
            outputs_info=[pt.zeros(n_units), pt.zeros(n_units), None],
            non_sequences=[beta_aw, beta_sl, decay],
        )
        pm.Deterministic("mu_awareness", mu_aw_all)
        pm.Deterministic("mu_sales", mu_sales_all)
        pm.Normal("awareness", mu=mu_aw_all, sigma=sigma_aw, shape=(n_time, n_units))
        pm.Normal("sales", mu=mu_sales_all, sigma=sigma_sl, shape=(n_time, n_units))

    est_model = pm.observe(
        gen_model, {"awareness": awareness_sorted, "sales": sales_sorted}
    )
    return _sample(est_model, draws, tune, chains, seed)


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    name: str
    method: str
    wall_time_s: float
    ess_bulk: dict[str, float]
    ess_per_sec: dict[str, float]
    divergences: int
    recovery: dict[str, tuple[float, float]]


def _ess_bulk(idata: az.InferenceData, params: list[str]) -> dict[str, float]:
    try:
        ess_ds = az.ess(idata, var_names=params, method="bulk")
    except Exception:
        return {p: float("nan") for p in params}
    result = {}
    for p in params:
        try:
            val = ess_ds[p]
            result[p] = float(np.atleast_1d(val.values).mean())
        except (KeyError, AttributeError):
            result[p] = float("nan")
    return result


def _divergences(idata: az.InferenceData) -> int:
    try:
        return int(idata.sample_stats["diverging"].sum())
    except (KeyError, AttributeError):
        return -1


def _posterior_mean(idata: az.InferenceData, param: str) -> float:
    try:
        return float(idata.posterior[param].mean())
    except (KeyError, AttributeError):
        return float("nan")


def run_benchmark(
    name: str,
    method: str,
    fit_fn,
    df: pd.DataFrame,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
    ess_params: list[str],
    recovery_map: dict[str, float],
) -> BenchmarkResult:
    print(f"\n{'=' * 60}")
    print(f"  {name} — {method}")
    print(f"{'=' * 60}")

    t0 = time.perf_counter()
    idata = fit_fn(df, draws, tune, chains, seed)
    wall = time.perf_counter() - t0

    ess = _ess_bulk(idata, ess_params)
    ess_s = {k: v / wall for k, v in ess.items()}
    divs = _divergences(idata)

    recovery = {}
    for param, true_val in recovery_map.items():
        post_mean = _posterior_mean(idata, param)
        recovery[param] = (true_val, post_mean)

    return BenchmarkResult(
        name=name,
        method=method,
        wall_time_s=wall,
        ess_bulk=ess,
        ess_per_sec=ess_s,
        divergences=divs,
        recovery=recovery,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_result(r: BenchmarkResult) -> None:
    print(f"\n--- {r.name} [{r.method}] ---")
    print(f"  Wall time:    {r.wall_time_s:.1f}s")
    print(f"  Divergences:  {r.divergences}")
    print("  ESS (bulk):")
    for p, v in r.ess_bulk.items():
        print(f"    {p:30s}  {v:8.0f}")
    print("  ESS/second:")
    for p, v in r.ess_per_sec.items():
        print(f"    {p:30s}  {v:8.1f}")
    print("  Recovery (true → posterior mean):")
    for p, (true, post) in r.recovery.items():
        print(f"    {p:30s}  {true:7.2f} → {post:7.2f}")


def print_comparison(results: list[BenchmarkResult]) -> None:
    print("\n" + "=" * 70)
    print("  GATE DECISION SUMMARY")
    print("=" * 70)

    by_name: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        by_name.setdefault(r.name, []).append(r)

    for name, rs in by_name.items():
        conv = next((r for r in rs if r.method == "convolution"), None)
        scan = next((r for r in rs if r.method == "scan"), None)
        if conv is None or scan is None:
            continue

        wall_ratio = (
            scan.wall_time_s / conv.wall_time_s
            if conv.wall_time_s > 0
            else float("inf")
        )

        shared_params = set(conv.ess_per_sec) & set(scan.ess_per_sec)
        valid_params = [
            p
            for p in shared_params
            if np.isfinite(conv.ess_per_sec[p])
            and np.isfinite(scan.ess_per_sec[p])
            and scan.ess_per_sec[p] > 0
        ]

        if valid_params:
            avg_conv_ess = np.mean([conv.ess_per_sec[p] for p in valid_params])
            avg_scan_ess = np.mean([scan.ess_per_sec[p] for p in valid_params])
            ess_ratio = avg_conv_ess / avg_scan_ess
        else:
            ess_ratio = float("nan")

        print(f"\n  {name}:")
        print(f"    Wall-time ratio (scan/conv):     {wall_ratio:.2f}x")
        if np.isfinite(ess_ratio):
            print(f"    ESS/s ratio (conv/scan):         {ess_ratio:.2f}x")
        else:
            print("    ESS/s ratio (conv/scan):         N/A")
        print(f"    Conv divergences:                {conv.divergences}")
        print(f"    Scan divergences:                {scan.divergences}")

        gate_metric = ess_ratio if np.isfinite(ess_ratio) else wall_ratio
        metric_name = "ESS/s" if np.isfinite(ess_ratio) else "wall-time"
        if gate_metric <= 3.0:
            verdict = f"PASS (≤3x on {metric_name}) — proceed to Phase 2"
        elif gate_metric <= 5.0:
            verdict = f"MARGINAL (3–5x on {metric_name}) — proceed with convolution fast-path note"
        else:
            verdict = f"FAIL (>5x on {metric_name}) — abort scan+do unification"
        print(f"    Gate verdict:                    {verdict}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan vs convolution benchmark")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    results: list[BenchmarkResult] = []
    ess_common = ["theta", "beta_sales", "sigma_sales"]

    # ---- Model 1: Adstock only ----
    df1 = generate_adstock_only(rng)
    for method, fn in [
        ("convolution", fit_conv_adstock_only),
        ("scan", fit_scan_adstock_only),
    ]:
        results.append(
            run_benchmark(
                "adstock_only",
                method,
                fn,
                df1,
                args.draws,
                args.tune,
                args.chains,
                args.seed,
                ess_params=ess_common,
                recovery_map={"theta": 0.7},
            )
        )
        print_result(results[-1])

    # ---- Model 2: Adstock + AR ----
    df2 = generate_adstock_ar(rng)
    for method, fn in [
        ("convolution", fit_conv_adstock_ar),
        ("scan", fit_scan_adstock_ar),
    ]:
        results.append(
            run_benchmark(
                "adstock_ar",
                method,
                fn,
                df2,
                args.draws,
                args.tune,
                args.chains,
                args.seed,
                ess_params=ess_common,
                recovery_map={"theta": 0.7},
            )
        )
        print_result(results[-1])

    # ---- Model 3: Multi-equation ----
    df3 = generate_multi_eq(rng)
    ess_multi = [
        "theta",
        "beta_awareness",
        "beta_sales",
        "sigma_awareness",
        "sigma_sales",
    ]
    for method, fn in [("convolution", fit_conv_multi_eq), ("scan", fit_scan_multi_eq)]:
        results.append(
            run_benchmark(
                "multi_eq",
                method,
                fn,
                df3,
                args.draws,
                args.tune,
                args.chains,
                args.seed,
                ess_params=ess_multi,
                recovery_map={"theta": 0.7},
            )
        )
        print_result(results[-1])

    print_comparison(results)


if __name__ == "__main__":
    main()
