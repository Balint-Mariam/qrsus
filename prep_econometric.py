import numpy as np
import pandas as pd
import matplotlib
import warnings
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tools.sm_exceptions import IterationLimitWarning

warnings.filterwarnings("ignore", category=IterationLimitWarning)

try:
    from arch.unitroot import PhillipsPerron
    _HAS_PP = True
except Exception:
    _HAS_PP = False

in_path = r"C:\Users\user\Desktop\Doctorat\simpozion\date_simp_concat.xlsx"

# Read the concatenated Excel (Date is first column/index)
df = pd.read_excel(in_path)

# Normalize column names (strip whitespace)
df.columns = [str(c).strip() for c in df.columns]

# If the first column is Date, set it as index
if df.columns[0].lower() == "date":
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.set_index("Date")

# Required columns
col_esg = "STOXX 600 ESG-X"
col_mkt = "STOXX 600"
col_vstoxx = "VSTOXX"
col_bund = "Bunds 10Y"
col_ttf = "TTF"
col_brent = "Brent"
col_eua = "EUA"

required = [col_esg, col_mkt, col_vstoxx, col_bund, col_ttf, col_brent, col_eua]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in input: {missing}")

# Compute transformations
r_esg = np.log(df[col_esg] / df[col_esg].shift(1))
r_mkt = np.log(df[col_mkt] / df[col_mkt].shift(1))
spread = r_esg - r_mkt

d_vstoxx = df[col_vstoxx] - df[col_vstoxx].shift(1)
d_bund = df[col_bund] - df[col_bund].shift(1)

r_ttf = np.log(df[col_ttf] / df[col_ttf].shift(1))
r_brent = np.log(df[col_brent] / df[col_brent].shift(1))
r_eua = np.log(df[col_eua] / df[col_eua].shift(1))

out = pd.DataFrame({
    "spread": spread,
    "d_vstoxx": d_vstoxx,
    "d_bund": d_bund,
    "r_ttf": r_ttf,
    "r_brent": r_brent,
    "r_eua": r_eua,
}, index=df.index)

out = out.dropna()

def run_adf(series):
    res = adfuller(series, autolag="AIC")
    return {
        "stat": res[0],
        "pvalue": res[1],
        "lags": res[2],
        "nobs": res[3],
    }

def run_pp(series):
    if not _HAS_PP:
        return None
    pp = PhillipsPerron(series)
    return {
        "stat": pp.stat,
        "pvalue": pp.pvalue,
        "lags": pp.lags,
        "nobs": pp.nobs,
    }

# Stationarity tests (ADF + PP)
rows = []
for col in out.columns:
    s = out[col].dropna()
    adf = run_adf(s)
    pp = run_pp(s)
    rows.append({
        "series": col,
        "adf_stat": adf["stat"],
        "adf_pvalue": adf["pvalue"],
        "adf_lags": adf["lags"],
        "adf_nobs": adf["nobs"],
        "pp_stat": None if pp is None else pp["stat"],
        "pp_pvalue": None if pp is None else pp["pvalue"],
        "pp_lags": None if pp is None else pp["lags"],
        "pp_nobs": None if pp is None else pp["nobs"],
    })

results = pd.DataFrame(rows)
print("ADF + PP results:")
print(results)

if not _HAS_PP:
    print("\nPP test not available. Install with: pip install arch")

# Descriptive stats
desc = out.describe().T
print("\nDescriptive statistics:")
print(desc)

# Correlation matrix
corr = out.corr()
print("\nCorrelation matrix:")
print(corr)

# VIF (exclude dependent variable: spread)
X = add_constant(out.drop(columns=["spread"]))
vif_rows = []
for i, col in enumerate(X.columns):
    if col == "const":
        continue
    vif_rows.append({
        "variable": col,
        "vif": variance_inflation_factor(X.values, i),
    })
vif = pd.DataFrame(vif_rows)
print("\nVIF:")
print(vif)

# Quantile Regression (with block bootstrap SE)
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

# Block bootstrap settings
block_len = 22
B = 1000
rng = np.random.default_rng(42)

def block_bootstrap_indices(n, block_len, rng):
    starts = rng.integers(0, n - block_len + 1, size=int(np.ceil(n / block_len)))
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
    return idx

def rho_tau(u, tau):
    return u * (tau - (u < 0))

def run_qr_block_bootstrap(out_df, label, B_override=None):
    y = out_df["spread"]
    X = out_df[["d_vstoxx", "d_bund", "r_ttf", "r_brent", "r_eua"]]
    X = add_constant(X)

    qr_rows = []
    qr_residuals = {}
    qr_boot_coefs = {q: [] for q in quantiles}
    qr_pseudo_r2_rows = []

    n = len(y)
    b_runs = B if B_override is None else B_override
    boot_indices = [block_bootstrap_indices(n, block_len, rng) for _ in range(b_runs)]

    for q in quantiles:
        model = QuantReg(y, X)
        res = model.fit(q=q)

        resid = y - res.predict(X)
        qr_residuals[q] = resid

        boot_coefs = []
        for idx in boot_indices:
            yb = y.iloc[idx]
            Xb = X.iloc[idx]
            boot_res = QuantReg(yb, Xb).fit(q=q)
            boot_coefs.append(boot_res.params.values)

        boot_coefs = np.asarray(boot_coefs)
        qr_boot_coefs[q] = boot_coefs
        boot_se = boot_coefs.std(axis=0, ddof=1)
        ci_low = np.percentile(boot_coefs, 2.5, axis=0)
        ci_high = np.percentile(boot_coefs, 97.5, axis=0)

        u = y - res.predict(X)
        qy = y.quantile(q)
        u0 = y - qy
        pseudo_r2 = 1 - (rho_tau(u, q).sum() / rho_tau(u0, q).sum())
        qr_pseudo_r2_rows.append({"quantile": q, "pseudo_r2": pseudo_r2})

        for i, param in enumerate(res.params.index):
            qr_rows.append({
                "quantile": q,
                "param": param,
                "coef": res.params[param],
                "boot_se": boot_se[i],
                "ci_low": ci_low[i],
                "ci_high": ci_high[i],
            })

    qr_results = pd.DataFrame(qr_rows)
    qr_pseudo_r2 = pd.DataFrame(qr_pseudo_r2_rows)

    # Quantile-difference tests
    qd_pairs = [(0.1, 0.5), (0.1, 0.9), (0.5, 0.9)]
    qd_rows = []
    param_names = list(X.columns)
    for q1, q2 in qd_pairs:
        b1 = qr_boot_coefs[q1]
        b2 = qr_boot_coefs[q2]
        diffs = b1 - b2
        for j, param in enumerate(param_names):
            d = diffs[:, j]
            pval = 2 * min((d >= 0).mean(), (d <= 0).mean())
            qd_rows.append({
                "q1": q1,
                "q2": q2,
                "param": param,
                "diff_mean": d.mean(),
                "diff_se": d.std(ddof=1),
                "pvalue": pval,
            })

    qd_results = pd.DataFrame(qd_rows)

    # Diagnostics on residuals
    diag_rows = []
    lb_lags = 10
    arch_lags = 10
    for q, resid in qr_residuals.items():
        resid = resid.dropna()
        lb = acorr_ljungbox(resid, lags=[lb_lags], return_df=True)
        arch = het_arch(resid, nlags=arch_lags)
        jb = jarque_bera(resid)
        diag_rows.append({
            "quantile": q,
            "ljungbox_lag": lb_lags,
            "ljungbox_stat": lb["lb_stat"].iloc[0],
            "ljungbox_pvalue": lb["lb_pvalue"].iloc[0],
            "arch_lags": arch_lags,
            "arch_stat": arch[0],
            "arch_pvalue": arch[1],
            "jb_stat": jb[0],
            "jb_pvalue": jb[1],
        })

    diag_results = pd.DataFrame(diag_rows)

    return qr_results, qr_pseudo_r2, qd_results, diag_results

def run_qr_rolling(out_df, window, label):
    y_all = out_df["spread"]
    X_all = out_df[["d_vstoxx", "d_bund", "r_ttf", "r_brent", "r_eua"]]
    rows = []
    for end_idx in range(window, len(out_df) + 1):
        y = y_all.iloc[end_idx - window:end_idx]
        X = X_all.iloc[end_idx - window:end_idx]
        X = add_constant(X)
        end_date = out_df.index[end_idx - 1]
        for q in quantiles:
            res = QuantReg(y, X).fit(q=q)
            for param, val in res.params.items():
                rows.append({
                    "window": window,
                    "end_date": end_date,
                    "quantile": q,
                    "param": param,
                    "coef": val,
                })
    return pd.DataFrame(rows)

def run_qlp_quantile(
    out_df,
    horizons,
    qlp_quantiles,
    B_qlp=500,
    block_len=22,
    standardize_shocks=True,
    label="full",
):
    # Main contemporaneous shock variables in the QLP equation.
    shock_vars = ["d_vstoxx", "d_bund", "r_ttf", "r_brent", "r_eua"]

    # Lag controls are kept in original units for transparent interpretation.
    work = out_df.copy()
    work["spread_lag1"] = work["spread"].shift(1)
    work["spread_lag2"] = work["spread"].shift(2)
    work["d_vstoxx_lag1"] = work["d_vstoxx"].shift(1)
    work["d_bund_lag1"] = work["d_bund"].shift(1)
    work["r_ttf_lag1"] = work["r_ttf"].shift(1)
    work["r_brent_lag1"] = work["r_brent"].shift(1)
    work["r_eua_lag1"] = work["r_eua"].shift(1)

    if standardize_shocks:
        # Standardize only contemporaneous shocks (within the current sample).
        for var in shock_vars:
            mu = work[var].mean()
            sigma = work[var].std(ddof=1)
            if pd.isna(sigma) or sigma == 0:
                work[var] = np.nan
            else:
                work[var] = (work[var] - mu) / sigma

    control_vars = [
        "spread_lag1",
        "spread_lag2",
        "d_vstoxx_lag1",
        "d_bund_lag1",
        "r_ttf_lag1",
        "r_brent_lag1",
        "r_eua_lag1",
    ]
    regressors = shock_vars + control_vars
    rng_local = np.random.default_rng(42)

    qlp_rows = []
    for h in horizons:
        # Cumulative future spread from t+1 to t+h.
        future_spread_h = sum(work["spread"].shift(-k) for k in range(1, h + 1))

        model_df = pd.DataFrame({"future_spread_h": future_spread_h}, index=work.index)
        for col in regressors:
            model_df[col] = work[col]
        model_df = model_df.dropna()

        if model_df.empty:
            continue

        y = model_df["future_spread_h"]
        X = add_constant(model_df[regressors], has_constant="add")
        nobs = int(len(model_df))
        block_len_eff = min(block_len, nobs)

        if block_len_eff < 2:
            continue

        # Pre-generate bootstrap block indices for each (horizon, quantile) design sample.
        boot_indices = [
            block_bootstrap_indices(nobs, block_len_eff, rng_local)
            for _ in range(B_qlp)
        ]

        for q in qlp_quantiles:
            try:
                res = QuantReg(y, X).fit(q=q)
            except Exception:
                continue

            param_index = list(res.params.index)
            boot_draws = []
            for idx in boot_indices:
                yb = y.iloc[idx]
                Xb = X.iloc[idx]
                try:
                    boot_res = QuantReg(yb, Xb).fit(q=q)
                    boot_draws.append(boot_res.params.reindex(param_index).values)
                except Exception:
                    continue

            valid_bootstrap = len(boot_draws)
            if valid_bootstrap > 0:
                boot_arr = np.asarray(boot_draws)
                boot_se = boot_arr.std(axis=0, ddof=1)
                ci_low = np.percentile(boot_arr, 2.5, axis=0)
                ci_high = np.percentile(boot_arr, 97.5, axis=0)
            else:
                boot_se = np.full(len(param_index), np.nan)
                ci_low = np.full(len(param_index), np.nan)
                ci_high = np.full(len(param_index), np.nan)

            for i, param in enumerate(param_index):
                qlp_rows.append({
                    "sample_label": label,
                    "horizon": h,
                    "quantile": q,
                    "param": param,
                    "coef": res.params[param],
                    "boot_se": boot_se[i],
                    "ci_low": ci_low[i],
                    "ci_high": ci_high[i],
                    "nobs": nobs,
                    "valid_bootstrap_replications": valid_bootstrap,
                })

    return pd.DataFrame(qlp_rows)


# Economic magnitude / standardized shock effects.
DRIVER_COLS = ["d_vstoxx", "d_bund", "r_ttf", "r_brent", "r_eua"]
DRIVER_DISPLAY = {
    "d_vstoxx": "\u0394VSTOXX",
    "d_bund": "\u0394Bund 10Y",
    "r_ttf": "TTF Return",
    "r_brent": "Brent Return",
    "r_eua": "EUA Return",
}


def console_driver_name(var):
    return DRIVER_DISPLAY[var].replace("\u0394", "Delta ")


def q_label(q):
    return f"q{float(q):.2f}"


def compute_driver_standard_deviations(out_df, driver_cols):
    missing = [col for col in driver_cols if col not in out_df.columns]
    if missing:
        raise ValueError(f"Missing drivers for shock-size computation: {missing}")

    sd_map = {}
    for col in driver_cols:
        sd = out_df[col].dropna().std(ddof=1)
        if pd.isna(sd) or sd <= 0:
            raise ValueError(f"Invalid standard deviation for {col}: {sd}")
        sd_map[col] = float(sd)
    return sd_map


def build_driver_shock_sizes(out_df, subsamples, driver_cols):
    rows = []
    sd_maps = {"full": compute_driver_standard_deviations(out_df, driver_cols)}

    for label, (start, end) in subsamples.items():
        sub = out_df.loc[start:] if end is None else out_df.loc[start:end]
        if len(sub) == 0:
            continue
        sd_maps[label] = compute_driver_standard_deviations(sub, driver_cols)

    for label, sd_map in sd_maps.items():
        for var, sd_value in sd_map.items():
            rows.append({
                "sample_label": label,
                "variable": var,
                "display_name": DRIVER_DISPLAY[var],
                "sd_value": sd_value,
                "shock_label": "+1 standard deviation shock",
                "interpretation_unit": "basis points of ESG-conventional spread",
            })

    return pd.DataFrame(rows), sd_maps


def ci_excludes_zero(row):
    ci_low = row.get("ci_low")
    ci_high = row.get("ci_high")
    return bool(pd.notna(ci_low) and pd.notna(ci_high) and ((ci_low > 0) or (ci_high < 0)))


def add_economic_magnitude(
    results_df,
    sd_map,
    sample_label=None,
    sd_type="full_sample_sd",
    coef_col="coef",
    param_col="param",
    qlp_shocks_already_standardized=False,
):
    if results_df is None or results_df.empty:
        return pd.DataFrame()

    work = results_df.copy()
    work = work[work[param_col].isin(DRIVER_COLS)].copy()
    if work.empty:
        return work

    if sample_label is not None and "sample_label" not in work.columns:
        work["sample_label"] = sample_label

    work["display_name"] = work[param_col].map(DRIVER_DISPLAY)
    work["sd_type"] = sd_type
    work["sd_value"] = work[param_col].map(sd_map)

    if qlp_shocks_already_standardized:
        # Existing QLP was estimated with standardize_shocks=True. Its main
        # driver coefficients are already responses to +1 sample-standard-
        # deviation shocks, so multiplying by SD again would double-scale them.
        work["impact_1sd"] = work[coef_col]
        if "ci_low" in work.columns:
            work["ci_low_1sd"] = work["ci_low"]
            work["ci_high_1sd"] = work["ci_high"]
        work["scaling_note"] = "QLP shocks already standardized in model"
    else:
        work["impact_1sd"] = work[coef_col] * work["sd_value"]
        if "ci_low" in work.columns:
            work["ci_low_1sd"] = work["ci_low"] * work["sd_value"]
            work["ci_high_1sd"] = work["ci_high"] * work["sd_value"]
        work["scaling_note"] = "coefficient multiplied by driver SD"

    work["impact_1sd_bps"] = work["impact_1sd"] * 10000
    if "ci_low_1sd" in work.columns:
        work["ci_low_1sd_bps"] = work["ci_low_1sd"] * 10000
        work["ci_high_1sd_bps"] = work["ci_high_1sd"] * 10000
    else:
        work["ci_low_1sd_bps"] = np.nan
        work["ci_high_1sd_bps"] = np.nan

    work["significant_95"] = work.apply(ci_excludes_zero, axis=1) if "ci_low" in work.columns else np.nan

    impact_cols = ["impact_1sd", "impact_1sd_bps", "ci_low_1sd_bps", "ci_high_1sd_bps"]
    for col in impact_cols:
        if col in work.columns and np.isinf(work[col].to_numpy(dtype=float, copy=True)).any():
            raise ValueError(f"Infinite values detected in {col}")

    return work


def format_ci_bps(row):
    low = row.get("ci_low_1sd_bps")
    high = row.get("ci_high_1sd_bps")
    if pd.isna(low) or pd.isna(high):
        return ""
    return f"[{low:.2f}, {high:.2f}]"


def effect_direction(value):
    if pd.isna(value):
        return "not available"
    if value < 0:
        return "compresses"
    if value > 0:
        return "widens"
    return "does not change"


def driver_interpretation(driver, impact_bps, context="qr"):
    direction = effect_direction(impact_bps)
    if context == "qlp":
        base = "changes the cumulative ESG-conventional spread"
    else:
        base = "changes the daily ESG-conventional spread"

    if driver == "r_brent":
        channel = "energy-exclusion wedge"
    elif driver == "d_vstoxx":
        channel = "market-stress channel"
    elif driver == "r_ttf":
        channel = "European gas shock channel"
    elif driver == "d_bund":
        channel = "interest-rate channel"
    else:
        channel = "carbon-price channel"

    return f"A +1sd {DRIVER_DISPLAY[driver]} shock {direction} the spread ({channel})."


def make_qr_full_key(econmag_qr_full):
    if econmag_qr_full.empty:
        return pd.DataFrame()

    key = econmag_qr_full[econmag_qr_full["significant_95"]].copy()
    if key.empty:
        return pd.DataFrame(columns=[
            "Driver",
            "Quantile",
            "Shock",
            "Impact on ESG spread (bps)",
            "95% CI (bps)",
            "Interpretation",
        ])

    rows = []
    for _, r in key.sort_values(["param", "quantile"]).iterrows():
        rows.append({
            "Driver": r["display_name"],
            "Quantile": q_label(r["quantile"]),
            "Shock": "+1 standard deviation shock",
            "Impact on ESG spread (bps)": r["impact_1sd_bps"],
            "95% CI (bps)": format_ci_bps(r),
            "Interpretation": driver_interpretation(r["param"], r["impact_1sd_bps"], "qr"),
        })
    return pd.DataFrame(rows)


def summarize_significant_effects(econmag_df, sample_col="sample_label", context="qr"):
    if econmag_df is None or econmag_df.empty:
        return pd.DataFrame()

    rows = []
    group_cols = [sample_col, "param"] if sample_col in econmag_df.columns else ["param"]
    for keys, grp in econmag_df.groupby(group_cols):
        if isinstance(keys, tuple):
            sample, driver = keys
        else:
            sample, driver = "full", keys
        sig = grp[grp["significant_95"] == True].copy()
        if sig.empty:
            continue

        if "horizon" in sig.columns:
            combos = ", ".join(
                f"h{int(r.horizon)}-{q_label(r.quantile)}"
                for r in sig.sort_values(["horizon", "quantile"]).itertuples()
            )
            sig_col_name = "Significant horizons/quantiles"
        else:
            combos = ", ".join(q_label(q) for q in sorted(sig["quantile"].unique()))
            sig_col_name = "Significant quantiles"

        signs = np.sign(sig["impact_1sd_bps"])
        if (signs > 0).all():
            sign_pattern = "positive"
        elif (signs < 0).all():
            sign_pattern = "negative"
        else:
            sign_pattern = "mixed"

        rows.append({
            "Sample": sample,
            "Driver": DRIVER_DISPLAY[driver],
            sig_col_name: combos,
            "Sign pattern": sign_pattern,
            "Avg. significant 1sd impact (bps)": sig["impact_1sd_bps"].abs().mean(),
            "Interpretation": driver_interpretation(driver, sig["impact_1sd_bps"].mean(), context),
        })

    return pd.DataFrame(rows)


def make_qlp_full_key(econmag_qlp_all):
    if econmag_qlp_all.empty:
        return pd.DataFrame()

    key = econmag_qlp_all[
        (econmag_qlp_all["sample_label"] == "full") &
        (econmag_qlp_all["significant_95"] == True)
    ].copy()
    if key.empty:
        return pd.DataFrame(columns=[
            "Driver",
            "Horizon",
            "Quantile",
            "Shock",
            "Cumulative impact (bps)",
            "95% CI (bps)",
            "Interpretation",
        ])

    rows = []
    for _, r in key.sort_values(["param", "horizon", "quantile"]).iterrows():
        rows.append({
            "Driver": r["display_name"],
            "Horizon": int(r["horizon"]),
            "Quantile": q_label(r["quantile"]),
            "Shock": "+1 standard deviation shock",
            "Cumulative impact (bps)": r["impact_1sd_bps"],
            "95% CI (bps)": format_ci_bps(r),
            "Interpretation": (
                f"A +1sd {r['display_name']} shock changes the cumulative ESG-conventional "
                f"spread over the next {int(r['horizon'])} days by {r['impact_1sd_bps']:.2f} bps."
            ),
        })
    return pd.DataFrame(rows)


def detect_bund_scenario(raw_df, bund_col):
    median_level = raw_df[bund_col].dropna().abs().median()
    if pd.isna(median_level):
        return 0.10, "Bund unit assumed percentage points; 10 bps = 0.10."
    if median_level > 0.5:
        return 0.10, "Bund yield appears measured in percentage points; 10 bps = 0.10."
    return 0.001, "Bund yield appears measured in decimal units; 10 bps = 0.001."


def build_scenario_tables(qr_results, raw_df, bund_col, include_all_scenarios=False):
    if qr_results.empty:
        return pd.DataFrame(), pd.DataFrame(), ""

    bund_shock, bund_note = detect_bund_scenario(raw_df, bund_col)
    scenarios = {
        "r_brent": ("Brent +3% daily shock", 0.03),
        "r_ttf": ("TTF +5% daily shock", 0.05),
        "r_eua": ("EUA +3% daily shock", 0.03),
        "d_vstoxx": ("VSTOXX +5 index points", 5.0),
        "d_bund": ("Bund 10Y +10 bps", bund_shock),
    }

    rows = []
    base = qr_results[qr_results["param"].isin(DRIVER_COLS)].copy()
    for _, r in base.iterrows():
        label, shock_value = scenarios[r["param"]]
        impact = r["coef"] * shock_value
        rows.append({
            "quantile": r["quantile"],
            "param": r["param"],
            "display_name": DRIVER_DISPLAY[r["param"]],
            "scenario_label": label,
            "shock_value": shock_value,
            "coef": r["coef"],
            "impact_scenario": impact,
            "impact_scenario_bps": impact * 10000,
            "ci_low_scenario_bps": r["ci_low"] * shock_value * 10000,
            "ci_high_scenario_bps": r["ci_high"] * shock_value * 10000,
            "significant_95": ci_excludes_zero(r),
            "unit_note": bund_note if r["param"] == "d_bund" else "",
        })
    scenario_df = pd.DataFrame(rows)

    focus = ["r_brent", "d_vstoxx", "r_ttf"]
    key = scenario_df[scenario_df["param"].isin(focus)].copy()
    if not include_all_scenarios:
        key = key[key["significant_95"] == True]

    key_rows = []
    for _, r in key.sort_values(["param", "quantile"]).iterrows():
        key_rows.append({
            "Driver": r["display_name"],
            "Quantile": q_label(r["quantile"]),
            "Scenario": r["scenario_label"],
            "Impact on ESG spread (bps)": r["impact_scenario_bps"],
            "95% CI (bps)": f"[{r['ci_low_scenario_bps']:.2f}, {r['ci_high_scenario_bps']:.2f}]",
            "Interpretation": driver_interpretation(r["param"], r["impact_scenario_bps"], "qr"),
        })
    return scenario_df, pd.DataFrame(key_rows), bund_note


def build_economic_magnitude_outputs(
    out_df,
    raw_df,
    subsamples,
    qr_results,
    qr_sub_results,
    qlp_key_results,
    rolling_252,
    rolling_504,
    bund_col,
):
    shock_sizes, sd_maps = build_driver_shock_sizes(out_df, subsamples, DRIVER_COLS)

    econmag_qr_full = add_economic_magnitude(
        qr_results,
        sd_maps["full"],
        sample_label="full",
        sd_type="full_sample_sd",
    )
    econmag_qr_full_key = make_qr_full_key(econmag_qr_full)

    sub_full_frames = []
    sub_local_frames = []
    for label, qr_sub in qr_sub_results.items():
        if qr_sub.empty:
            continue
        full_scaled = add_economic_magnitude(
            qr_sub,
            sd_maps["full"],
            sample_label=label,
            sd_type="full_sample_sd",
        )
        local_scaled = add_economic_magnitude(
            qr_sub,
            sd_maps[label],
            sample_label=label,
            sd_type="local_sample_sd",
        )
        sub_full_frames.append(full_scaled)
        sub_local_frames.append(local_scaled)

    econmag_qr_sub_fullsd = pd.concat(sub_full_frames, ignore_index=True) if sub_full_frames else pd.DataFrame()
    econmag_qr_sub_localsd = pd.concat(sub_local_frames, ignore_index=True) if sub_local_frames else pd.DataFrame()
    econmag_qr_sub_summary = summarize_significant_effects(econmag_qr_sub_localsd, context="qr")

    if qlp_key_results is not None and not qlp_key_results.empty:
        qlp_frames = []
        for label in ["full"] + list(subsamples.keys()):
            part = qlp_key_results[qlp_key_results["sample_label"] == label].copy()
            if part.empty:
                continue
            qlp_frames.append(add_economic_magnitude(
                part,
                sd_maps.get(label, sd_maps["full"]),
                sample_label=label,
                sd_type="model_standardized_shock",
                qlp_shocks_already_standardized=True,
            ))
        econmag_qlp_all = pd.concat(qlp_frames, ignore_index=True) if qlp_frames else pd.DataFrame()
    else:
        econmag_qlp_all = pd.DataFrame()

    econmag_qlp_full_key = make_qlp_full_key(econmag_qlp_all)
    econmag_qlp_sub_summary = summarize_significant_effects(
        econmag_qlp_all[econmag_qlp_all["sample_label"] != "full"].copy()
        if not econmag_qlp_all.empty else pd.DataFrame(),
        context="qlp",
    )

    rolling_252_econmag = add_economic_magnitude(
        rolling_252,
        sd_maps["full"],
        sd_type="full_sample_sd",
    )
    rolling_504_econmag = add_economic_magnitude(
        rolling_504,
        sd_maps["full"],
        sd_type="full_sample_sd",
    )

    scenarios, scenarios_key, bund_note = build_scenario_tables(qr_results, raw_df, bund_col)

    outputs = {
        "driver_shock_sizes": shock_sizes,
        "econmag_qr_full": econmag_qr_full,
        "econmag_qr_full_key": econmag_qr_full_key,
        "econmag_qr_sub_fullsd": econmag_qr_sub_fullsd,
        "econmag_qr_sub_localsd": econmag_qr_sub_localsd,
        "econmag_qr_sub_summary": econmag_qr_sub_summary,
        "econmag_qlp_all": econmag_qlp_all,
        "econmag_qlp_full_key": econmag_qlp_full_key,
        "econmag_qlp_sub_summary": econmag_qlp_sub_summary,
        "rolling_252_econmag": rolling_252_econmag,
        "rolling_504_econmag": rolling_504_econmag,
        "econmag_qr_scenarios": scenarios,
        "econmag_qr_scenarios_key": scenarios_key,
    }
    return outputs, sd_maps, bund_note


def render_table_png(df, path, title, max_rows=18):
    if df is None or df.empty:
        return False

    display = df.head(max_rows).copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

    fig_height = max(2.0, 0.38 * (len(display) + 2))
    fig_width = min(15.0, max(8.0, 1.25 * len(display.columns)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", color="#26418F", pad=12)
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.25)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D9D9D9")
        if row == 0:
            cell.set_facecolor("#26418F")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def export_econ_figure(fig, out_dir, base):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        path = out_dir / f"{base}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")


def plot_econmag_qr_full(econmag_qr_full, out_dir):
    sig = econmag_qr_full[econmag_qr_full["significant_95"] == True].copy()
    if sig.empty:
        return 0

    sig = sig.sort_values("impact_1sd_bps")
    labels = [f"{DRIVER_DISPLAY[p]} {q_label(q)}" for p, q in zip(sig["param"], sig["quantile"])]
    colors = ["#86BC25" if v > 0 else "#B5121B" for v in sig["impact_1sd_bps"]]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(range(len(sig)), sig["impact_1sd_bps"], color=colors)
    ax.axhline(0, color="#222222", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(sig)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Impact of +1sd shock (bps)")
    ax.set_title("Economic Magnitude of Full-Sample QR Effects", color="#26418F")
    ax.grid(axis="y", color="#D9D9D9", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    export_econ_figure(fig, out_dir, "figure_econmag_qr_full_bps")
    plt.close(fig)
    return 3


def plot_econmag_qlp_full(econmag_qlp_all, out_dir):
    full = econmag_qlp_all[
        (econmag_qlp_all["sample_label"] == "full") &
        (econmag_qlp_all["significant_95"] == True)
    ].copy()
    if full.empty:
        return 0

    colors = {
        "d_vstoxx": "#26418F",
        "d_bund": "#5B5B5B",
        "r_ttf": "#ED7D31",
        "r_brent": "#B5121B",
        "r_eua": "#86BC25",
    }
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for var in DRIVER_COLS:
        vdf = full[full["param"] == var].copy()
        if vdf.empty:
            continue
        # Plot significant cumulative effects as points; horizontal jitter keeps
        # multiple quantiles at the same horizon readable.
        for q, offset in [(0.1, -0.35), (0.5, 0.0), (0.9, 0.35)]:
            qdf = vdf[np.isclose(vdf["quantile"], q)]
            if qdf.empty:
                continue
            ax.scatter(
                qdf["horizon"] + offset,
                qdf["impact_1sd_bps"],
                s=45,
                color=colors[var],
                label=DRIVER_DISPLAY[var] if q == 0.1 else None,
                alpha=0.9,
            )
    ax.axhline(0, color="#222222", linewidth=1, linestyle="--")
    ax.set_xticks([1, 5, 10, 20])
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Cumulative impact of +1sd shock (bps)")
    ax.set_title("Economic Magnitude of Significant QLP Effects", color="#26418F")
    ax.grid(axis="y", color="#D9D9D9", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    export_econ_figure(fig, out_dir, "figure_econmag_qlp_full_bps")
    plt.close(fig)
    return 3


def write_final_visual_package_econmag(outputs, final_dir):
    final_dir = Path(final_dir)
    if not final_dir.exists():
        return 0, 0, []

    tables_excel_dir = final_dir / "tables_excel"
    tables_png_dir = final_dir / "tables_png"
    figures_main_dir = final_dir / "figures_main"
    tables_excel_dir.mkdir(parents=True, exist_ok=True)
    tables_png_dir.mkdir(parents=True, exist_ok=True)
    figures_main_dir.mkdir(parents=True, exist_ok=True)

    table_outputs = {
        "qr_full_key": outputs.get("econmag_qr_full_key", pd.DataFrame()),
        "qlp_full_key": outputs.get("econmag_qlp_full_key", pd.DataFrame()),
        "qr_scenarios_key": outputs.get("econmag_qr_scenarios_key", pd.DataFrame()),
    }

    table_count = 0
    excel_path = tables_excel_dir / "economic_magnitude_tables.xlsx"
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for sheet, table in table_outputs.items():
                table.to_excel(writer, sheet_name=sheet[:31], index=False)
                table_count += 1
    except Exception:
        with pd.ExcelWriter(excel_path) as writer:
            for sheet, table in table_outputs.items():
                table.to_excel(writer, sheet_name=sheet[:31], index=False)
                table_count += 1

    png_tables = [
        ("table_econmag_qr_full_key.png", "QR Full Sample: +1sd Effects", table_outputs["qr_full_key"]),
        ("table_econmag_qlp_full_key.png", "QLP Full Sample: +1sd Cumulative Effects", table_outputs["qlp_full_key"]),
        ("table_econmag_qr_scenarios_key.png", "QR Scenario-Based Effects", table_outputs["qr_scenarios_key"]),
    ]
    for filename, title, table in png_tables:
        if render_table_png(table, tables_png_dir / filename, title):
            table_count += 1

    figure_count = 0
    if "econmag_qr_full" in outputs:
        figure_count += plot_econmag_qr_full(outputs["econmag_qr_full"], figures_main_dir)
    if "econmag_qlp_all" in outputs:
        figure_count += plot_econmag_qlp_full(outputs["econmag_qlp_all"], figures_main_dir)

    generated = [
        str(excel_path),
        str(tables_png_dir / "table_econmag_qr_full_key.png"),
        str(tables_png_dir / "table_econmag_qlp_full_key.png"),
        str(tables_png_dir / "table_econmag_qr_scenarios_key.png"),
        str(figures_main_dir / "figure_econmag_qr_full_bps.png"),
        str(figures_main_dir / "figure_econmag_qlp_full_bps.png"),
    ]
    return table_count, figure_count, generated


def write_economic_magnitude_summary(summary_path, outputs, sd_maps, bund_note, generated_visuals, warnings):
    lines = []
    lines.append("Economic Magnitude / Standardized Shock Effects Summary")
    lines.append("======================================================")
    lines.append("")
    lines.append("Formula for non-standardized QR and rolling QR coefficients:")
    lines.append("impact_1sd_bps = beta * sd(driver) * 10000")
    lines.append("")
    lines.append("QLP note:")
    lines.append("Existing QLP models use standardize_shocks=True, so main-driver QLP coefficients")
    lines.append("are already responses to +1 sample-standard-deviation shocks. Therefore QLP")
    lines.append("economic magnitudes are converted to bps as coef * 10000, not multiplied by SD again.")
    lines.append("")
    lines.append("Interpretation notes:")
    lines.append("- QR impacts are contemporaneous changes in the daily ESG-conventional spread.")
    lines.append("- QLP impacts are cumulative future ESG-conventional spread changes over horizon h.")
    lines.append("- Effects are economically scaled conditional associations, not causal impulse responses.")
    lines.append("")
    lines.append("Full-sample driver standard deviations:")
    for var, sd in sd_maps.get("full", {}).items():
        lines.append(f"- {DRIVER_DISPLAY[var]} ({var}): {sd:.8f}")
    lines.append("")
    if bund_note:
        lines.append(f"Bund scenario unit note: {bund_note}")
        lines.append("")

    qr_full = outputs.get("econmag_qr_full", pd.DataFrame())
    if not qr_full.empty:
        lines.append("Top 10 absolute QR full-sample +1sd impacts (bps):")
        top = qr_full.reindex(qr_full["impact_1sd_bps"].abs().sort_values(ascending=False).index).head(10)
        for _, r in top.iterrows():
            lines.append(
                f"- {DRIVER_DISPLAY[r['param']]} {q_label(r['quantile'])}: "
                f"{r['impact_1sd_bps']:.2f} bps; significant={bool(r['significant_95'])}"
            )
        lines.append("")

    qlp = outputs.get("econmag_qlp_all", pd.DataFrame())
    if not qlp.empty:
        lines.append("Top 10 absolute QLP cumulative +1sd impacts (bps):")
        top = qlp.reindex(qlp["impact_1sd_bps"].abs().sort_values(ascending=False).index).head(10)
        for _, r in top.iterrows():
            lines.append(
                f"- {r['sample_label']} {DRIVER_DISPLAY[r['param']]} h{int(r['horizon'])} "
                f"{q_label(r['quantile'])}: {r['impact_1sd_bps']:.2f} bps; "
                f"significant={bool(r['significant_95'])}"
            )
        lines.append("")

    lines.append("Excel sheets added:")
    for sheet in outputs:
        lines.append(f"- {sheet}")
    lines.append("")
    lines.append("Final visual package files:")
    for item in generated_visuals:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Warnings / skipped outputs:")
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


qr_results, qr_pseudo_r2, qd_results, diag_results = run_qr_block_bootstrap(out, "full")
print("\nQuantile Regression results:")
print(qr_results)

# Shared sample splits and QLP configuration.
subsamples = {
    "2012_2019": ("2012-01-01", "2019-12-31"),
    "2020_2022": ("2020-01-01", "2022-12-31"),
    "2023_present": ("2023-01-01", None),
}
horizons = [1, 5, 10, 20]
qlp_quantiles = [0.1, 0.5, 0.9]

# Save outputs (explicitly requested)
out_base = r"C:\Users\user\Desktop\Doctorat\simpozion\output\\output_stats"
out_txt = out_base + ".txt"
out_xlsx = out_base + ".xlsx"
out_dir = out_base.rsplit("\\", 1)[0]

qlp_full = pd.DataFrame()
qlp_sub_results = {}
qlp_key_results = pd.DataFrame()
qr_sub_results = {}
economic_outputs = {}
economic_sd_maps = {}
economic_bund_note = ""
economic_warnings = []
economic_table_count = 0
economic_figure_count = 0
economic_generated_visuals = []

try:
    writer = pd.ExcelWriter(out_xlsx, engine="xlsxwriter")
except Exception:
    writer = pd.ExcelWriter(out_xlsx, engine="openpyxl")

with writer as w:
    results.to_excel(w, sheet_name="adf_pp", index=False)
    desc.to_excel(w, sheet_name="descriptive", index=True)
    corr.to_excel(w, sheet_name="correlation", index=True)
    vif.to_excel(w, sheet_name="vif", index=False)
    qr_results.to_excel(w, sheet_name="quantile_reg", index=False)
    qr_pseudo_r2.to_excel(w, sheet_name="qr_pseudo_r2", index=False)
    qd_results.to_excel(w, sheet_name="qr_quantile_diff", index=False)
    diag_results.to_excel(w, sheet_name="diagnostics", index=False)

    # Sub-samples
    for label, (start, end) in subsamples.items():
        if end is None:
            sub = out.loc[start:]
        else:
            sub = out.loc[start:end]
        if len(sub) < 50:
            continue
        qr_s, r2_s, qd_s, diag_s = run_qr_block_bootstrap(sub, label, B_override=300)
        qr_sub_results[label] = qr_s
        qr_s.to_excel(w, sheet_name=f"qr_{label}", index=False)
        r2_s.to_excel(w, sheet_name=f"r2_{label}", index=False)
        qd_s.to_excel(w, sheet_name=f"qd_{label}", index=False)
        diag_s.to_excel(w, sheet_name=f"diag_{label}", index=False)

    # Rolling windows (QR only)
    rolling_252 = run_qr_rolling(out, 252, "rolling_252")
    rolling_504 = run_qr_rolling(out, 504, "rolling_504")
    rolling_252.to_excel(w, sheet_name="rolling_252", index=False)
    rolling_504.to_excel(w, sheet_name="rolling_504", index=False)

    # Quantile Local Projections (added after rolling QR).
    qlp_full = run_qlp_quantile(
        out,
        horizons=horizons,
        qlp_quantiles=qlp_quantiles,
        B_qlp=500,
        block_len=22,
        standardize_shocks=True,
        label="full",
    )
    qlp_full.to_excel(w, sheet_name="qlp_full", index=False)

    for label, (start, end) in subsamples.items():
        if end is None:
            sub = out.loc[start:]
        else:
            sub = out.loc[start:end]
        if len(sub) < 50:
            qlp_sub_results[label] = pd.DataFrame()
            continue
        qlp_sub = run_qlp_quantile(
            sub,
            horizons=horizons,
            qlp_quantiles=qlp_quantiles,
            B_qlp=300,
            block_len=22,
            standardize_shocks=True,
            label=label,
        )
        qlp_sub_results[label] = qlp_sub
        qlp_sub.to_excel(w, sheet_name=f"qlp_{label}", index=False)

    qlp_frames = [qlp_full] + [df for df in qlp_sub_results.values() if not df.empty]
    if qlp_frames:
        qlp_all = pd.concat(qlp_frames, ignore_index=True)
        qlp_key_results = qlp_all[
            qlp_all["param"].isin(["d_vstoxx", "d_bund", "r_ttf", "r_brent", "r_eua"])
        ].copy()
    else:
        qlp_key_results = pd.DataFrame()
    qlp_key_results.to_excel(w, sheet_name="qlp_key_results", index=False)

    # Economic magnitude / standardized shock effects.
    economic_outputs, economic_sd_maps, economic_bund_note = build_economic_magnitude_outputs(
        out_df=out,
        raw_df=df,
        subsamples=subsamples,
        qr_results=qr_results,
        qr_sub_results=qr_sub_results,
        qlp_key_results=qlp_key_results,
        rolling_252=rolling_252,
        rolling_504=rolling_504,
        bund_col=col_bund,
    )
    for sheet_name, sheet_df in economic_outputs.items():
        if sheet_df is None:
            economic_warnings.append(f"Skipped {sheet_name}: object is None.")
            continue
        sheet_df.to_excel(w, sheet_name=sheet_name, index=False)

economic_summary_path = f"{out_dir}\\economic_magnitude_summary.txt"
final_visual_dir = Path(in_path).resolve().parent / "final_visual_package"
if economic_outputs:
    economic_table_count, economic_figure_count, economic_generated_visuals = write_final_visual_package_econmag(
        economic_outputs,
        final_visual_dir,
    )
    write_economic_magnitude_summary(
        economic_summary_path,
        economic_outputs,
        economic_sd_maps,
        economic_bund_note,
        economic_generated_visuals,
        economic_warnings,
    )
    if final_visual_dir.exists():
        write_economic_magnitude_summary(
            final_visual_dir / "economic_magnitude_outputs_summary.txt",
            economic_outputs,
            economic_sd_maps,
            economic_bund_note,
            economic_generated_visuals,
            economic_warnings,
        )

# Full-sample QLP plots for key drivers.
if not qlp_full.empty:
    for driver in ["d_vstoxx", "d_bund", "r_ttf", "r_brent", "r_eua"]:
        plot_df = qlp_full[qlp_full["param"] == driver].copy()
        if plot_df.empty:
            continue
        plot_df["horizon"] = plot_df["horizon"].astype(int)
        fig, ax = plt.subplots(figsize=(7, 4))
        for q in qlp_quantiles:
            qdf = plot_df[plot_df["quantile"] == q].sort_values("horizon")
            if qdf.empty:
                continue
            ax.plot(qdf["horizon"], qdf["coef"], marker="o", label=f"tau={q}")
            if qdf["ci_low"].notna().all() and qdf["ci_high"].notna().all():
                ax.fill_between(
                    qdf["horizon"].to_numpy(),
                    qdf["ci_low"].to_numpy(),
                    qdf["ci_high"].to_numpy(),
                    alpha=0.15,
                )
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("QLP coefficient")
        ax.set_title(f"QLP Full Sample - {driver}")
        ax.set_xticks(horizons)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(f"{out_dir}\\qlp_full_{driver}.png", dpi=150)
        plt.close(fig)

with open(out_txt, "w", encoding="utf-8") as f:
    f.write("ADF + PP results\n")
    f.write(results.to_string(index=False))
    f.write("\n\nDescriptive statistics\n")
    f.write(desc.to_string())
    f.write("\n\nCorrelation matrix\n")
    f.write(corr.to_string())
    f.write("\n\nVIF\n")
    f.write(vif.to_string(index=False))
    f.write("\n\nQuantile Regression\n")
    f.write(qr_results.to_string(index=False))
    f.write("\n\nQuantile Regression Pseudo R^2\n")
    f.write(qr_pseudo_r2.to_string(index=False))
    f.write("\n\nQuantile-Difference Tests (adjacent quantiles)\n")
    f.write(qd_results.to_string(index=False))
    f.write("\n\nDiagnostics (Ljung-Box, ARCH-LM, Jarque-Bera)\n")
    f.write(diag_results.to_string(index=False))
    f.write("\n\nSub-sample results saved to Excel sheets.\n")
    f.write("\nRolling QR saved to Excel sheets: rolling_252, rolling_504.\n")
    if not qlp_key_results.empty:
        f.write("\n\nQLP key results (main drivers)\n")
        f.write(qlp_key_results.to_string(index=False))
    f.write("\n\nQLP sheets saved: qlp_full, qlp_2012_2019, qlp_2020_2022, qlp_2023_present, qlp_key_results.\n")
    if economic_outputs:
        f.write("\nEconomic magnitude sheets saved:\n")
        for sheet_name in economic_outputs:
            f.write(f"- {sheet_name}\n")
        f.write(f"\nEconomic magnitude summary saved to: {economic_summary_path}\n")
    if not _HAS_PP:
        f.write("\n\nPP test not available. Install with: pip install arch\n")

# Console summary for QLP execution diagnostics.
qlp_frames_summary = [qlp_full] + [df for df in qlp_sub_results.values() if not df.empty]
if qlp_frames_summary:
    qlp_all_summary = pd.concat(qlp_frames_summary, ignore_index=True)
    combo_cols = [
        "sample_label",
        "horizon",
        "quantile",
        "nobs",
        "valid_bootstrap_replications",
    ]
    qlp_combo = qlp_all_summary[combo_cols].drop_duplicates()
    full_combo = qlp_combo[qlp_combo["sample_label"] == "full"]
    expected_models = len(horizons) * len(qlp_quantiles) * (1 + len(subsamples))
    estimated_models = len(qlp_combo)
    model_fit_failures = expected_models - estimated_models
    qlp_combo["target_bootstrap"] = np.where(
        qlp_combo["sample_label"] == "full", 500, 300
    )
    qlp_combo["bootstrap_failures"] = (
        qlp_combo["target_bootstrap"] - qlp_combo["valid_bootstrap_replications"]
    ).clip(lower=0)
    total_bootstrap_failures = int(qlp_combo["bootstrap_failures"].sum())
    min_valid = int(qlp_combo["valid_bootstrap_replications"].min())
    max_valid = int(qlp_combo["valid_bootstrap_replications"].max())
    full_nobs_min = int(full_combo["nobs"].min()) if not full_combo.empty else 0
    full_nobs_max = int(full_combo["nobs"].max()) if not full_combo.empty else 0

    print("\nQLP summary:")
    print(f"Full-sample QLP nobs (min/max across horizons): {full_nobs_min} / {full_nobs_max}")
    print(f"QLP models estimated: {estimated_models}")
    print(f"QLP outputs saved to: {out_xlsx}")
    print(f"QLP plots saved to: {out_dir}")
    print(f"QLP model fit failures: {model_fit_failures}")
    print(f"QLP bootstrap failures (total): {total_bootstrap_failures}")
    print(f"QLP valid bootstrap draws (min/max): {min_valid} / {max_valid}")
else:
    print("\nQLP summary:")
    print("No QLP models were estimated.")

if economic_outputs:
    print("\nEconomic magnitude summary:")
    print(f"New economic magnitude sheets: {len(economic_outputs)}")
    print(f"Economic magnitude tables created: {economic_table_count}")
    print(f"Economic magnitude figures created: {economic_figure_count}")
    print(f"Output workbook: {out_xlsx}")
    print(f"Economic magnitude summary: {economic_summary_path}")

    qr_full_print = economic_outputs.get("econmag_qr_full", pd.DataFrame())
    if not qr_full_print.empty:
        top_qr = qr_full_print.reindex(
            qr_full_print["impact_1sd_bps"].abs().sort_values(ascending=False).index
        ).head(10)
        print("\nTop 10 QR full-sample +1sd impacts (bps):")
        for _, r in top_qr.iterrows():
            print(
                f"- {console_driver_name(r['param'])} {q_label(r['quantile'])}: "
                f"{r['impact_1sd_bps']:.2f} bps; significant={bool(r['significant_95'])}"
            )

    qlp_print = economic_outputs.get("econmag_qlp_all", pd.DataFrame())
    if not qlp_print.empty:
        top_qlp = qlp_print.reindex(
            qlp_print["impact_1sd_bps"].abs().sort_values(ascending=False).index
        ).head(10)
        print("\nTop 10 QLP +1sd cumulative impacts (bps):")
        for _, r in top_qlp.iterrows():
            print(
                f"- {r['sample_label']} {console_driver_name(r['param'])} h{int(r['horizon'])} "
                f"{q_label(r['quantile'])}: {r['impact_1sd_bps']:.2f} bps; "
                f"significant={bool(r['significant_95'])}"
            )

    if economic_warnings:
        print("\nEconomic magnitude warnings:")
        for warning in economic_warnings:
            print(f"- {warning}")
