import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera

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
