# -*- coding: utf-8 -*-
"""
Modul peramalan permintaan (Rilis Realisasi) semen.

Isi mengikuti syntax analisis: Bayesian Hierarchical (partial pooling),
Holt Damped Exponential Smoothing, Random Forest Pooled, perbandingan dengan
RKAP, evaluasi data training dan testing, serta Rolling-Origin Cross-Validation.
"""

import logging
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")
logging.getLogger("pymc").setLevel(logging.ERROR)

TIPE_URUT = ["SprintPro", "UltraPro", "PowerPro", "EzPro"]
METODE_MODEL = ["Bayesian Hierarchical", "Holt Damped", "Random Forest Pooled"]
METODE_SEMUA = METODE_MODEL + ["RKAP (Perusahaan)"]
WARNA_METODE = {
    "Bayesian Hierarchical": "#2a78d6",
    "Holt Damped": "#1baf7a",
    "Random Forest Pooled": "#eb6834",
    "RKAP (Perusahaan)": "#8a8f9a",
}
PLOT_LAYOUT = dict(
    font=dict(family="Plus Jakarta Sans, Segoe UI, Arial, sans-serif", size=13, color="#2b2b2b"),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=56, b=86),
    legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
    hoverlabel=dict(font_size=12),
)


# ----------------------------------------------------------------------
# Persiapan data
# ----------------------------------------------------------------------
def siapkan_deret(data):
    """
    Ambil Rilis, plant Total, dalam bentuk per tipe semen dan bulan.
    Mengembalikan:
      data_long : Tipe Semen, Bulan, Permintaan (Realisasi)
      data_rkap : Tipe Semen, Bulan, Aktual, RKAP
    """
    d = data[(data["Parameter"] == "Rilis") & (data["Plant"] == "Total")]

    real = (
        d[d["Source"] == "Realisasi"]
        .groupby(["Tipe Semen", "NoBulan"], as_index=False)["Nilai"].sum()
        .rename(columns={"NoBulan": "Bulan", "Nilai": "Permintaan"})
    )
    rkap = (
        d[d["Source"] == "RKAP"]
        .groupby(["Tipe Semen", "NoBulan"], as_index=False)["Nilai"].sum()
        .rename(columns={"NoBulan": "Bulan", "Nilai": "RKAP"})
    )
    real = real[real["Tipe Semen"].isin(TIPE_URUT)].sort_values(["Tipe Semen", "Bulan"]).reset_index(drop=True)
    rkap = rkap[rkap["Tipe Semen"].isin(TIPE_URUT)]

    data_rkap = (
        real.merge(rkap, on=["Tipe Semen", "Bulan"], how="left")
        .rename(columns={"Permintaan": "Aktual"})
        .sort_values(["Tipe Semen", "Bulan"])
        .reset_index(drop=True)
    )
    return real, data_rkap


# ----------------------------------------------------------------------
# Fungsi evaluasi error
# ----------------------------------------------------------------------
def hitung_error(actual, pred, history=None):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    error = actual - pred

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    mask = actual != 0
    mape = np.mean(np.abs(error[mask] / actual[mask])) * 100 if mask.sum() > 0 else np.nan

    denominator = np.abs(actual) + np.abs(pred)
    smape_values = np.divide(
        2 * np.abs(error), denominator,
        out=np.zeros_like(error), where=denominator != 0,
    )
    smape = np.mean(smape_values) * 100

    # MASE memakai naive lag 1 dari satu tipe semen
    if history is not None:
        history = np.asarray(history, dtype=float)
        scale = np.mean(np.abs(np.diff(history)))
        mase = mae / scale if scale > 0 else np.nan
    else:
        mase = np.nan

    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "sMAPE (%)": smape, "MASE": mase}


def evaluasi_tabel(df_eval, history_df):
    """df_eval berisi: Metode, Tipe Semen, Bulan, Aktual, Forecast. Skala MASE per tipe."""
    baris = []
    for (metode, tipe), g in df_eval.groupby(["Metode", "Tipe Semen"]):
        g = g.sort_values("Bulan")
        hist = history_df[history_df["Tipe Semen"] == tipe].sort_values("Bulan")["Permintaan"].values
        baris.append({"Metode": metode, "Tipe Semen": tipe,
                      **hitung_error(g["Aktual"], g["Forecast"], hist)})
    per_tipe = pd.DataFrame(baris)
    global_ = per_tipe.drop(columns="Tipe Semen").groupby("Metode").mean().reset_index()
    return per_tipe, global_


# ----------------------------------------------------------------------
# Metode 1: Bayesian Hierarchical (partial pooling)
# ----------------------------------------------------------------------
def bayesian_hierarchical(data_train, forecast_steps=3, draws=1000, tune=2000,
                          target_accept=0.99, random_seed=42):
    import arviz as az
    import pymc as pm

    groups = sorted(data_train["Tipe Semen"].unique())
    group_map = {name: i for i, name in enumerate(groups)}

    d = data_train.sort_values(["Tipe Semen", "Bulan"]).copy()
    d["group_id"] = d["Tipe Semen"].map(group_map)

    # Standardisasi per tipe semen
    stat = d.groupby("Tipe Semen")["Permintaan"].agg(["mean", "std"])
    stat["std"] = stat["std"].fillna(1.0).replace(0, 1.0)
    d["y_scaled"] = (
        (d["Permintaan"] - d["Tipe Semen"].map(stat["mean"]))
        / d["Tipe Semen"].map(stat["std"])
    )

    y = d["y_scaled"].values
    group_idx = d["group_id"].values
    t = d["Bulan"].values.astype(float) / 12
    n_groups = len(groups)

    with pm.Model():
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=1)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)
        mu_beta = pm.Normal("mu_beta", mu=0, sigma=1)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1)

        # Non-centered parameterization
        z_alpha = pm.Normal("z_alpha", mu=0, sigma=1, shape=n_groups)
        z_beta = pm.Normal("z_beta", mu=0, sigma=1, shape=n_groups)

        alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * z_alpha)
        beta = pm.Deterministic("beta", mu_beta + sigma_beta * z_beta)
        sigma = pm.HalfNormal("sigma", sigma=1)

        mu = alpha[group_idx] + beta[group_idx] * t
        pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=draws, tune=tune, chains=4, cores=1,
            target_accept=target_accept, random_seed=random_seed,
            progressbar=False,
        )

    # Diagnostik konvergensi
    diag = {"Divergences": int(idata.sample_stats["diverging"].sum())}
    try:
        ringkas = az.summary(idata, var_names=["mu_alpha", "sigma_alpha", "mu_beta", "sigma_beta", "sigma"])
        diag["R-hat maks"] = float(ringkas["r_hat"].max())
        diag["ESS bulk min"] = float(ringkas["ess_bulk"].min())
    except Exception:  # noqa: BLE001
        diag["R-hat maks"] = np.nan
        diag["ESS bulk min"] = np.nan

    alpha_s = idata.posterior["alpha"].stack(sample=("chain", "draw")).values
    beta_s = idata.posterior["beta"].stack(sample=("chain", "draw")).values
    sigma_s = idata.posterior["sigma"].stack(sample=("chain", "draw")).values

    rng = np.random.default_rng(random_seed)
    hasil = []
    last_month = int(data_train["Bulan"].max())

    for group, idx in group_map.items():
        m, s = stat.loc[group, "mean"], stat.loc[group, "std"]
        for step in range(1, forecast_steps + 1):
            future_t = (last_month + step) / 12
            pred = (alpha_s[idx, :] + beta_s[idx, :] * future_t) * s + m
            # Interval prediksi 80% (posterior prediktif, termasuk noise)
            pred_pp = pred + rng.normal(0, sigma_s * s)
            hasil.append({
                "Tipe Semen": group,
                "Bulan": last_month + step,
                "Forecast": max(0, np.mean(pred)),
                "Lower80": max(0, np.percentile(pred_pp, 10)),
                "Upper80": np.percentile(pred_pp, 90),
            })

    # Nilai fitted pada data training
    fitted = []
    for group, idx in group_map.items():
        m, s = stat.loc[group, "mean"], stat.loc[group, "std"]
        for bulan in sorted(d.loc[d["Tipe Semen"] == group, "Bulan"]):
            f = (alpha_s[idx, :] + beta_s[idx, :] * (bulan / 12)) * s + m
            fitted.append({"Tipe Semen": group, "Bulan": int(bulan), "Forecast": max(0, np.mean(f))})

    return pd.DataFrame(hasil), pd.DataFrame(fitted), diag


# ----------------------------------------------------------------------
# Metode 2: Holt Damped Exponential Smoothing
# ----------------------------------------------------------------------
def holt_damped(data_train, forecast_steps=3):
    hasil, fitted = [], []
    for tipe in sorted(data_train["Tipe Semen"].unique()):
        subset = data_train[data_train["Tipe Semen"] == tipe].sort_values("Bulan")
        y = subset["Permintaan"].values.astype(float)

        fit = ExponentialSmoothing(
            y, trend="add", damped_trend=True, seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True)

        forecast = fit.forecast(forecast_steps)
        for i, nilai in enumerate(forecast):
            hasil.append({"Tipe Semen": tipe,
                          "Bulan": int(subset["Bulan"].max()) + i + 1,
                          "Forecast": max(0, float(nilai))})

        for bulan, f in zip(subset["Bulan"].values, fit.fittedvalues):
            fitted.append({"Tipe Semen": tipe, "Bulan": int(bulan), "Forecast": max(0, float(f))})

    return pd.DataFrame(hasil), pd.DataFrame(fitted)


# ----------------------------------------------------------------------
# Metode 3: Random Forest Pooled
# ----------------------------------------------------------------------
def buat_fitur_rf(data):
    hasil = []
    for tipe in sorted(data["Tipe Semen"].unique()):
        subset = data[data["Tipe Semen"] == tipe].sort_values("Bulan")
        values = subset.set_index("Bulan")["Permintaan"]
        for bulan in values.index:
            if bulan < 4:
                continue
            if not all((bulan - lag) in values.index for lag in [1, 2, 3]):
                continue
            hasil.append({
                "Tipe Semen": tipe,
                "Bulan": bulan,
                "Lag1": values.loc[bulan - 1],
                "Lag2": values.loc[bulan - 2],
                "Lag3": values.loc[bulan - 3],
                "Target": values.loc[bulan],
            })
    return pd.DataFrame(hasil)


def random_forest_pooled(data_train, forecast_steps=3, n_estimators=300):
    rf_data = buat_fitur_rf(data_train)

    X = pd.get_dummies(
        rf_data[["Tipe Semen", "Bulan", "Lag1", "Lag2", "Lag3"]],
        columns=["Tipe Semen"], dtype=int,
    )
    y = rf_data["Target"]

    model = RandomForestRegressor(
        n_estimators=n_estimators, max_features=1.0,
        min_samples_leaf=1, random_state=42, n_jobs=1,
    )
    model.fit(X, y)

    hasil = []
    for tipe in sorted(data_train["Tipe Semen"].unique()):
        subset = data_train[data_train["Tipe Semen"] == tipe].sort_values("Bulan")
        history = list(subset["Permintaan"].values.astype(float))
        last_month = int(subset["Bulan"].max())

        for step in range(1, forecast_steps + 1):
            bulan_pred = last_month + step
            fitur = {"Bulan": bulan_pred, "Lag1": history[-1], "Lag2": history[-2], "Lag3": history[-3]}
            fitur["Tipe Semen_" + tipe] = 1
            X_pred = pd.DataFrame([fitur]).reindex(columns=X.columns, fill_value=0)
            pred = max(0, float(model.predict(X_pred)[0]))
            hasil.append({"Tipe Semen": tipe, "Bulan": bulan_pred, "Forecast": pred})
            history.append(pred)

    fitted = pd.DataFrame({
        "Tipe Semen": rf_data["Tipe Semen"].values,
        "Bulan": rf_data["Bulan"].astype(int).values,
        "Forecast": model.predict(X),
    })
    return pd.DataFrame(hasil), fitted, model


# ----------------------------------------------------------------------
# Rolling-Origin Cross-Validation
# ----------------------------------------------------------------------
def rolling_origin_cv(data_long, data_rkap, origins=(6, 7, 8, 9, 10, 11),
                      bayes_draws=500, bayes_tune=1000):
    cv_results = []
    max_month = int(data_long["Bulan"].max())

    for fold, origin in enumerate(origins, start=1):
        horizon = min(3, max_month - origin)
        if horizon <= 0:
            continue

        fold_train = data_long[data_long["Bulan"] <= origin].copy()
        fold_test = data_long[(data_long["Bulan"] > origin) & (data_long["Bulan"] <= origin + horizon)].copy()

        pred_bayes, _, _ = bayesian_hierarchical(
            fold_train, forecast_steps=horizon,
            draws=bayes_draws, tune=bayes_tune, random_seed=42 + fold,
        )
        pred_holt, _ = holt_damped(fold_train, forecast_steps=horizon)
        pred_rf, _, _ = random_forest_pooled(fold_train, forecast_steps=horizon, n_estimators=300)

        pred_rkap = data_rkap[
            (data_rkap["Bulan"] > origin) & (data_rkap["Bulan"] <= origin + horizon)
        ][["Tipe Semen", "Bulan", "RKAP"]].rename(columns={"RKAP": "Forecast"})

        pred_all = pd.concat([
            pred_bayes.assign(Metode="Bayesian Hierarchical"),
            pred_holt.assign(Metode="Holt Damped"),
            pred_rf.assign(Metode="Random Forest Pooled"),
            pred_rkap.assign(Metode="RKAP (Perusahaan)"),
        ], ignore_index=True)

        pred_all = pred_all.merge(
            fold_test[["Tipe Semen", "Bulan", "Permintaan"]].rename(columns={"Permintaan": "Aktual"}),
            on=["Tipe Semen", "Bulan"], how="inner",
        )
        pred_all["Fold"] = fold
        pred_all["Origin"] = origin
        cv_results.append(pred_all)

    return pd.concat(cv_results, ignore_index=True)


def ringkas_cv(hasil_cv, data_long):
    """Error per fold dan ringkasan rata-rata serta simpang baku antar fold."""
    list_eval = []
    for (metode, fold, tipe), group in hasil_cv.groupby(["Metode", "Fold", "Tipe Semen"]):
        origin = int(group["Origin"].iloc[0])
        history = data_long[
            (data_long["Tipe Semen"] == tipe) & (data_long["Bulan"] <= origin)
        ].sort_values("Bulan")["Permintaan"].values
        metrik = hitung_error(group["Aktual"].values, group["Forecast"].values, history)
        list_eval.append({"Metode": metode, "Fold": fold, "Tipe Semen": tipe, **metrik})

    cv_eval = pd.DataFrame(list_eval)
    ringkasan = cv_eval.groupby("Metode").agg(
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        MAPE_mean=("MAPE (%)", "mean"), MAPE_std=("MAPE (%)", "std"),
        sMAPE_mean=("sMAPE (%)", "mean"), sMAPE_std=("sMAPE (%)", "std"),
        MASE_mean=("MASE", "mean"), MASE_std=("MASE", "std"),
    ).reset_index()
    return cv_eval, ringkasan


def buat_ranking(df, kolom):
    r = df.copy()
    for k in kolom:
        r[k + "_Rank"] = r[k].rank(method="min", ascending=True)
    r["Total_Rank"] = r[[k + "_Rank" for k in kolom]].sum(axis=1)
    return r.sort_values("Total_Rank")


# ----------------------------------------------------------------------
# Penjalan model yang disimpan di cache (Bayesian memakan waktu)
# ----------------------------------------------------------------------
def _setelan_bayes(cepat):
    # Mode normal = setelan notebook. Mode cepat = sampel lebih sedikit.
    return (500, 1000) if cepat else (1000, 2000)


@st.cache_data(show_spinner=False, max_entries=8)
def jalankan_testing(data_long, data_rkap, cepat):
    draws, tune = _setelan_bayes(cepat)
    train = data_long[data_long["Bulan"].between(1, 9)].sort_values(["Tipe Semen", "Bulan"]).reset_index(drop=True)
    test = data_long[data_long["Bulan"].between(10, 12)].sort_values(["Tipe Semen", "Bulan"]).reset_index(drop=True)

    hb, fb, diag = bayesian_hierarchical(train, forecast_steps=3, draws=draws, tune=tune)
    hh, fh = holt_damped(train, forecast_steps=3)
    hr, fr, _ = random_forest_pooled(train, forecast_steps=3)

    rkap_test = data_rkap[data_rkap["Bulan"].between(10, 12)][["Tipe Semen", "Bulan", "RKAP"]].rename(columns={"RKAP": "Forecast"})
    rkap_train = data_rkap[data_rkap["Bulan"].between(1, 9)][["Tipe Semen", "Bulan", "RKAP"]].rename(columns={"RKAP": "Forecast"})

    fc_test = pd.concat([
        hb.assign(Metode="Bayesian Hierarchical"),
        hh.assign(Metode="Holt Damped"),
        hr.assign(Metode="Random Forest Pooled"),
        rkap_test.assign(Metode="RKAP (Perusahaan)"),
    ], ignore_index=True)

    fit_train = pd.concat([
        fb.assign(Metode="Bayesian Hierarchical"),
        fh.assign(Metode="Holt Damped"),
        fr.assign(Metode="Random Forest Pooled"),
        rkap_train.assign(Metode="RKAP (Perusahaan)"),
    ], ignore_index=True)

    eval_test = fc_test.merge(
        test[["Tipe Semen", "Bulan", "Permintaan"]].rename(columns={"Permintaan": "Aktual"}),
        on=["Tipe Semen", "Bulan"], how="inner",
    )
    eval_train = fit_train.merge(
        train[["Tipe Semen", "Bulan", "Permintaan"]].rename(columns={"Permintaan": "Aktual"}),
        on=["Tipe Semen", "Bulan"], how="inner",
    )

    per_tipe_test, global_test = evaluasi_tabel(eval_test, train)
    per_tipe_train, global_train = evaluasi_tabel(eval_train, train)

    b = eval_test[eval_test["Metode"] == "Bayesian Hierarchical"]
    cakupan = float(((b["Aktual"] >= b["Lower80"]) & (b["Aktual"] <= b["Upper80"])).mean() * 100)

    return {
        "eval_test": eval_test,
        "per_tipe_test": per_tipe_test,
        "global_test": global_test,
        "per_tipe_train": per_tipe_train,
        "global_train": global_train,
        "cakupan": cakupan,
        "diag": diag,
    }


@st.cache_data(show_spinner=False, max_entries=8)
def jalankan_forecast(data_long, cepat):
    draws, tune = _setelan_bayes(cepat)
    hb, _, diag = bayesian_hierarchical(data_long, forecast_steps=3, draws=draws, tune=tune)
    hh, _ = holt_damped(data_long, forecast_steps=3)
    hr, _, _ = random_forest_pooled(data_long, forecast_steps=3)
    hasil = pd.concat([
        hb.assign(Metode="Bayesian Hierarchical"),
        hh.assign(Metode="Holt Damped"),
        hr.assign(Metode="Random Forest Pooled"),
    ], ignore_index=True)
    return hasil, diag


@st.cache_data(show_spinner=False, max_entries=8)
def jalankan_cv(data_long, data_rkap, cepat):
    bd, bt = (300, 600) if cepat else (500, 1000)
    hasil_cv = rolling_origin_cv(data_long, data_rkap, bayes_draws=bd, bayes_tune=bt)
    cv_eval, ringkasan = ringkas_cv(hasil_cv, data_long)
    return hasil_cv, cv_eval, ringkasan


# ----------------------------------------------------------------------
# Tampilan (dipanggil dari app.py)
# ----------------------------------------------------------------------
def _f_ton(v):
    return f"{v:,.0f}".replace(",", ".")


def _grafik_forecast(tipe, data_long, hasil_fc):
    fig = go.Figure()
    h = data_long[data_long["Tipe Semen"] == tipe].sort_values("Bulan")
    fig.add_scatter(
        x=h["Bulan"], y=h["Permintaan"], name="Data historis",
        mode="lines+markers", line=dict(color="#2b2b2b", width=2),
        marker=dict(size=7), hovertemplate="Bulan %{x}<br>%{y:,.0f} ton<extra></extra>",
    )
    for metode in METODE_MODEL:
        p = hasil_fc[(hasil_fc["Tipe Semen"] == tipe) & (hasil_fc["Metode"] == metode)].sort_values("Bulan")
        if p.empty:
            continue
        if metode == "Bayesian Hierarchical":
            fig.add_scatter(
                x=list(p["Bulan"]) + list(p["Bulan"])[::-1],
                y=list(p["Upper80"]) + list(p["Lower80"])[::-1],
                fill="toself", fillcolor="rgba(42,120,214,0.12)", line=dict(width=0),
                name="Interval 80% Bayesian", hoverinfo="skip",
            )
        fig.add_scatter(
            x=p["Bulan"], y=p["Forecast"], name=metode, mode="lines+markers",
            line=dict(color=WARNA_METODE[metode], width=2, dash="dash"),
            marker=dict(size=7), hovertemplate=metode + "<br>Bulan %{x}<br>%{y:,.0f} ton<extra></extra>",
        )
    fig.add_vline(x=12.5, line_dash="dot", line_color="#e34948")
    fig.update_layout(
        **PLOT_LAYOUT,
        title=f"Forecast Bulan 13 sampai 15, semen {tipe} (ton)",
        xaxis=dict(title="Bulan ke-", tickmode="linear", dtick=1, range=[0.8, 15.2]),
        yaxis=dict(gridcolor="#ececec", zeroline=False),
    )
    return fig


def _grafik_rkap_real(tipe, data_rkap):
    g = data_rkap[data_rkap["Tipe Semen"] == tipe].sort_values("Bulan")
    fig = go.Figure()
    fig.add_scatter(x=g["Bulan"], y=g["Aktual"], name="Realisasi", mode="lines+markers",
                    line=dict(color="#2b2b2b", width=2), marker=dict(size=7))
    fig.add_scatter(x=g["Bulan"], y=g["RKAP"], name="RKAP", mode="lines+markers",
                    line=dict(color="#eb6834", width=2, dash="dash"), marker=dict(size=7, symbol="square"))
    fig.update_layout(
        **PLOT_LAYOUT, title=f"RKAP vs Realisasi, semen {tipe} (ton)",
        xaxis=dict(title="Bulan ke-", tickmode="linear", dtick=1),
        yaxis=dict(gridcolor="#ececec", zeroline=False),
    )
    return fig


def render_peramalan(data):
    """Tampilkan seluruh analisis peramalan sesuai syntax KP."""
    data_long, data_rkap = siapkan_deret(data)
    if data_long.empty:
        st.warning("Data Rilis Realisasi dengan Plant Total tidak ditemukan.")
        return

    st.markdown(
        "Analisis ini memakai **Rilis Realisasi** dengan **Plant Total**, untuk empat tipe semen. "
        "Data dibagi berurutan: **training Bulan 1 sampai 9** dan **testing Bulan 10 sampai 12**. "
        "Ketiga model dibandingkan dengan **RKAP** sebagai pembanding perusahaan."
    )
    with st.expander("Penjelasan istilah (untuk pembaca awam)"):
        st.markdown(
            """
**RKAP** adalah rencana atau target perusahaan. **Realisasi** adalah angka yang benar-benar tercapai.

**Training dan testing.** Model dilatih dulu dengan data Bulan 1 sampai 9 (training), lalu diuji meramal Bulan 10 sampai 12 yang belum pernah dilihat (testing). Cara ini menguji apakah model benar-benar bisa meramal, bukan hanya cocok dengan data lama.

**Bayesian Hierarchical.** Model yang menggabungkan informasi antar tipe semen tetapi tetap memberi ruang untuk perbedaan tiap tipe. Ketidakpastian dihitung dengan cara Bayesian, sehingga hasilnya berupa rentang (interval 80 persen), bukan satu angka saja.

**Holt Damped.** Pemulusan eksponensial yang mengikuti arah tren, dengan tren yang makin melandai di masa depan. Sederhana dan sering menjadi pembanding yang kuat.

**Random Forest Pooled.** Metode pembelajaran mesin yang belajar dari tiga bulan sebelumnya. Akurat pada data latih, tetapi bisa terlalu menghafal data (overfitting) bila data sedikit.

**MAPE.** Rata-rata persentase meleset ramalan dari kenyataan. Makin kecil makin akurat.

**sMAPE.** Versi MAPE yang lebih adil saat nilai aktual dan ramalan sama-sama kecil atau besar.

**MAE dan RMSE.** Rata-rata besar kesalahan dalam satuan ton. RMSE lebih menghukum kesalahan yang sangat besar.

**MASE.** Perbandingan error model dengan tebakan paling sederhana (nilai bulan sebelumnya). Nilai di bawah 1 berarti model lebih baik dari tebakan sederhana itu.

**Interval 80 persen.** Rentang yang diperkirakan memuat nilai sebenarnya dengan peluang 80 persen. Cakupan menunjukkan berapa banyak data testing yang benar-benar masuk ke rentang itu.

**Rolling-Origin Cross-Validation.** Titik awal peramalan digeser beberapa kali untuk melihat apakah performa model stabil atau hanya bagus di satu periode saja.

**Ranking.** Setiap metode diberi peringkat per ukuran error, lalu peringkatnya dijumlahkan. Total rank terkecil berarti paling baik.
"""
        )

    cepat = st.toggle(
        "Mode cepat (untuk server gratis)", value=False,
        help="Mengurangi jumlah sampel Bayesian agar lebih cepat. Hasil bisa sedikit berbeda dari notebook. "
             "Mode normal memakai setelan notebook dan bisa butuh beberapa menit saat pertama dibuka.",
    )

    # 1. RKAP vs Realisasi
    st.subheader("1. Evaluasi RKAP terhadap Realisasi (Bulan 1 sampai 12)")
    baris = []
    for tipe in TIPE_URUT:
        g = data_rkap[data_rkap["Tipe Semen"] == tipe].sort_values("Bulan")
        if g.empty:
            continue
        m = hitung_error(g["Aktual"], g["RKAP"], g["Aktual"].values)
        baris.append({
            "Tipe Semen": tipe,
            "Total RKAP (ton)": _f_ton(g["RKAP"].sum()),
            "Total Realisasi (ton)": _f_ton(g["Aktual"].sum()),
            "Selisih Total (%)": round((g["Aktual"].sum() - g["RKAP"].sum()) / g["RKAP"].sum() * 100, 2),
            "Bulan Realisasi < RKAP": int((g["Aktual"] < g["RKAP"]).sum()),
            "MAPE (%)": round(m["MAPE (%)"], 2),
            "MAE (ton)": _f_ton(m["MAE"]),
        })
    st.dataframe(pd.DataFrame(baris), use_container_width=True, hide_index=True)

    tipe_rk = st.selectbox("Tipe semen untuk grafik RKAP", TIPE_URUT, key="tipe_rkap")
    with st.container(border=True):
        st.plotly_chart(_grafik_rkap_real(tipe_rk, data_rkap), use_container_width=True,
                        config={"displaylogo": False})

    # Jalankan model
    with st.spinner("Menjalankan model testing (Bayesian, Holt, Random Forest)..."):
        tes = jalankan_testing(data_long, data_rkap, cepat)
    with st.spinner("Menjalankan model forecast Bulan 13 sampai 15..."):
        fc, diag_fc = jalankan_forecast(data_long, cepat)

    # 2. Error training dan testing
    st.subheader("2. Error data training (Bulan 1 sampai 9) dan testing (Bulan 10 sampai 12)")
    pilihan = ["Rata-rata semua tipe"] + TIPE_URUT
    pil_err = st.selectbox("Tampilkan error untuk", pilihan, key="tipe_err")

    def _tabel_err(glob, per):
        if pil_err == "Rata-rata semua tipe":
            t = glob.copy()
        else:
            t = per[per["Tipe Semen"] == pil_err].drop(columns="Tipe Semen").copy()
        t = t[["Metode", "MAE", "RMSE", "MAPE (%)", "sMAPE (%)", "MASE"]]
        t = t.set_index("Metode").loc[[m for m in METODE_SEMUA if m in set(t["Metode"])]].reset_index()
        return t.round(2)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Training (Bulan 1 sampai 9)**")
        st.dataframe(_tabel_err(tes["global_train"], tes["per_tipe_train"]),
                     use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Testing (Bulan 10 sampai 12)**")
        st.dataframe(_tabel_err(tes["global_test"], tes["per_tipe_test"]),
                     use_container_width=True, hide_index=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Cakupan interval 80% Bayesian", f"{tes['cakupan']:.0f}%",
              help="Persentase data testing yang berada di dalam interval prediksi 80 persen.")
    m2.metric("Divergences (Bayesian, testing)", f"{tes['diag']['Divergences']}")
    m3.metric("R-hat maks (Bayesian, testing)",
              f"{tes['diag']['R-hat maks']:.3f}" if tes["diag"]["R-hat maks"] == tes["diag"]["R-hat maks"] else "n/a",
              help="Nilai mendekati 1 menunjukkan rantai sampel sudah konvergen.")

    # 3. Forecast Bulan 13 sampai 15
    st.subheader("3. Forecast Bulan 13 sampai 15 (memakai seluruh 12 bulan)")
    tipe_fc = st.selectbox("Tipe semen untuk forecast", TIPE_URUT, key="tipe_fc")
    with st.container(border=True):
        st.plotly_chart(_grafik_forecast(tipe_fc, data_long, fc), use_container_width=True,
                        config={"displaylogo": False})
    tbl = fc[fc["Tipe Semen"] == tipe_fc].copy()
    tbl["Bulan"] = tbl["Bulan"].map(lambda b: f"Bulan {b}")
    tbl = tbl.rename(columns={"Forecast": "Forecast (ton)"})
    tbl = tbl[["Metode", "Bulan", "Forecast (ton)"] + [c for c in ["Lower80", "Upper80"] if c in tbl.columns]]
    tbl = tbl.rename(columns={"Lower80": "Batas bawah 80%", "Upper80": "Batas atas 80%"})
    st.dataframe(tbl.round(0), use_container_width=True, hide_index=True)

    # 4. Rolling-Origin CV
    st.subheader("4. Rolling-Origin Cross-Validation")
    st.caption("Titik awal peramalan digeser beberapa kali untuk mengukur kestabilan model. "
               "Proses ini memakan waktu paling lama karena Bayesian dijalankan untuk setiap fold.")
    with st.spinner("Menjalankan Rolling-Origin CV, mohon tunggu..."):
        hasil_cv, cv_eval, ringkasan = jalankan_cv(data_long, data_rkap, cepat)

    ring_tampil = ringkasan.set_index("Metode").loc[[m for m in METODE_SEMUA if m in set(ringkasan["Metode"])]]
    st.dataframe(ring_tampil.round(2), use_container_width=True)

    # Ranking
    rank_test = buat_ranking(tes["global_test"], ["MAE", "RMSE", "MAPE (%)", "sMAPE (%)", "MASE"])
    rank_cv = buat_ranking(ringkasan, ["MAE_mean", "RMSE_mean", "MAPE_mean", "sMAPE_mean", "MASE_mean"])
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Ranking berdasarkan testing** (urutan terbaik di atas)")
        st.dataframe(rank_test[["Metode", "Total_Rank"]].rename(columns={"Total_Rank": "Total rank"}),
                     use_container_width=True, hide_index=True)
    with r2:
        st.markdown("**Ranking berdasarkan Rolling-Origin CV**")
        st.dataframe(rank_cv[["Metode", "Total_Rank"]].rename(columns={"Total_Rank": "Total rank"}),
                     use_container_width=True, hide_index=True)

    with st.expander("Detail error per fold"):
        st.dataframe(cv_eval.round(2), use_container_width=True, hide_index=True)

    st.info(
        "Data hanya 12 bulan dengan 12 observasi testing, sehingga selisih antar metode bersifat indikatif. "
        "Penelitian lanjutan disarankan memakai data multi-tahun."
    )
