# -*- coding: utf-8 -*-
"""
Dashboard RKAP vs Realisasi Semen - PT Semen Indonesia
Perbandingan target (RKAP) dan aktual (Realisasi) untuk Produksi dan Rilis
per tipe semen, per bulan, hingga level Plant dan Finish Mill.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt

# ----------------------------------------------------------------------
# Konfigurasi halaman
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard RKAP vs Realisasi Semen",
    page_icon="bar_chart",
    layout="wide",
    initial_sidebar_state="expanded",
)

def temukan_data() -> Path:
    """Cari file Excel data, baik di folder data/ maupun langsung di root repo."""
    here = Path(__file__).parent
    kandidat = [
        here / "data" / "RKAP_Realisasi_Semen_2023.xlsx",
        here / "RKAP_Realisasi_Semen_2023.xlsx",
    ]
    for p in kandidat:
        if p.exists():
            return p
    for folder in (here / "data", here):
        if folder.exists():
            hits = sorted(folder.glob("*.xlsx"))
            if hits:
                return hits[0]
    return kandidat[0]  # biar pesan error tetap jelas bila benar benar tidak ada


DATA_DEFAULT = temukan_data()

# Palet warna konsisten per tipe semen (lolos validasi colorblind-safe)
WARNA_TIPE = {
    "SprintPro": "#2a78d6",
    "UltraPro": "#eb6834",
    "PowerPro": "#1baf7a",
    "EzPro": "#eda100",
}
WARNA_RKAP = "#b8b5a8"
WARNA_REAL = "#2a78d6"
URUTAN_TIPE = ["SprintPro", "UltraPro", "PowerPro", "EzPro"]

BULAN_URUT = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
              "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
BULAN_PANJANG = {
    "Jan": "Januari", "Feb": "Februari", "Mar": "Maret", "Apr": "April",
    "Mei": "Mei", "Jun": "Juni", "Jul": "Juli", "Agu": "Agustus",
    "Sep": "September", "Okt": "Oktober", "Nov": "November", "Des": "Desember",
}

LAYOUT_DASAR = dict(
    font=dict(family="Segoe UI, Arial, sans-serif", size=13, color="#2b2b2b"),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=54, b=72),
    hoverlabel=dict(font_size=12),
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
    title=dict(x=0, xanchor="left", font=dict(size=15)),
)
PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


# ----------------------------------------------------------------------
# Gaya tampilan (CSS)
# ----------------------------------------------------------------------
def pasang_gaya():
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(170deg, #fff4cf 0%, #fdeaef 48%, #e8f1fc 100%);
            background-attachment: fixed;
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }
        footer { visibility: hidden; }

        .hero {
            background: linear-gradient(100deg, #ffd24d 0%, #ffb347 52%, #ff8f5e 100%);
            border-radius: 18px;
            padding: 24px 28px;
            margin-bottom: 20px;
            box-shadow: 0 6px 20px rgba(237,161,0,0.28);
        }
        .hero h1 { font-size: 1.75rem; margin: 0 0 4px 0; color: #3a2600; }
        .hero p  { margin: 0; color: #5c3d00; font-size: 0.95rem; }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #eee7cf;
            border-left: 6px solid #eda100;
            padding: 16px 18px;
            border-radius: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            transition: transform .12s ease, box-shadow .12s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.10);
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(1) [data-testid="stMetric"] { border-left-color: #eda100; }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(2) [data-testid="stMetric"] { border-left-color: #2a78d6; }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(3) [data-testid="stMetric"] { border-left-color: #1baf7a; }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(4) [data-testid="stMetric"] { border-left-color: #eb6834; }
        [data-testid="stMetricLabel"] p { font-size: 0.82rem; color: #7a7a7a; }
        [data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; }

        [data-testid="stSidebar"] { background: #fffaf0; border-right: 1px solid #f0e6c4; }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border-radius: 14px;
            box-shadow: 0 2px 12px rgba(31,50,80,0.06);
        }

        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            padding: 6px 18px;
        }
        .stTabs [aria-selected="true"] {
            background: #fff1cc;
            border-bottom: 3px solid #eda100;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Pemuatan dan perapian data
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def muat_data(sumber) -> pd.DataFrame:
    df = pd.read_excel(sumber, sheet_name=0)
    kolom_id = ["Source", "Parameter", "Plant", "FM", "Tipe Semen", "UoM"]
    kolom_bulan = [c for c in df.columns if c not in kolom_id and c != "Total"]

    for k in kolom_id:
        df[k] = df[k].astype(str).str.strip()

    long = df.melt(
        id_vars=kolom_id, value_vars=kolom_bulan,
        var_name="TanggalBulan", value_name="Nilai",
    )
    long["Nilai"] = pd.to_numeric(long["Nilai"], errors="coerce").fillna(0.0)
    long["TanggalBulan"] = pd.to_datetime(long["TanggalBulan"], errors="coerce")
    long["NoBulan"] = long["TanggalBulan"].dt.month
    long["Bulan"] = long["NoBulan"].map(lambda m: BULAN_URUT[m - 1])
    return long


# ----------------------------------------------------------------------
# Komponen grafik
# ----------------------------------------------------------------------
def grafik_rkap_vs_real(rkap_bln, real_bln):
    fig = go.Figure()
    fig.add_bar(
        x=rkap_bln["Bulan"], y=rkap_bln["Nilai"], name="RKAP (Target)",
        marker_color=WARNA_RKAP, marker_line_width=0,
        hovertemplate="RKAP %{x}<br>%{y:,.0f} ton<extra></extra>",
    )
    fig.add_bar(
        x=real_bln["Bulan"], y=real_bln["Nilai"], name="Realisasi (Aktual)",
        marker_color=WARNA_REAL, marker_line_width=0,
        hovertemplate="Realisasi %{x}<br>%{y:,.0f} ton<extra></extra>",
    )
    fig.update_layout(
        **LAYOUT_DASAR, barmode="group", bargap=0.25, bargroupgap=0.08,
        title="RKAP vs Realisasi per Bulan (ton)",
        xaxis=dict(categoryorder="array", categoryarray=BULAN_URUT),
        yaxis=dict(title=None, gridcolor="#ececec", zeroline=False),
    )
    return fig


def grafik_pencapaian(df_capai):
    warna = [WARNA_TIPE[t] for t in df_capai["Tipe Semen"]]
    fig = go.Figure()
    fig.add_bar(
        x=df_capai["Tipe Semen"], y=df_capai["Pencapaian"],
        marker_color=warna, marker_line_width=0,
        text=[f"{v:,.1f}%" for v in df_capai["Pencapaian"]], textposition="outside",
        hovertemplate="%{x}<br>Pencapaian %{y:,.1f}%<extra></extra>",
    )
    fig.add_hline(y=100, line_dash="dash", line_color="#8a8a8a",
                  annotation_text="Target 100%", annotation_position="top left")
    maks = max(120, df_capai["Pencapaian"].max() * 1.15)
    fig.update_layout(
        **LAYOUT_DASAR, showlegend=False,
        title="Pencapaian Realisasi terhadap RKAP per Tipe Semen",
        xaxis=dict(categoryorder="array", categoryarray=URUTAN_TIPE),
        yaxis=dict(title="Persen (%)", range=[0, maks], gridcolor="#ececec", zeroline=False),
    )
    return fig


def grafik_tren(real, rkap, mode):
    fig = go.Figure()
    for tipe in URUTAN_TIPE:
        if mode in ("Realisasi", "Keduanya"):
            sub = real[real["Tipe Semen"] == tipe].sort_values("NoBulan")
            if not sub.empty:
                fig.add_scatter(
                    x=sub["Bulan"], y=sub["Nilai"],
                    name=(tipe if mode == "Realisasi" else f"{tipe} (Real)"),
                    mode="lines+markers",
                    line=dict(color=WARNA_TIPE[tipe], width=2),
                    marker=dict(size=7, line=dict(width=1, color="#fcfcfb")),
                    hovertemplate=tipe + " %{x}<br>%{y:,.0f} ton<extra></extra>",
                )
        if mode in ("RKAP", "Keduanya"):
            subr = rkap[rkap["Tipe Semen"] == tipe].sort_values("NoBulan")
            if not subr.empty:
                fig.add_scatter(
                    x=subr["Bulan"], y=subr["Nilai"],
                    name=(tipe if mode == "RKAP" else f"{tipe} (RKAP)"),
                    mode="lines",
                    line=dict(color=WARNA_TIPE[tipe], width=2,
                              dash=("solid" if mode == "RKAP" else "dot")),
                    hovertemplate=tipe + " RKAP %{x}<br>%{y:,.0f} ton<extra></extra>",
                )
    fig.update_layout(
        **LAYOUT_DASAR,
        title=f"Tren Bulanan per Tipe Semen ({mode}, ton)",
        xaxis=dict(categoryorder="array", categoryarray=BULAN_URUT),
        yaxis=dict(title=None, gridcolor="#ececec", zeroline=False),
    )
    return fig


def grafik_komposisi(df_komposisi):
    warna = [WARNA_TIPE[t] for t in df_komposisi["Tipe Semen"]]
    fig = go.Figure(go.Pie(
        labels=df_komposisi["Tipe Semen"], values=df_komposisi["Nilai"],
        hole=0.55, marker=dict(colors=warna, line=dict(color="#fcfcfb", width=2)),
        sort=False, direction="clockwise", textinfo="label+percent",
        hovertemplate="%{label}<br>%{value:,.0f} ton (%{percent})<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_DASAR, showlegend=False,
                      title="Komposisi Total per Tipe Semen")
    return fig


def grafik_plant(df_plant):
    fig = go.Figure()
    for tipe in URUTAN_TIPE:
        sub = df_plant[df_plant["Tipe Semen"] == tipe]
        if sub["Nilai"].sum() == 0:
            continue
        fig.add_bar(
            x=sub["Plant"], y=sub["Nilai"], name=tipe,
            marker_color=WARNA_TIPE[tipe], marker_line=dict(color="#fcfcfb", width=1),
            hovertemplate=tipe + " - %{x}<br>%{y:,.0f} ton<extra></extra>",
        )
    fig.update_layout(
        **LAYOUT_DASAR, barmode="stack", bargap=0.3,
        title="Realisasi Produksi per Plant (ton)",
        xaxis=dict(title=None),
        yaxis=dict(title=None, gridcolor="#ececec", zeroline=False),
    )
    return fig


def grafik_fm(df_fm):
    df_fm = df_fm.sort_values("Nilai")
    fig = go.Figure(go.Bar(
        x=df_fm["Nilai"], y=df_fm["FM"], orientation="h",
        marker_color=WARNA_REAL, marker_line_width=0,
        text=[f"{v:,.0f}" for v in df_fm["Nilai"]], textposition="auto",
        hovertemplate="%{y}<br>%{x:,.0f} ton<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_DASAR, showlegend=False,
        title="Total Realisasi Produksi per Finish Mill (ton)",
        xaxis=dict(title="ton", gridcolor="#ececec", zeroline=False),
        yaxis=dict(title=None), height=380,
    )
    return fig


# ----------------------------------------------------------------------
# Peramalan (forecasting)
# ----------------------------------------------------------------------
def _mape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    m = a != 0
    return float(np.mean(np.abs((a[m] - f[m]) / a[m])) * 100) if m.any() else float("nan")


def _rmse(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    return float(np.sqrt(np.mean((a - f) ** 2)))


def ramal(y, metode, h, **kw):
    """Hasilkan dict: fitted, fc (ramalan), lo, hi (interval), mape, rmse, catatan."""
    y = np.asarray(y, float)
    lo = hi = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if metode == "Moving Average":
            w = int(kw.get("window", 3))
            fitted = pd.Series(y).rolling(w).mean().to_numpy()
            fc = np.repeat(float(np.mean(y[-w:])), h)
            catatan = (f"Rata-rata bergerak (moving average) jendela {w} bulan. "
                       f"Ramalan ditahan datar pada rata-rata {w} bulan terakhir.")
        elif metode == "Exponential Smoothing (Holt)":
            damped = bool(kw.get("damped", False))
            model = Holt(y, damped_trend=damped, initialization_method="estimated").fit()
            fitted = np.asarray(model.fittedvalues)
            fc = np.asarray(model.forecast(h))
            catatan = "Holt (pemulusan eksponensial dengan tren)" + (
                ", tren teredam." if damped else ".")
        elif metode == "Holt-Winters":
            p = int(kw.get("periods", 4))
            if len(y) < 2 * p:
                raise ValueError(
                    f"Holt-Winters musiman periode {p} butuh minimal {2 * p} titik, "
                    f"data hanya {len(y)} bulan.")
            model = ExponentialSmoothing(
                y, trend="add", seasonal="add", seasonal_periods=p,
                initialization_method="estimated").fit()
            fitted = np.asarray(model.fittedvalues)
            fc = np.asarray(model.forecast(h))
            catatan = f"Holt-Winters aditif dengan periode musiman {p} bulan."
        elif metode == "ARIMA":
            order = kw.get("order")
            if order is None:
                best = None
                for pp in range(3):
                    for dd in range(2):
                        for qq in range(3):
                            try:
                                m = ARIMA(y, order=(pp, dd, qq)).fit()
                                if best is None or m.aic < best[1]:
                                    best = ((pp, dd, qq), m.aic, m)
                            except Exception:  # noqa: BLE001
                                pass
                order, _, model = best
            else:
                model = ARIMA(y, order=order).fit()
            fitted = np.asarray(model.fittedvalues)
            res = model.get_forecast(h)
            fc = np.asarray(res.predicted_mean)
            ci = np.asarray(res.conf_int(alpha=0.05))
            lo, hi = ci[:, 0], ci[:, 1]
            catatan = f"ARIMA order {tuple(order)} (dipilih berdasarkan AIC bila otomatis)."
        else:
            raise ValueError("Metode tidak dikenal.")

    mask = ~np.isnan(fitted)
    mask[:1] = False  # abaikan titik awal (inisialisasi) saat menghitung galat
    mp = _mape(y[mask], fitted[mask]) if mask.any() else float("nan")
    rm = _rmse(y[mask], fitted[mask]) if mask.any() else float("nan")
    return dict(fitted=fitted, fc=fc, lo=lo, hi=hi, mape=mp, rmse=rm, catatan=catatan)


def label_bulan_lanjut(h):
    """Label bulan untuk horizon ramalan, lanjut dari Des 2023."""
    return [f"{BULAN_URUT[j % 12]} '{24 + j // 12:02d}" for j in range(h)]


def grafik_ramal(hist_labels, y, fc_labels, fc, lo, hi, warna, judul):
    fig = go.Figure()
    fig.add_scatter(
        x=hist_labels, y=y, name="Historis", mode="lines+markers",
        line=dict(color=warna, width=2),
        marker=dict(size=7, line=dict(width=1, color="#fcfcfb")),
        hovertemplate="%{x}<br>%{y:,.0f} ton<extra></extra>",
    )
    x_fc = [hist_labels[-1]] + list(fc_labels)
    y_fc = [y[-1]] + list(fc)
    fig.add_scatter(
        x=x_fc, y=y_fc, name="Ramalan", mode="lines+markers",
        line=dict(color="#e34948", width=2, dash="dash"),
        marker=dict(size=7, line=dict(width=1, color="#fcfcfb")),
        hovertemplate="%{x}<br>%{y:,.0f} ton<extra></extra>",
    )
    if lo is not None:
        fig.add_scatter(
            x=list(fc_labels) + list(fc_labels)[::-1],
            y=list(hi) + list(lo)[::-1],
            fill="toself", fillcolor="rgba(227,73,72,0.12)",
            line=dict(width=0), name="Interval 95%", hoverinfo="skip",
        )
    fig.update_layout(
        **LAYOUT_DASAR, title=judul,
        xaxis=dict(title=None),
        yaxis=dict(title=None, gridcolor="#ececec", zeroline=False),
    )
    return fig


# ----------------------------------------------------------------------
# Antarmuka
# ----------------------------------------------------------------------
pasang_gaya()

st.markdown(
    """
    <div class="hero">
      <h1>Dashboard RKAP vs Realisasi Semen</h1>
      <p>PT Semen Indonesia (Persero) Tbk &nbsp;.&nbsp; Tahun 2023 &nbsp;.&nbsp; satuan ton</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Filter")
    unggah = st.file_uploader("Ganti data (opsional, .xlsx)", type=["xlsx"])
    sumber = unggah if unggah is not None else DATA_DEFAULT

try:
    data = muat_data(sumber)
except Exception as e:  # noqa: BLE001
    st.error(f"Gagal membaca data: {e}")
    st.stop()

with st.sidebar:
    parameter = st.radio("Parameter", ["Produksi", "Rilis"], horizontal=True)
    tipe_tersedia = [t for t in URUTAN_TIPE if t in data["Tipe Semen"].unique()]
    tipe_pilih = st.multiselect("Tipe Semen", tipe_tersedia, default=tipe_tersedia)
    rentang = st.select_slider("Rentang Bulan", options=BULAN_URUT,
                               value=(BULAN_URUT[0], BULAN_URUT[-1]))
    idx_awal, idx_akhir = BULAN_URUT.index(rentang[0]), BULAN_URUT.index(rentang[1])
    if idx_awal > idx_akhir:
        idx_awal, idx_akhir = idx_akhir, idx_awal
    bulan_pilih = BULAN_URUT[idx_awal: idx_akhir + 1]
    st.divider()
    st.caption("Dibuat dengan Streamlit")

if not tipe_pilih:
    st.warning("Pilih minimal satu tipe semen pada panel kiri.")
    st.stop()

mask = (
    data["Parameter"].eq(parameter)
    & data["Tipe Semen"].isin(tipe_pilih)
    & data["Bulan"].isin(bulan_pilih)
)
d = data[mask].copy()
rkap = d[d["Source"].eq("RKAP")]
real = d[d["Source"].eq("Realisasi")]

# KPI
total_rkap = rkap["Nilai"].sum()
total_real = real["Nilai"].sum()
pencapaian = (total_real / total_rkap * 100) if total_rkap else 0.0
selisih = total_real - total_rkap

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total RKAP (Target)", f"{total_rkap:,.0f} ton")
k2.metric("Total Realisasi", f"{total_real:,.0f} ton")
k3.metric("Pencapaian", f"{pencapaian:,.1f}%", delta=f"{pencapaian - 100:,.1f}% vs target")
k4.metric("Selisih", f"{selisih:,.0f} ton")

st.write("")

tab1, tab2, tab3, tab5, tab4 = st.tabs(
    ["Ringkasan", "Tren Bulanan", "Plant & Finish Mill", "Peramalan", "Data"]
)

with tab1:
    rkap_bln = (rkap.groupby(["NoBulan", "Bulan"], as_index=False)["Nilai"].sum()
                .sort_values("NoBulan"))
    real_bln = (real.groupby(["NoBulan", "Bulan"], as_index=False)["Nilai"].sum()
                .sort_values("NoBulan"))

    c1, c2 = st.columns([3, 2])
    with c1:
        with st.container(border=True):
            st.plotly_chart(grafik_rkap_vs_real(rkap_bln, real_bln),
                            use_container_width=True, config=PLOTLY_CONFIG)
    with c2:
        with st.container(border=True):
            komp = real.groupby("Tipe Semen", as_index=False)["Nilai"].sum()
            komp["urut"] = komp["Tipe Semen"].map({t: i for i, t in enumerate(URUTAN_TIPE)})
            komp = komp.sort_values("urut")
            st.plotly_chart(grafik_komposisi(komp),
                            use_container_width=True, config=PLOTLY_CONFIG)

    rk_tipe = rkap.groupby("Tipe Semen", as_index=False)["Nilai"].sum().rename(columns={"Nilai": "RKAP"})
    rl_tipe = real.groupby("Tipe Semen", as_index=False)["Nilai"].sum().rename(columns={"Nilai": "Real"})
    capai = rk_tipe.merge(rl_tipe, on="Tipe Semen", how="outer").fillna(0)
    capai["Pencapaian"] = capai.apply(
        lambda r: (r["Real"] / r["RKAP"] * 100) if r["RKAP"] else 0, axis=1)
    capai["urut"] = capai["Tipe Semen"].map({t: i for i, t in enumerate(URUTAN_TIPE)})
    capai = capai.sort_values("urut")
    with st.container(border=True):
        st.plotly_chart(grafik_pencapaian(capai),
                        use_container_width=True, config=PLOTLY_CONFIG)

with tab2:
    mode = st.radio("Tampilkan", ["Realisasi", "RKAP", "Keduanya"],
                    horizontal=True, key="mode_tren")
    rkap_t = rkap.groupby(["Tipe Semen", "NoBulan", "Bulan"], as_index=False)["Nilai"].sum()
    real_t = real.groupby(["Tipe Semen", "NoBulan", "Bulan"], as_index=False)["Nilai"].sum()
    with st.container(border=True):
        st.plotly_chart(grafik_tren(real_t, rkap_t, mode),
                        use_container_width=True, config=PLOTLY_CONFIG)
    st.caption("Mode Keduanya menampilkan RKAP sebagai garis titik-titik sebagai pembanding target.")

with tab3:
    if parameter != "Produksi":
        st.info("Rincian Plant dan Finish Mill hanya tersedia untuk parameter Produksi. "
                "Pilih Produksi pada panel kiri untuk melihatnya.")
    else:
        real_plant = real[real["Plant"].str.startswith("Tuban")]
        if real_plant.empty:
            st.info("Tidak ada data Plant untuk filter saat ini.")
        else:
            cc1, cc2 = st.columns(2)
            with cc1:
                with st.container(border=True):
                    pl = real_plant.groupby(["Plant", "Tipe Semen"], as_index=False)["Nilai"].sum()
                    st.plotly_chart(grafik_plant(pl),
                                    use_container_width=True, config=PLOTLY_CONFIG)
            with cc2:
                with st.container(border=True):
                    fm = real_plant.groupby("FM", as_index=False)["Nilai"].sum()
                    st.plotly_chart(grafik_fm(fm),
                                    use_container_width=True, config=PLOTLY_CONFIG)

with tab5:
    st.caption("Peramalan memakai seluruh 12 bulan data Realisasi untuk parameter terpilih "
               "(mengabaikan filter Rentang Bulan).")

    base_fc = data[
        data["Source"].eq("Realisasi")
        & data["Parameter"].eq(parameter)
        & data["Tipe Semen"].isin(tipe_pilih)
    ]
    opsi_seri = ["Total (tipe terpilih)"] + [t for t in URUTAN_TIPE if t in tipe_pilih]

    cfa, cfb, cfc = st.columns(3)
    with cfa:
        seri = st.selectbox("Seri yang diramal", opsi_seri)
    with cfb:
        metode = st.selectbox(
            "Metode",
            ["Moving Average", "Exponential Smoothing (Holt)", "Holt-Winters", "ARIMA"],
        )
    with cfc:
        h = st.slider("Horizon (bulan ke depan)", 1, 12, 6)

    kw = {}
    if metode == "Moving Average":
        kw["window"] = st.slider("Jendela MA (bulan)", 2, 6, 3)
    elif metode == "Exponential Smoothing (Holt)":
        kw["damped"] = st.checkbox("Tren teredam (damped)")
    elif metode == "Holt-Winters":
        kw["periods"] = st.select_slider("Periode musiman (bulan)", options=[3, 4, 6], value=4)
    elif metode == "ARIMA":
        auto = st.checkbox("Cari order otomatis (AIC terbaik)", value=True)
        if not auto:
            cp, cd, cq = st.columns(3)
            pp = cp.number_input("p (AR)", 0, 5, 1)
            dd = cd.number_input("d (diff)", 0, 2, 1)
            qq = cq.number_input("q (MA)", 0, 5, 0)
            kw["order"] = (int(pp), int(dd), int(qq))

    if seri == "Total (tipe terpilih)":
        s = base_fc.groupby("NoBulan")["Nilai"].sum()
        warna_seri = WARNA_REAL
    else:
        s = base_fc[base_fc["Tipe Semen"].eq(seri)].groupby("NoBulan")["Nilai"].sum()
        warna_seri = WARNA_TIPE.get(seri, WARNA_REAL)
    y = s.reindex(range(1, 13), fill_value=0).to_numpy(float)

    try:
        r = ramal(y, metode, h, **kw)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Metode ini belum bisa dijalankan pada data saat ini. {e}")
    else:
        hist_labels = [f"{m} '23" for m in BULAN_URUT]
        fc_labels = label_bulan_lanjut(h)

        m1, m2, m3 = st.columns(3)
        m1.metric("MAPE in-sample",
                  f"{r['mape']:.1f}%" if r["mape"] == r["mape"] else "n/a")
        m2.metric("RMSE in-sample",
                  f"{r['rmse']:,.0f} ton" if r["rmse"] == r["rmse"] else "n/a")
        m3.metric("Rata-rata ramalan", f"{np.mean(r['fc']):,.0f} ton")

        with st.container(border=True):
            st.plotly_chart(
                grafik_ramal(hist_labels, y, fc_labels, r["fc"], r["lo"], r["hi"],
                             warna_seri, f"Peramalan {seri} - {parameter} ({metode})"),
                use_container_width=True, config=PLOTLY_CONFIG,
            )
        st.caption(r["catatan"])

        tbl = pd.DataFrame({"Bulan": fc_labels,
                            "Ramalan (ton)": np.round(r["fc"]).astype(int)})
        if r["lo"] is not None:
            tbl["Batas bawah 95%"] = np.round(r["lo"]).astype(int)
            tbl["Batas atas 95%"] = np.round(r["hi"]).astype(int)
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        st.download_button(
            "Unduh ramalan (CSV)", tbl.to_csv(index=False).encode("utf-8"),
            file_name="ramalan.csv", mime="text/csv",
        )
        st.info("Data hanya 12 bulan, sehingga hasil peramalan bersifat indikatif. "
                "Holt-Winters musiman penuh (periode 12 bulan) membutuhkan minimal 24 bulan data.")

with tab4:
    tampil = d.copy()
    tampil["Bulan"] = tampil["Bulan"].map(BULAN_PANJANG)
    tampil = tampil[["Source", "Parameter", "Plant", "FM", "Tipe Semen",
                     "Bulan", "Nilai"]].rename(columns={"Nilai": "Nilai (ton)"})
    st.dataframe(tampil, use_container_width=True, hide_index=True)
    st.download_button(
        "Unduh data terfilter (CSV)",
        tampil.to_csv(index=False).encode("utf-8"),
        file_name="data_terfilter.csv", mime="text/csv",
    )
