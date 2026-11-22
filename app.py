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
from forecast import render_peramalan

# ----------------------------------------------------------------------
# Konfigurasi halaman
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard RKAP vs Realisasi Semen",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="auto",
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
    margin=dict(l=10, r=10, t=58, b=86),
    hoverlabel=dict(font_size=12),
    legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
)
PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


# ----------------------------------------------------------------------
# Gaya tampilan (CSS)
# ----------------------------------------------------------------------
def pasang_gaya():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        :root { color-scheme: light; }
        html, body, [data-testid="stAppViewContainer"], [class*="css"] {
            font-family: 'Plus Jakarta Sans', 'Segoe UI', system-ui, -apple-system, sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(1200px 520px at 10% -8%, #fff2d2 0%, rgba(255,242,210,0) 60%),
                radial-gradient(1000px 520px at 100% 0%, #e9f1fb 0%, rgba(233,241,251,0) 55%),
                linear-gradient(180deg, #faf9f6 0%, #f3f5f9 100%);
            background-attachment: fixed;
        }
        [data-testid="stHeader"] { background: transparent; }
        footer, #MainMenu { visibility: hidden; }
        .block-container { padding: 1.4rem 1.3rem 3rem; max-width: 1320px; }
        * { font-feature-settings: "tnum" 1; }
        h1, h2, h3, h4 { color: #1f2430; letter-spacing: -0.01em; }

        .hero {
            position: relative; overflow: hidden;
            background: linear-gradient(120deg, #ffd24a 0%, #ffab3f 50%, #ff8a5c 100%);
            border-radius: 22px; padding: 26px 30px; margin-bottom: 22px;
            box-shadow: 0 14px 34px rgba(237,161,0,0.30);
        }
        .hero::after {
            content: ""; position: absolute; right: -40px; top: -60px;
            width: 220px; height: 220px; border-radius: 50%;
            background: rgba(255,255,255,0.18);
        }
        .hero-badge {
            display: inline-block; background: rgba(255,255,255,0.30); color: #3a2600;
            font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em;
            padding: 4px 12px; border-radius: 999px; margin-bottom: 10px;
        }
        .hero h1 { font-size: clamp(1.35rem, 4.6vw, 1.95rem); margin: 0 0 6px 0; color: #3a2600; line-height: 1.18; }
        .hero p  { margin: 0; color: #5c3d00; font-size: clamp(0.82rem, 2.6vw, 0.96rem); }

        [data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #ecedf1; border-left: 6px solid #eda100;
            padding: 18px 20px; border-radius: 16px;
            box-shadow: 0 6px 18px rgba(31,50,80,0.08);
            transition: transform .15s ease, box-shadow .15s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-3px); box-shadow: 0 12px 26px rgba(31,50,80,0.13);
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(1) [data-testid="stMetric"] { border-left-color: #eda100; }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(2) [data-testid="stMetric"] { border-left-color: #2a78d6; }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(3) [data-testid="stMetric"] { border-left-color: #1baf7a; }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(4) [data-testid="stMetric"] { border-left-color: #eb6834; }
        [data-testid="stMetricLabel"] p { font-size: 0.8rem; color: #6b7280; font-weight: 500; }
        [data-testid="stMetricValue"] { font-size: clamp(1.15rem, 3.4vw, 1.55rem); font-weight: 700; color: #1f2430; }

        [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #ecedf1; }
        [data-testid="stSidebar"] h2 { font-size: 1.05rem; }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff; border: 1px solid #edeef2; border-radius: 18px;
            box-shadow: 0 6px 20px rgba(31,50,80,0.07); padding: 8px 12px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 4px; flex-wrap: wrap; background: #ffffff; border: 1px solid #ecedf1;
            padding: 6px; border-radius: 14px; box-shadow: 0 4px 14px rgba(31,50,80,0.05);
        }
        .stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 7px 16px; color: #5b6472; }
        .stTabs [data-baseweb="tab"]:hover { background: #faf6ea; }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(120deg, #ffd24a, #ffb347); color: #3a2600 !important;
            font-weight: 600; box-shadow: 0 3px 10px rgba(237,161,0,0.28);
        }
        .stTabs [data-baseweb="tab-highlight"] { background: transparent; }

        .stDownloadButton button, .stButton button {
            border-radius: 10px; border: 1px solid #e7cf8f; background: #fff8e8;
            color: #3a2600; font-weight: 600;
        }
        .stDownloadButton button:hover, .stButton button:hover { background: #ffefc4; border-color: #eda100; }

        /* Kartu KPI custom */
        .kpi-grid {
            display: grid; grid-template-columns: repeat(4, 1fr);
            gap: 16px; margin: 4px 0 8px;
        }
        .kpi-card {
            position: relative; background: #ffffff; border: 1px solid #ecedf1;
            border-radius: 18px; padding: 18px 20px;
            box-shadow: 0 6px 20px rgba(31,50,80,0.07);
            transition: transform .16s ease, box-shadow .16s ease;
        }
        .kpi-card:hover { transform: translateY(-4px); box-shadow: 0 14px 30px rgba(31,50,80,0.14); }
        .kpi-card::before {
            content: ""; position: absolute; left: 0; top: 16px; bottom: 16px;
            width: 5px; border-radius: 0 6px 6px 0; background: var(--c, #eda100);
        }
        .kpi-head { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
        .kpi-ic {
            width: 36px; height: 36px; border-radius: 11px; display: grid; place-items: center;
            background: var(--cbg, #fff3d6); color: var(--c, #eda100); flex: none;
        }
        .kpi-ic { font-size: 18px; font-weight: 800; line-height: 1; }
        .kpi-label { font-size: 0.78rem; color: #6b7280; font-weight: 600; line-height: 1.2; }
        .kpi-value { font-size: clamp(1.25rem, 3vw, 1.7rem); font-weight: 800; color: #1f2430; letter-spacing: -0.02em; }
        .kpi-unit { font-size: 0.8rem; font-weight: 600; color: #9aa1ac; }
        .kpi-sub { margin-top: 8px; font-size: 0.76rem; color: #8b92a0; }
        .kpi-chip {
            display: inline-block; margin-top: 8px; font-size: 0.74rem; font-weight: 700;
            padding: 3px 10px; border-radius: 999px;
        }
        .chip-up { background: #e6f5ee; color: #0f7a53; }
        .chip-down { background: #fdeceb; color: #c0392b; }

        .explain {
            background: #f4f8fd; border: 1px solid #dceaf8; border-left: 4px solid #2a78d6;
            border-radius: 12px; padding: 12px 16px; margin: 2px 0 12px;
        }
        .explain-title { display: block; font-weight: 700; color: #1f2430; font-size: 0.86rem; margin-bottom: 3px; }
        .explain-body { display: block; color: #4b5563; font-size: 0.82rem; line-height: 1.5; }

        @media (max-width: 980px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 640px) {
            .kpi-grid { grid-template-columns: repeat(2, 1fr); gap: 10px; }
            .block-container { padding: 1rem 0.8rem 2rem; }
            .hero { padding: 20px 18px; border-radius: 18px; }
            [data-testid="stMetric"] { padding: 14px 16px; }
            div[data-testid="stVerticalBlockBorderWrapper"] { padding: 4px 6px; }
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
# Antarmuka
# ----------------------------------------------------------------------
pasang_gaya()

st.markdown(
    """
    <div class="hero">
      <span class="hero-badge">PT SEMEN INDONESIA (PERSERO) TBK</span>
      <h1>Dashboard RKAP vs Realisasi Semen</h1>
      <p>Monitoring produksi dan rilis semen &nbsp;.&nbsp; Tahun 2023 &nbsp;.&nbsp; satuan ton</p>
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

def _fmt(v):
    return f"{v:,.0f}".replace(",", ".")


def _pct(v):
    return f"{v:.1f}".replace(".", ",")


_pen_up = pencapaian >= 100
_sel_up = selisih >= 0
_chip_pen = "chip-up" if _pen_up else "chip-down"
_chip_sel = "chip-up" if _sel_up else "chip-down"
_delta_txt = f"{'+' if pencapaian - 100 >= 0 else ''}{_pct(pencapaian - 100)}% vs target"

st.markdown(
    f"""
    <div class="kpi-grid">
      <div class="kpi-card" style="--c:#eda100;--cbg:#fff3d6;">
        <div class="kpi-head"><span class="kpi-ic">&#9678;</span><span class="kpi-label">TOTAL RKAP<br>Target</span></div>
        <div class="kpi-value">{_fmt(total_rkap)} <span class="kpi-unit">ton</span></div>
        <div class="kpi-sub">Target {parameter.lower()} tahun 2023</div>
      </div>
      <div class="kpi-card" style="--c:#2a78d6;--cbg:#e4effb;">
        <div class="kpi-head"><span class="kpi-ic">&#9638;</span><span class="kpi-label">TOTAL REALISASI<br>Aktual</span></div>
        <div class="kpi-value">{_fmt(total_real)} <span class="kpi-unit">ton</span></div>
        <div class="kpi-sub">Capaian {parameter.lower()} aktual</div>
      </div>
      <div class="kpi-card" style="--c:#1baf7a;--cbg:#e3f5ee;">
        <div class="kpi-head"><span class="kpi-ic">&#10003;</span><span class="kpi-label">PENCAPAIAN<br>vs Target</span></div>
        <div class="kpi-value">{_pct(pencapaian)}<span class="kpi-unit">%</span></div>
        <div><span class="kpi-chip {_chip_pen}">{_delta_txt}</span></div>
      </div>
      <div class="kpi-card" style="--c:#eb6834;--cbg:#fdeae0;">
        <div class="kpi-head"><span class="kpi-ic">&#8645;</span><span class="kpi-label">SELISIH<br>Realisasi - RKAP</span></div>
        <div class="kpi-value">{'+' if _sel_up else '-'}{_fmt(abs(selisih))} <span class="kpi-unit">ton</span></div>
        <div><span class="kpi-chip {_chip_sel}">{'surplus' if _sel_up else 'di bawah target'}</span></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
    render_peramalan(data)

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
