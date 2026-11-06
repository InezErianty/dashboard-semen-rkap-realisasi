# -*- coding: utf-8 -*-
"""
Dashboard RKAP vs Realisasi Semen - PT Semen Indonesia
Perbandingan target (RKAP) dan aktual (Realisasi) untuk Produksi dan Rilis
per tipe semen, per bulan, hingga level Plant dan Finish Mill.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
    margin=dict(l=10, r=10, t=54, b=10),
    hoverlabel=dict(font_size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


# ----------------------------------------------------------------------
# Gaya tampilan (CSS)
# ----------------------------------------------------------------------
def pasang_gaya():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }
        footer { visibility: hidden; }

        .hero {
            background: linear-gradient(100deg, #fff6da 0%, #ffffff 65%);
            border: 1px solid #f0e6c4;
            border-radius: 16px;
            padding: 22px 26px;
            margin-bottom: 18px;
        }
        .hero h1 { font-size: 1.7rem; margin: 0 0 4px 0; color: #1f1f1f; }
        .hero p  { margin: 0; color: #6b6b6b; font-size: 0.95rem; }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #eee7cf;
            border-left: 5px solid #eda100;
            padding: 16px 18px;
            border-radius: 12px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
            transition: transform .12s ease, box-shadow .12s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        [data-testid="stMetricLabel"] p { font-size: 0.82rem; color: #7a7a7a; }
        [data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; }

        [data-testid="stSidebar"] { background: #fffdf6; border-right: 1px solid #f0e6c4; }

        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 6px 16px;
        }
        .stTabs [aria-selected="true"] {
            background: #fff6da;
            border-bottom: 2px solid #eda100;
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

tab1, tab2, tab3, tab4 = st.tabs(
    ["Ringkasan", "Tren Bulanan", "Plant & Finish Mill", "Data"]
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
