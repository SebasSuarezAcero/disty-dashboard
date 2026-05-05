import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import os
import numpy as np
from scipy import stats

st.set_page_config(page_title="Petroleum Products Dashboard", page_icon="⛽", layout="wide")

try:
    API_KEY = st.secrets["EIA_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    API_KEY = os.getenv("EIA_API_KEY")

PADD_MAP = {
    "R10": "PADD 1 — East Coast",
    "R20": "PADD 2 — Midwest",
    "R30": "PADD 3 — Gulf Coast",
    "R40": "PADD 4 — Rocky Mountain",
    "R50": "PADD 5 — West Coast",
}

PADD_COLORS = {
    "R10": "#4e9af1",
    "R20": "#f1c84e",
    "R30": "#4ef18a",
    "R40": "#f16b4e",
    "R50": "#c44ef1",
}

# ── Fetch functions ────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_inventory_national(product_code, product_label):
    url = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
    headers = {
        "X-Params": '{"frequency":"weekly","data":["value"],"facets":{"product":["' + product_code + '"],"duoarea":["NUS"]},"sort":[{"column":"period","direction":"desc"}],"offset":0,"length":520}'
    }
    r = requests.get(url, params={"api_key": API_KEY}, headers=headers)
    data = r.json()["response"]["data"]
    df = pd.DataFrame(data)[["period", "value"]].rename(
        columns={"period": "date", "value": f"{product_label}_inventory_mbbls"}
    )
    df[f"{product_label}_inventory_mbbls"] = pd.to_numeric(df[f"{product_label}_inventory_mbbls"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

@st.cache_data(ttl=86400)
def fetch_inventory_padd(product_code, padd_code, product_label):
    url = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
    headers = {
        "X-Params": '{"frequency":"weekly","data":["value"],"facets":{"product":["' + product_code + '"],"duoarea":["' + padd_code + '"]},"sort":[{"column":"period","direction":"desc"}],"offset":0,"length":520}'
    }
    r = requests.get(url, params={"api_key": API_KEY}, headers=headers)
    data = r.json()["response"]["data"]
    df = pd.DataFrame(data)[["period", "value"]].rename(
        columns={"period": "date", "value": f"{product_label}_{padd_code}_mbbls"}
    )
    df[f"{product_label}_{padd_code}_mbbls"] = pd.to_numeric(df[f"{product_label}_{padd_code}_mbbls"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

@st.cache_data(ttl=86400)
def fetch_spot_price(series_code, col_name):
    url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
    headers = {
        "X-Params": '{"frequency":"weekly","data":["value"],"facets":{"series":["' + series_code + '"]},"sort":[{"column":"period","direction":"desc"}],"offset":0,"length":520}'
    }
    r = requests.get(url, params={"api_key": API_KEY}, headers=headers)
    data = r.json()["response"]["data"]
    df = pd.DataFrame(data)[["period", "value"]].rename(
        columns={"period": "date", "value": col_name}
    )
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

@st.cache_data(ttl=86400)
def load_all_data():
    dist_nat   = fetch_inventory_national("EPD0", "distillate")
    jet_nat    = fetch_inventory_national("EPJK", "jetfuel")
    dist_price = fetch_spot_price("EER_EPD2F_PF4_Y35NY_DPG", "distillate_spot_price")
    jet_price  = fetch_spot_price("EER_EPJK_PF4_RGC_DPG",    "jetfuel_spot_price")

    df = dist_nat.merge(jet_nat, on="date", how="outer")
    df = df.merge(dist_price, on="date", how="left")
    df = df.merge(jet_price,  on="date", how="left")

    for padd in PADD_MAP:
        for prod_code, prod_label in [("EPD0", "distillate"), ("EPJK", "jetfuel")]:
            padd_df = fetch_inventory_padd(prod_code, padd, prod_label)
            df = df.merge(padd_df, on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True)

    for prod in ["distillate", "jetfuel"]:
        col = f"{prod}_inventory_mbbls"
        df[f"{prod}_rolling_mean"] = df[col].rolling(156, min_periods=52).mean()
        df[f"{prod}_rolling_std"]  = df[col].rolling(156, min_periods=52).std()
        df[f"{prod}_upper_2sd"]    = df[f"{prod}_rolling_mean"] + 2 * df[f"{prod}_rolling_std"]
        df[f"{prod}_lower_2sd"]    = df[f"{prod}_rolling_mean"] - 2 * df[f"{prod}_rolling_std"]
        df[f"{prod}_upper_1sd"]    = df[f"{prod}_rolling_mean"] + df[f"{prod}_rolling_std"]
        df[f"{prod}_lower_1sd"]    = df[f"{prod}_rolling_mean"] - df[f"{prod}_rolling_std"]
        df[f"{prod}_zscore"]       = (df[col] - df[f"{prod}_rolling_mean"]) / df[f"{prod}_rolling_std"]
        conditions = [
            df[col] > df[f"{prod}_upper_2sd"],
            df[col] > df[f"{prod}_upper_1sd"],
            df[col] < df[f"{prod}_lower_2sd"],
            df[col] < df[f"{prod}_lower_1sd"],
        ]
        choices = ["🔴 Bearish Outlier", "🟡 Elevated", "🟢 Bullish Outlier", "🟡 Tight"]
        df[f"{prod}_signal"] = np.select(conditions, choices, default="⚪ Neutral")

    for padd in PADD_MAP:
        for prod_label in ["distillate", "jetfuel"]:
            col = f"{prod_label}_{padd}_mbbls"
            if col in df.columns:
                df[f"{prod_label}_{padd}_mean"]   = df[col].rolling(156, min_periods=52).mean()
                df[f"{prod_label}_{padd}_std"]    = df[col].rolling(156, min_periods=52).std()
                df[f"{prod_label}_{padd}_upper2"] = df[f"{prod_label}_{padd}_mean"] + 2 * df[f"{prod_label}_{padd}_std"]
                df[f"{prod_label}_{padd}_lower2"] = df[f"{prod_label}_{padd}_mean"] - 2 * df[f"{prod_label}_{padd}_std"]
                df[f"{prod_label}_{padd}_zscore"] = (df[col] - df[f"{prod_label}_{padd}_mean"]) / df[f"{prod_label}_{padd}_std"]

    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"]         = df["date"].dt.year
    return df

with st.spinner("Fetching latest EIA data..."):
    df = load_all_data()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("⛽ U.S. Petroleum Products Dashboard")
st.caption("Weekly inventory levels and spot prices for Distillate Fuel Oil & Kerosene-Type Jet Fuel | Source: EIA Weekly Petroleum Status Report")

product = st.radio("Select Product", ["Distillate Fuel Oil", "Kerosene-Jet Fuel", "Both"], horizontal=True)
st.divider()

product_map = {
    "Distillate Fuel Oil": {
        "inv_col":     "distillate_inventory_mbbls",
        "price_col":   "distillate_spot_price",
        "inv_label":   "Distillate Stocks (Thousand Barrels)",
        "price_label": "No.2 Heating Oil Spot Price ($/Barrel)",
        "color":       "steelblue",
        "prefix":      "distillate"
    },
    "Kerosene-Jet Fuel": {
        "inv_col":     "jetfuel_inventory_mbbls",
        "price_col":   "jetfuel_spot_price",
        "inv_label":   "Jet Fuel Stocks (Thousand Barrels)",
        "price_label": "Kerosene-Jet Fuel Spot Price ($/Barrel)",
        "color":       "darkorange",
        "prefix":      "jetfuel"
    }
}

# ── KPI Cards ──────────────────────────────────────────────────────────────────
def kpi_cards(cfg, label):
    p      = cfg["prefix"]
    latest = df.iloc[-1]
    prior  = df.iloc[-2]
    inv    = latest[cfg["inv_col"]]
    inv_ch = inv - prior[cfg["inv_col"]]
    avg    = df[cfg["inv_col"]].mean()
    vs_avg = ((inv - avg) / avg) * 100
    price  = latest[cfg["price_col"]]
    pr_ch  = price - prior[cfg["price_col"]]
    signal = latest[f"{p}_signal"]
    zscore = latest[f"{p}_zscore"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        label=f"📦 {label} Stocks",
        value=f"{inv:,.0f} Mbbl",
        delta=f"{inv_ch:+,.0f} Mbbl vs. last week",
        help="Total U.S. weekly ending stocks in thousands of barrels (Mbbl)"
    )
    c2.metric(
        label="📊 vs. 52-Week Average",
        value=f"{vs_avg:+.1f}%",
        delta="Surplus above avg" if vs_avg > 0 else "Deficit below avg",
        help="How current stocks compare to the rolling 52-week average"
    )
    c3.metric(
        label="💵 Spot Price",
        value=f"${price:.2f}/bbl",
        delta=f"{pr_ch:+.2f} vs. last week",
        help="Weekly average spot price in dollars per barrel"
    )
    c4.metric(
        label="📐 3-Year Z-Score",
        value=f"{zscore:+.2f}σ",
        delta="Above 3-yr norm" if zscore > 0 else "Below 3-yr norm",
        help="How many standard deviations current stocks are from the 3-year rolling mean. >+2 = historically high (bearish), <-2 = historically low (bullish)"
    )
    c5.metric(
        label="🚦 Market Signal",
        value=signal,
        help="Signal derived from z-score: Bearish Outlier (>+2σ), Elevated (+1 to +2σ), Neutral (±1σ), Tight (-1 to -2σ), Bullish Outlier (<-2σ)"
    )

# ── PADD Latest Bar ────────────────────────────────────────────────────────────
def padd_latest_bar(cfg, label):
    p = cfg["prefix"]
    latest = df.iloc[-1]
    padd_vals, padd_lbls, padd_cols, padd_zs = [], [], [], []

    for padd in PADD_MAP:
        col   = f"{p}_{padd}_mbbls"
        z_col = f"{p}_{padd}_zscore"
        if col in df.columns:
            val = latest[col]
            z   = latest[z_col] if z_col in df.columns else np.nan
            padd_vals.append(val)
            padd_lbls.append(PADD_MAP[padd])
            padd_cols.append(PADD_COLORS[padd])
            padd_zs.append(z)

    hover_text = [
        f"<b>{lbl}</b><br>"
        f"Stocks: {v:,.0f} Mbbl<br>"
        f"Z-Score: {z:+.2f}σ<br>"
        f"{'🔴 Above normal' if z > 2 else ('🟢 Below normal' if z < -2 else '⚪ Near normal')}"
        for lbl, v, z in zip(padd_lbls, padd_vals, padd_zs)
    ]

    fig = go.Figure(go.Bar(
        x=padd_lbls,
        y=padd_vals,
        marker_color=padd_cols,
        text=[f"{v:,.0f} Mbbl<br><b>{z:+.2f}σ</b>" for v, z in zip(padd_vals, padd_zs)],
        textposition="outside",
        hovertext=hover_text,
        hoverinfo="text",
        opacity=0.85
    ))

    # Add a horizontal line for national average per PADD (total / 5 as rough guide)
    nat_latest = latest[cfg["inv_col"]]
    fig.add_hline(
        y=nat_latest / 5,
        line_dash="dot", line_color="white",
        annotation_text=f"Avg PADD share (~{nat_latest/5:,.0f} Mbbl)",
        annotation_position="top right"
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{label} — Latest Weekly Stocks by PADD District</b><br>"
                 f"<sup>Bar labels show stock level (Mbbl) and z-score vs. 3-year rolling mean. "
                 f"Z > +2σ = historically high | Z < -2σ = historically low</sup>",
            font=dict(size=14)
        ),
        template="plotly_dark",
        height=420,
        yaxis_title="Stocks (Thousand Barrels)",
        xaxis_title="PADD District",
        showlegend=False,
        margin=dict(t=90)
    )
    st.plotly_chart(fig, use_container_width=True)

# ── PADD Stacked Bar ───────────────────────────────────────────────────────────
def padd_stacked_chart(cfg, label):
    p = cfg["prefix"]

    selected_padds = st.multiselect(
        "Select PADDs to display",
        options=list(PADD_MAP.keys()),
        default=list(PADD_MAP.keys()),
        format_func=lambda x: PADD_MAP[x],
        key=f"padd_select_{p}"
    )

    fig = go.Figure()
    for padd in selected_padds:
        col = f"{p}_{padd}_mbbls"
        if col in df.columns:
            fig.add_trace(go.Bar(
                x=df["date"],
                y=df[col],
                name=PADD_MAP[padd],
                marker_color=PADD_COLORS[padd],
                opacity=0.85,
                hovertemplate=(
                    f"<b>{PADD_MAP[padd]}</b><br>"
                    "Week: %{x|%b %d, %Y}<br>"
                    "Stocks: %{y:,.0f} Mbbl<extra></extra>"
                )
            ))

    fig.update_layout(
        barmode="stack",
        title=dict(
            text=f"<b>{label} — Weekly Inventory by PADD District (Stacked)</b><br>"
                 f"<sup>Each color represents one PADD. Stack height = total U.S. inventory. "
                 f"Use the multiselect above to isolate specific regions.</sup>",
            font=dict(size=14)
        ),
        template="plotly_dark",
        height=460,
        legend=dict(
            title="PADD District",
            orientation="h", yanchor="bottom", y=1.02
        ),
        hovermode="x unified",
        xaxis_title="Week Ending Date",
        yaxis_title="Stocks (Thousand Barrels)",
        margin=dict(t=90)
    )
    st.plotly_chart(fig, use_container_width=True)

# ── PADD SD Band Charts (grid) ─────────────────────────────────────────────────
def padd_sd_charts(cfg, label):
    p = cfg["prefix"]

    st.caption(
        "📌 Each chart shows one PADD's inventory (bars) overlaid with a **±2σ shaded band** "
        "based on a 3-year rolling mean. The dotted white line is the rolling mean. "
        "Bars above the top band = historically high stock levels. "
        "Bars below the bottom band = historically low stock levels."
    )

    cols = st.columns(2)
    for i, (padd, padd_label) in enumerate(PADD_MAP.items()):
        inv_col  = f"{p}_{padd}_mbbls"
        mean_col = f"{p}_{padd}_mean"
        u2_col   = f"{p}_{padd}_upper2"
        l2_col   = f"{p}_{padd}_lower2"
        z_col    = f"{p}_{padd}_zscore"

        if inv_col not in df.columns:
            continue

        d = df.dropna(subset=[mean_col])

        latest_z   = df[z_col].iloc[-1] if z_col in df.columns else np.nan
        latest_inv = df[inv_col].iloc[-1]

        if not np.isnan(latest_z):
            if latest_z > 2:
                signal_emoji = "🔴"
                signal_text  = "Bearish — stocks historically HIGH"
            elif latest_z < -2:
                signal_emoji = "🟢"
                signal_text  = "Bullish — stocks historically LOW"
            elif latest_z > 1:
                signal_emoji = "🟡"
                signal_text  = "Elevated above normal"
            elif latest_z < -1:
                signal_emoji = "🟡"
                signal_text  = "Tight below normal"
            else:
                signal_emoji = "⚪"
                signal_text  = "Near normal range"
        else:
            signal_emoji = "❓"
            signal_text  = "Insufficient history"

        fig = go.Figure()

        # ±2σ band
        fig.add_trace(go.Scatter(
            x=pd.concat([d["date"], d["date"][::-1]]),
            y=pd.concat([d[u2_col], d[l2_col][::-1]]),
            fill="toself",
            fillcolor="rgba(100,150,255,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="±2σ Normal Range",
            hoverinfo="skip"
        ))
        # Rolling mean line
        fig.add_trace(go.Scatter(
            x=d["date"], y=d[mean_col],
            line=dict(color="white", width=1.5, dash="dot"),
            name="3-Year Rolling Mean",
            hovertemplate="3-Yr Mean: %{y:,.0f} Mbbl<extra></extra>"
        ))
        # Inventory bars
        fig.add_trace(go.Bar(
            x=df["date"], y=df[inv_col],
            name="Weekly Stocks",
            marker_color=PADD_COLORS[padd],
            opacity=0.8,
            hovertemplate=(
                f"<b>{padd_label}</b><br>"
                "Week: %{x|%b %d, %Y}<br>"
                "Stocks: %{y:,.0f} Mbbl<extra></extra>"
            )
        ))

        # Upper/lower band labels on the right edge
        if len(d) > 0:
            last_date = d["date"].iloc[-1]
            fig.add_annotation(
                x=last_date, y=d[u2_col].iloc[-1],
                text="+2σ", showarrow=False,
                font=dict(color="rgba(150,150,255,0.8)", size=10),
                xanchor="left"
            )
            fig.add_annotation(
                x=last_date, y=d[l2_col].iloc[-1],
                text="−2σ", showarrow=False,
                font=dict(color="rgba(150,150,255,0.8)", size=10),
                xanchor="left"
            )

        fig.update_layout(
            title=dict(
                text=f"<b>{signal_emoji} {padd_label}</b>   "
                     f"<span style='color:gray'>Latest: {latest_inv:,.0f} Mbbl | "
                     f"Z-Score: {latest_z:+.2f}σ | {signal_text}</span>",
                font=dict(size=12)
            ),
            template="plotly_dark",
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9)),
            hovermode="x unified",
            xaxis_title="Week Ending Date",
            yaxis_title="Stocks (Mbbl)",
            margin=dict(t=70, b=40, l=50, r=40)
        )
        with cols[i % 2]:
            st.plotly_chart(fig, use_container_width=True)

# ── PADD Z-Score Heatmap ───────────────────────────────────────────────────────
def padd_zscore_heatmap(cfg, label):
    p = cfg["prefix"]
    recent = df.tail(52)

    z_matrix  = []
    padd_lbls = []
    for padd in PADD_MAP:
        col = f"{p}_{padd}_zscore"
        if col in df.columns:
            z_matrix.append(recent[col].values)
            padd_lbls.append(PADD_MAP[padd])

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=recent["date"].dt.strftime("%b %d '%y"),
        y=padd_lbls,
        colorscale=[
            [0.0,  "rgba(0,200,80,0.9)"],
            [0.25, "rgba(0,200,80,0.3)"],
            [0.5,  "rgba(180,180,180,0.2)"],
            [0.75, "rgba(255,80,80,0.3)"],
            [1.0,  "rgba(255,50,50,0.9)"]
        ],
        zmid=0, zmin=-3, zmax=3,
        colorbar=dict(
            title=dict(text="Z-Score<br><sup>(std devs from 3-yr mean)</sup>", side="right"),
            tickvals=[-3, -2, -1, 0, 1, 2, 3],
            ticktext=["−3σ Bullish", "−2σ", "−1σ", "0 Normal", "+1σ", "+2σ", "+3σ Bearish"]
        ),
        hoverongaps=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Week: %{x}<br>"
            "Z-Score: %{z:+.2f}σ<br>"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{label} — PADD Z-Score Heatmap (Last 52 Weeks)</b><br>"
                 f"<sup>Green = inventory historically LOW (bullish for prices) | "
                 f"Red = inventory historically HIGH (bearish for prices) | "
                 f"Gray = near normal</sup>",
            font=dict(size=14)
        ),
        template="plotly_dark",
        height=340,
        xaxis=dict(title="Week Ending Date", tickangle=-45),
        yaxis_title="PADD District",
        margin=dict(t=90, b=80)
    )
    st.plotly_chart(fig, use_container_width=True)

# ── National SD Band Chart ─────────────────────────────────────────────────────
def sd_band_chart(cfg, label):
    p = cfg["prefix"]
    d = df.dropna(subset=[f"{p}_rolling_mean"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pd.concat([d["date"], d["date"][::-1]]),
        y=pd.concat([d[f"{p}_upper_2sd"], d[f"{p}_lower_2sd"][::-1]]),
        fill="toself", fillcolor="rgba(100,100,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±2σ Normal Range",
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([d["date"], d["date"][::-1]]),
        y=pd.concat([d[f"{p}_upper_1sd"], d[f"{p}_lower_1sd"][::-1]]),
        fill="toself", fillcolor="rgba(100,100,255,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1σ Normal Range",
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=d["date"], y=d[f"{p}_rolling_mean"],
        line=dict(color="white", width=1.5, dash="dot"),
        name="3-Year Rolling Mean",
        hovertemplate="3-Yr Mean: %{y:,.0f} Mbbl<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=df["date"], y=df[cfg["inv_col"]],
        name="Weekly Stocks",
        marker_color=cfg["color"], opacity=0.8,
        hovertemplate="Week: %{x|%b %d, %Y}<br>Stocks: %{y:,.0f} Mbbl<extra></extra>"
    ))

    bearish = df[df[f"{p}_signal"] == "🔴 Bearish Outlier"]
    bullish = df[df[f"{p}_signal"] == "🟢 Bullish Outlier"]
    if not bearish.empty:
        fig.add_trace(go.Scatter(
            x=bearish["date"], y=bearish[cfg["inv_col"]],
            mode="markers", marker=dict(color="red", size=8, symbol="circle"),
            name="🔴 Bearish Outlier (>+2σ)",
            hovertemplate="Bearish Outlier<br>Week: %{x|%b %d, %Y}<br>Stocks: %{y:,.0f} Mbbl<extra></extra>"
        ))
    if not bullish.empty:
        fig.add_trace(go.Scatter(
            x=bullish["date"], y=bullish[cfg["inv_col"]],
            mode="markers", marker=dict(color="lime", size=8, symbol="circle"),
            name="🟢 Bullish Outlier (<-2σ)",
            hovertemplate="Bullish Outlier<br>Week: %{x|%b %d, %Y}<br>Stocks: %{y:,.0f} Mbbl<extra></extra>"
        ))

    # Annotate the most recent data point
    last = df.iloc[-1]
    fig.add_annotation(
        x=last["date"], y=last[cfg["inv_col"]],
        text=f"Latest:<br>{last[cfg['inv_col']]:,.0f} Mbbl",
        showarrow=True, arrowhead=2, arrowcolor="white",
        font=dict(color="white", size=10),
        bgcolor="rgba(0,0,0,0.5)",
        xanchor="right"
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{label} — U.S. Total Inventory with 3-Year Rolling SD Bands</b><br>"
                 f"<sup>Shaded bands show ±1σ (inner) and ±2σ (outer) ranges from the 3-year rolling mean. "
                 f"Red dots = historically high stock weeks. Green dots = historically low stock weeks.</sup>",
            font=dict(size=14)
        ),
        template="plotly_dark",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        barmode="overlay",
        xaxis_title="Week Ending Date",
        yaxis_title=cfg["inv_label"],
        margin=dict(t=90)
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Combo Chart ────────────────────────────────────────────────────────────────
def combo_chart(cfg, label):
    fig  = make_subplots(
        specs=[[{"secondary_y": True}]],
        subplot_titles=[""]
    )
    avg  = df[cfg["inv_col"]].mean()

    fig.add_trace(go.Bar(
        x=df["date"], y=df[cfg["inv_col"]],
        name="Weekly Stocks (left axis)",
        marker_color=cfg["color"], opacity=0.75,
        hovertemplate="Week: %{x|%b %d, %Y}<br>Stocks: %{y:,.0f} Mbbl<extra></extra>"
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df[cfg["price_col"]],
        name="Spot Price (right axis)",
        line=dict(color="orangered", width=2),
        hovertemplate="Week: %{x|%b %d, %Y}<br>Price: $%{y:.2f}/bbl<extra></extra>"
    ), secondary_y=True)

    fig.add_hline(
        y=avg, line_dash="dot", line_color="gray",
        annotation_text=f"52-Wk Avg Stocks: {avg:,.0f} Mbbl",
        annotation_position="top left",
        secondary_y=False
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{label} — Weekly Inventory vs. Spot Price</b><br>"
                 f"<sup>Bars = stock levels (left axis, Mbbl). Line = spot price (right axis, $/bbl). "
                 f"High stocks typically pressure prices lower; low stocks support higher prices.</sup>",
            font=dict(size=14)
        ),
        template="plotly_dark",
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        xaxis_title="Week Ending Date",
        margin=dict(t=90)
    )
    fig.update_yaxes(title_text=cfg["inv_label"], secondary_y=False)
    fig.update_yaxes(title_text=cfg["price_label"], secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# ── Z-Score Chart ──────────────────────────────────────────────────────────────
def zscore_chart(cfg, label):
    p = cfg["prefix"]
    d = df.dropna(subset=[f"{p}_zscore"])
    colors = d[f"{p}_zscore"].apply(
        lambda z: "red" if z > 2 else ("lime" if z < -2 else ("orange" if abs(z) > 1 else "steelblue"))
    )

    fig = go.Figure()

    fig.add_hrect(
        y0=2, y1=5, fillcolor="red", opacity=0.07, line_width=0,
        annotation_text="Bearish Zone: stocks unusually HIGH (>+2σ)",
        annotation_position="top left",
        annotation_font=dict(color="rgba(255,100,100,0.8)", size=10)
    )
    fig.add_hrect(
        y0=-5, y1=-2, fillcolor="lime", opacity=0.07, line_width=0,
        annotation_text="Bullish Zone: stocks unusually LOW (<-2σ)",
        annotation_position="bottom left",
        annotation_font=dict(color="rgba(100,255,100,0.8)", size=10)
    )
    fig.add_hrect(
        y0=-1, y1=1, fillcolor="gray", opacity=0.05, line_width=0,
        annotation_text="Normal Range (±1σ)",
        annotation_position="top right",
        annotation_font=dict(color="rgba(200,200,200,0.6)", size=10)
    )

    fig.add_trace(go.Bar(
        x=d["date"], y=d[f"{p}_zscore"],
        marker_color=colors,
        name="Weekly Z-Score",
        hovertemplate=(
            "Week: %{x|%b %d, %Y}<br>"
            "Z-Score: %{y:+.2f}σ<br>"
            "<extra></extra>"
        )
    ))

    fig.add_hline(y=0,  line_dash="dash", line_color="white", line_width=1,
                  annotation_text="Mean (0σ)", annotation_position="right")
    fig.add_hline(y=2,  line_dash="dot",  line_color="red",   line_width=1,
                  annotation_text="+2σ threshold", annotation_position="right")
    fig.add_hline(y=-2, line_dash="dot",  line_color="lime",  line_width=1,
                  annotation_text="−2σ threshold", annotation_position="right")

    fig.update_layout(
        title=dict(
            text=f"<b>{label} — National Inventory Z-Score (3-Year Rolling Window)</b><br>"
                 f"<sup>Z-Score measures how far current stocks are from the 3-year rolling mean in standard deviations. "
                 f"Red bars (>+2σ) = bearish for prices. Green bars (<-2σ) = bullish for prices.</sup>",
            font=dict(size=14)
        ),
        template="plotly_dark",
        height=400,
        xaxis_title="Week Ending Date",
        yaxis_title="Standard Deviations from 3-Year Mean (σ)",
        hovermode="x unified",
        margin=dict(t=90)
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Seasonal Chart ─────────────────────────────────────────────────────────────
def seasonal_chart(cfg, label):
    col        = cfg["inv_col"]
    current_yr = df["year"].max()
    fig        = go.Figure()

    for yr in sorted(df["year"].unique()):
        yr_df = df[df["year"] == yr].groupby("week_of_year")[col].mean().reset_index()
        if yr == current_yr:
            fig.add_trace(go.Scatter(
                x=yr_df["week_of_year"], y=yr_df[col],
                name=f"{yr} (Current)",
                line=dict(color="white", width=3),
                hovertemplate=f"{yr} — Wk %{{x}}: %{{y:,.0f}} Mbbl<extra></extra>"
            ))
        else:
            fig.add_trace(go.Scatter(
                x=yr_df["week_of_year"], y=yr_df[col],
                name=str(yr),
                line=dict(width=1),
                opacity=0.35,
                hovertemplate=f"{yr} — Wk %{{x}}: %{{y:,.0f}} Mbbl<extra></extra>"
            ))

    seasonal_avg = df.groupby("week_of_year")[col].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=seasonal_avg["week_of_year"], y=seasonal_avg[col],
        name="Historical Average (All Years)",
        line=dict(color="yellow", width=2, dash="dot"),
        hovertemplate="Hist Avg — Wk %{x}: %{y:,.0f} Mbbl<extra></extra>"
    ))

    # Summer/winter demand annotations
    fig.add_vrect(x0=22, x1=35, fillcolor="orange", opacity=0.05, line_width=0,
                  annotation_text="Summer Draw", annotation_position="top",
                  annotation_font=dict(color="orange", size=9))
    fig.add_vrect(x0=44, x1=52, fillcolor="steelblue", opacity=0.05, line_width=0,
                  annotation_text="Winter Build", annotation_position="top",
                  annotation_font=dict(color="steelblue", size=9))

    fig.update_layout(
        title=dict(
            text=f"<b>{label} — Seasonal Inventory Pattern by Week of Year</b><br>"
                 f"<sup>Each line = one calendar year. Bold white = current year. "
                 f"Yellow dotted = historical average. Compare current trajectory to prior years to spot early/late builds or draws.</sup>",
            font=dict(size=14)
        ),
        template="plotly_dark",
        height=440,
        xaxis=dict(
            title="Week of Year (1 = first week of January, 52 = last week of December)",
            tickvals=list(range(1, 53, 4)),
        ),
        yaxis_title=cfg["inv_label"],
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9)),
        margin=dict(t=90)
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Fair Value Chart ───────────────────────────────────────────────────────────
def fair_value_chart(cfg, label):
    p     = cfg["prefix"]
    clean = df.dropna(subset=[f"{p}_zscore", cfg["price_col"]])
    if len(clean) < 30:
        st.info("Not enough data for regression model yet.")
        return

    slope, intercept, r, _, _ = stats.linregress(clean[f"{p}_zscore"], clean[cfg["price_col"]])
    clean = clean.copy()
    clean["fair_value"] = intercept + slope * clean[f"{p}_zscore"]

    latest_z     = clean[f"{p}_zscore"].iloc[-1]
    latest_price = clean[cfg["price_col"]].iloc[-1]
    fair_val     = intercept + slope * latest_z
    premium      = latest_price - fair_val

    c1, c2, c3 = st.columns(3)
    c1.metric("💵 Current Spot Price",     f"${latest_price:.2f}/bbl", "",
              help="Most recent weekly average spot price")
    c2.metric("🎯 Model Fair Value",       f"${fair_val:.2f}/bbl",     "",
              help="Price implied by the regression of inventory z-score vs. historical price")
    c3.metric("📏 Premium / Discount",    f"${premium:+.2f}/bbl",
              "Market overpriced vs. inventory signal" if premium > 0 else "Market underpriced vs. inventory signal",
              help="Positive = spot is trading above what inventory levels historically imply. Negative = spot is below fair value.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=clean["date"], y=clean[cfg["price_col"]],
        name="Actual Spot Price",
        line=dict(color="orangered", width=2),
        hovertemplate="Week: %{x|%b %d, %Y}<br>Spot: $%{y:.2f}/bbl<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=clean["date"], y=clean["fair_value"],
        name="Model Fair Value (inventory-implied)",
        line=dict(color="cyan", width=2, dash="dash"),
        hovertemplate="Week: %{x|%b %d, %Y}<br>Fair Value: $%{y:.2f}/bbl<extra></extra>"
    ))

    # Annotate latest gap
    last = clean.iloc[-1]
    fig.add_annotation(
        x=last["date"],
        y=(last[cfg["price_col"]] + last["fair_value"]) / 2,
        text=f"Gap: ${premium:+.2f}/bbl",
        showarrow=False,
        font=dict(color="white", size=11),
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor="white",
        borderwidth=1
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{label} — Spot Price vs. Inventory-Implied Fair Value  (R² = {r**2:.2f})</b><br>"
                 f"<sup>Fair value is estimated via linear regression of inventory z-score vs. historical spot price. "
                 f"R² = {r**2:.2f} means {r**2*100:.0f}% of price variation is explained by inventory levels. "
                 f"When spot > fair value, market may be pricing in factors beyond inventory (geopolitics, demand shocks).</sup>",
            font=dict(size=13)
        ),
        template="plotly_dark",
        height=400,
        xaxis_title="Week Ending Date",
        yaxis_title=cfg["price_label"],
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=100)
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Render ─────────────────────────────────────────────────────────────────────
def render_product(key, label):
    cfg = product_map[key]
    st.subheader(f"📊 {label}")
    kpi_cards(cfg, label)
    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ By PADD District",
        "📦 National SD Bands",
        "📈 Inventory vs. Price",
        "📉 Z-Score",
        "🌀 Seasonal"
    ])

    with tab1:
        padd_latest_bar(cfg, label)
        st.divider()
        padd_stacked_chart(cfg, label)
        st.divider()
        st.subheader("Individual PADD — Inventory vs. 3-Year SD Bands")
        padd_sd_charts(cfg, label)
        st.divider()
        padd_zscore_heatmap(cfg, label)
    with tab2:
        sd_band_chart(cfg, label)
    with tab3:
        combo_chart(cfg, label)
        st.divider()
        st.subheader("💡 Inventory-Implied Fair Value Model")
        fair_value_chart(cfg, label)
    with tab4:
        zscore_chart(cfg, label)
    with tab5:
        seasonal_chart(cfg, label)

if product == "Both":
    for key, label in [("Distillate Fuel Oil","Distillate Fuel Oil"),("Kerosene-Jet Fuel","Kerosene-Jet Fuel")]:
        render_product(key, label)
        st.divider()
else:
    render_product(product, product)

with st.expander("📋 View Raw Data Table — All Fields"):
    base_cols = ["date",
        "distillate_inventory_mbbls","distillate_zscore","distillate_signal","distillate_spot_price",
        "jetfuel_inventory_mbbls","jetfuel_zscore","jetfuel_signal","jetfuel_spot_price"]
    padd_cols = [f"{p}_{padd}_mbbls" for p in ["distillate","jetfuel"] for padd in PADD_MAP if f"{p}_{padd}_mbbls" in df.columns]
    display = df[base_cols + padd_cols].copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display.sort_values("date", ascending=False), use_container_width=True)

st.caption("Built with Streamlit + Plotly | Data: EIA Open Data API | All inventory in Thousand Barrels (Mbbl) | Prices in USD per Barrel")