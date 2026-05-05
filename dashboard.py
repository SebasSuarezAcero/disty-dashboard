import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import os
import numpy as np
from scipy import stats

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Petroleum Products Dashboard",
    page_icon="⛽",
    layout="wide"
)

# ── Load API key ──────────────────────────────────────────────────────────────
try:
    API_KEY = st.secrets["EIA_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    API_KEY = os.getenv("EIA_API_KEY")

# ── Fetch functions ───────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_inventory(product_code, product_label):
    url = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
    headers = {
        "X-Params": '{"frequency":"weekly","data":["value"],"facets":{"product":["' + product_code + '"],"duoarea":["NUS"]},"sort":[{"column":"period","direction":"desc"}],"offset":0,"length":520}'
    }
    params = {"api_key": API_KEY}
    r = requests.get(url, params=params, headers=headers)
    data = r.json()["response"]["data"]
    df = pd.DataFrame(data)
    df = df[["period", "value"]].rename(
        columns={"period": "date", "value": f"{product_label}_inventory_mbbls"}
    )
    df[f"{product_label}_inventory_mbbls"] = pd.to_numeric(
        df[f"{product_label}_inventory_mbbls"], errors="coerce"
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

@st.cache_data(ttl=86400)
def fetch_spot_price(series_code, col_name):
    url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
    headers = {
        "X-Params": '{"frequency":"weekly","data":["value"],"facets":{"series":["' + series_code + '"]},"sort":[{"column":"period","direction":"desc"}],"offset":0,"length":520}'
    }
    params = {"api_key": API_KEY}
    r = requests.get(url, params=params, headers=headers)
    data = r.json()["response"]["data"]
    df = pd.DataFrame(data)
    df = df[["period", "value"]].rename(
        columns={"period": "date", "value": col_name}
    )
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

@st.cache_data(ttl=86400)
def load_all_data():
    distillate_df = fetch_inventory("EPD0", "distillate")
    jetfuel_df    = fetch_inventory("EPJK", "jetfuel")
    dist_price    = fetch_spot_price("EER_EPD2F_PF4_Y35NY_DPG", "distillate_spot_price")
    jet_price     = fetch_spot_price("EER_EPJK_PF4_RGC_DPG",    "jetfuel_spot_price")

    df = distillate_df.merge(jetfuel_df, on="date", how="outer")
    df = df.merge(dist_price, on="date", how="left")
    df = df.merge(jet_price,  on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    # ── Rolling 3-year (156 weeks) stats ──────────────────────────────────────
    for prod in ["distillate", "jetfuel"]:
        col = f"{prod}_inventory_mbbls"
        df[f"{prod}_rolling_mean"] = df[col].rolling(156, min_periods=52).mean()
        df[f"{prod}_rolling_std"]  = df[col].rolling(156, min_periods=52).std()
        df[f"{prod}_upper_2sd"]    = df[f"{prod}_rolling_mean"] + 2 * df[f"{prod}_rolling_std"]
        df[f"{prod}_lower_2sd"]    = df[f"{prod}_rolling_mean"] - 2 * df[f"{prod}_rolling_std"]
        df[f"{prod}_upper_1sd"]    = df[f"{prod}_rolling_mean"] + df[f"{prod}_rolling_std"]
        df[f"{prod}_lower_1sd"]    = df[f"{prod}_rolling_mean"] - df[f"{prod}_rolling_std"]
        df[f"{prod}_zscore"]       = (df[col] - df[f"{prod}_rolling_mean"]) / df[f"{prod}_rolling_std"]

        # ── Signal flags ──────────────────────────────────────────────────────
        conditions = [
            df[col] > df[f"{prod}_upper_2sd"],
            df[col] > df[f"{prod}_upper_1sd"],
            df[col] < df[f"{prod}_lower_2sd"],
            df[col] < df[f"{prod}_lower_1sd"],
        ]
        choices = ["🔴 Bearish Outlier", "🟡 Elevated", "🟢 Bullish Outlier", "🟡 Tight"]
        df[f"{prod}_signal"] = np.select(conditions, choices, default="⚪ Neutral")

    # ── Seasonal (week of year) ───────────────────────────────────────────────
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"]         = df["date"].dt.year

    return df

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Fetching latest EIA data..."):
    df = load_all_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⛽ U.S. Petroleum Products Dashboard")
st.caption("Distillate Fuel Oil & Kerosene-Type Jet Fuel | Source: EIA Weekly Petroleum Status Report")

product = st.radio(
    "Select Product",
    options=["Distillate Fuel Oil", "Kerosene-Jet Fuel", "Both"],
    horizontal=True
)
st.divider()

# ── Config map ────────────────────────────────────────────────────────────────
product_map = {
    "Distillate Fuel Oil": {
        "inv_col":     "distillate_inventory_mbbls",
        "price_col":   "distillate_spot_price",
        "inv_label":   "Distillate Stocks (Mbbl)",
        "price_label": "No.2 Heating Oil Spot ($/bbl)",
        "color":       "steelblue",
        "prefix":      "distillate"
    },
    "Kerosene-Jet Fuel": {
        "inv_col":     "jetfuel_inventory_mbbls",
        "price_col":   "jetfuel_spot_price",
        "inv_label":   "Jet Fuel Stocks (Mbbl)",
        "price_label": "Jet Fuel Spot ($/bbl)",
        "color":       "darkorange",
        "prefix":      "jetfuel"
    }
}

# ── KPI Cards ─────────────────────────────────────────────────────────────────
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

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(f"{label} Stocks",     f"{inv:,.0f} Mbbl",  f"{inv_ch:+,.0f} Mbbl WoW")
    col2.metric("vs. 52-Week Average", f"{vs_avg:+.1f}%",   "Surplus" if vs_avg > 0 else "Deficit")
    col3.metric("Spot Price",          f"${price:.2f}/bbl", f"{pr_ch:+.2f} WoW")
    col4.metric("3-Year Z-Score",      f"{zscore:+.2f}σ",   "Above norm" if zscore > 0 else "Below norm")
    col5.metric("Market Signal",       signal,              "")

# ── SD Band Chart ─────────────────────────────────────────────────────────────
def sd_band_chart(cfg, label):
    p   = cfg["prefix"]
    d   = df.dropna(subset=[f"{p}_rolling_mean"])
    fig = go.Figure()

    # 2SD band (shaded)
    fig.add_trace(go.Scatter(
        x=pd.concat([d["date"], d["date"][::-1]]),
        y=pd.concat([d[f"{p}_upper_2sd"], d[f"{p}_lower_2sd"][::-1]]),
        fill="toself", fillcolor="rgba(100,100,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±2σ Band", hoverinfo="skip"
    ))
    # 1SD band (shaded)
    fig.add_trace(go.Scatter(
        x=pd.concat([d["date"], d["date"][::-1]]),
        y=pd.concat([d[f"{p}_upper_1sd"], d[f"{p}_lower_1sd"][::-1]]),
        fill="toself", fillcolor="rgba(100,100,255,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1σ Band", hoverinfo="skip"
    ))
    # Rolling mean
    fig.add_trace(go.Scatter(
        x=d["date"], y=d[f"{p}_rolling_mean"],
        line=dict(color="white", width=1.5, dash="dot"),
        name="3-Yr Rolling Mean"
    ))
    # Actual inventory bars
    fig.add_trace(go.Bar(
        x=df["date"], y=df[cfg["inv_col"]],
        name=cfg["inv_label"],
        marker_color=cfg["color"], opacity=0.8
    ))

    # Outlier dots
    bearish = df[df[f"{p}_signal"] == "🔴 Bearish Outlier"]
    bullish = df[df[f"{p}_signal"] == "🟢 Bullish Outlier"]
    if not bearish.empty:
        fig.add_trace(go.Scatter(
            x=bearish["date"], y=bearish[cfg["inv_col"]],
            mode="markers", marker=dict(color="red", size=7, symbol="circle"),
            name="Bearish Outlier"
        ))
    if not bullish.empty:
        fig.add_trace(go.Scatter(
            x=bullish["date"], y=bullish[cfg["inv_col"]],
            mode="markers", marker=dict(color="lime", size=7, symbol="circle"),
            name="Bullish Outlier"
        ))

    fig.update_layout(
        title=f"{label} — Inventory with 3-Year Rolling SD Bands",
        template="plotly_dark", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified", barmode="overlay"
    )
    fig.update_yaxes(title_text=cfg["inv_label"])
    st.plotly_chart(fig, use_container_width=True)

# ── Combo Chart ───────────────────────────────────────────────────────────────
def combo_chart(cfg, label):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    avg = df[cfg["inv_col"]].mean()
    fig.add_trace(go.Bar(x=df["date"], y=df[cfg["inv_col"]], name=cfg["inv_label"], marker_color=cfg["color"], opacity=0.75), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["date"], y=df[cfg["price_col"]], name=cfg["price_label"], line=dict(color="orangered", width=2)), secondary_y=True)
    fig.add_hline(y=avg, line_dash="dot", line_color="gray", annotation_text="Avg Stocks", secondary_y=False)
    fig.update_layout(title=f"{label} — Inventory vs. Spot Price", template="plotly_dark", height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02), hovermode="x unified")
    fig.update_yaxes(title_text=cfg["inv_label"], secondary_y=False)
    fig.update_yaxes(title_text=cfg["price_label"], secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# ── Z-Score Chart ─────────────────────────────────────────────────────────────
def zscore_chart(cfg, label):
    p = cfg["prefix"]
    d = df.dropna(subset=[f"{p}_zscore"])
    colors = d[f"{p}_zscore"].apply(
        lambda z: "red" if z > 2 else ("lime" if z < -2 else ("orange" if abs(z) > 1 else "steelblue"))
    )
    fig = go.Figure()
    fig.add_hrect(y0=2,  y1=4,  fillcolor="red",  opacity=0.07, line_width=0, annotation_text="Bearish Zone")
    fig.add_hrect(y0=-4, y1=-2, fillcolor="lime", opacity=0.07, line_width=0, annotation_text="Bullish Zone")
    fig.add_hrect(y0=-1, y1=1,  fillcolor="gray", opacity=0.05, line_width=0)
    fig.add_trace(go.Bar(
        x=d["date"], y=d[f"{p}_zscore"],
        marker_color=colors, name="Z-Score"
    ))
    fig.add_hline(y=0,  line_dash="dash", line_color="white",  line_width=1)
    fig.add_hline(y=2,  line_dash="dot",  line_color="red",    line_width=1)
    fig.add_hline(y=-2, line_dash="dot",  line_color="lime",   line_width=1)
    fig.update_layout(
        title=f"{label} — Inventory Z-Score (3-Year Rolling)",
        template="plotly_dark", height=350,
        yaxis_title="Standard Deviations from Mean",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Seasonal Chart ────────────────────────────────────────────────────────────
def seasonal_chart(cfg, label):
    col        = cfg["inv_col"]
    current_yr = df["year"].max()
    fig        = go.Figure()

    for yr in sorted(df["year"].unique()):
        yr_df = df[df["year"] == yr].groupby("week_of_year")[col].mean().reset_index()
        if yr == current_yr:
            fig.add_trace(go.Scatter(
                x=yr_df["week_of_year"], y=yr_df[col],
                name=str(yr), line=dict(color="white", width=3)
            ))
        else:
            fig.add_trace(go.Scatter(
                x=yr_df["week_of_year"], y=yr_df[col],
                name=str(yr), line=dict(width=1),
                opacity=0.35
            ))

    seasonal_avg = df.groupby("week_of_year")[col].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=seasonal_avg["week_of_year"], y=seasonal_avg[col],
        name="All-Year Avg", line=dict(color="yellow", width=2, dash="dot")
    ))
    fig.update_layout(
        title=f"{label} — Seasonal Inventory Pattern by Week of Year",
        template="plotly_dark", height=400,
        xaxis_title="Week of Year", yaxis_title=cfg["inv_label"],
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Fair Value Regression ─────────────────────────────────────────────────────
def fair_value_chart(cfg, label):
    p     = cfg["prefix"]
    clean = df.dropna(subset=[f"{p}_zscore", cfg["price_col"]])
    if len(clean) < 30:
        st.info("Not enough data for regression model yet.")
        return

    slope, intercept, r, pval, _ = stats.linregress(clean[f"{p}_zscore"], clean[cfg["price_col"]])
    clean = clean.copy()
    clean["fair_value"] = intercept + slope * clean[f"{p}_zscore"]

    latest_z     = clean[f"{p}_zscore"].iloc[-1]
    latest_price = clean[cfg["price_col"]].iloc[-1]
    fair_val     = intercept + slope * latest_z
    premium      = latest_price - fair_val

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Spot",  f"${latest_price:.2f}/bbl", "")
    col2.metric("Model Fair Value", f"${fair_val:.2f}/bbl", "")
    col3.metric("Premium / Discount", f"${premium:+.2f}/bbl",
                "Overpriced vs inventory" if premium > 0 else "Underpriced vs inventory")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=clean["date"], y=clean[cfg["price_col"]],
        name="Actual Spot Price", line=dict(color="orangered", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=clean["date"], y=clean["fair_value"],
        name="Model Fair Value", line=dict(color="cyan", width=2, dash="dash")
    ))
    fig.update_layout(
        title=f"{label} — Spot Price vs. Inventory-Implied Fair Value (R²={r**2:.2f})",
        template="plotly_dark", height=380,
        yaxis_title=cfg["price_label"],
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Render ────────────────────────────────────────────────────────────────────
def render_product(key, label):
    cfg = product_map[key]
    st.subheader(f"📊 {label}")
    kpi_cards(cfg, label)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📦 SD Bands & Outliers",
        "📈 Inventory vs. Price",
        "📉 Z-Score",
        "🌀 Seasonal Pattern"
    ])
    with tab1:
        sd_band_chart(cfg, label)
    with tab2:
        combo_chart(cfg, label)
        st.subheader("💡 Inventory-Implied Fair Value")
        fair_value_chart(cfg, label)
    with tab3:
        zscore_chart(cfg, label)
    with tab4:
        seasonal_chart(cfg, label)

if product == "Both":
    for key, label in [
        ("Distillate Fuel Oil", "Distillate Fuel Oil"),
        ("Kerosene-Jet Fuel",   "Kerosene-Jet Fuel")
    ]:
        render_product(key, label)
        st.divider()
else:
    render_product(product, product)

# ── Raw Data ──────────────────────────────────────────────────────────────────
with st.expander("📋 View Raw Data Table"):
    cols = [
        "date",
        "distillate_inventory_mbbls", "distillate_zscore", "distillate_signal", "distillate_spot_price",
        "jetfuel_inventory_mbbls",    "jetfuel_zscore",    "jetfuel_signal",    "jetfuel_spot_price"
    ]
    display_df = df[cols].copy()
    display_df.columns = [
        "Date",
        "Distillate Stocks (Mbbl)", "Distillate Z-Score", "Distillate Signal", "Distillate Spot ($/bbl)",
        "Jet Fuel Stocks (Mbbl)",   "Jet Fuel Z-Score",   "Jet Fuel Signal",   "Jet Fuel Spot ($/bbl)"
    ]
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display_df.sort_values("Date", ascending=False), use_container_width=True)

st.caption("Built with Streamlit + Plotly | Data: EIA Open Data API")