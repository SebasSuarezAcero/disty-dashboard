import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Petroleum Products Dashboard",
    page_icon="⛽",
    layout="wide"
)

# ── Load API key (works both locally and on Streamlit Cloud) ──────────────────
try:
    API_KEY = st.secrets["EIA_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    API_KEY = os.getenv("EIA_API_KEY")

# ── Fetch data directly from EIA API ─────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_inventory(product_code, product_label):
    url = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
    headers = {
        "X-Params": '{"frequency":"weekly","data":["value"],"facets":{"product":["' + product_code + '"],"duoarea":["NUS"]},"sort":[{"column":"period","direction":"desc"}],"offset":0,"length":260}'
    }
    params = {"api_key": API_KEY}
    response = requests.get(url, params=params, headers=headers)
    data = response.json()["response"]["data"]
    df = pd.DataFrame(data)
    df = df[["period", "value"]].rename(
        columns={"period": "date", "value": f"{product_label}_inventory_mbbls"}
    )
    df[f"{product_label}_inventory_mbbls"] = pd.to_numeric(
        df[f"{product_label}_inventory_mbbls"], errors="coerce"
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")

@st.cache_data(ttl=86400)
def fetch_spot_price(series_code, col_name):
    url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
    headers = {
        "X-Params": '{"frequency":"weekly","data":["value"],"facets":{"series":["' + series_code + '"]},"sort":[{"column":"period","direction":"desc"}],"offset":0,"length":260}'
    }
    params = {"api_key": API_KEY}
    response = requests.get(url, params=params, headers=headers)
    data = response.json()["response"]["data"]
    df = pd.DataFrame(data)
    df = df[["period", "value"]].rename(
        columns={"period": "date", "value": col_name}
    )
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")

@st.cache_data(ttl=86400)
def load_all_data():
    distillate_df = fetch_inventory("EPD0", "distillate")
    jetfuel_df    = fetch_inventory("EPJK", "jetfuel")
    dist_price    = fetch_spot_price("EER_EPD2F_PF4_Y35NY_DPG", "distillate_spot_price")
    jet_price     = fetch_spot_price("EER_EPJK_PF4_RGC_DPG",    "jetfuel_spot_price")
    df = distillate_df.merge(jetfuel_df, on="date", how="outer")
    df = df.merge(dist_price, on="date", how="left")
    df = df.merge(jet_price,  on="date", how="left")
    return df.sort_values("date")

# ── Load data with spinner ────────────────────────────────────────────────────
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

product_map = {
    "Distillate Fuel Oil": {
        "inv_col":     "distillate_inventory_mbbls",
        "price_col":   "distillate_spot_price",
        "inv_label":   "Distillate Stocks (Mbbl)",
        "price_label": "No.2 Heating Oil Spot ($/bbl)",
        "color":       "steelblue"
    },
    "Kerosene-Jet Fuel": {
        "inv_col":     "jetfuel_inventory_mbbls",
        "price_col":   "jetfuel_spot_price",
        "inv_label":   "Jet Fuel Stocks (Mbbl)",
        "price_label": "Jet Fuel Spot ($/bbl)",
        "color":       "darkorange"
    }
}

def kpi_cards(cfg, label):
    latest = df.iloc[-1]
    prior  = df.iloc[-2]
    inv    = latest[cfg["inv_col"]]
    inv_ch = inv - prior[cfg["inv_col"]]
    avg    = df[cfg["inv_col"]].mean()
    vs_avg = ((inv - avg) / avg) * 100
    price  = latest[cfg["price_col"]]
    pr_ch  = price - prior[cfg["price_col"]]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"{label} Stocks",     f"{inv:,.0f} Mbbl",  f"{inv_ch:+,.0f} Mbbl WoW")
    col2.metric("vs. 52-Week Average", f"{vs_avg:+.1f}%",   "Surplus" if vs_avg > 0 else "Deficit")
    col3.metric("Spot Price",          f"${price:.2f}/bbl", f"{pr_ch:+.2f} WoW")
    col4.metric("52-Week Avg Stocks",  f"{avg:,.0f} Mbbl",  "")

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

def scatter_chart(cfg, label):
    clean = df.dropna(subset=[cfg["inv_col"], cfg["price_col"]])
    fig = px.scatter(clean, x=cfg["inv_col"], y=cfg["price_col"], color=cfg["price_col"], color_continuous_scale="RdYlGn_r", hover_data={"date": True}, trendline="ols", labels={cfg["inv_col"]: cfg["inv_label"], cfg["price_col"]: cfg["price_label"], "date": "Week"}, template="plotly_dark", title=f"{label} — Inventory vs. Price Correlation")
    fig.update_layout(height=380, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

if product == "Both":
    col_a, col_b = st.columns(2)
    for col, key, label in [(col_a, "Distillate Fuel Oil", "Distillate Fuel Oil"), (col_b, "Kerosene-Jet Fuel", "Kerosene-Jet Fuel")]:
        with col:
            st.subheader(label)
            cfg    = product_map[key]
            latest = df.iloc[-1]
            prior  = df.iloc[-2]
            st.metric(f"{label} Stocks", f"{latest[cfg['inv_col']]:,.0f} Mbbl", f"{latest[cfg['inv_col']] - prior[cfg['inv_col']]:+,.0f} WoW")
            st.metric("Spot Price", f"${latest[cfg['price_col']]:.2f}/bbl", f"{latest[cfg['price_col']] - prior[cfg['price_col']]:+.2f} WoW")
    st.divider()
    for key, label in [("Distillate Fuel Oil", "Distillate Fuel Oil"), ("Kerosene-Jet Fuel", "Kerosene-Jet Fuel")]:
        combo_chart(product_map[key], label)
        scatter_chart(product_map[key], label)
        st.divider()
else:
    cfg = product_map[product]
    kpi_cards(cfg, product)
    st.divider()
    combo_chart(cfg, product)
    st.divider()
    scatter_chart(cfg, product)
    st.divider()

with st.expander("📋 View Raw Data Table"):
    display_df = df[["date", "distillate_inventory_mbbls", "jetfuel_inventory_mbbls", "distillate_spot_price", "jetfuel_spot_price"]].copy()
    display_df.columns = ["Date", "Distillate Stocks (Mbbl)", "Jet Fuel Stocks (Mbbl)", "Distillate Spot ($/bbl)", "Jet Fuel Spot ($/bbl)"]
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display_df.sort_values("Date", ascending=False), use_container_width=True)

st.caption("Built with Streamlit + Plotly | Data: EIA Open Data API")