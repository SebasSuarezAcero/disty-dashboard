
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("EIA_API_KEY", "DEMO_KEY")
BASE = "https://api.eia.gov/v2"

# ── EIA Series Map ──────────────────────────────────────────────────────────────
# Distillate Inventory by PADD  (weekly stocks, petroleum/stoc/wstk)
# duoarea codes: NUS=US total, R10=PADD1, R20=PADD2, R30=PADD3, R40=PADD4, R50=PADD5
# product: EPD0 = distillate fuel oil total

PADD_AREAS = {
    "US Total": "NUS",
    "PADD 1 (East Coast)":   "R10",
    "PADD 2 (Midwest)":      "R20",
    "PADD 3 (Gulf Coast)":   "R30",
    "PADD 4 (Rocky Mtn)":    "R40",
    "PADD 5 (West Coast)":   "R50",
}

# Days of supply series IDs (petroleum/sum/sndw, process=VSD, product=EPD0)
DOS_SERIES = {
    "US Total": "WDISTUS2",   # Weekly U.S. Distillate Days of Supply
}

# Product supplied (demand proxy) – weekly, mbblpd
# petroleum/sum/sndw, process=VPP, product=EPD0
DEMAND_SERIES_ID = "WDISUPUS2"  # Weekly U.S. Distillate Product Supplied

# Refinery utilization – weekly, percent
# petroleum/pnp/wiup, product=EPXXX, duoarea
UTIL_SERIES = {
    "US Total":             "WPULEUS3",
    "PADD 1 (East Coast)":  "WPUR1US3",
    "PADD 2 (Midwest)":     "WPUR2US3",
    "PADD 3 (Gulf Coast)":  "WPUR3US3",
    "PADD 4 (Rocky Mtn)":   "WPUR4US3",
    "PADD 5 (West Coast)":  "WPUR5US3",
}

# Net production of distillate – refinery & blender net production weekly mbblpd
# petroleum/sum/sndw, process=FPF, EPD0
PROD_SERIES_ID = "WDIRPUS2"   # Weekly U.S. Distillate Fuel Oil Refinery Production

# Imports weekly mbblpd
IMPORT_SERIES_ID = "WDIIMUS2"  # Weekly U.S. Distillate Imports

# Exports weekly mbblpd
EXPORT_SERIES_ID = "WDIEEUS2"  # Weekly U.S. Distillate Exports


# ── Data Fetcher ────────────────────────────────────────────────────────────────
def fetch_series(series_id: str, length: int = 156) -> pd.DataFrame:
    """Fetch a single EIA v2 series (petroleum/stoc/wstk or sum/sndw) by series_id."""
    url = f"{BASE}/seriesid/{series_id}"
    params = {"api_key": API_KEY, "data[0]": "value",
              "sort[0][column]": "period", "sort[0][direction]": "asc",
              "length": length}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        rows = data.get("response", {}).get("data", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["period"] = pd.to_datetime(df["period"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["period", "value"]].dropna().sort_values("period").reset_index(drop=True)
    except Exception as e:
        print(f"Error fetching {series_id}: {e}")
        return pd.DataFrame()


def fetch_inventory_padd(area_code: str, length: int = 156) -> pd.DataFrame:
    """Fetch distillate inventory for a PADD using v2 facets."""
    url = f"{BASE}/petroleum/stoc/wstk/data/"
    params = {
        "api_key": API_KEY,
        "data[0]": "value",
        "facets[product][]": "EPD0",
        "facets[duoarea][]": area_code,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": length,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        rows = r.json().get("response", {}).get("data", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["period"] = pd.to_datetime(df["period"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["period", "value"]].dropna().sort_values("period").reset_index(drop=True)
    except Exception as e:
        print(f"Error fetching inventory {area_code}: {e}")
        return pd.DataFrame()


# ── Predictive Model ─────────────────────────────────────────────────────────
def build_forecast(df: pd.DataFrame, weeks_ahead: int = 12) -> pd.DataFrame:
    """
    Simple polynomial regression (degree 2) on inventory levels.
    Uses the last 52 weeks of data to project forward.
    Returns a DataFrame with period + forecast_value + lower + upper.
    """
    if df.empty or len(df) < 10:
        return pd.DataFrame()
    recent = df.tail(52).copy()
    recent["t"] = np.arange(len(recent))
    X = recent[["t"]].values
    y = recent["value"].values

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)

    # Residual std for confidence interval
    y_pred_train = model.predict(X_poly)
    residual_std = np.std(y - y_pred_train)

    # Future periods
    last_date = recent["period"].iloc[-1]
    future_t = np.arange(len(recent), len(recent) + weeks_ahead).reshape(-1, 1)
    future_X = poly.transform(future_t)
    future_vals = model.predict(future_X)
    future_dates = [last_date + timedelta(weeks=i+1) for i in range(weeks_ahead)]

    forecast_df = pd.DataFrame({
        "period": future_dates,
        "forecast": future_vals,
        "lower": future_vals - 1.645 * residual_std,
        "upper": future_vals + 1.645 * residual_std,
    })
    return forecast_df


# ── App Layout ───────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="Disty Dashboard")
server = app.server

PADDS = list(PADD_AREAS.keys())

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Distillate Fuel Oil Dashboard", style={"margin": 0, "fontSize": "1.4rem", "fontWeight": 700}),
        html.P("EIA Weekly Data · Inventory · Supply · Production · Refinery Utilization",
               style={"margin": 0, "opacity": 0.7, "fontSize": "0.85rem"}),
    ], style={"background": "#0f172a", "color": "#e2e8f0", "padding": "16px 24px",
              "borderBottom": "1px solid #1e293b"}),

    # Controls
    html.Div([
        html.Div([
            html.Label("PADD Region", style={"fontSize": "0.75rem", "fontWeight": 600, "textTransform": "uppercase",
                                              "color": "#64748b", "marginBottom": "4px", "display": "block"}),
            dcc.Dropdown(
                id="padd-select",
                options=[{"label": p, "value": p} for p in PADDS],
                value="US Total",
                clearable=False,
                style={"fontSize": "0.875rem"}
            ),
        ], style={"flex": "1", "minWidth": "200px", "maxWidth": "280px"}),

        html.Div([
            html.Label("History (weeks)", style={"fontSize": "0.75rem", "fontWeight": 600, "textTransform": "uppercase",
                                                  "color": "#64748b", "marginBottom": "4px", "display": "block"}),
            dcc.Slider(id="weeks-slider", min=26, max=260, step=26, value=104,
                       marks={26: "6mo", 52: "1yr", 104: "2yr", 156: "3yr", 208: "4yr", 260: "5yr"},
                       tooltip={"always_visible": False}),
        ], style={"flex": "2", "minWidth": "260px"}),

        html.Div([
            html.Label("Forecast (weeks)", style={"fontSize": "0.75rem", "fontWeight": 600, "textTransform": "uppercase",
                                                   "color": "#64748b", "marginBottom": "4px", "display": "block"}),
            dcc.Slider(id="forecast-slider", min=4, max=26, step=2, value=12,
                       marks={4: "4w", 8: "8w", 12: "12w", 18: "18w", 26: "26w"},
                       tooltip={"always_visible": False}),
        ], style={"flex": "1", "minWidth": "220px"}),

        html.Button("Refresh Data", id="refresh-btn", n_clicks=0,
                    style={"background": "#0ea5e9", "color": "white", "border": "none",
                           "borderRadius": "6px", "padding": "8px 18px", "cursor": "pointer",
                           "fontWeight": 600, "alignSelf": "flex-end", "height": "36px"}),
    ], style={"display": "flex", "gap": "24px", "padding": "16px 24px",
              "background": "#f8fafc", "borderBottom": "1px solid #e2e8f0",
              "alignItems": "flex-end", "flexWrap": "wrap"}),

    # KPI Row
    html.Div(id="kpi-row", style={"display": "flex", "gap": "16px", "padding": "16px 24px",
                                   "flexWrap": "wrap"}),

    # Charts grid
    html.Div([
        # Row 1: Inventory (large) + Days of Supply
        html.Div([
            html.Div([dcc.Graph(id="inventory-chart", config={"displayModeBar": False})],
                     style={"flex": "2", "minWidth": "400px", "background": "white",
                            "borderRadius": "8px", "boxShadow": "0 1px 4px rgba(0,0,0,0.08)", "padding": "8px"}),
            html.Div([dcc.Graph(id="dos-chart", config={"displayModeBar": False})],
                     style={"flex": "1", "minWidth": "300px", "background": "white",
                            "borderRadius": "8px", "boxShadow": "0 1px 4px rgba(0,0,0,0.08)", "padding": "8px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # Row 2: Supply Balance + Refinery Utilization
        html.Div([
            html.Div([dcc.Graph(id="supply-chart", config={"displayModeBar": False})],
                     style={"flex": "1", "minWidth": "340px", "background": "white",
                            "borderRadius": "8px", "boxShadow": "0 1px 4px rgba(0,0,0,0.08)", "padding": "8px"}),
            html.Div([dcc.Graph(id="util-chart", config={"displayModeBar": False})],
                     style={"flex": "1", "minWidth": "340px", "background": "white",
                            "borderRadius": "8px", "boxShadow": "0 1px 4px rgba(0,0,0,0.08)", "padding": "8px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginTop": "16px"}),

        # Row 3: PADD Inventory Comparison + Forecast
        html.Div([
            html.Div([dcc.Graph(id="padd-compare-chart", config={"displayModeBar": False})],
                     style={"flex": "1", "minWidth": "340px", "background": "white",
                            "borderRadius": "8px", "boxShadow": "0 1px 4px rgba(0,0,0,0.08)", "padding": "8px"}),
            html.Div([dcc.Graph(id="forecast-chart", config={"displayModeBar": False})],
                     style={"flex": "1", "minWidth": "340px", "background": "white",
                            "borderRadius": "8px", "boxShadow": "0 1px 4px rgba(0,0,0,0.08)", "padding": "8px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginTop": "16px"}),

    ], style={"padding": "0 24px 24px 24px"}),

], style={"fontFamily": "'Inter', 'Helvetica Neue', sans-serif", "background": "#f1f5f9", "minHeight": "100vh"})


# ── Callbacks ────────────────────────────────────────────────────────────────
@app.callback(
    Output("kpi-row", "children"),
    Output("inventory-chart", "figure"),
    Output("dos-chart", "figure"),
    Output("supply-chart", "figure"),
    Output("util-chart", "figure"),
    Output("padd-compare-chart", "figure"),
    Output("forecast-chart", "figure"),
    Input("refresh-btn", "n_clicks"),
    Input("padd-select", "value"),
    Input("weeks-slider", "value"),
    Input("forecast-slider", "value"),
)
def update_all(n_clicks, padd_name, weeks, forecast_weeks):
    area_code = PADD_AREAS[padd_name]
    length = weeks

    # ── Fetch data ──
    inv_df = fetch_inventory_padd(area_code, length)
    prod_df = fetch_series(PROD_SERIES_ID, length)
    imp_df  = fetch_series(IMPORT_SERIES_ID, length)
    exp_df  = fetch_series(EXPORT_SERIES_ID, length)
    dem_df  = fetch_series(DEMAND_SERIES_ID, length)

    # Days of supply: calculate from US inventory + demand if series unavailable
    # DOS = inventory / (demand * 7/1000) — inv in mbbls, demand in mbblpd
    dos_df = pd.DataFrame()
    if not inv_df.empty and not dem_df.empty:
        merged = pd.merge(inv_df, dem_df, on="period", suffixes=("_inv", "_dem"))
        merged["dos"] = merged["value_inv"] / merged["value_dem"]
        dos_df = merged[["period", "dos"]].rename(columns={"dos": "value"})

    # Refinery utilization — use US or PADD-specific
    util_series = UTIL_SERIES.get(padd_name, UTIL_SERIES["US Total"])
    util_df = fetch_series(util_series, length)

    # Net supply balance = production + imports - exports (mbblpd)
    supply_df = pd.DataFrame()
    if not prod_df.empty:
        supply_df = prod_df.copy().rename(columns={"value": "production"})
        if not imp_df.empty:
            supply_df = pd.merge(supply_df, imp_df.rename(columns={"value": "imports"}),
                                 on="period", how="outer")
        if not exp_df.empty:
            supply_df = pd.merge(supply_df, exp_df.rename(columns={"value": "exports"}),
                                 on="period", how="outer")
        supply_df = supply_df.fillna(0)
        if "imports" in supply_df.columns and "exports" in supply_df.columns:
            supply_df["net_supply"] = supply_df["production"] + supply_df["imports"] - supply_df["exports"]
        else:
            supply_df["net_supply"] = supply_df["production"]
        if not dem_df.empty:
            supply_df = pd.merge(supply_df, dem_df.rename(columns={"value": "demand"}),
                                 on="period", how="left")
            supply_df["balance"] = supply_df["net_supply"] - supply_df.get("demand", 0)
        supply_df = supply_df.sort_values("period").tail(weeks)

    # ── KPIs ──
    def latest_val(df, col="value"):
        if df.empty or col not in df.columns:
            return None
        v = df[col].dropna()
        return float(v.iloc[-1]) if len(v) else None

    def yoy_delta(df, col="value"):
        if df.empty or col not in df.columns or len(df) < 53:
            return None
        v = df[col].dropna()
        if len(v) < 53:
            return None
        return float(v.iloc[-1]) - float(v.iloc[-53])

    inv_val  = latest_val(inv_df)
    dos_val  = latest_val(dos_df)
    prod_val = latest_val(prod_df)
    util_val = latest_val(util_df)
    inv_yoy  = yoy_delta(inv_df)

    def kpi_card(title, value, unit, delta=None, delta_label="vs yr ago"):
        delta_color = "#16a34a" if (delta or 0) >= 0 else "#dc2626"
        delta_html = html.Span(
            f"{'▲' if (delta or 0)>=0 else '▼'} {abs(delta or 0):,.0f} {delta_label}",
            style={"fontSize": "0.72rem", "color": delta_color, "marginTop": "2px", "display": "block"}
        ) if delta is not None else html.Span()
        return html.Div([
            html.P(title, style={"fontSize": "0.7rem", "fontWeight": 600, "textTransform": "uppercase",
                                  "color": "#64748b", "margin": "0 0 4px 0"}),
            html.P(f"{value:,.1f} {unit}" if value is not None else "—",
                   style={"fontSize": "1.5rem", "fontWeight": 700, "color": "#0f172a", "margin": 0}),
            delta_html,
        ], style={"background": "white", "borderRadius": "8px", "padding": "14px 18px",
                  "boxShadow": "0 1px 4px rgba(0,0,0,0.08)", "minWidth": "160px", "flex": "1"})

    kpis = [
        kpi_card("Inventory", inv_val, "mbbls", inv_yoy),
        kpi_card("Days of Supply", dos_val, "days"),
        kpi_card("Refinery Production", prod_val, "mbbl/d"),
        kpi_card("Refinery Utilization", util_val, "%"),
    ]

    # ── Chart helpers ──
    COLORS = {"inventory": "#0ea5e9", "production": "#10b981", "imports": "#8b5cf6",
              "exports": "#f59e0b", "demand": "#ef4444", "balance": "#0ea5e9",
              "util": "#6366f1", "dos": "#f97316", "forecast": "#0ea5e9"}

    def empty_fig(msg="No data available"):
        fig = go.Figure()
        fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font={"size": 14, "color": "#94a3b8"})
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                          margin={"t": 40, "l": 10, "r": 10, "b": 10})
        return fig

    def base_layout(fig, title, yaxis_title="", yaxis2=False):
        fig.update_layout(
            title={"text": title, "font": {"size": 13, "color": "#0f172a"}, "x": 0.01, "xanchor": "left"},
            paper_bgcolor="white", plot_bgcolor="white",
            margin={"t": 44, "l": 10, "r": 16, "b": 10},
            legend={"orientation": "h", "y": -0.15, "font": {"size": 11}},
            hovermode="x unified",
            xaxis={"showgrid": False, "color": "#64748b", "tickfont": {"size": 10}},
            yaxis={"gridcolor": "#f1f5f9", "color": "#64748b", "tickfont": {"size": 10},
                   "title": {"text": yaxis_title, "font": {"size": 10}}},
        )
        return fig

    # 1. Inventory chart with 5yr range shading
    fig_inv = go.Figure()
    if not inv_df.empty:
        # 5-yr seasonal range (±1 std by week-of-year)
        inv_df["week"] = inv_df["period"].dt.isocalendar().week.astype(int)
        seasonal = inv_df.groupby("week")["value"].agg(["mean","std"]).reset_index()
        inv_with_season = pd.merge(inv_df, seasonal, on="week")
        fig_inv.add_trace(go.Scatter(
            x=inv_df["period"], y=inv_with_season["mean"] + inv_with_season["std"],
            fill=None, line={"width": 0}, showlegend=False, name="Seasonal Hi"))
        fig_inv.add_trace(go.Scatter(
            x=inv_df["period"], y=inv_with_season["mean"] - inv_with_season["std"],
            fill="tonexty", fillcolor="rgba(14,165,233,0.10)",
            line={"width": 0}, name="Seasonal Range", showlegend=True))
        fig_inv.add_trace(go.Scatter(
            x=inv_df["period"], y=inv_df["value"],
            line={"color": COLORS["inventory"], "width": 2},
            name="Inventory (mbbls)"))
    base_layout(fig_inv, f"Distillate Inventory — {padd_name}", "mbbls")

    # 2. Days of supply
    fig_dos = go.Figure()
    if not dos_df.empty:
        fig_dos.add_trace(go.Scatter(
            x=dos_df["period"], y=dos_df["value"],
            line={"color": COLORS["dos"], "width": 2}, name="Days of Supply",
            fill="tozeroy", fillcolor="rgba(249,115,22,0.08)"))
        # 20-day reference line
        fig_dos.add_hline(y=20, line_dash="dot", line_color="#ef4444",
                          annotation_text="20-day threshold", annotation_position="bottom right",
                          annotation_font_size=10)
    base_layout(fig_dos, "Days of Supply (US)", "days")

    # 3. Supply balance: production + imports - exports vs demand
    fig_sup = go.Figure()
    if not supply_df.empty:
        if "production" in supply_df.columns:
            fig_sup.add_trace(go.Bar(x=supply_df["period"], y=supply_df["production"],
                                     name="Production", marker_color=COLORS["production"], opacity=0.8))
        if "imports" in supply_df.columns:
            fig_sup.add_trace(go.Bar(x=supply_df["period"], y=supply_df["imports"],
                                     name="Imports", marker_color=COLORS["imports"], opacity=0.8))
        if "exports" in supply_df.columns:
            fig_sup.add_trace(go.Bar(x=supply_df["period"], y=-supply_df["exports"],
                                     name="Exports (−)", marker_color=COLORS["exports"], opacity=0.8))
        if "demand" in supply_df.columns:
            fig_sup.add_trace(go.Scatter(x=supply_df["period"], y=supply_df["demand"],
                                          line={"color": COLORS["demand"], "width": 2, "dash": "dot"},
                                          name="Product Supplied (Demand)"))
        fig_sup.update_layout(barmode="relative")
    base_layout(fig_sup, "Supply Balance (US, mbbl/d)", "mbbl/d")

    # 4. Refinery utilization
    fig_util = go.Figure()
    if not util_df.empty:
        fig_util.add_trace(go.Scatter(
            x=util_df["period"], y=util_df["value"],
            line={"color": COLORS["util"], "width": 2}, name="Utilization %",
            fill="tozeroy", fillcolor="rgba(99,102,241,0.08)"))
        fig_util.add_hline(y=90, line_dash="dot", line_color="#10b981",
                           annotation_text="90% threshold", annotation_position="top right",
                           annotation_font_size=10)
    base_layout(fig_util, f"Refinery Utilization — {padd_name}", "%")

    # 5. PADD comparison (last value per PADD)
    fig_compare = go.Figure()
    padd_latest = []
    for pname, pcode in PADD_AREAS.items():
        if pname == "US Total":
            continue
        df_p = fetch_inventory_padd(pcode, 4)
        if not df_p.empty:
            padd_latest.append({"PADD": pname, "inventory": float(df_p["value"].iloc[-1])})
    if padd_latest:
        df_c = pd.DataFrame(padd_latest).sort_values("inventory", ascending=True)
        colors_bar = ["#0ea5e9" if p == padd_name else "#cbd5e1" for p in df_c["PADD"]]
        fig_compare.add_trace(go.Bar(
            x=df_c["inventory"], y=df_c["PADD"],
            orientation="h", marker_color=colors_bar, name="Inventory"))
    base_layout(fig_compare, "Current Inventory by PADD (mbbls)", "mbbls")

    # 6. Forecast chart
    fig_fc = go.Figure()
    if not inv_df.empty:
        forecast_df = build_forecast(inv_df, weeks_ahead=forecast_weeks)
        # Historical (last 52 weeks only for clarity)
        hist_tail = inv_df.tail(52)
        fig_fc.add_trace(go.Scatter(
            x=hist_tail["period"], y=hist_tail["value"],
            line={"color": "#0ea5e9", "width": 2}, name="Historical"))
        if not forecast_df.empty:
            fig_fc.add_trace(go.Scatter(
                x=forecast_df["period"].tolist() + forecast_df["period"].tolist()[::-1],
                y=forecast_df["upper"].tolist() + forecast_df["lower"].tolist()[::-1],
                fill="toself", fillcolor="rgba(14,165,233,0.12)",
                line={"width": 0}, name="90% CI", showlegend=True))
            fig_fc.add_trace(go.Scatter(
                x=forecast_df["period"], y=forecast_df["forecast"],
                line={"color": "#f97316", "width": 2, "dash": "dash"}, name="Forecast"))
    base_layout(fig_fc, f"Inventory Forecast — {padd_name} ({forecast_weeks}w)", "mbbls")

    return kpis, fig_inv, fig_dos, fig_sup, fig_util, fig_compare, fig_fc


if __name__ == "__main__":
    app.run(debug=True)
