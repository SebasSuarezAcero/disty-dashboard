import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
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

# ══════════════════════════════════════════════════════════════════════════════
# CONFIRMED EIA SERIES IDs  (verified from live WPSR table 5/6/2026)
# ══════════════════════════════════════════════════════════════════════════════

UTIL_SERIES = {
    "US Total":            "WPULEUS3",
    "PADD 1 (East Coast)": "WPUR1US3",
    "PADD 2 (Midwest)":    "WPUR2US3",
    "PADD 3 (Gulf Coast)": "WPUR3US3",
    "PADD 4 (Rocky Mtn)":  "WPUR4US3",
    "PADD 5 (West Coast)": "WPUR5US3",
}

PADD_AREAS = {
    "US Total":            "NUS",
    "PADD 1 (East Coast)": "R10",
    "PADD 2 (Midwest)":    "R20",
    "PADD 3 (Gulf Coast)": "R30",
    "PADD 4 (Rocky Mtn)":  "R40",
    "PADD 5 (West Coast)": "R50",
}

# Product-specific series
PRODUCTS = {
    "Distillate Fuel Oil": {
        "prod":         "WDIRPUS2",
        "imp":          "WDIIMUS2",
        "exp":          "WDIEEUS2",
        "dem":          "WDISUPUS2",
        "dos":          "WDISDUS2",
        "product_code": "EPD0",
        "color":        "#0ea5e9",
    },
    "Jet Fuel (Kerosene)": {
        "prod":         "WKJRPUS2",
        "imp":          "WKJIMUS2",
        "exp":          "WKJEEUS2",
        "dem":          "WKJUPUS2",
        "dos":          "WKJSDUS2",
        "product_code": "EPJK",
        "color":        "#8b5cf6",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_series(series_id, length=156):
    url = f"{BASE}/seriesid/{series_id}"
    params = {"api_key": API_KEY, "data[0]": "value",
              "sort[0][column]": "period", "sort[0][direction]": "asc",
              "length": length, "offset": 0}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        rows = r.json().get("response", {}).get("data", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["period"] = pd.to_datetime(df["period"])
        df["value"]  = pd.to_numeric(df["value"], errors="coerce")
        return df[["period","value"]].dropna().sort_values("period").reset_index(drop=True)
    except Exception as e:
        print(f"[WARN] fetch_series({series_id}): {e}")
        return pd.DataFrame()


def fetch_inventory_padd(area_code, product_code, length=156):
    url = f"{BASE}/petroleum/stoc/wstk/data/"
    params = {"api_key": API_KEY, "data[0]": "value",
              "facets[product][]": product_code,
              "facets[duoarea][]": area_code,
              "sort[0][column]": "period", "sort[0][direction]": "asc",
              "length": length, "offset": 0}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        rows = r.json().get("response", {}).get("data", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["period"] = pd.to_datetime(df["period"])
        df["value"]  = pd.to_numeric(df["value"], errors="coerce")
        return df[["period","value"]].dropna().sort_values("period").reset_index(drop=True)
    except Exception as e:
        print(f"[WARN] fetch_inventory_padd({area_code},{product_code}): {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# FORECASTING
# ══════════════════════════════════════════════════════════════════════════════

def build_forecast(df, weeks_ahead=12):
    if df.empty or len(df) < 10:
        return pd.DataFrame()
    recent = df.tail(52).copy()
    recent["t"] = np.arange(len(recent))
    X, y = recent[["t"]].values, recent["value"].values
    poly = PolynomialFeatures(degree=2)
    Xp   = poly.fit_transform(X)
    model = LinearRegression().fit(Xp, y)
    std   = np.std(y - model.predict(Xp))
    last  = recent["period"].iloc[-1]
    ft    = np.arange(len(recent), len(recent)+weeks_ahead).reshape(-1,1)
    fv    = model.predict(poly.transform(ft))
    return pd.DataFrame({
        "period":   [last + timedelta(weeks=i+1) for i in range(weeks_ahead)],
        "forecast": fv,
        "lower":    fv - 1.645*std,
        "upper":    fv + 1.645*std,
    })


# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

def chart_card(cid):
    return html.Div(
        [dcc.Graph(id=cid, config={"displayModeBar": False})],
        style={"flex":"1","minWidth":"340px","background":"white",
               "borderRadius":"8px","boxShadow":"0 1px 4px rgba(0,0,0,0.08)","padding":"8px"}
    )

def lbl(txt):
    return html.Label(txt, style={"fontSize":"0.72rem","fontWeight":600,
                                   "textTransform":"uppercase","color":"#64748b",
                                   "marginBottom":"4px","display":"block"})


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

app = dash.Dash(__name__, title="Disty Dashboard")
server = app.server

app.layout = html.Div([

    html.Div([
        html.H1("Petroleum Supply Dashboard",
                style={"margin":0,"fontSize":"1.35rem","fontWeight":700}),
        html.P("EIA Weekly · Distillate & Jet Fuel · Inventory · Production · Days of Supply · Utilization",
               style={"margin":0,"opacity":.65,"fontSize":"0.82rem"}),
    ], style={"background":"#0f172a","color":"#e2e8f0",
              "padding":"14px 24px","borderBottom":"2px solid #1e293b"}),

    html.Div([
        html.Div([lbl("Product"),
                  dcc.Dropdown(id="product-select",
                               options=[{"label":p,"value":p} for p in PRODUCTS],
                               value="Distillate Fuel Oil", clearable=False,
                               style={"fontSize":"0.875rem"})],
                 style={"flex":"1","minWidth":"220px","maxWidth":"280px"}),

        html.Div([lbl("PADD Region"),
                  dcc.Dropdown(id="padd-select",
                               options=[{"label":p,"value":p} for p in PADD_AREAS],
                               value="US Total", clearable=False,
                               style={"fontSize":"0.875rem"})],
                 style={"flex":"1","minWidth":"200px","maxWidth":"260px"}),

        html.Div([lbl("History (weeks)"),
                  dcc.Slider(id="weeks-slider", min=26, max=260, step=26, value=104,
                             marks={26:"6mo",52:"1yr",104:"2yr",156:"3yr",208:"4yr",260:"5yr"},
                             tooltip={"always_visible":False})],
                 style={"flex":"2","minWidth":"240px"}),

        html.Div([lbl("Forecast (weeks)"),
                  dcc.Slider(id="forecast-slider", min=4, max=26, step=2, value=12,
                             marks={4:"4w",8:"8w",12:"12w",18:"18w",26:"26w"},
                             tooltip={"always_visible":False})],
                 style={"flex":"1","minWidth":"200px"}),

        html.Button("↻ Refresh", id="refresh-btn", n_clicks=0,
                    style={"background":"#0ea5e9","color":"white","border":"none",
                           "borderRadius":"6px","padding":"8px 18px","cursor":"pointer",
                           "fontWeight":600,"alignSelf":"flex-end","height":"36px"}),
    ], style={"display":"flex","gap":"20px","padding":"14px 24px",
              "background":"#f8fafc","borderBottom":"1px solid #e2e8f0",
              "alignItems":"flex-end","flexWrap":"wrap"}),

    html.Div(id="kpi-row", style={"display":"flex","gap":"14px",
                                   "padding":"14px 24px","flexWrap":"wrap"}),

    html.Div([
        html.Div([chart_card("inventory-chart"), chart_card("dos-chart")],
                 style={"display":"flex","gap":"14px","flexWrap":"wrap"}),
        html.Div([chart_card("supply-chart"), chart_card("util-chart")],
                 style={"display":"flex","gap":"14px","flexWrap":"wrap","marginTop":"14px"}),
        html.Div([chart_card("padd-compare-chart"), chart_card("forecast-chart")],
                 style={"display":"flex","gap":"14px","flexWrap":"wrap","marginTop":"14px"}),
    ], style={"padding":"0 24px 24px 24px"}),

], style={"fontFamily":"'Inter','Helvetica Neue',sans-serif",
          "background":"#f1f5f9","minHeight":"100vh"})


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACK
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("kpi-row",            "children"),
    Output("inventory-chart",    "figure"),
    Output("dos-chart",          "figure"),
    Output("supply-chart",       "figure"),
    Output("util-chart",         "figure"),
    Output("padd-compare-chart", "figure"),
    Output("forecast-chart",     "figure"),
    Input("refresh-btn",      "n_clicks"),
    Input("product-select",   "value"),
    Input("padd-select",      "value"),
    Input("weeks-slider",     "value"),
    Input("forecast-slider",  "value"),
)
def update(n_clicks, product_name, padd_name, weeks, fw):
    cfg       = PRODUCTS[product_name]
    area_code = PADD_AREAS[padd_name]
    prod_code = cfg["product_code"]
    accent    = cfg["color"]

    # hex to rgb helper for rgba strings
    def hex_rgba(h, a):
        h = h.lstrip("#")
        r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f"rgba({r},{g},{b},{a})"

    # Fetch
    inv_df  = fetch_inventory_padd(area_code, prod_code, weeks)
    prod_df = fetch_series(cfg["prod"],  weeks)
    imp_df  = fetch_series(cfg["imp"],   weeks)
    exp_df  = fetch_series(cfg["exp"],   weeks)
    dem_df  = fetch_series(cfg["dem"],   weeks)
    dos_df  = fetch_series(cfg["dos"],   weeks)
    util_df = fetch_series(UTIL_SERIES.get(padd_name, UTIL_SERIES["US Total"]), weeks)

    # Fallback: calculate DOS if series empty
    if dos_df.empty and not inv_df.empty and not dem_df.empty:
        m = pd.merge(inv_df, dem_df, on="period", suffixes=("_i","_d"))
        m["dos"] = m["value_i"] / m["value_d"]
        dos_df = m[["period","dos"]].rename(columns={"dos":"value"})

    # Supply balance
    sup = pd.DataFrame()
    if not prod_df.empty:
        sup = prod_df.rename(columns={"value":"production"})
        for col, df in [("imports",imp_df),("exports",exp_df),("demand",dem_df)]:
            if not df.empty:
                sup = pd.merge(sup, df.rename(columns={"value":col}), on="period", how="outer")
        sup = sup.fillna(0).sort_values("period").tail(weeks)

    # KPIs
    def lv(df, col="value"):
        s = df[col].dropna() if (not df.empty and col in df.columns) else pd.Series(dtype=float)
        return float(s.iloc[-1]) if len(s) else None

    def yoy_d(df):
        s = df["value"].dropna() if not df.empty else pd.Series(dtype=float)
        return float(s.iloc[-1])-float(s.iloc[-53]) if len(s)>=53 else None

    def kpi(title, val, unit, delta=None):
        dc = "#16a34a" if (delta or 0)>=0 else "#dc2626"
        return html.Div([
            html.P(title, style={"fontSize":"0.68rem","fontWeight":600,
                                  "textTransform":"uppercase","color":"#64748b","margin":"0 0 3px 0"}),
            html.P(f"{val:,.1f} {unit}" if val is not None else "—",
                   style={"fontSize":"1.45rem","fontWeight":700,"color":"#0f172a","margin":0}),
            html.Span(f"{'▲' if (delta or 0)>=0 else '▼'} {abs(delta or 0):,.0f} vs yr ago",
                      style={"fontSize":"0.7rem","color":dc}) if delta is not None else html.Span(),
        ], style={"background":"white","borderRadius":"8px","padding":"12px 16px",
                  "boxShadow":"0 1px 4px rgba(0,0,0,0.08)","minWidth":"148px","flex":"1"})

    net_imp = None
    if lv(imp_df) is not None and lv(exp_df) is not None:
        net_imp = lv(imp_df) - lv(exp_df)

    kpis = [
        kpi("Inventory",            lv(inv_df),  "mbbls",   yoy_d(inv_df)),
        kpi("Days of Supply",       lv(dos_df),  "days"),
        kpi("Refinery Production",  lv(prod_df), "mbbl/d"),
        kpi("Refinery Utilization", lv(util_df), "%"),
        kpi("Product Supplied",     lv(dem_df),  "mbbl/d"),
        kpi("Net Imports",          net_imp,      "mbbl/d"),
    ]

    # Chart base style
    def base(fig, title, ytitle=""):
        fig.update_layout(
            title={"text":title,"font":{"size":12,"color":"#0f172a"},"x":0.01,"xanchor":"left"},
            paper_bgcolor="white", plot_bgcolor="white",
            margin={"t":42,"l":10,"r":14,"b":10},
            legend={"orientation":"h","y":-0.18,"font":{"size":10}},
            hovermode="x unified",
            xaxis={"showgrid":False,"color":"#64748b","tickfont":{"size":9}},
            yaxis={"gridcolor":"#f1f5f9","color":"#64748b","tickfont":{"size":9},
                   "title":{"text":ytitle,"font":{"size":9}}},
        )
        return fig

    def empty_fig(msg="No data available"):
        fig = go.Figure()
        fig.add_annotation(text=msg, xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font={"size":13,"color":"#94a3b8"})
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                          margin={"t":40,"l":10,"r":10,"b":10})
        return fig

    # 1 Inventory
    fig_inv = go.Figure()
    if not inv_df.empty:
        inv_df["woy"] = inv_df["period"].dt.isocalendar().week.astype(int)
        sea = inv_df.groupby("woy")["value"].agg(["mean","std"]).reset_index()
        md  = pd.merge(inv_df, sea, on="woy")
        fig_inv.add_trace(go.Scatter(x=inv_df["period"], y=(md["mean"]+md["std"]).values,
                                      fill=None, line={"width":0}, showlegend=False, name="_hi"))
        fig_inv.add_trace(go.Scatter(x=inv_df["period"], y=(md["mean"]-md["std"]).values,
                                      fill="tonexty", fillcolor=hex_rgba(accent,0.10),
                                      line={"width":0}, name="Seasonal ±1σ"))
        fig_inv.add_trace(go.Scatter(x=inv_df["period"], y=inv_df["value"],
                                      line={"color":accent,"width":2}, name="Inventory"))
        base(fig_inv, f"{product_name} Inventory — {padd_name}", "mbbls")
    else:
        fig_inv = empty_fig(f"No inventory data · {product_name} / {padd_name}")

    # 2 Days of Supply
    fig_dos = go.Figure()
    if not dos_df.empty:
        fig_dos.add_trace(go.Scatter(x=dos_df["period"], y=dos_df["value"],
                                      line={"color":"#f97316","width":2},
                                      fill="tozeroy", fillcolor="rgba(249,115,22,0.07)",
                                      name="Days of Supply"))
        fig_dos.add_hline(y=20, line_dash="dot", line_color="#ef4444",
                          annotation_text="20-day floor",
                          annotation_position="bottom right", annotation_font_size=9)
        base(fig_dos, f"{product_name} Days of Supply (US)", "days")
    else:
        fig_dos = empty_fig("No days of supply data")

    # 3 Supply Balance
    fig_sup = go.Figure()
    if not sup.empty:
        if "production" in sup.columns:
            fig_sup.add_trace(go.Bar(x=sup["period"], y=sup["production"],
                                      name="Production", marker_color="#10b981", opacity=.85))
        if "imports" in sup.columns:
            fig_sup.add_trace(go.Bar(x=sup["period"], y=sup["imports"],
                                      name="Imports", marker_color="#8b5cf6", opacity=.85))
        if "exports" in sup.columns:
            fig_sup.add_trace(go.Bar(x=sup["period"], y=-sup["exports"],
                                      name="Exports (−)", marker_color="#f59e0b", opacity=.85))
        if "demand" in sup.columns:
            fig_sup.add_trace(go.Scatter(x=sup["period"], y=sup["demand"],
                                          line={"color":"#ef4444","width":2,"dash":"dot"},
                                          name="Product Supplied"))
        fig_sup.update_layout(barmode="relative")
        base(fig_sup, f"{product_name} Supply Balance (US, mbbl/d)", "mbbl/d")
    else:
        fig_sup = empty_fig("No supply balance data")

    # 4 Refinery Utilization
    fig_util = go.Figure()
    if not util_df.empty:
        fig_util.add_trace(go.Scatter(x=util_df["period"], y=util_df["value"],
                                       line={"color":"#6366f1","width":2},
                                       fill="tozeroy", fillcolor="rgba(99,102,241,0.07)",
                                       name="Utilization %"))
        fig_util.add_hline(y=90, line_dash="dot", line_color="#10b981",
                           annotation_text="90% benchmark",
                           annotation_position="top right", annotation_font_size=9)
        base(fig_util, f"Refinery Utilization — {padd_name}", "%")
    else:
        fig_util = empty_fig("No refinery utilization data")

    # 5 PADD Comparison
    fig_cmp = go.Figure()
    padd_rows = []
    for pname, pcode in PADD_AREAS.items():
        if pname == "US Total":
            continue
        df_p = fetch_inventory_padd(pcode, prod_code, 4)
        if not df_p.empty:
            padd_rows.append({"PADD": pname, "inv": float(df_p["value"].iloc[-1])})
    if padd_rows:
        dc = pd.DataFrame(padd_rows).sort_values("inv", ascending=True)
        bar_colors = [accent if p==padd_name else "#cbd5e1" for p in dc["PADD"]]
        fig_cmp.add_trace(go.Bar(x=dc["inv"], y=dc["PADD"],
                                  orientation="h", marker_color=bar_colors,
                                  name="Inventory"))
        base(fig_cmp, f"Latest Inventory by PADD — {product_name} (mbbls)", "mbbls")
    else:
        fig_cmp = empty_fig("No PADD comparison data")

    # 6 Forecast
    fig_fc = go.Figure()
    if not inv_df.empty:
        hist = inv_df.tail(52)
        fc   = build_forecast(inv_df, fw)
        fig_fc.add_trace(go.Scatter(x=hist["period"], y=hist["value"],
                                     line={"color":accent,"width":2}, name="Historical"))
        if not fc.empty:
            fig_fc.add_trace(go.Scatter(
                x=fc["period"].tolist()+fc["period"].tolist()[::-1],
                y=fc["upper"].tolist()+fc["lower"].tolist()[::-1],
                fill="toself", fillcolor=hex_rgba(accent,0.12),
                line={"width":0}, name="90% CI"))
            fig_fc.add_trace(go.Scatter(x=fc["period"], y=fc["forecast"],
                                         line={"color":"#f97316","width":2,"dash":"dash"},
                                         name="Forecast"))
        base(fig_fc, f"Inventory Forecast — {padd_name} ({fw}w)", "mbbls")
    else:
        fig_fc = empty_fig("No forecast data")

    return kpis, fig_inv, fig_dos, fig_sup, fig_util, fig_cmp, fig_fc


if __name__ == "__main__":
    app.run(debug=True)
