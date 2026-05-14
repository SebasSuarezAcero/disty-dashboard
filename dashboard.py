import time
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from flask_caching import Cache
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("EIA_API_KEY", "DEMO_KEY")
V2_BASE = "https://api.eia.gov/v2"

PRODUCTS = {
    "Distillate (HO/ULSD)": {
        "product_code": "EPD0",
        "color":        "#38bdf8",
        "prod":  "WDIRPUS2",
        "imp":   "WDIIMUS2",
        "exp":   "WDIEXUS2",
        "dem":   "WDIUPUS2",
    },
    "Jet Fuel (Kerosene)": {
        "product_code": "EPJK",
        "color":        "#a78bfa",
        "prod":  "WKJRPUS2",
        "imp":   "WKJIMUS2",
        "exp":   "WKJEXUS2",
        "dem":   "WKJUPUS2",
    },
}

PADD_AREAS = {
    "US Total":  "NUS",
    "PADD 1":    "R10",
    "PADD 2":    "R20",
    "PADD 3":    "R30",
    "PADD 4":    "R40",
    "PADD 5":    "R50",
}

UTIL_SERIES = {
    "US Total": "WPULEUS3",
    "PADD 1":   "WPUR1US3",
    "PADD 2":   "WPUR2US3",
    "PADD 3":   "WPUR3US3",
    "PADD 4":   "WPUR4US3",
    "PADD 5":   "WPUR5US3",
}

CRUDE_INPUT_SERIES = {
    "US Total": "WCRRIUS2",
    "PADD 1":   "WCRRIR12",
    "PADD 2":   "WCRRIR22",
    "PADD 3":   "WCRRIR32",
    "PADD 4":   "WCRRIR42",
    "PADD 5":   "WCRRIR52",
}

V2_ROUTES = [
    "/petroleum/sum/sndw/data/",
    "/petroleum/cons/wpsup/data/",
    "/petroleum/sum/crdsnd/data/",
    "/petroleum/cons/psup/data/",
    "/petroleum/move/wkly/data/",
    "/petroleum/stoc/wstk/data/",
    "/petroleum/pnp/wiup/data/",
]

app    = dash.Dash(__name__, title="Distillate & Jet Fuel Trader Dashboard")
server = app.server

cache = Cache(server, config={
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 1800
})


@cache.memoize(timeout=1800)
def fetch_series(series_id: str, length: int = 260) -> pd.DataFrame:
    clean = series_id.strip()
    for route in V2_ROUTES:
        url = f"{V2_BASE}{route}"
        params = {
            "api_key":            API_KEY,
            "data[0]":            "value",
            "facets[series][]":   clean,
            "sort[0][column]":    "period",
            "sort[0][direction]": "asc",
            "length":             length,
            "offset":             0,
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(2)
                continue
            if r.status_code == 400:
                continue
            r.raise_for_status()
            rows = r.json().get("response", {}).get("data", [])
            if rows:
                df = pd.DataFrame(rows)
                df["period"] = pd.to_datetime(df["period"], errors="coerce")
                df["value"]  = pd.to_numeric(df["value"], errors="coerce")
                df = df[["period","value"]].dropna().sort_values("period").reset_index(drop=True)
                if not df.empty:
                    return df
        except Exception as e:
            print(f"[WARN] fetch_series({clean}) {route}: {e}")
    print(f"[WARN] No data: {clean}")
    return pd.DataFrame()


@cache.memoize(timeout=1800)
def fetch_inventory(area_code: str, product_code: str, length: int = 260) -> pd.DataFrame:
    url = f"{V2_BASE}/petroleum/stoc/wstk/data/"
    params = {
        "api_key":            API_KEY,
        "data[0]":            "value",
        "facets[product][]":  product_code,
        "facets[duoarea][]":  area_code,
        "sort[0][column]":    "period",
        "sort[0][direction]": "asc",
        "length":             length,
        "offset":             0,
    }
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
        print(f"[WARN] fetch_inventory({area_code},{product_code}): {e}")
        return pd.DataFrame()


def build_forecast(df: pd.DataFrame, weeks_ahead: int = 12) -> pd.DataFrame:
    if df.empty or len(df) < 10:
        return pd.DataFrame()
    recent = df.tail(52).copy()
    recent["t"] = np.arange(len(recent))
    X, y  = recent[["t"]].values, recent["value"].values
    poly  = PolynomialFeatures(degree=2)
    Xp    = poly.fit_transform(X)
    model = LinearRegression().fit(Xp, y)
    std   = np.std(y - model.predict(Xp))
    last  = recent["period"].iloc[-1]
    ft    = np.arange(len(recent), len(recent) + weeks_ahead).reshape(-1, 1)
    fv    = model.predict(poly.transform(ft))
    return pd.DataFrame({
        "period":   [last + timedelta(weeks=i+1) for i in range(weeks_ahead)],
        "forecast": fv,
        "lower":    fv - 1.645 * std,
        "upper":    fv + 1.645 * std,
    })


def seasonal_bands(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["woy"]  = df["period"].dt.isocalendar().week.astype(int)
    df["year"] = df["period"].dt.year
    cutoff     = df["year"].max() - 5
    hist       = df[df["year"] <= cutoff]
    if hist.empty:
        hist = df
    sea = hist.groupby("woy")["value"].agg(["mean","std"]).reset_index()
    sea["std"] = sea["std"].fillna(0)
    return sea


BG       = "#07090f"
SURFACE  = "#0d1117"
SURFACE2 = "#131820"
SURFACE3 = "#1a2235"
BORDER   = "#1e2d4a"
TEXT     = "#e2e8f0"
MUTED    = "#5a6a85"
GRID     = "#131c2e"
GREEN    = "#22c55e"
RED      = "#ef4444"
ORANGE   = "#f97316"
YELLOW   = "#eab308"


def rgba(h, a):
    h = h.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"


def base_layout(title="", ytitle="", ytitle2="", two_axis=False):
    layout = dict(
        title={"text": title, "font": {"size": 11, "color": TEXT, "family": "Inter"},
               "x": 0.01, "xanchor": "left", "y": 0.97},
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font={"color": MUTED, "family": "Inter", "size": 10},
        margin={"t": 38, "l": 8, "r": 8, "b": 8},
        legend={"orientation": "h", "y": -0.18, "x": 0,
                "font": {"size": 9, "color": MUTED},
                "bgcolor": "rgba(0,0,0,0)"},
        hovermode="x unified",
        hoverlabel={"bgcolor": SURFACE3, "font_color": TEXT,
                    "bordercolor": BORDER, "font_size": 11},
        xaxis={"showgrid": False, "color": MUTED, "tickfont": {"size": 9},
               "linecolor": BORDER, "zerolinecolor": BORDER},
        yaxis={"gridcolor": GRID, "color": MUTED, "tickfont": {"size": 9},
               "title": {"text": ytitle, "font": {"size": 9}},
               "linecolor": BORDER, "zerolinecolor": BORDER},
    )
    if two_axis:
        layout["yaxis2"] = {
            "gridcolor": GRID, "color": MUTED, "tickfont": {"size": 9},
            "title": {"text": ytitle2, "font": {"size": 9}},
            "overlaying": "y", "side": "right", "showgrid": False,
        }
    return layout


def empty_fig(msg="No data"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font={"size": 11, "color": MUTED})
    fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      margin={"t": 32, "l": 8, "r": 8, "b": 8})
    return fig


def card(children, flex="1", min_w="320px"):
    return html.Div(children, style={
        "flex": flex, "minWidth": min_w,
        "background": SURFACE, "borderRadius": "10px",
        "border": f"1px solid {BORDER}", "padding": "4px",
        "boxSizing": "border-box",
    })


def kpi(title, val, unit, delta=None, invert=False):
    if val is None:
        return html.Div([
            html.P(title, style={"fontSize":"0.62rem","fontWeight":700,
                                  "textTransform":"uppercase","letterSpacing":"0.07em",
                                  "color":MUTED,"margin":"0 0 4px 0"}),
            html.P("—", style={"fontSize":"1.4rem","fontWeight":800,"color":TEXT,"margin":0}),
        ], style={"background":SURFACE,"borderRadius":"8px",
                  "border":f"1px solid {BORDER}","padding":"12px 16px",
                  "flex":"1","minWidth":"120px"})
    dcolor = GREEN
    arrow  = "▲"
    if delta is not None:
        up     = delta >= 0
        dcolor = (RED if up else GREEN) if invert else (GREEN if up else RED)
        arrow  = "▲" if up else "▼"
    return html.Div([
        html.P(title, style={"fontSize":"0.62rem","fontWeight":700,
                              "textTransform":"uppercase","letterSpacing":"0.07em",
                              "color":MUTED,"margin":"0 0 4px 0"}),
        html.P(f"{val:,.1f} {unit}", style={"fontSize":"1.4rem","fontWeight":800,
                                              "color":TEXT,"margin":0,"lineHeight":"1.1"}),
        html.Div([
            html.Span(f"{arrow} {abs(delta):,.1f}",
                      style={"color":dcolor,"fontWeight":700,"fontSize":"0.72rem"}),
            html.Span(" YoY", style={"color":MUTED,"fontSize":"0.68rem"}),
        ], style={"marginTop":"4px"}) if delta is not None else html.Span(),
    ], style={"background":SURFACE,"borderRadius":"8px",
              "border":f"1px solid {BORDER}",
              "borderBottom":f"3px solid {dcolor if delta is not None else BORDER}",
              "padding":"12px 16px","flex":"1","minWidth":"130px"})


def lbl(t):
    return html.Label(t, style={"fontSize":"0.65rem","fontWeight":700,
                                 "textTransform":"uppercase","letterSpacing":"0.08em",
                                 "color":MUTED,"marginBottom":"4px","display":"block"})


app.layout = html.Div([

    html.Div([
        html.Div([
            html.Span("▣ ", style={"color":"#38bdf8","fontSize":"1.2rem"}),
            html.Span("Distillate & Jet Fuel  ",
                      style={"fontWeight":800,"fontSize":"1rem","color":TEXT}),
            html.Span("Trader Dashboard",
                      style={"fontWeight":300,"fontSize":"1rem","color":MUTED}),
        ], style={"display":"flex","alignItems":"center","gap":"4px"}),
        html.Div([
            html.Span("EIA Weekly  ·  ", style={"color":MUTED,"fontSize":"0.72rem"}),
            html.Span(id="last-updated", style={"color":MUTED,"fontSize":"0.72rem"}),
        ]),
    ], style={"background":SURFACE,"borderBottom":f"1px solid {BORDER}",
              "padding":"12px 24px","display":"flex",
              "justifyContent":"space-between","alignItems":"center"}),

    html.Div([
        html.Div([lbl("Product"),
                  dcc.Dropdown(id="product-dd",
                               options=[{"label":p,"value":p} for p in PRODUCTS],
                               value="Distillate (HO/ULSD)", clearable=False,
                               style={"fontSize":"0.82rem"})],
                 style={"flex":"1","minWidth":"200px","maxWidth":"260px"}),

        html.Div([lbl("PADD Region"),
                  dcc.Dropdown(id="padd-dd",
                               options=[{"label":p,"value":p} for p in PADD_AREAS],
                               value="US Total", clearable=False,
                               style={"fontSize":"0.82rem"})],
                 style={"flex":"1","minWidth":"170px","maxWidth":"220px"}),

        html.Div([lbl("History Window"),
                  dcc.Slider(id="weeks-sl", min=52, max=260, step=26, value=156,
                             marks={52:"1yr",104:"2yr",156:"3yr",208:"4yr",260:"5yr"},
                             tooltip={"always_visible":False})],
                 style={"flex":"2","minWidth":"220px"}),

        html.Div([lbl("Forecast"),
                  dcc.Slider(id="fc-sl", min=4, max=26, step=2, value=12,
                             marks={4:"4w",12:"12w",26:"26w"},
                             tooltip={"always_visible":False})],
                 style={"flex":"1","minWidth":"160px"}),

        html.Button("↻ Refresh", id="refresh-btn", n_clicks=0, style={
            "background":"linear-gradient(135deg,#0ea5e9,#6366f1)",
            "color":"white","border":"none","borderRadius":"7px",
            "padding":"8px 18px","cursor":"pointer","fontWeight":700,
            "fontSize":"0.8rem","alignSelf":"flex-end","height":"34px",
            "whiteSpace":"nowrap",
        }),
    ], style={"display":"flex","gap":"16px","padding":"12px 24px",
              "background":SURFACE2,"borderBottom":f"1px solid {BORDER}",
              "alignItems":"flex-end","flexWrap":"wrap"}),

    html.Div(id="kpi-row", style={"display":"flex","gap":"10px",
                                   "padding":"14px 24px","flexWrap":"wrap"}),

    html.Div([
        card([dcc.Graph(id="inv-chart",   config={"displayModeBar":False})], flex="3", min_w="340px"),
        card([dcc.Graph(id="util-chart",  config={"displayModeBar":False})], flex="2", min_w="300px"),
    ], style={"display":"flex","gap":"10px","padding":"0 24px","flexWrap":"wrap"}),

    html.Div(style={"height":"10px"}),

    html.Div([
        card([dcc.Graph(id="balance-chart", config={"displayModeBar":False})], flex="3", min_w="340px"),
        card([dcc.Graph(id="padd-chart",    config={"displayModeBar":False})], flex="2", min_w="300px"),
    ], style={"display":"flex","gap":"10px","padding":"0 24px","flexWrap":"wrap"}),

    html.Div(style={"height":"10px"}),

    html.Div([
        card([dcc.Graph(id="flow-chart",     config={"displayModeBar":False})], flex="1", min_w="320px"),
        card([dcc.Graph(id="forecast-chart", config={"displayModeBar":False})], flex="1", min_w="320px"),
    ], style={"display":"flex","gap":"10px","padding":"0 24px","flexWrap":"wrap"}),

    html.Div(style={"height":"24px"}),

], style={"fontFamily":"'Inter','Segoe UI',sans-serif",
          "background":BG,"minHeight":"100vh"})


@app.callback(
    Output("last-updated",   "children"),
    Output("kpi-row",        "children"),
    Output("inv-chart",      "figure"),
    Output("util-chart",     "figure"),
    Output("balance-chart",  "figure"),
    Output("padd-chart",     "figure"),
    Output("flow-chart",     "figure"),
    Output("forecast-chart", "figure"),
    Input("refresh-btn",  "n_clicks"),
    Input("product-dd",   "value"),
    Input("padd-dd",      "value"),
    Input("weeks-sl",     "value"),
    Input("fc-sl",        "value"),
)
def update(n_clicks, product_name, padd_name, weeks, fw):
    from datetime import datetime
    from functools import reduce

    if n_clicks and n_clicks > 0:
        cache.clear()

    cfg       = PRODUCTS[product_name]
    area_code = PADD_AREAS[padd_name]
    prod_code = cfg["product_code"]
    accent    = cfg["color"]

    inv_df   = fetch_inventory(area_code, prod_code, weeks)
    prod_df  = fetch_series(cfg["prod"], weeks)
    imp_df   = fetch_series(cfg["imp"],  weeks)
    exp_df   = fetch_series(cfg["exp"],  weeks)
    dem_df   = fetch_series(cfg["dem"],  weeks)
    util_df  = fetch_series(UTIL_SERIES.get(padd_name, UTIL_SERIES["US Total"]), weeks)
    crude_df = fetch_series(CRUDE_INPUT_SERIES.get(padd_name, CRUDE_INPUT_SERIES["US Total"]), weeks)

    def lv(df):
        return float(df["value"].dropna().iloc[-1]) if not df.empty else None

    def yoy(df):
        s = df["value"].dropna() if not df.empty else pd.Series(dtype=float)
        return float(s.iloc[-1]) - float(s.iloc[-53]) if len(s) >= 53 else None

    net_imp = None
    if lv(imp_df) is not None and lv(exp_df) is not None:
        net_imp = lv(imp_df) - lv(exp_df)

    kpis = html.Div([
        kpi("Inventory",           lv(inv_df),   "mbbls",  yoy(inv_df)),
        kpi("Refinery Production", lv(prod_df),  "mbbl/d", yoy(prod_df)),
        kpi("Product Demand",      lv(dem_df),   "mbbl/d", yoy(dem_df)),
        kpi("Imports",             lv(imp_df),   "mbbl/d", yoy(imp_df)),
        kpi("Exports",             lv(exp_df),   "mbbl/d", yoy(exp_df), invert=True),
        kpi("Net Imports",         net_imp,      "mbbl/d"),
        kpi("Refinery Util",       lv(util_df),  "%",      yoy(util_df)),
        kpi("Crude Inputs",        lv(crude_df), "mbbl/d", yoy(crude_df)),
    ], style={"display":"flex","gap":"10px","flexWrap":"wrap","width":"100%"})

    # 1 — Inventory with seasonal bands
    fig_inv = go.Figure()
    if not inv_df.empty:
        tail = inv_df.tail(weeks)
        sea  = seasonal_bands(inv_df)
        if not sea.empty:
            tail2      = tail.copy()
            tail2["woy"] = tail2["period"].dt.isocalendar().week.astype(int)
            m = pd.merge(tail2, sea, on="woy", how="left")
            fig_inv.add_trace(go.Scatter(
                x=tail["period"], y=(m["mean"]+m["std"]).values,
                fill=None, line={"width":0}, showlegend=False,
                name="_hi", hoverinfo="skip"))
            fig_inv.add_trace(go.Scatter(
                x=tail["period"], y=(m["mean"]-m["std"]).values,
                fill="tonexty", fillcolor=rgba(accent, 0.10),
                line={"width":0}, name="5yr Seasonal ±1σ"))
            fig_inv.add_trace(go.Scatter(
                x=tail["period"], y=m["mean"].values,
                line={"color":rgba(accent,0.4),"width":1,"dash":"dot"},
                name="5yr Avg"))
        fig_inv.add_trace(go.Scatter(
            x=tail["period"], y=tail["value"],
            line={"color":accent,"width":2.2}, name="Inventory"))
        fig_inv.update_layout(**base_layout(
            f"{product_name} Inventory — {padd_name}", "mbbls"))
    else:
        fig_inv = empty_fig("No inventory data")

    # 2 — Refinery Utilization + Crude Inputs
    fig_util = go.Figure()
    if not util_df.empty or not crude_df.empty:
        if not util_df.empty:
            ut = util_df.tail(weeks)
            fig_util.add_trace(go.Scatter(
                x=ut["period"], y=ut["value"],
                line={"color":"#818cf8","width":2.2},
                name="Utilization %", yaxis="y"))
            fig_util.add_hline(y=90, line_dash="dot",
                               line_color=rgba(GREEN, 0.5),
                               annotation_text="90%",
                               annotation_font={"color":GREEN,"size":9})
        if not crude_df.empty:
            cr = crude_df.tail(weeks)
            fig_util.add_trace(go.Scatter(
                x=cr["period"], y=cr["value"],
                line={"color":ORANGE,"width":1.8},
                name="Crude Inputs (mbbl/d)", yaxis="y2"))
        fig_util.update_layout(**base_layout(
            f"Refinery Ops — {padd_name}", "%", "mbbl/d", two_axis=True))
    else:
        fig_util = empty_fig("No refinery data")

    # 3 — Supply Balance
    fig_bal = go.Figure()
    frames = []
    if not prod_df.empty: frames.append(prod_df.rename(columns={"value":"prod"}))
    if not imp_df.empty:  frames.append(imp_df.rename(columns={"value":"imp"}))
    if not exp_df.empty:  frames.append(exp_df.rename(columns={"value":"exp"}))
    if not dem_df.empty:  frames.append(dem_df.rename(columns={"value":"dem"}))
    if frames:
        bal = reduce(lambda a,b: pd.merge(a,b,on="period",how="outer"), frames)
        bal = bal.fillna(0).sort_values("period").tail(weeks)
        if "prod" in bal.columns:
            fig_bal.add_trace(go.Bar(x=bal["period"], y=bal["prod"],
                name="Production", marker_color=GREEN, opacity=0.85))
        if "imp" in bal.columns:
            fig_bal.add_trace(go.Bar(x=bal["period"], y=bal["imp"],
                name="Imports", marker_color="#38bdf8", opacity=0.85))
        if "exp" in bal.columns:
            fig_bal.add_trace(go.Bar(x=bal["period"], y=-bal["exp"],
                name="Exports (−)", marker_color=YELLOW, opacity=0.85))
        if "dem" in bal.columns:
            fig_bal.add_trace(go.Scatter(x=bal["period"], y=bal["dem"],
                line={"color":RED,"width":2,"dash":"dot"},
                name="Demand / Product Supplied"))
        fig_bal.update_layout(barmode="relative",
            **base_layout(f"{product_name} Supply Balance (US, mbbl/d)", "mbbl/d"))
    else:
        fig_bal = empty_fig("No supply balance data")

    # 4 — PADD Inventory Comparison
    fig_padd = go.Figure()
    rows = []
    for pname, pcode in PADD_AREAS.items():
        if pname == "US Total":
            continue
        df_p = fetch_inventory(pcode, prod_code, 4)
        if not df_p.empty:
            rows.append({"PADD": pname, "inv": float(df_p["value"].iloc[-1])})
    if rows:
        dc     = pd.DataFrame(rows).sort_values("inv", ascending=True)
        colors = [accent if p == padd_name else rgba(accent, 0.28) for p in dc["PADD"]]
        fig_padd.add_trace(go.Bar(
            x=dc["inv"], y=dc["PADD"], orientation="h",
            marker={"color":colors,"line":{"width":0}},
            text=[f"{v:,.0f}" for v in dc["inv"]],
            textposition="outside",
            textfont={"size":9,"color":TEXT},
            name="Inventory"))
        fig_padd.update_layout(
            **base_layout(f"Inventory by PADD — {product_name} (mbbls)"))
        fig_padd.update_xaxes(showticklabels=False)
    else:
        fig_padd = empty_fig("No PADD comparison data")

    # 5 — Trade Flows
    fig_flow = go.Figure()
    if not imp_df.empty or not exp_df.empty:
        if not imp_df.empty:
            im = imp_df.tail(weeks)
            fig_flow.add_trace(go.Scatter(
                x=im["period"], y=im["value"],
                line={"color":"#38bdf8","width":2},
                fill="tozeroy", fillcolor=rgba("#38bdf8",0.07),
                name="Imports"))
        if not exp_df.empty:
            ex = exp_df.tail(weeks)
            fig_flow.add_trace(go.Scatter(
                x=ex["period"], y=ex["value"],
                line={"color":YELLOW,"width":2},
                fill="tozeroy", fillcolor=rgba(YELLOW,0.07),
                name="Exports"))
        if not imp_df.empty and not exp_df.empty:
            net = pd.merge(imp_df.tail(weeks), exp_df.tail(weeks),
                           on="period", suffixes=("_i","_e"), how="inner")
            net["net"] = net["value_i"] - net["value_e"]
            fig_flow.add_trace(go.Scatter(
                x=net["period"], y=net["net"],
                line={"color":"#a78bfa","width":1.5,"dash":"dot"},
                name="Net Imports"))
        fig_flow.update_layout(
            **base_layout(f"{product_name} Trade Flows (US, mbbl/d)", "mbbl/d"))
    else:
        fig_flow = empty_fig("No trade flow data")

    # 6 — Inventory Forecast
    fig_fc = go.Figure()
    if not inv_df.empty:
        hist = inv_df.tail(52)
        fc   = build_forecast(inv_df, fw)
        fig_fc.add_trace(go.Scatter(
            x=hist["period"], y=hist["value"],
            line={"color":accent,"width":2.2}, name="Historical"))
        if not fc.empty:
            fig_fc.add_trace(go.Scatter(
                x=fc["period"].tolist() + fc["period"].tolist()[::-1],
                y=fc["upper"].tolist() + fc["lower"].tolist()[::-1],
                fill="toself", fillcolor=rgba(accent,0.10),
                line={"width":0}, name="90% CI"))
            fig_fc.add_trace(go.Scatter(
                x=fc["period"], y=fc["forecast"],
                line={"color":ORANGE,"width":2,"dash":"dash"},
                name=f"{fw}w Forecast"))
        fig_fc.update_layout(
            **base_layout(f"Inventory Forecast — {padd_name} ({fw}w)", "mbbls"))
    else:
        fig_fc = empty_fig("No forecast data")

    ts = datetime.now().strftime("Updated %b %d %Y  %I:%M %p")
    return ts, kpis, fig_inv, fig_util, fig_bal, fig_padd, fig_flow, fig_fc


if __name__ == "__main__":
    app.run(debug=True)