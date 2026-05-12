import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("EIA_API_KEY", "DEMO_KEY")

# ══════════════════════════════════════════════════════════════════════════════
# EIA SERIES DEFINITIONS
# All fetched via v1 API: http://api.eia.gov/series/?series_id=XXX&api_key=KEY
# Inventory uses v2 facets route (confirmed working)
# ══════════════════════════════════════════════════════════════════════════════

V1_BASE = "https://api.eia.gov/series/"
V2_BASE = "https://api.eia.gov/v2"

# Refinery utilization — weekly %
UTIL_SERIES = {
    "US Total":            "PET.WPULEUS3.W",
    "PADD 1 (East Coast)": "PET.WPUR1US3.W",
    "PADD 2 (Midwest)":    "PET.WPUR2US3.W",
    "PADD 3 (Gulf Coast)": "PET.WPUR3US3.W",
    "PADD 4 (Rocky Mtn)":  "PET.WPUR4US3.W",
    "PADD 5 (West Coast)": "PET.WPUR5US3.W",
}

PADD_AREAS = {
    "US Total":            "NUS",
    "PADD 1 (East Coast)": "R10",
    "PADD 2 (Midwest)":    "R20",
    "PADD 3 (Gulf Coast)": "R30",
    "PADD 4 (Rocky Mtn)":  "R40",
    "PADD 5 (West Coast)": "R50",
}

PRODUCTS = {
    "Distillate Fuel Oil": {
        "prod":         "PET.WDIRPUS2.W",   # Refinery & blender net production mbbl/d
        "imp":          "PET.WDIIMUS2.W",   # Imports mbbl/d
        "exp":          "PET.WDIEEUS2.W",   # Exports mbbl/d
        "dem":          "PET.WDISUPUS2.W",  # Product supplied mbbl/d
        "dos":          "PET.WDISDUS2.W",   # Days of supply
        "product_code": "EPD0",
        "color":        "#38bdf8",           # sky blue
    },
    "Jet Fuel (Kerosene)": {
        "prod":         "PET.WKJRPUS2.W",
        "imp":          "PET.WKJIMUS2.W",
        "exp":          "PET.WKJEEUS2.W",
        "dem":          "PET.WKJUPUS2.W",
        "dos":          "PET.WKJSDUS2.W",
        "product_code": "EPJK",
        "color":        "#a78bfa",           # violet
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_v1(series_id: str, length: int = 200) -> pd.DataFrame:
    """Fetch via EIA v1 legacy API — handles PET.XXXX.W format."""
    params = {"series_id": series_id, "api_key": API_KEY, "num": length}
    try:
        r = requests.get(V1_BASE, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        series = data.get("series", [])
        if not series:
            print(f"[WARN] No data for {series_id}: {data.get('data', {})}")
            return pd.DataFrame()
        rows = series[0].get("data", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["period", "value"])
        # EIA v1 weekly dates format: "20240101"
        df["period"] = pd.to_datetime(df["period"], format="%Y%m%d", errors="coerce")
        df["value"]  = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna().sort_values("period").reset_index(drop=True)
    except Exception as e:
        print(f"[WARN] fetch_v1({series_id}): {e}")
        return pd.DataFrame()


def fetch_inventory_padd(area_code: str, product_code: str, length: int = 200) -> pd.DataFrame:
    """Fetch inventory via EIA v2 facets (confirmed working)."""
    url = f"{V2_BASE}/petroleum/stoc/wstk/data/"
    params = {
        "api_key":          API_KEY,
        "data[0]":          "value",
        "facets[product][]": product_code,
        "facets[duoarea][]": area_code,
        "sort[0][column]":  "period",
        "sort[0][direction]": "asc",
        "length":           length,
        "offset":           0,
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
        print(f"[WARN] fetch_inventory_padd({area_code},{product_code}): {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# FORECASTING  — polynomial regression (degree 2) on last 52 weeks
# ══════════════════════════════════════════════════════════════════════════════

def build_forecast(df: pd.DataFrame, weeks_ahead: int = 12) -> pd.DataFrame:
    if df.empty or len(df) < 10:
        return pd.DataFrame()
    recent = df.tail(52).copy()
    recent["t"] = np.arange(len(recent))
    X, y = recent[["t"]].values, recent["value"].values
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


# ══════════════════════════════════════════════════════════════════════════════
# DARK THEME TOKENS
# ══════════════════════════════════════════════════════════════════════════════

BG        = "#0a0e1a"        # page background
SURFACE   = "#0f1629"        # card background
SURFACE2  = "#161d35"        # slightly lighter card
BORDER    = "#1e2d4a"        # card border
TEXT      = "#e2e8f0"        # primary text
MUTED     = "#64748b"        # secondary text
GRID      = "#1a2540"        # chart gridlines
ACCENT    = "#38bdf8"        # default accent (overridden per product)


def hex_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def chart_card(cid):
    return html.Div(
        [dcc.Graph(id=cid, config={"displayModeBar": False})],
        style={
            "flex": "1", "minWidth": "340px",
            "background": SURFACE,
            "borderRadius": "12px",
            "border": f"1px solid {BORDER}",
            "padding": "4px",
        }
    )


def lbl(txt):
    return html.Label(txt, style={
        "fontSize": "0.68rem", "fontWeight": 700,
        "textTransform": "uppercase", "letterSpacing": "0.08em",
        "color": MUTED, "marginBottom": "5px", "display": "block"
    })


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

app = dash.Dash(__name__, title="Disty Dashboard")
server = app.server

app.layout = html.Div([

    # ── Header ────────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Div("◈", style={"fontSize": "1.4rem", "color": ACCENT,
                                  "marginRight": "10px", "lineHeight": "1"}),
            html.Div([
                html.H1("Petroleum Supply Dashboard",
                        style={"margin": 0, "fontSize": "1.1rem", "fontWeight": 700,
                               "color": TEXT, "letterSpacing": "0.02em"}),
                html.P("EIA Weekly · Distillate & Jet Fuel · Live Data",
                       style={"margin": 0, "fontSize": "0.72rem", "color": MUTED}),
            ]),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div(id="last-updated", style={"fontSize": "0.7rem", "color": MUTED}),
    ], style={
        "background": SURFACE,
        "borderBottom": f"1px solid {BORDER}",
        "padding": "14px 28px",
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
    }),

    # ── Controls ──────────────────────────────────────────────────────────────
    html.Div([
        html.Div([lbl("Product"),
                  dcc.Dropdown(id="product-select",
                               options=[{"label": p, "value": p} for p in PRODUCTS],
                               value="Distillate Fuel Oil", clearable=False,
                               style={"fontSize": "0.82rem"})],
                 style={"flex": "1", "minWidth": "210px", "maxWidth": "270px"}),

        html.Div([lbl("PADD Region"),
                  dcc.Dropdown(id="padd-select",
                               options=[{"label": p, "value": p} for p in PADD_AREAS],
                               value="US Total", clearable=False,
                               style={"fontSize": "0.82rem"})],
                 style={"flex": "1", "minWidth": "200px", "maxWidth": "255px"}),

        html.Div([lbl("History (weeks)"),
                  dcc.Slider(id="weeks-slider", min=26, max=260, step=26, value=104,
                             marks={26:"6mo",52:"1yr",104:"2yr",156:"3yr",208:"4yr",260:"5yr"},
                             tooltip={"always_visible": False})],
                 style={"flex": "2", "minWidth": "240px"}),

        html.Div([lbl("Forecast (weeks)"),
                  dcc.Slider(id="forecast-slider", min=4, max=26, step=2, value=12,
                             marks={4:"4w",8:"8w",12:"12w",18:"18w",26:"26w"},
                             tooltip={"always_visible": False})],
                 style={"flex": "1", "minWidth": "190px"}),

        html.Button("↻  Refresh", id="refresh-btn", n_clicks=0,
                    style={
                        "background": "linear-gradient(135deg,#0ea5e9,#6366f1)",
                        "color": "white", "border": "none",
                        "borderRadius": "8px", "padding": "8px 20px",
                        "cursor": "pointer", "fontWeight": 700,
                        "fontSize": "0.82rem", "alignSelf": "flex-end",
                        "height": "36px", "whiteSpace": "nowrap",
                        "boxShadow": "0 0 12px rgba(56,189,248,0.3)",
                    }),
    ], style={
        "display": "flex", "gap": "20px", "padding": "14px 28px",
        "background": SURFACE2,
        "borderBottom": f"1px solid {BORDER}",
        "alignItems": "flex-end", "flexWrap": "wrap",
    }),

    # ── KPI Row ───────────────────────────────────────────────────────────────
    html.Div(id="kpi-row", style={
        "display": "flex", "gap": "12px",
        "padding": "16px 28px", "flexWrap": "wrap",
    }),

    # ── Charts ────────────────────────────────────────────────────────────────
    html.Div([
        html.Div([chart_card("inventory-chart"), chart_card("dos-chart")],
                 style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        html.Div([chart_card("supply-chart"), chart_card("util-chart")],
                 style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginTop": "12px"}),
        html.Div([chart_card("padd-compare-chart"), chart_card("forecast-chart")],
                 style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginTop": "12px"}),
    ], style={"padding": "0 28px 28px 28px"}),

], style={
    "fontFamily": "'Inter','Segoe UI','Helvetica Neue',sans-serif",
    "background": BG,
    "minHeight": "100vh",
})


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACK
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("last-updated",       "children"),
    Output("kpi-row",            "children"),
    Output("inventory-chart",    "figure"),
    Output("dos-chart",          "figure"),
    Output("supply-chart",       "figure"),
    Output("util-chart",         "figure"),
    Output("padd-compare-chart", "figure"),
    Output("forecast-chart",     "figure"),
    Input("refresh-btn",     "n_clicks"),
    Input("product-select",  "value"),
    Input("padd-select",     "value"),
    Input("weeks-slider",    "value"),
    Input("forecast-slider", "value"),
)
def update(n_clicks, product_name, padd_name, weeks, fw):
    from datetime import datetime
    cfg       = PRODUCTS[product_name]
    area_code = PADD_AREAS[padd_name]
    prod_code = cfg["product_code"]
    accent    = cfg["color"]

    # ── Fetch ─────────────────────────────────────────────────────────────────
    inv_df  = fetch_inventory_padd(area_code, prod_code, weeks)
    prod_df = fetch_v1(cfg["prod"],  weeks)
    imp_df  = fetch_v1(cfg["imp"],   weeks)
    exp_df  = fetch_v1(cfg["exp"],   weeks)
    dem_df  = fetch_v1(cfg["dem"],   weeks)
    dos_df  = fetch_v1(cfg["dos"],   weeks)
    util_df = fetch_v1(UTIL_SERIES.get(padd_name, UTIL_SERIES["US Total"]), weeks)

    # Fallback DOS calculation
    if dos_df.empty and not inv_df.empty and not dem_df.empty:
        m = pd.merge(inv_df, dem_df, on="period", suffixes=("_i","_d"))
        m["dos"] = m["value_i"] / m["value_d"]
        dos_df = m[["period","dos"]].rename(columns={"dos":"value"})

    # Supply balance table
    sup = pd.DataFrame()
    if not prod_df.empty:
        sup = prod_df.rename(columns={"value":"production"})
        for col, df in [("imports",imp_df),("exports",exp_df),("demand",dem_df)]:
            if not df.empty:
                sup = pd.merge(sup, df.rename(columns={"value":col}),
                               on="period", how="outer")
        sup = sup.fillna(0).sort_values("period").tail(weeks)

    # ── KPI helpers ───────────────────────────────────────────────────────────
    def lv(df, col="value"):
        s = df[col].dropna() if (not df.empty and col in df.columns) else pd.Series(dtype=float)
        return float(s.iloc[-1]) if len(s) else None

    def yoy_d(df):
        s = df["value"].dropna() if not df.empty else pd.Series(dtype=float)
        return float(s.iloc[-1]) - float(s.iloc[-53]) if len(s) >= 53 else None

    def kpi_card(title, val, unit, delta=None):
        delta_color  = "#34d399" if (delta or 0) >= 0 else "#f87171"
        arrow        = "▲" if (delta or 0) >= 0 else "▼"
        val_display  = f"{val:,.1f}" if val is not None else "—"
        return html.Div([
            html.P(title, style={
                "fontSize": "0.65rem", "fontWeight": 700,
                "textTransform": "uppercase", "letterSpacing": "0.07em",
                "color": MUTED, "margin": "0 0 6px 0",
            }),
            html.P(f"{val_display} {unit}" if val is not None else "—", style={
                "fontSize": "1.55rem", "fontWeight": 800,
                "color": TEXT, "margin": 0, "lineHeight": "1",
            }),
            html.Div([
                html.Span(f"{arrow} {abs(delta or 0):,.0f}",
                          style={"color": delta_color, "fontWeight": 700}),
                html.Span(" vs yr ago", style={"color": MUTED}),
            ], style={"fontSize": "0.7rem", "marginTop": "6px"}) if delta is not None else html.Span(),
            # Accent bottom border
        ], style={
            "background": SURFACE,
            "borderRadius": "10px",
            "border": f"1px solid {BORDER}",
            "borderBottom": f"3px solid {accent}",
            "padding": "14px 18px",
            "minWidth": "148px",
            "flex": "1",
        })

    net_imp = None
    if lv(imp_df) is not None and lv(exp_df) is not None:
        net_imp = lv(imp_df) - lv(exp_df)

    kpis = [
        kpi_card("Inventory",            lv(inv_df),  "mbbls",   yoy_d(inv_df)),
        kpi_card("Days of Supply",        lv(dos_df),  "days"),
        kpi_card("Refinery Production",   lv(prod_df), "mbbl/d"),
        kpi_card("Refinery Utilization",  lv(util_df), "%"),
        kpi_card("Product Supplied",      lv(dem_df),  "mbbl/d"),
        kpi_card("Net Imports",           net_imp,     "mbbl/d"),
    ]

    # ── Chart base style ──────────────────────────────────────────────────────
    def base(fig, title, ytitle=""):
        fig.update_layout(
            title={
                "text": title,
                "font": {"size": 12, "color": TEXT, "family": "Inter"},
                "x": 0.01, "xanchor": "left",
            },
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            font={"color": MUTED, "family": "Inter"},
            margin={"t": 46, "l": 12, "r": 16, "b": 8},
            legend={
                "orientation": "h", "y": -0.2,
                "font": {"size": 10, "color": MUTED},
                "bgcolor": "rgba(0,0,0,0)",
            },
            hovermode="x unified",
            hoverlabel={"bgcolor": "#1e2d4a", "font_color": TEXT,
                        "bordercolor": BORDER},
            xaxis={
                "showgrid": False,
                "color": MUTED,
                "tickfont": {"size": 9},
                "linecolor": BORDER,
                "zerolinecolor": BORDER,
            },
            yaxis={
                "gridcolor": GRID,
                "color": MUTED,
                "tickfont": {"size": 9},
                "title": {"text": ytitle, "font": {"size": 9, "color": MUTED}},
                "linecolor": BORDER,
                "zerolinecolor": BORDER,
            },
        )
        return fig

    def empty_fig(msg="No data available"):
        fig = go.Figure()
        fig.add_annotation(
            text=msg, xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font={"size": 12, "color": MUTED},
        )
        fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                          margin={"t": 40, "l": 10, "r": 10, "b": 10})
        return fig

    # 1 ── Inventory + seasonal band ──────────────────────────────────────────
    fig_inv = go.Figure()
    if not inv_df.empty:
        inv_df["woy"] = inv_df["period"].dt.isocalendar().week.astype(int)
        sea = inv_df.groupby("woy")["value"].agg(["mean","std"]).reset_index()
        md  = pd.merge(inv_df, sea, on="woy")
        fig_inv.add_trace(go.Scatter(
            x=inv_df["period"], y=(md["mean"]+md["std"]).values,
            fill=None, line={"width": 0}, showlegend=False, name="_hi"))
        fig_inv.add_trace(go.Scatter(
            x=inv_df["period"], y=(md["mean"]-md["std"]).values,
            fill="tonexty", fillcolor=hex_rgba(accent, 0.12),
            line={"width": 0}, name="Seasonal ±1σ"))
        fig_inv.add_trace(go.Scatter(
            x=inv_df["period"], y=inv_df["value"],
            line={"color": accent, "width": 2.5},
            name="Inventory (mbbls)"))
        base(fig_inv, f"{product_name} Inventory — {padd_name}", "mbbls")
    else:
        fig_inv = empty_fig(f"No inventory data · {product_name} / {padd_name}")

    # 2 ── Days of Supply ─────────────────────────────────────────────────────
    fig_dos = go.Figure()
    if not dos_df.empty:
        fig_dos.add_trace(go.Scatter(
            x=dos_df["period"], y=dos_df["value"],
            line={"color": "#fb923c", "width": 2.5},
            fill="tozeroy", fillcolor="rgba(251,146,60,0.08)",
            name="Days of Supply"))
        fig_dos.add_hline(y=20, line_dash="dot", line_color="#f87171",
                          annotation_text="20-day floor",
                          annotation_position="bottom right",
                          annotation_font={"color": "#f87171", "size": 9})
        base(fig_dos, f"{product_name} Days of Supply (US)", "days")
    else:
        fig_dos = empty_fig("No days of supply data")

    # 3 ── Supply Balance ─────────────────────────────────────────────────────
    fig_sup = go.Figure()
    if not sup.empty:
        if "production" in sup.columns:
            fig_sup.add_trace(go.Bar(
                x=sup["period"], y=sup["production"],
                name="Production", marker_color="#34d399", opacity=0.85))
        if "imports" in sup.columns:
            fig_sup.add_trace(go.Bar(
                x=sup["period"], y=sup["imports"],
                name="Imports", marker_color="#a78bfa", opacity=0.85))
        if "exports" in sup.columns:
            fig_sup.add_trace(go.Bar(
                x=sup["period"], y=-sup["exports"],
                name="Exports (−)", marker_color="#fbbf24", opacity=0.85))
        if "demand" in sup.columns:
            fig_sup.add_trace(go.Scatter(
                x=sup["period"], y=sup["demand"],
                line={"color": "#f87171", "width": 2, "dash": "dot"},
                name="Product Supplied"))
        fig_sup.update_layout(barmode="relative")
        base(fig_sup, f"{product_name} Supply Balance (US, mbbl/d)", "mbbl/d")
    else:
        fig_sup = empty_fig("No supply balance data")

    # 4 ── Refinery Utilization ───────────────────────────────────────────────
    fig_util = go.Figure()
    if not util_df.empty:
        fig_util.add_trace(go.Scatter(
            x=util_df["period"], y=util_df["value"],
            line={"color": "#818cf8", "width": 2.5},
            fill="tozeroy", fillcolor="rgba(129,140,248,0.08)",
            name="Utilization %"))
        fig_util.add_hline(y=90, line_dash="dot", line_color="#34d399",
                           annotation_text="90% benchmark",
                           annotation_position="top right",
                           annotation_font={"color": "#34d399", "size": 9})
        base(fig_util, f"Refinery Utilization — {padd_name}", "%")
    else:
        fig_util = empty_fig("No refinery utilization data")

    # 5 ── PADD Comparison ────────────────────────────────────────────────────
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
        bar_colors = [accent if p == padd_name else hex_rgba(accent, 0.3) for p in dc["PADD"]]
        fig_cmp.add_trace(go.Bar(
            x=dc["inv"], y=dc["PADD"],
            orientation="h",
            marker={"color": bar_colors, "line": {"width": 0}},
            name="Inventory"))
        base(fig_cmp, f"Latest Inventory by PADD — {product_name} (mbbls)", "mbbls")
    else:
        fig_cmp = empty_fig("No PADD comparison data")

    # 6 ── Forecast ───────────────────────────────────────────────────────────
    fig_fc = go.Figure()
    if not inv_df.empty:
        hist = inv_df.tail(52)
        fc   = build_forecast(inv_df, fw)
        fig_fc.add_trace(go.Scatter(
            x=hist["period"], y=hist["value"],
            line={"color": accent, "width": 2.5},
            name="Historical"))
        if not fc.empty:
            fig_fc.add_trace(go.Scatter(
                x=fc["period"].tolist() + fc["period"].tolist()[::-1],
                y=fc["upper"].tolist() + fc["lower"].tolist()[::-1],
                fill="toself", fillcolor=hex_rgba(accent, 0.12),
                line={"width": 0}, name="90% CI", showlegend=True))
            fig_fc.add_trace(go.Scatter(
                x=fc["period"], y=fc["forecast"],
                line={"color": "#fb923c", "width": 2, "dash": "dash"},
                name="Forecast"))
        base(fig_fc, f"Inventory Forecast — {padd_name} ({fw}w ahead)", "mbbls")
    else:
        fig_fc = empty_fig("No forecast data")

    ts = datetime.now().strftime("Updated %b %d, %Y %I:%M %p")
    return ts, kpis, fig_inv, fig_dos, fig_sup, fig_util, fig_cmp, fig_fc


if __name__ == "__main__":
    app.run(debug=True)
