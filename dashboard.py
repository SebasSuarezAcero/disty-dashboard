import os
import requests
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, callback
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("EIA_API_KEY", "")

PADD_MAP = {
    "R10": "PADD 1 - East Coast",
    "R20": "PADD 2 - Midwest",
    "R30": "PADD 3 - Gulf Coast",
    "R40": "PADD 4 - Rocky Mountain",
    "R50": "PADD 5 - West Coast",
}
PADD_COLORS = {
    "R10": "#4e79a7",
    "R20": "#f28e2b",
    "R30": "#59a14f",
    "R40": "#e15759",
    "R50": "#af7aa1",
}
PRODUCTS = {
    "Distillate Fuel Oil": {
        "product_code": "EPD0",
        "pfx_short": "dist",
        "inventory_label": "Distillate Stocks (Mbbl)",
        "price_label": "U.S. Retail Diesel ($/gal)",
        "color": "#4e79a7",
    },
    "Kerosene-Jet Fuel": {
        "product_code": "EPJK",
        "pfx_short": "jet",
        "inventory_label": "Kerosene-Jet Stocks (Mbbl)",
        "price_label": "U.S. Retail Diesel ($/gal)",
        "color": "#f28e2b",
    },
}
DIESEL_SERIES = "EMD_EPD2D_PTE_NUS_DPG"


# ── Data fetching ──────────────────────────────────────────────────────────────

def eia_get(url, params, timeout=60):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", {}).get("data", [])


def fetch_national(product_code, pfx_short):
    rows = eia_get(
        "https://api.eia.gov/v2/petroleum/stoc/wstk/data/",
        {"api_key": API_KEY, "frequency": "weekly", "data[0]": "value",
         "facets[product][]": product_code, "facets[duoarea][]": "NUS",
         "sort[0][column]": "period", "sort[0][direction]": "desc",
         "offset": 0, "length": 520},
    )
    col = pfx_short + "_inv"
    d = pd.DataFrame(rows).rename(columns={"period": "date", "value": col})
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d["date"] = pd.to_datetime(d["date"])
    return d[["date", col]].sort_values("date").reset_index(drop=True)


def fetch_padd(product_code, padd_code, pfx_short):
    col = pfx_short + "_" + padd_code
    rows = eia_get(
        "https://api.eia.gov/v2/petroleum/stoc/wstk/data/",
        {"api_key": API_KEY, "frequency": "weekly", "data[0]": "value",
         "facets[product][]": product_code, "facets[duoarea][]": padd_code,
         "sort[0][column]": "period", "sort[0][direction]": "desc",
         "offset": 0, "length": 520},
    )
    if not rows:
        return pd.DataFrame(columns=["date", col])
    d = pd.DataFrame(rows).rename(columns={"period": "date", "value": col})
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d["date"] = pd.to_datetime(d["date"])
    return d[["date", col]].sort_values("date").reset_index(drop=True)


def fetch_price():
    rows = eia_get(
        "https://api.eia.gov/v2/petroleum/pri/gnd/data/",
        {"api_key": API_KEY, "frequency": "weekly", "data[0]": "value",
         "facets[series][]": DIESEL_SERIES,
         "sort[0][column]": "period", "sort[0][direction]": "desc",
         "offset": 0, "length": 1500},
    )
    if not rows:
        return pd.DataFrame(columns=["date", "price_gal"])
    d = pd.DataFrame(rows)
    date_col = next((c for c in d.columns if "period" in c.lower()), d.columns[0])
    val_col  = next((c for c in d.columns if c.lower() == "value"), None)
    if val_col is None:
        num_cols = d.select_dtypes(include="number").columns.tolist()
        val_col  = num_cols[-1] if num_cols else d.columns[-1]
    d = d[[date_col, val_col]].rename(columns={date_col: "date", val_col: "price_gal"})
    d["price_gal"] = pd.to_numeric(d["price_gal"], errors="coerce")
    d["date"] = pd.to_datetime(d["date"])
    return d.dropna(subset=["price_gal"]).sort_values("date").reset_index(drop=True)


def build_data():
    print("Fetching EIA data...")
    dist = fetch_national("EPD0", "dist")
    jet  = fetch_national("EPJK", "jet")
    px   = fetch_price()
    df   = dist.merge(jet, on="date", how="outer")
    df   = df.merge(px, on="date", how="left") if not px.empty else df.assign(price_gal=np.nan)
    for padd in PADD_MAP:
        df = df.merge(fetch_padd("EPD0", padd, "dist"), on="date", how="left")
        df = df.merge(fetch_padd("EPJK", padd, "jet"),  on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    for col, pfx in [("dist_inv", "dist_nat"), ("jet_inv", "jet_nat")]:
        m = df[col].rolling(156, min_periods=52).mean()
        s = df[col].rolling(156, min_periods=52).std()
        df[pfx+"_mean"]=m; df[pfx+"_std"]=s
        df[pfx+"_u2"]=m+2*s; df[pfx+"_l2"]=m-2*s
        df[pfx+"_u1"]=m+s;   df[pfx+"_l1"]=m-s
        df[pfx+"_z"]=(df[col]-m)/s
        cond=[df[pfx+"_z"]>2, df[pfx+"_z"]>1, df[pfx+"_z"]<-2, df[pfx+"_z"]<-1]
        df[pfx+"_sig"]=np.select(cond,["Bearish Outlier","Elevated","Bullish Outlier","Tight"],default="Neutral")

    for padd in PADD_MAP:
        for ps in ["dist","jet"]:
            col=ps+"_"+padd
            if col in df.columns and df[col].notna().sum()>52:
                m=df[col].rolling(156,min_periods=52).mean()
                s=df[col].rolling(156,min_periods=52).std()
                df[col+"_mean"]=m; df[col+"_std"]=s
                df[col+"_u2"]=m+2*s; df[col+"_l2"]=m-2*s
                df[col+"_u1"]=m+s;   df[col+"_l1"]=m-s
                df[col+"_z"]=(df[col]-m)/s

    df["week_of_year"]=df["date"].dt.isocalendar().week.astype(int)
    if "price_gal" in df.columns:
        df["dist_corr"]=df["dist_inv"].rolling(26).corr(df["price_gal"])
        df["jet_corr"] =df["jet_inv"].rolling(26).corr(df["price_gal"])
    else:
        df["dist_corr"]=np.nan; df["jet_corr"]=np.nan
    print("Data ready.")
    return df


def forecast_series(df, col, periods=12):
    tmp=df[["date",col]].dropna().copy()
    if len(tmp)<104: return pd.DataFrame()
    tmp["t"]=np.arange(len(tmp))
    tmp["wk"]=tmp["date"].dt.isocalendar().week.astype(int)
    sea=tmp.groupby("wk")[col].mean()-tmp[col].mean()
    m=LinearRegression().fit(tmp[["t"]],tmp[col])
    resid=tmp[col]-m.predict(tmp[["t"]])-tmp["wk"].map(sea).fillna(0)
    sigma=float(np.nanstd(resid))
    lt=int(tmp["t"].max()); ld=tmp["date"].iloc[-1]
    fd=pd.date_range(ld+pd.Timedelta(days=7),periods=periods,freq="W-FRI")
    ft=np.arange(lt+1,lt+1+periods)
    fwk=fd.isocalendar().week.astype(int)
    fc=m.predict(ft.reshape(-1,1))+np.array([sea.get(w,0) for w in fwk])
    return pd.DataFrame({"date":fd,"f":fc,"upper":fc+1.96*sigma,"lower":fc-1.96*sigma})


# ── Chart builders (return figures, not render) ────────────────────────────────

def make_sd_band_fig(df, val_col, pfx, title, color):
    mc=pfx+"_mean"
    if mc not in df.columns or df.dropna(subset=[val_col,mc]).empty:
        return go.Figure().update_layout(title=title+" (insufficient data)",template="plotly_dark")
    d=df.dropna(subset=[val_col,mc])
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=pd.concat([d["date"],d["date"][::-1]]),
        y=pd.concat([d[pfx+"_u2"],d[pfx+"_l2"][::-1]]),
        fill="toself",fillcolor="rgba(100,150,255,0.10)",
        line=dict(color="rgba(0,0,0,0)"),name="+/-2sd",hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=pd.concat([d["date"],d["date"][::-1]]),
        y=pd.concat([d[pfx+"_u1"],d[pfx+"_l1"][::-1]]),
        fill="toself",fillcolor="rgba(100,150,255,0.18)",
        line=dict(color="rgba(0,0,0,0)"),name="+/-1sd",hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["date"],y=d[mc],
        line=dict(color="white",width=2,dash="dot"),name="3-yr mean",
        hovertemplate="Mean: %{y:,.0f} Mbbl<extra></extra>"))
    fig.add_trace(go.Bar(x=d["date"],y=d[val_col],
        marker_color=color,opacity=0.8,name="Inventory",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Inventory: %{y:,.0f} Mbbl<extra></extra>"))
    fig.update_layout(title=title,template="plotly_dark",height=450,
        barmode="overlay",hovermode="x unified",
        xaxis_title="Week",yaxis_title="Inventory (Mbbl)",
        legend=dict(orientation="h",yanchor="bottom",y=1.02))
    return fig


def make_combo_fig(df, inv_col, price_col, title, inv_label, price_label, color):
    if price_col not in df.columns or df[price_col].notna().sum()==0:
        fig=go.Figure()
        fig.add_trace(go.Bar(x=df["date"],y=df[inv_col],marker_color=color,opacity=0.8,name=inv_label))
        fig.update_layout(title=title,template="plotly_dark",height=430,
            xaxis_title="Week",yaxis_title="Inventory (Mbbl)")
        return fig
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=df["date"],y=df[inv_col],name=inv_label,
        marker_color=color,opacity=0.75,
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Inv: %{y:,.0f} Mbbl<extra></extra>"),secondary_y=False)
    fig.add_trace(go.Scatter(x=df["date"],y=df[price_col],name=price_label,
        line=dict(color="orangered",width=2),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Price: $%{y:.3f}/gal<extra></extra>"),secondary_y=True)
    fig.update_layout(title=title,template="plotly_dark",height=430,
        hovermode="x unified",legend=dict(orientation="h",yanchor="bottom",y=1.02))
    fig.update_xaxes(title_text="Week")
    fig.update_yaxes(title_text="Inventory (Mbbl)",secondary_y=False)
    fig.update_yaxes(title_text="Price ($/gal)",secondary_y=True)
    return fig


def make_scatter_fig(df, inv_col, price_col, title):
    if price_col not in df.columns: return go.Figure()
    d=df[[inv_col,price_col,"date"]].dropna()
    if d.empty: return go.Figure()
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=d[inv_col],y=d[price_col],mode="markers",
        marker=dict(size=7,color=d.index,colorscale="Viridis"),
        text=d["date"].dt.strftime("%Y-%m-%d"),
        hovertemplate="Date: %{text}<br>Inv: %{x:,.0f}<br>Price: $%{y:.3f}<extra></extra>"))
    tmp=d[[inv_col,price_col]].dropna()
    if len(tmp)>=12:
        reg=stats.linregress(tmp[inv_col],tmp[price_col])
        xl=np.linspace(d[inv_col].min(),d[inv_col].max(),100)
        fig.add_trace(go.Scatter(x=xl,y=reg.intercept+reg.slope*xl,
            mode="lines",line=dict(color="white",dash="dash"),
            name="R2="+str(round(reg.rvalue**2,2)),hoverinfo="skip"))
    fig.update_layout(title=title,template="plotly_dark",height=400,
        xaxis_title="Inventory (Mbbl)",yaxis_title="Diesel Price ($/gal)")
    return fig


def make_corr_fig(df, corr_col, title):
    d=df.dropna(subset=[corr_col])
    if d.empty: return go.Figure()
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=d["date"],y=d[corr_col],
        line=dict(color="cyan",width=2),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Corr: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0,line_dash="dash",line_color="white")
    fig.update_layout(title=title,template="plotly_dark",height=320,
        xaxis_title="Week",yaxis_title="26-week Rolling Correlation")
    return fig


def make_padd_stack_fig(df, pfx_short, title):
    fig=go.Figure()
    for padd,name in PADD_MAP.items():
        col=pfx_short+"_"+padd
        if col in df.columns:
            fig.add_trace(go.Bar(x=df["date"],y=df[col],name=name,
                marker_color=PADD_COLORS[padd],
                hovertemplate=name+"<br>%{x|%Y-%m-%d}: %{y:,.0f} Mbbl<extra></extra>"))
    fig.update_layout(title=title,template="plotly_dark",height=430,
        barmode="stack",hovermode="x unified",
        xaxis_title="Week",yaxis_title="Inventory (Mbbl)",
        legend=dict(orientation="h",yanchor="bottom",y=1.02))
    return fig


def make_forecast_fig(df, col, fdf, title, y_label, color):
    hist=df[["date",col]].dropna()
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"],y=hist[col],
        line=dict(color=color,width=2),name="History",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Value: %{y:,.3f}<extra></extra>"))
    if not fdf.empty:
        fig.add_trace(go.Scatter(x=fdf["date"],y=fdf["f"],
            line=dict(color="white",width=2,dash="dash"),name="12-week forecast",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Forecast: %{y:,.3f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=pd.concat([fdf["date"],fdf["date"][::-1]]),
            y=pd.concat([fdf["upper"],fdf["lower"][::-1]]),
            fill="toself",fillcolor="rgba(255,255,255,0.08)",
            line=dict(color="rgba(0,0,0,0)"),name="95% CI",hoverinfo="skip"))
    fig.update_layout(title=title,template="plotly_dark",height=400,
        xaxis_title="Week",yaxis_title=y_label)
    return fig


# ── Load data once at startup ──────────────────────────────────────────────────
DF = build_data()

# ── Dash app ───────────────────────────────────────────────────────────────────
app = Dash(__name__, title="U.S. Distillates Dashboard")
server = app.server

CARD = {"background":"#1e1e2e","borderRadius":"8px","padding":"20px","textAlign":"center"}
LABEL_STYLE = {"color":"#888","fontSize":"13px","marginBottom":"4px"}
VALUE_STYLE = {"color":"#fff","fontSize":"22px","fontWeight":"600"}
DELTA_STYLE_POS = {"color":"#59a14f","fontSize":"13px"}
DELTA_STYLE_NEG = {"color":"#e15759","fontSize":"13px"}


def metric_card(label, value, delta=None, delta_val=None):
    delta_style = DELTA_STYLE_POS if (delta_val or 0) >= 0 else DELTA_STYLE_NEG
    return html.Div([
        html.Div(label, style=LABEL_STYLE),
        html.Div(value, style=VALUE_STYLE),
        html.Div(delta or "", style=delta_style),
    ], style=CARD)


app.layout = html.Div(style={"backgroundColor":"#0e0e1a","minHeight":"100vh",
                               "fontFamily":"Inter, sans-serif","color":"#ccc","padding":"24px"}, children=[
    html.H1("U.S. Distillates Market Dashboard",
            style={"color":"#fff","fontSize":"24px","marginBottom":"4px"}),
    html.P("EIA weekly inventory | U.S. retail diesel price | PADD SD bands | Forecasting",
           style={"color":"#888","fontSize":"13px","marginBottom":"24px"}),

    # Product toggle
    html.Div([
        html.Label("Product:", style={"color":"#aaa","marginRight":"12px","fontSize":"14px"}),
        dcc.RadioItems(
            id="product-radio",
            options=[{"label":"Distillate Fuel Oil","value":"Distillate Fuel Oil"},
                     {"label":"Kerosene-Jet Fuel",  "value":"Kerosene-Jet Fuel"}],
            value="Distillate Fuel Oil",
            inline=True,
            labelStyle={"marginRight":"20px","color":"#ccc","fontSize":"14px"},
        )
    ], style={"marginBottom":"20px"}),

    # KPI cards
    html.Div(id="kpi-row", style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)",
                                   "gap":"16px","marginBottom":"24px"}),

    html.Hr(style={"borderColor":"#333","marginBottom":"24px"}),

    # Tabs
    dcc.Tabs(id="tabs", value="tab-national", style={"marginBottom":"16px"},
             colors={"border":"#333","primary":"#4e79a7","background":"#1e1e2e"},
             children=[
        dcc.Tab(label="National SD Bands", value="tab-national",
                style={"color":"#aaa","backgroundColor":"#1e1e2e"},
                selected_style={"color":"#fff","backgroundColor":"#2e2e3e","borderTop":"2px solid #4e79a7"}),
        dcc.Tab(label="PADD Filter",       value="tab-padd",
                style={"color":"#aaa","backgroundColor":"#1e1e2e"},
                selected_style={"color":"#fff","backgroundColor":"#2e2e3e","borderTop":"2px solid #4e79a7"}),
        dcc.Tab(label="Price Comparison",  value="tab-price",
                style={"color":"#aaa","backgroundColor":"#1e1e2e"},
                selected_style={"color":"#fff","backgroundColor":"#2e2e3e","borderTop":"2px solid #4e79a7"}),
        dcc.Tab(label="Forecasting",       value="tab-forecast",
                style={"color":"#aaa","backgroundColor":"#1e1e2e"},
                selected_style={"color":"#fff","backgroundColor":"#2e2e3e","borderTop":"2px solid #4e79a7"}),
        dcc.Tab(label="Raw Data",          value="tab-raw",
                style={"color":"#aaa","backgroundColor":"#1e1e2e"},
                selected_style={"color":"#fff","backgroundColor":"#2e2e3e","borderTop":"2px solid #4e79a7"}),
    ]),

    html.Div(id="tab-content"),

    html.P("Price: EIA weekly U.S. retail on-highway diesel (EMD_EPD2D_PTE_NUS_DPG, $/gal)",
           style={"color":"#555","fontSize":"11px","marginTop":"32px"}),
])


@callback(Output("kpi-row","children"), Input("product-radio","value"))
def update_kpis(product):
    cfg = PRODUCTS[product]
    pfx = cfg["pfx_short"]
    inv_col  = pfx+"_inv"
    stat_pfx = pfx+"_nat"

    def slast(col, n=1):
        tmp=DF.dropna(subset=[col])
        return tmp.iloc[-n] if len(tmp)>=n else None

    lat=slast(inv_col); prv=slast(inv_col,2)
    lpx=slast("price_gal"); ppx=slast("price_gal",2)
    lz=slast(stat_pfx+"_z") if stat_pfx+"_z" in DF.columns else None

    inv_val   = f"{lat[inv_col]:,.0f} Mbbl" if lat is not None else "N/A"
    inv_d     = f"{lat[inv_col]-prv[inv_col]:+,.0f} WoW" if lat is not None and prv is not None else ""
    inv_dv    = (lat[inv_col]-prv[inv_col]) if lat is not None and prv is not None else 0
    px_val    = f"${lpx['price_gal']:.3f}/gal" if lpx is not None else "N/A"
    px_d      = f"{lpx['price_gal']-ppx['price_gal']:+.3f} WoW" if lpx is not None and ppx is not None else ""
    px_dv     = (lpx["price_gal"]-ppx["price_gal"]) if lpx is not None and ppx is not None else 0
    zv        = lz[stat_pfx+"_z"] if lz is not None and pd.notna(lz[stat_pfx+"_z"]) else None
    z_val     = f"{zv:+.2f}σ" if zv is not None else "N/A"
    sig_val   = lz[stat_pfx+"_sig"] if lz is not None else "N/A"

    return [
        metric_card("Inventory",        inv_val, inv_d, inv_dv),
        metric_card("U.S. Retail Diesel", px_val, px_d,  px_dv),
        metric_card("Z-Score (3yr rolling)", z_val),
        metric_card("Signal", str(sig_val)),
    ]


@callback(Output("tab-content","children"),
          Input("tabs","value"), Input("product-radio","value"))
def render_tab(tab, product):
    cfg      = PRODUCTS[product]
    pfx      = cfg["pfx_short"]
    inv_col  = pfx+"_inv"
    stat_pfx = pfx+"_nat"
    corr_col = pfx+"_corr"
    color    = cfg["color"]

    if tab == "tab-national":
        return html.Div([
            dcc.Graph(figure=make_sd_band_fig(DF, inv_col, stat_pfx,
                product+" - National inventory with 3-year SD bands", color)),
            dcc.Graph(figure=make_combo_fig(DF, inv_col, "price_gal",
                product+" - National inventory vs U.S. retail diesel price",
                cfg["inventory_label"], cfg["price_label"], color)),
        ])

    elif tab == "tab-padd":
        return html.Div([
            html.Div([
                html.Label("Select PADD:", style={"color":"#aaa","fontSize":"13px","marginRight":"10px"}),
                dcc.Dropdown(
                    id="padd-dropdown",
                    options=[{"label":v,"value":k} for k,v in PADD_MAP.items()],
                    value="R10",
                    clearable=False,
                    style={"width":"320px","backgroundColor":"#2e2e3e","color":"#000"},
                ),
            ], style={"display":"flex","alignItems":"center","marginBottom":"16px"}),
            html.Div(id="padd-chart-area"),
            dcc.Graph(figure=make_padd_stack_fig(DF, pfx, product+" - All PADDs stacked")),
        ])

    elif tab == "tab-price":
        return html.Div([
            dcc.Graph(figure=make_combo_fig(DF, inv_col, "price_gal",
                product+" - Inventory vs U.S. retail diesel price",
                cfg["inventory_label"], cfg["price_label"], color)),
            dcc.Graph(figure=make_scatter_fig(DF, inv_col, "price_gal",
                product+" - Price vs inventory scatter (darker = more recent)")),
            dcc.Graph(figure=make_corr_fig(DF, corr_col,
                product+" - 26-week rolling correlation: inventory vs price")),
        ])

    elif tab == "tab-forecast":
        ifc=forecast_series(DF, inv_col, 12)
        pfc=forecast_series(DF, "price_gal", 12)
        return html.Div([
            html.P("Linear trend + weekly seasonal pattern. Directional only - not a trading model.",
                   style={"color":"#888","fontSize":"12px","marginBottom":"8px"}),
            dcc.Graph(figure=make_forecast_fig(DF, inv_col, ifc,
                product+" - 12-week inventory forecast","Inventory (Mbbl)", color)),
            dcc.Graph(figure=make_forecast_fig(DF, "price_gal", pfc,
                product+" - 12-week diesel price forecast","Price ($/gal)","orangered")),
        ])

    elif tab == "tab-raw":
        pfx2 = pfx
        sc=["date", inv_col, "price_gal", stat_pfx+"_z", stat_pfx+"_sig"]
        for padd in PADD_MAP:
            c=pfx2+"_"+padd
            if c in DF.columns: sc.append(c)
        dff=DF[sc].sort_values("date",ascending=False).head(200)
        dff["date"]=dff["date"].dt.strftime("%Y-%m-%d")
        cols=[{"name":c,"id":c} for c in dff.columns]
        from dash import dash_table
        return dash_table.DataTable(
            data=dff.to_dict("records"),
            columns=cols,
            style_table={"overflowX":"auto"},
            style_header={"backgroundColor":"#2e2e3e","color":"#fff","fontWeight":"bold","fontSize":"12px"},
            style_cell={"backgroundColor":"#1e1e2e","color":"#ccc","fontSize":"12px",
                        "border":"1px solid #333","padding":"6px"},
            style_data_conditional=[
                {"if":{"row_index":"odd"},"backgroundColor":"#252535"}],
            page_size=30,
        )


@callback(Output("padd-chart-area","children"),
          Input("padd-dropdown","value"), Input("product-radio","value"))
def update_padd_chart(padd_code, product):
    cfg   = PRODUCTS[product]
    pfx   = cfg["pfx_short"]
    pc    = pfx+"_"+padd_code
    color = PADD_COLORS[padd_code]
    fig   = make_sd_band_fig(DF, pc, pc,
        product+" - "+PADD_MAP[padd_code]+" with 3-year SD bands", color)
    return dcc.Graph(figure=fig)


if __name__ == "__main__":
    app.run(debug=False, port=8050)
