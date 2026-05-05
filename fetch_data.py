import requests
import pandas as pd
import os
from dotenv import load_dotenv

# Load API key from .env file
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
API_KEY = os.getenv("EIA_API_KEY")

BASE_URL = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"

def fetch_inventory(product_code, product_label):
    """Fetch weekly petroleum inventory for a given EIA product code."""
    headers = {
        "X-Params": '{"frequency":"weekly","data":["value"],"facets":{"product":["' + product_code + '"],"duoarea":["NUS"]},"sort":[{"column":"period","direction":"desc"}],"offset":0,"length":260}'
    }
    params = {"api_key": API_KEY}
    
    response = requests.get(BASE_URL, params=params, headers=headers)
    
    print(f"\nStatus code: {response.status_code}")
    
    json_data = response.json()
    
    # Check for errors in response
    if "error" in json_data:
        print(f"API Error: {json_data['error']}")
        return pd.DataFrame()
    
    if "response" not in json_data:
        print(f"Unexpected response: {json_data}")
        return pd.DataFrame()
    
    data = json_data["response"]["data"]
    
    if not data:
        print(f"No data returned for product code: {product_code}")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    print(f"Columns returned: {list(df.columns)}")
    print(df.head(3))
    
    df = df[["period", "value"]].rename(
        columns={"period": "date", "value": f"{product_label}_inventory_mbbls"}
    )
    df[f"{product_label}_inventory_mbbls"] = pd.to_numeric(
        df[f"{product_label}_inventory_mbbls"], errors="coerce"
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def fetch_spot_prices():
    """Fetch weekly spot prices for No.2 Heating Oil and Jet Fuel."""
    BASE_SPOT_URL = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
    
    series_map = {
        "EER_EPD2F_PF4_Y35NY_DPG": "distillate_spot_price",
        "EER_EPJK_PF4_RGC_DPG":    "jetfuel_spot_price"
    }
    
    all_dfs = []
    for series_code, col_name in series_map.items():
        headers = {
            "X-Params": '{"frequency":"weekly","data":["value"],"facets":{"series":["' + series_code + '"]},"sort":[{"column":"period","direction":"desc"}],"offset":0,"length":52}'
        }
        params = {"api_key": API_KEY}
        response = requests.get(BASE_SPOT_URL, params=params, headers=headers)
        json_data = response.json()
        
        if "response" not in json_data or not json_data["response"]["data"]:
            print(f"No spot price data for series: {series_code}")
            continue
        
        data = json_data["response"]["data"]
        df = pd.DataFrame(data)
        df = df[["period", "value"]].rename(
            columns={"period": "date", "value": col_name}
        )
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        all_dfs.append(df)
    
    if len(all_dfs) == 2:
        return all_dfs[0].merge(all_dfs[1], on="date", how="outer")
    elif len(all_dfs) == 1:
        return all_dfs[0]
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    print("API Key loaded:", API_KEY[:8] + "..." if API_KEY else "NOT FOUND")
    
    print("\n--- Fetching Distillate Fuel Oil inventory (EPD0) ---")
    distillate_df = fetch_inventory("EPD0", "distillate")
    
    print("\n--- Fetching Kerosene-Jet Fuel inventory (EPJK) ---")
    jetfuel_df = fetch_inventory("EPJK", "jetfuel")
    
    print("\n--- Fetching Spot Prices ---")
    prices_df = fetch_spot_prices()
    
    if distillate_df.empty and jetfuel_df.empty:
        print("\n❌ No inventory data returned. Check your API key and internet connection.")
    else:
        # Merge all datasets on date
        df = distillate_df.merge(jetfuel_df, on="date", how="outer")
        if not prices_df.empty:
            df = df.merge(prices_df, on="date", how="left")
        df = df.sort_values("date")
        df.to_csv("data.csv", index=False)
        print(f"\n✅ data.csv saved — {len(df)} weeks of data.")
        print(df.tail(5))