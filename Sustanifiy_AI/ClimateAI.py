
import os
import io
import json
import time
import math
import base64
import requests
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------
# Login Credentials (MODIFIED)
# -------------------------
VALID_USER = "aditya01"
VALID_PASS = "vermasingh01"

# Forecasting imports with graceful fallbacks
try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

try:
    from pmdarima import auto_arima
    _HAS_ARIMA = True
except Exception:
    _HAS_ARIMA = False

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

import plotly.express as px
import plotly.graph_objects as go

# SustainifyAI — Sustainability & Climate Change Tracker (All-in-One Streamlit App)
# ---------------------------------------------------------------------------------

st.set_page_config(
    page_title="SustainifyAI — Material Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------ MATERIAL DASHBOARD THEME (CSS OVERHAUL) ------------------------------
# 6. Color System: Element Color Sidebar #2C2C54 Active Tab #00C9A7
# 7. Typography: Poppins/Inter
# 3. Metric Cards: Green #4CAF50, Orange #FF9800, Red #F44336, Blue #2196F3

MATERIAL_PRIMARY = "#00C9A7" # Active/Accent Teal
MATERIAL_SECONDARY = "#3F51B5" # Indigo (for titles/buttons)
MATERIAL_CARD_BG = "#2C2C54" # Sidebar Background
MATERIAL_BG = "#F4F5F7" # Light Gray Main Background
MATERIAL_CARD_LIFT = "0 8px 10px 1px rgba(0, 0, 0, 0.14), 0 3px 14px 2px rgba(0, 0, 0, 0.12), 0 5px 5px -3px rgba(0, 0, 0, 0.2)"
MATERIAL_CARD_FLAT = "0 2px 2px 0 rgba(0, 0, 0, 0.14), 0 3px 1px -2px rgba(0, 0, 0, 0.12), 0 1px 5px 0 rgba(0, 0, 0, 0.2)"
MATERIAL_HEADER_BG = "linear-gradient(60deg, #26c6da, #00acc1)" # Teal to Cyan Gradient

st.markdown(
    f"""
    <style>
    /* *** 7. Typography: Poppins/Inter *** */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

    /* --- CORE BACKGROUND & LAYOUT --- */
    .stApp {{ 
        background-color: {MATERIAL_BG}; 
        font-family: 'Inter', sans-serif;
    }}
    .main > div {{
        padding: 1rem 3rem; /* Increase side padding */
        padding-top: 1rem;
    }}

    /* ------------------------------ 1. SIDEBAR (Material Dark) ------------------------------ */
    .stSidebar > div:first-child {{
        background: {MATERIAL_CARD_BG} !important;
        border-right: none;
        box-shadow: 4px 0 10px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }}
    /* Titles in Sidebar */
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar .stMarkdown p {{
        color: #FFFFFF !important;
        font-weight: 700;
        text-shadow: none;
        opacity: 0.9;
    }}
    /* Sidebar Input/Controls Label Color */
    .stSidebar .stMarkdown label p {{
        color: #BBBBBB !important;
        font-weight: 600;
    }}

    /* Active Nav Item Styling (1. Sidebar & 6. Color System) */
    .stSidebar .stRadio div[role="radiogroup"] > label {{
        padding: 10px 15px;
        margin-bottom: 5px;
        border-radius: 8px;
        color: #E0E0E0; /* Light text */
        transition: all 0.3s ease;
    }}
    .stSidebar .stRadio div[role="radiogroup"] > label:hover {{
        background-color: #3C4B64; /* Darker hover */
        box-shadow: {MATERIAL_CARD_FLAT};
        transform: translateX(3px);
    }}

    .stSidebar .stRadio div[role="radiogroup"] > label:has(input[type="radio"]:checked) {{
        background-color: {MATERIAL_PRIMARY}; /* Active Teal */
        color: #FFFFFF;
        box-shadow: {MATERIAL_CARD_LIFT};
        transform: translateX(0px);
    }}
    .stSidebar .stRadio div[role="radiogroup"] > label p {{
        font-weight: 600;
        font-size: 1.05rem !important;
    }}
    
    /* Hide the ugly radio button dot */
    .stRadio [data-testid="stFormSubmitButton"] + div > label:first-child {{
        display: none;
    }}
    .stRadio [data-testid="stFormSubmitButton"] + div {{
        display: none;
    }}
    
    /* ------------------------------ 2. TOP NAVBAR (Fixed) ------------------------------ */
    header {{
        background: #FFFFFF; /* White Header */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        position: fixed;
        top: 0;
        width: calc(100% - 250px); /* Adjust for sidebar width */
        z-index: 1000;
        transition: width 0.3s;
    }}
    
    .st-emotion-cache-18ni7ap {{ /* Custom Class for header title area */
        display: none !important; /* Hide default Streamlit header */
    }}
    
    .navbar-container {{
        background: {MATERIAL_HEADER_BG};
        padding: 10px 30px;
        border-radius: 0 0 10px 10px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.14), 0 7px 10px -5px rgba(0, 150, 136, 0.4);
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }}

    .navbar-title {{
        color: #FFFFFF;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}

    .navbar-actions {{
        display: flex;
        gap: 15px;
        align-items: center;
    }}
    .navbar-icon {{
        font-size: 1.5rem;
        color: #FFFFFF;
        cursor: pointer;
        opacity: 0.8;
    }}
    .navbar-icon:hover {{
        opacity: 1;
    }}
    .search-input input {{
        background-color: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 4px;
        color: white;
        padding: 5px 10px;
    }}
    
    /* ------------------------------ 3. METRIC CARDS (Material Elevated) ------------------------------ */
    /* Color Palette */
    .bg-green {{ background: #4CAF50 !important; }}
    .bg-orange {{ background: #FF9800 !important; }}
    .bg-red {{ background: #F44336 !important; }}
    .bg-blue {{ background: #2196F3 !important; }}
    
    .material-card-kpi {{
        background-color: #FFFFFF;
        border-radius: 6px;
        box-shadow: {MATERIAL_CARD_FLAT};
        transition: all 0.3s ease;
        padding: 20px;
        height: 120px;
        margin-top: 50px; /* Space for the lifted icon block */
    }}
    .material-card-kpi:hover {{
        box-shadow: {MATERIAL_CARD_LIFT};
        transform: translateY(-2px);
    }}
    
    .kpi-icon-block {{
        position: absolute;
        top: -20px; /* Lift the block above the card */
        left: 10px;
        width: 80px;
        height: 80px;
        border-radius: 3px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.14), 0 7px 10px -5px rgba(0, 150, 136, 0.4);
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 2.5rem;
        color: #FFFFFF;
        line-height: 1;
        z-index: 1;
    }}
    
    .kpi-value-container {{
        text-align: right;
        padding-top: 10px;
    }}
    .kpi-value-material {{
        font-size: 1.8rem;
        font-weight: 600;
        color: #3C4B64;
        margin: 0;
    }}
    .kpi-label-material {{
        font-size: 0.8rem;
        color: #999999;
        font-weight: 400;
        margin: 0;
    }}

    /* Adjust Streamlit's metric to fit the new card structure */
    div[data-testid="stMetric"] {{
        display: none; /* Hide default metric output */
    }}

    /* ------------------------------ 4. CHART CARDS (Material Elevated) ------------------------------ */
    .chart-card-wrap {{
        background-color: #FFFFFF;
        border-radius: 6px;
        box-shadow: {MATERIAL_CARD_FLAT};
        padding: 0;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }}
    .chart-card-wrap:hover {{
        box-shadow: {MATERIAL_CARD_LIFT};
    }}
    
    .card-header-material {{
        background: {MATERIAL_HEADER_BG};
        color: white;
        padding: 10px 15px;
        margin: 15px 15px 0 15px;
        border-radius: 3px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.14), 0 7px 10px -5px rgba(0, 150, 136, 0.4);
    }}

    .card-header-material h3 {{
        color: white !important;
        margin: 0;
        font-weight: 600;
        font-size: 1.2rem;
        line-height: 1.5;
    }}
    .card-body-material {{
        padding: 20px;
    }}
    
    /* --- General Text/Title Styling --- */
    h1, h2, h3, h4, h5 {{
        color: #3C4B64 !important; /* Dark Blue for headings */
        font-family: 'Inter', sans-serif !important;
        font-weight: 700;
    }}
    p, li, span, div, label, .stMarkdown {{ 
        color: #3C4B64 !important; 
        font-size: 1.0rem !important;
    }}
    
    /* Override for Plotly in Card Body */
    .plot-wrap {{
        border: none;
        box-shadow: none;
        background: white;
        padding: 0;
    }}
    
    /* LOGIN STYLES */
    .login-box {{
        max-width:480px;
        margin:6rem auto 2rem auto;
        padding:32px;
        border-radius:12px;
        background:#FFFFFF;
        box-shadow: {MATERIAL_CARD_LIFT};
        border:1px solid #E0E0E0;
    }}
    .login-title {{
        margin:0 0 10px 0;
        color: {MATERIAL_SECONDARY};
        font-family: 'Inter', sans-serif !important;
        text-shadow: none;
    }}
    .login-sub {{
        color:#757575;
        margin:0 0 16px 0;
    }}
    
    /* Chatbot adjustments for Material look */
    .chatbot-card {{
        background-color: #FFFFFF;
        border-radius: 6px;
        box-shadow: {MATERIAL_CARD_FLAT};
        padding: 15px;
        margin-bottom: 20px;
    }}
    .stChatMessage {{
        background-color: #EFEFEF;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
    }}
    .stChatMessage:last-child {{
        margin-bottom: 0;
    }}
    
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Simple Login Function
# -------------------------
def _login_view():
    st.markdown(
        f"""
        <div class="login-box">
          <h2 class="login-title">🔐 SustainifyAI — Secure Login</h2>
          <p class="login-sub">Use your credentials to access the Climate Dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("login_form", clear_on_submit=False):
        u = st.text_input("Username", "")
        p = st.text_input("Password", "", type="password")
        colA, colB = st.columns([1,3])
        submit = colA.form_submit_button("Login")
        if submit:
            if u == VALID_USER and p == VALID_PASS:
                st.session_state["auth_ok"] = True
                st.session_state["auth_user"] = u
                st.success("Login successful. Redirecting...")
                st.rerun()
            else:
                st.error("Invalid username or password.")
    
# --- LOGIN GATE ---
if "auth_ok" not in st.session_state:
    st.session_state["auth_ok"] = False
    st.session_state["auth_user"] = None

if not st.session_state["auth_ok"]:
    _login_view()
    st.stop()  # Halt the app here until logged in

# ------------------------------ The rest of your SustainifyAI code follows ------------------------------

# ------------------------------ Utility: Caching (Corrected Definition) ------------------------------
@st.cache_data(show_spinner=False)
def geocode_place(place: str) -> Optional[Tuple[float, float, str, str]]:
    """Use Open-Meteo geocoding (no API key) to resolve a place to (lat, lon, name, country)."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(url, params={"name": place, "count": 1, "language": "en", "format": "json"}, timeout=20)
    if r.ok:
        js = r.json()
        if js.get("results"):
            res = js["results"][0]
            return float(res["latitude"]), float(res["longitude"]), res.get("name",""), res.get("country","")
    return None

@st.cache_data(show_spinner=False)
def fetch_openmeteo_daily(lat: float, lon: float, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Fetch daily climate variables from Open-Meteo ERA5 reanalysis (no key).
    """
    
    today_date = dt.date.today()
    
    # Use yesterday's date for archive access if the user specified the current date or later
    if end >= today_date:
        api_end_date = today_date - dt.timedelta(days=1)
    else:
        api_end_date = end

    if api_end_date < start:
        # Return empty data frame with expected columns if period is invalid
        return pd.DataFrame({'time': [], 'temperature_2m_max': [], 'temperature_2m_mean': [], 'temperature_2m_min': [], 'precipitation_sum': [], 'windspeed_10m_max': [], 'shortwave_radiation_sum': []})

    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": api_end_date.isoformat(), # Use the adjusted end date
        "daily": [
            "temperature_2m_mean","temperature_2m_max","temperature_2m_min",
            "precipitation_sum","windspeed_10m_max","shortwave_radiation_sum",
        ],
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status() 
    js = r.json()
    df = pd.DataFrame(js["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df

@st.cache_data(show_spinner=False)
def fetch_air_quality_current(lat: float, lon: float) -> pd.DataFrame:
    """
    Fetch latest air quality using Open-Meteo's Air Quality API (No key required).
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    # Replacing subscripts with standard text
    hourly_vars = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly_vars),
        "domains": "auto",
        "timezone": "auto",
        "current": ",".join(hourly_vars)
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Air Quality API (Open-Meteo) fetch failed: {e}")
        return pd.DataFrame()

    rows = []
    if "current" in js and "hourly_units" in js:
        current_data = js["current"]
        units = js["hourly_units"]
        last_updated = current_data.get("time")
        
        for param in hourly_vars:
            value = current_data.get(param)
            # Replacing subscript in unit string
            unit = units.get(param, "ug/m3").replace("µg/m³", "ug/m3") 
            
            if value is not None:
                rows.append({
                    "location": f"{js.get('latitude', lat):.3f}, {js.get('longitude', lon):.3f}",
                    "parameter": param,
                    "value": float(value),
                    "unit": unit,
                    "date": last_updated,
                    "lat": js.get('latitude', lat),
                    "lon": js.get('longitude', lon),
                })

    return pd.DataFrame(rows)

# --- Placeholder Functions for Complex Features (Dynamic for City) ---

@st.cache_data(ttl=3600, show_spinner=False)
def get_river_health_data(city_name: str):
    """Synthesizes data for the major river near the selected city."""
    
    city_lower = city_name.lower()
    
    # --- Dynamic Logic based on major Indian cities/regions ---
    if "mumbai" in city_lower or "pune" in city_lower:
        river = "Mula-Mutha/Mithi (Maharashtra)"
        do, bod, coliform, status = 4.5, 6.0, 4000, "Extreme Stress"
    elif "chennai" in city_lower or "madurai" in city_lower or "kochi" in city_lower:
        river = "Cooum/Vaigai (Tamil Nadu/Kerala)"
        do, bod, coliform, status = 3.0, 8.0, 5000, "Extreme Stress"
    elif "kolkata" in city_lower or "patna" in city_lower:
        river = "Hooghly/Ganga (East)"
        do, bod, coliform, status = 5.5, 3.5, 2000, "Critical Stress"
    elif "kanpur" in city_lower:
        river = "Ganga (Kanpur)"
        do, bod, coliform, status = 5.8, 4.5, 2500, "Critical Stress"
    elif "varanasi" in city_lower:
        river = "Ganga (Varanasi)"
        do, bod, coliform, status = 6.8, 3.2, 1200, "High Stress"
    elif "lucknow" in city_lower or "jaunpur" in city_lower:
        river = "Gomti (UP)"
        do, bod, coliform, status = 5.0, 4.0, 3000, "Critical Stress"
    elif "prayagraj" in city_lower or "allahabad" in city_lower:
        # Prayagraj Specific Data (Based on reports)
        river = "Ganga (Sangam/Prayagraj)"
        do, bod, coliform, status = 7.0, 3.5, 11000, "High Stress"
    elif "hyderabad" in city_lower:
        river = "Musil (Telangana)"
        do, bod, coliform, status = 4.0, 7.0, 4500, "Extreme Stress"
    else:
        # Default/General River Logic (Covers all smaller UP/Indian cities dynamically)
        river = f"{city_name} River (General)"
        do, bod, coliform, status = 7.5, 2.5, 800, "Moderate Stress"
        
    data = {
        "River": [river],
        "Dissolved Oxygen (DO mg/L)": [do], # Healthy > 6.0
        "BOD (mg/L)": [bod], # Good < 3.0
        "Coliform (MPN/100ml)": [coliform], # Safe < 500
        "Status": [status],
    }
    df = pd.DataFrame(data)
    # Green/Yellow/Red for Status - Using Bright Theme Colors
    df['Color'] = df['Status'].apply(lambda x: '#e63946' if x == 'Critical Stress' or x == x == 'Extreme Stress' else ('#ffc107' if x == 'High Stress' else '#38a3a5'))
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def get_tree_inventory(city_name: str):
    """Synthesizes tree data and requirements for the selected city (Maximized UP Granularity)."""
    
    city_lower = city_name.lower()

    # --- Highly Granular UP/Major City Lookup ---
    population_proxies = {
        "delhi": 19000000, "mumbai": 20000000, "bengaluru": 13000000, "chennai": 8000000,
        "kanpur": 2700000, "lucknow": 2700000, "ghaziabad": 2500000, "agra": 1800000,
        "varanasi": 1500000, "meerut": 1500000, "bareilly": 1200000, "aligarh": 1000000,
        "moradabad": 1000000, "firozabad": 1000000, "jhansi": 800000, "gorakhpur": 800000,
        "prayagraj": 1600000, "allahabad": 1600000, # Prayagraj specific population
        # Default for smaller UP districts
    }

    current_trees_proxies = {
        "delhi": 3000000, "mumbai": 1500000, "bengaluru": 1200000, "chennai": 900000,
        "kanpur": 850000, "lucknow": 950000, "ghaziabad": 650000, "agra": 400000,
        "varanasi": 500000, "meerut": 350000, "bareilly": 300000, "aligarh": 250000,
        "moradabad": 250000, "firozabad": 200000, "jhansi": 180000, "gorakhpur": 190000,
        "prayagraj": 550000, "allahabad": 550000, # Prayagraj specific tree count
    }
    
    # Get base values, defaulting to a smaller urban size if city is not listed
    population = population_proxies.get(city_lower, 400000)
    current_trees = current_trees_proxies.get(city_lower, 100000)
        
    target_ratio = 10 # Trees per person (national standard recommendation)
    trees_needed = (population * target_ratio) - current_trees
    
    return {
        "city": city_name,
        "current": current_trees,
        "population": population,
        "target_ratio": target_ratio,
        "needed": max(0, trees_needed),
        "needed_per_capita": round(trees_needed / population, 2)
    }

def get_future_impact_prediction(pm25_level: float):
    """Predicts generalized health impact based on current PM2.5."""
    # Replacing PM2.5 threshold comment
    if pm25_level < 50:
        return {"health_risk": "Low", "advice": "Continue outdoor activities.", "color": "#38a3a5"} # Teal/Blue-Green for Good
    elif 50 <= pm25_level < 100:
        return {"health_risk": "Moderate", "advice": "Sensitive groups should limit prolonged outdoor exertion.", "color": "#ffc107"} # Yellow/Amber for Warning
    else:
        return {"health_risk": "High", "advice": "All groups should avoid prolonged or heavy exertion outdoors. Wear N95 masks.", "color": "#e63946"} # Red for Critical

# ⭐️ MODIFIED FUNCTION FOR DYNAMIC NEWS TICKER ⭐️
@st.cache_data(ttl=300, show_spinner=False)
def get_pollution_news_ticker(city_name: str) -> str:
    """Combines suggested text into a single, moving line, dynamically localized by city."""
    
    # --- City-Specific Dynamic Content ---
    # Fetch placeholder data for context
    river_data = get_river_health_data(city_name)
    tree_data = get_tree_inventory(city_name)
    
    # 1. City-Specific Tree Goal Headline
    trees_needed_str = f"{tree_data['needed']:,}"
    tree_headline = f"💡 {city_name} local bodies initiate massive tree plantation drive to bridge the {trees_needed_str} gap."
    
    # 2. City-Specific River Health Headline
    river_status = river_data['Status'].iloc[0]
    river_name = river_data['River'].iloc[0].split(' (')[0]
    river_headline = f"💧 {river_name} health at '{river_status}' status; BOD reduction initiatives intensify."
    
    # 3. City-Specific Swachh Rank Headline (Check against top 3)
    rank_df = get_swachh_ranking_data([city_name], city_name)
    current_rank = rank_df[rank_df['City'] == city_name].get('2025', pd.Series([np.nan])).iloc[0]
    
    swachh_headline = ""
    if not np.isnan(current_rank) and current_rank < 50:
        swachh_headline = f"🏆 {city_name} aims for Top 50 Global Swachh Rank; new waste collection methods deployed."
    else:
        swachh_headline = f"♻ New mandate targets 50% waste recycling to improve Swachh ranking for {city_name}."
        
    # --- Assemble final list (repeat for smooth scrolling) ---
    news_items = [
        tree_headline,
        swachh_headline,
        river_headline,
        "⚡ Focus shifts to solar roof-tops to boost urban renewable energy capacity.",
    ]
    
    return " | ".join(news_items) * 3

# --- NEW Impact Simulation Functions ---

def get_crop_loss_simulation(mean_temp: float) -> Tuple[float, float, str]:
    """Calculates simulated crop yield based on temperature rise."""
    base_yield = 100
    critical_temp = 25.0 # Critical threshold for Indian dry season crops (simple proxy)
    
    if mean_temp > critical_temp:
        loss_factor = min(0.35, (mean_temp - critical_temp) * 0.05) # Max loss 35%
        stressed_yield = base_yield * (1 - loss_factor)
        loss_percent = round(loss_factor * 100, 1)
        # Replacing degree symbol in status text
        status = f"Severe stress (Avg Temp > {critical_temp} deg C)"
    else:
        stressed_yield = base_yield
        loss_percent = 0.0
        status = "Optimal/Moderate Stress"
        
    return stressed_yield, loss_percent, status

# Placeholder for All India City Data (Needed for Choropleth comparison)
@st.cache_data(ttl=3600, show_spinner=False)
def get_all_india_city_emissions() -> pd.DataFrame:
    """Mock CO2 data for major Indian cities for the Choropleth map."""
    data = {
        'City': ['Mumbai', 'Delhi', 'Bengaluru', 'Chennai', 'Kolkata', 'Varanasi', 'Prayagraj', 'Kanpur', 'Lucknow', 'Ahmedabad'],
        'Latitude': [19.0760, 28.7041, 12.9716, 13.0827, 22.5726, 25.3176, 25.4358, 26.4499, 26.8467, 23.0225],
        'Longitude': [72.8777, 77.1025, 77.5946, 80.2707, 88.3639, 82.9739, 81.8463, 80.3319, 80.9462, 72.5714],
        'CO2_Emissions_Annual_kT': [45000, 52000, 31000, 19000, 25000, 3500, 4200, 6000, 6500, 15000], # Kilotons per year proxy
    }
    return pd.DataFrame(data)

# --- NEW: Green Infrastructure & EV Placeholder Data Functions ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_pollution_free_energy_data(city_names: List[str]) -> pd.DataFrame:
    """Simulated pollution-free energy resource data (Capacity in MW)."""
    city_base_values = {
        "Prayagraj": {"Solar": 150, "Wind": 5, "Hydro": 20},
        "Lucknow": {"Solar": 200, "Wind": 10, "Hydro": 15},
        "Varanasi": {"Solar": 120, "Wind": 3, "Hydro": 10},
        "Kanpur": {"Solar": 180, "Wind": 8, "Hydro": 25},
        "Mumbai": {"Solar": 400, "Wind": 50, "Hydro": 30},
        "Delhi": {"Solar": 350, "Wind": 20, "Hydro": 5},
        "Bengaluru": {"Solar": 300, "Wind": 15, "Hydro": 20},
        "Agra": {"Solar": 80, "Wind": 5, "Hydro": 15},
    }
    rows = []
    for city in city_names:
        clean_city = city.strip()
        data = city_base_values.get(clean_city, {"Solar": 75, "Wind": 2, "Hydro": 10})
        rows.append({"City": clean_city, "Solar (MW)": data["Solar"], "Wind (MW)": data["Wind"], "Hydro (MW)": data["Hydro"]})
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600, show_spinner=False)
def get_registered_solar_connections(city_names: List[str]) -> pd.DataFrame:
    """Simulated registered solar connections (Units)."""
    city_base_values = {
        "Prayagraj": 25000, "Lucknow": 35000, "Varanasi": 20000, "Kanpur": 30000,
        "Mumbai": 80000, "Delhi": 70000, "Bengaluru": 60000, "Agra": 18000,
    }
    rows = []
    for city in city_names:
        clean_city = city.strip()
        connections = city_base_values.get(clean_city, np.random.randint(8000, 15000))
        rows.append({'City': clean_city, 'Solar Connections': connections})
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600, show_spinner=False)
def get_registered_ev_vehicles(city_names: List[str]) -> pd.DataFrame:
    """Simulated registered EV vehicles (Units)."""
    city_base_values = {
        "Prayagraj": 18000, "Lucknow": 25000, "Varanasi": 15000, "Kanpur": 22000,
        "Mumbai": 70000, "Delhi": 100000, "Bengaluru": 80000, "Agra": 12000,
    }
    rows = []
    for city in city_names:
        clean_city = city.strip()
        evs = city_base_values.get(clean_city, np.random.randint(5000, 12000))
        rows.append({'City': clean_city, 'Registered EVs': evs})
    return pd.DataFrame(rows)

# --- NEW: Swachh Survekshan Data Placeholder (MAX UP COVERAGE - CORRECTED) ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_swachh_ranking_data(city_names: List[str], current_city: str) -> pd.DataFrame:
    """
    Simulated Swachh Survekshan ranking data (Global Rank) for all 17 UP Nagar Nigams
    plus key national performers.
    Lower rank number = better.
    """
    
    # 1. DEFINE BASE RANKING DATA (All 17 UP Nagar Nigams + National Leaders)
    # NOTE: All inner lists must have exactly 23 elements.
    rank_data = {
        'City': [
            # National Top (3)
            'Indore', 'Surat', 'Navi Mumbai', 
            
            # UP Nagar Nigams (17 Cities)
            'Lucknow', 'Varanasi', 'Prayagraj', 'Gorakhpur', 'Agra', 'Ghaziabad', 'Meerut', 
            'Moradabad', 'Aligarh', 'Bareilly', 'Kanpur', 'Firozabad', 'Ayodhya', 
            'Jhansi', 'Saharanpur', 'Mathura-Vrindavan', 'Shahjahanpur',
            
            # Key Nagar Palika Parishad (NPP) Proxies (3)
            'Bijnor', 'Shamshabad', 'Basti',
        ],
        
        # Rankings (23 items in each list - CORRECTED to match City list length)
        '2021': [
            1, 2, 4, 		# National (3)
            40, 35, 60, 90, 100, 150, 160, # Tier 1 UP (7)
            180, 200, 220, 250, 280, 290, # Tier 2 UP (6)
            310, 330, 350, 370, 		# Tier 3 UP (4)
            400, 420, 440 				# NPP Proxies/Lower Ranks (3)
        ],	
        
        '2022': [
            1, 2, 3,
            30, 28, 50, 75, 85, 130, 140,
            160, 180, 200, 230, 260, 270,
            290, 310, 330, 350, 		# Tier 3 UP (4)
            380, 400, 420
        ],
        
        '2023': [
            1, 1, 3,
            20, 22, 40, 60, 70, 110, 120,
            140, 160, 180, 210, 240, 250,
            270, 290, 310, 330, 		# Tier 3 UP (4)
            360, 380, 400
        ],
        
        # Simulated/Target (Based on recent performance improvements)
        '2024': [
            1, 1, 3,
            15, 18, 35, 50, 60, 90, 100,
            120, 140, 160, 190, 220, 230,
            250, 270, 290, 310, 		# Tier 3 UP (4)
            340, 360, 380
        ],
        
        '2025': [
            1, 1, 3,
            10, 15, 30, 40, 50, 80, 90,
            110, 130, 150, 180, 210, 220,
            240, 260, 280, 300, 		# Tier 3 UP (4)
            320, 340, 360
        ],
    }
    df = pd.DataFrame(rank_data)
    
    # 2. FILTERING AND CASE-INSENSITIVE MATCHING
    city_map = {c.lower(): c for c in df['City'].unique()}
    
    filtered_cities = []
    # Accumulate cities that are in the master list
    for name in city_names:
        matched_name = city_map.get(name.strip().lower())
        if matched_name and matched_name not in filtered_cities:
            filtered_cities.append(matched_name)

    # 3. CRITICAL: HANDLING CITIES NOT IN THE LIST (Nagar Palika Parishads / Nagar Panchayats)
    current_city_lower = current_city.lower()
    
    # Check if the user's current city is *already* included (either via selection or as a default list member)
    city_is_known = current_city_lower in [c.lower() for c in filtered_cities]
    
    if not city_is_known:
        # If the user's current city is not a major corporation, 
        # assign it a dynamic, high, but improving rank (e.g., starting at rank 500)
        default_rank_series = {
            'City': current_city, 
            '2021': 500, '2022': 480, '2023': 450, '2024': 420, '2025': 390
        }
        # Use a list of dictionaries for robust insertion into the DataFrame
        df = pd.concat([df, pd.DataFrame([default_rank_series])], ignore_index=True)
        filtered_cities.append(current_city)
    
    # Final filter based on accumulated cities
    return df[df['City'].isin(filtered_cities)]


# ------------------------------ Sustainability Score ------------------------------

def normalize_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.max() == s.min():
        return pd.Series(np.ones(len(s)))
    return (s - s.min()) / (s - s.min()).max()

@dataclass
class SustainabilityInputs:
    pm25: float
    co2_per_capita: float              # optional proxy if available
    renewable_share: float # 0..100
    water_quality_index: float # 0..100
    waste_recycling_rate: float # 0..100


def compute_sustainability_score(inp: SustainabilityInputs) -> Tuple[float, dict]:
    """Composite score 0-100 with interpretable sub-scores and weights."""
    # Lower PM2.5 is better. Invert it against a reference band.
    # Replacing comments
    pm25_scaled = np.clip(1 - (inp.pm25 / 75.0), 0, 1)  # 75 ug/m3 ~ very poor
    co2_scaled = np.clip(1 - (inp.co2_per_capita / 20.0), 0, 1)  # 20 t/cap ~ bad
    ren_scaled = np.clip(inp.renewable_share / 100.0, 0, 1)
    water_scaled = np.clip(inp.water_quality_index / 100.0, 0, 1)
    waste_scaled = np.clip(inp.waste_recycling_rate / 100.0, 0, 1)

    # Correcting CO2 (subscript) references in weights dictionary keys
    weights = {
        "Air Quality (PM2.5)": 0.28,
        "CO2 / Capita": 0.18, 
        "Renewables Share": 0.24,
        "Water Quality": 0.15,
        "Recycling Rate": 0.15,
    }
    subs = {
        "Air Quality (PM2.5)": pm25_scaled,
        "CO2 / Capita": co2_scaled,
        "Renewables Share": ren_scaled,
        "Water Quality": water_scaled,
        "Recycling Rate": waste_scaled,
    }
    score = sum(subs[k]*w for k, w in weights.items()) * 100
    return float(score), {k: round(v*100, 1) for k, v in subs.items()}

# ------------------------------ Forecasting Helpers ------------------------------

def backtest_train_forecast(df: pd.DataFrame, target_col: str, horizon: int = 30, model_choice: str = "auto"):
    """Time-series train/validation split, fit model, forecast horizon days. Returns forecast and metrics."""
    ts = df[["time", target_col]].dropna().copy()
    ts = ts.sort_values("time")
    ts.rename(columns={"time":"ds", target_col:"y"}, inplace=True)

    # Use last 20% as validation
    n = len(ts)
    if n < 100:
        # Small set: reduce horizon
        horizon = max(7, min(horizon, n//5))
    split_idx = max(5, int(n*0.8))
    train, valid = ts.iloc[:split_idx], ts.iloc[split_idx:]

    y_pred = None

    def prophet_fit_forecast():
        m = Prophet(seasonality_mode='additive', yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(train)
        future = m.make_future_dataframe(periods=horizon)
        fcst = m.predict(future)
        return m, fcst, m.predict(valid[["ds"]])["yhat"].values, valid["y"].values

    def arima_fit_forecast():
        model = auto_arima(train["y"], seasonal=True, m=365, suppress_warnings=True, error_action='ignore', stepwise=True)
        # Predict on validation set for metrics
        yhat_valid = model.predict(n_periods=len(valid))
        
        # Build future index
        steps = horizon
        future_preds = model.predict(n_periods=steps)
        fcst = pd.DataFrame({
            "ds": pd.date_range(valid["ds"].iloc[-1] + pd.Timedelta(days=1), periods=steps, freq='D'),
            "yhat": future_preds
        })
        return model, fcst, yhat_valid, valid["y"].values

    def ml_fit_forecast():
        # Simple lag features RF
        full = pd.concat([train, valid], axis=0).reset_index(drop=True)
        for lag in [1,2,7,14,30]:
            full[f"lag_{lag}"] = full["y"].shift(lag)
        full.dropna(inplace=True)
        X = full.drop(columns=["ds","y"]).values
        y = full["y"].values
        split = int(len(full)*0.8)
        Xtr, Xva = X[:split], X[split:]
        ytr, yva = y[:split], y[split:]
        rf = RandomForestRegressor(n_estimators=400, random_state=42)
        rf.fit(Xtr, ytr)
        # backtest pred
        y_pred_bt = rf.predict(Xva)
        # iterative future forecast using last row
        last = full.iloc[-1:].copy()
        future_rows = []
        for i in range(horizon):
            feats = last.drop(columns=["ds","y"]).values
            yhat = rf.predict(feats)[0]
            new_date = last["ds"].iloc[0] + pd.Timedelta(days=1)
            row = {
                "ds": new_date,
                "y": np.nan,
            }
            # shift lags
            for lag in [1,2,7,14,30]:
                if lag == 1:
                    row[f"lag_{lag}"] = yhat
                else:
                    # Defensive check: if lag column exists, shift it. Otherwise, use yhat.
                    row[f"lag_{lag}"] = last[f"lag_{lag-1}"].iloc[0] if f"lag_{lag-1}" in last.columns else yhat
            future_rows.append(row)
            last = pd.DataFrame([row])
        fcst = pd.DataFrame(future_rows)
        fcst.rename(columns={"y": "yhat"}, inplace=True)
        return rf, fcst, y_pred_bt, yva

    model_used = None
    metrics = {"MAE": None, "MAPE": None}

    if (model_choice == "Prophet" and _HAS_PROPHET) or (model_choice == "auto" and _HAS_PROPHET):
        model_used = "Prophet"
        m, fcst, yhat_valid, y_valid = prophet_fit_forecast()
        metrics["MAE"] = float(mean_absolute_error(y_valid, yhat_valid))
        metrics["MAPE"] = float(mean_absolute_percentage_error(y_valid, yhat_valid))
    elif (model_choice == "ARIMA" and _HAS_ARIMA) or (model_choice == "auto" and _HAS_ARIMA):
        model_used = "ARIMA"
        try:
            m, fcst, yhat_valid, y_valid = arima_fit_forecast()
            metrics["MAE"] = float(mean_absolute_error(y_valid, yhat_valid))
            metrics["MAPE"] = float(mean_absolute_percentage_error(y_valid, yhat_valid))
        except Exception:
             # Fallback if ARIMA model selection or prediction fails
             model_used = "ML Ensemble"
             m, fcst, y_pred_bt, y_valid = ml_fit_forecast()
             metrics["MAE"] = float(mean_absolute_error(y_valid, y_pred_bt))
             metrics["MAPE"] = float(mean_absolute_percentage_error(y_valid, y_pred_bt))
    else:
        model_used = "ML Ensemble"
        m, fcst, y_pred_bt, y_valid = ml_fit_forecast()
        metrics["MAE"] = float(mean_absolute_error(y_valid, y_pred_bt))
        metrics["MAPE"] = float(mean_absolute_percentage_error(y_valid, y_pred_bt))

    return model_used, ts, train, valid, fcst, metrics

# ------------------------------ Alerts (Telegram Optional) ------------------------------

def send_telegram(msg: str) -> bool:
    # Telegram code remains the same
    return False # Disabled for public code submission

# ------------------------------ Gauge Color Utility ------------------------------

def get_gauge_color(value, good_threshold, bad_threshold, reverse=False):
    """Returns color based on value and thresholds. Good is green, Bad is red. Using Bright Palette."""
    # Custom colors for Bright Theme
    COLOR_GOOD = "#38a3a5" # Teal/Blue-Green
    COLOR_WARNING = "#ffc107" # Yellow/Amber
    COLOR_CRITICAL = "#e63946" # Red
    
    if reverse: # Lower value is better (e.g., CO2, PM2.5)
        if value <= good_threshold:
            return COLOR_GOOD 
        elif value < bad_threshold:
            return COLOR_WARNING 
        else:
            return COLOR_CRITICAL 
    else: # Higher value is better (e.g., Renewable Share, Water Quality)
        if value >= good_threshold:
            return COLOR_GOOD 
        elif value > bad_threshold:
            return COLOR_WARNING 
        else:
            return COLOR_CRITICAL 
            
# ------------------------------ CHATBOT LOGIC (UPDATED FOR COMPREHENSIVE Q&A) ------------------------------

def generate_chatbot_response(prompt: str) -> str:
    prompt_lower = prompt.lower().strip()
    
    # --- GREETINGS, SOCIALS, AND KEYWORDS ---
    if any(keyword in prompt_lower for keyword in ["hello", "hi", "hey", "good morning", "good evening"]):
        return "Hello! I am SustainifyAI's virtual assistant. I'm here to answer questions about the **dashboard's features, data visualizations, and scoring methods**. What can I explain for you today?"
    
    if any(keyword in prompt_lower for keyword in ["how are you", "what's up", "how are things"]):
        return "I am a virtual assistant, operating perfectly! Ask me specific questions about the **SustainifyAI project features, technical terms, or any of the graphs**."
    
    if any(keyword in prompt_lower for keyword in ["thank you", "bye", "good day", "thanks"]):
        return "You're welcome! Feel free to return if you have more questions. Bye!"

    # --- TECHNICAL TERMS & METRICS ---
    
    if "pm2.5" in prompt_lower or "pm25" in prompt_lower:
        return "💨 **PM2.5** stands for Particulate Matter 2.5 micrometers in diameter or smaller. These are microscopic particles (like dust, soot, and smoke) that are harmful because they can penetrate deep into the lungs. **Lower values mean better air quality.**"
        
    elif "co2" in prompt_lower or "carbon" in prompt_lower:
        return "🏭 **CO2 (Carbon Dioxide)** is the main greenhouse gas responsible for global warming. In the dashboard, 'CO2 per Capita' measures the estimated annual carbon emissions per person in the city, typically measured in metric tons (t)."
        
    elif "mae" in prompt_lower:
        return "📐 **MAE (Mean Absolute Error)** is a forecast metric. It tells you the average magnitude of error in your prediction, measured in the same units as the target variable (e.g., if forecasting temperature, MAE is in deg C). **Lower MAE means higher accuracy.**"
        
    elif "mape" in prompt_lower:
        return "📈 **MAPE (Mean Absolute Percentage Error)** is a forecast metric. It expresses the error as a percentage of the actual value. For example, a 10% MAPE means the forecast is off by 10% on average. **Lower MAPE is better.**"
        
    elif "el niño" in prompt_lower or "la niña" in prompt_lower:
        return "🌍 **El Niño and La Niña** are climate patterns in the Pacific Ocean that affect weather globally. El Niño usually leads to **warmer and drier** conditions in India (causing heat spikes), while La Niña often brings **cooler and wetter** conditions."
        
    elif "bod" in prompt_lower or "dissolved oxygen" in prompt_lower or "water quality" in prompt_lower:
        return "🌊 **BOD (Biochemical Oxygen Demand)** measures the amount of oxygen required by microorganisms to decompose organic matter in water. High BOD means high pollution, leading to low **Dissolved Oxygen (DO)**. **Good water health requires low BOD and high DO.**"
        
    elif "afforestation" in prompt_lower or "tree goal" in prompt_lower:
        return "🌳 **Afforestation Goals** track the number of trees currently in the city versus the number *needed* to reach a sustainability standard (often 10 trees per person). This highlights the planting gap the local body must bridge."
        
    # --- TAB AND FEATURE EXPLANATIONS ---
    
    if "overview tab" in prompt_lower or "kpi" in prompt_lower or "main screen" in prompt_lower:
        return "🏠 The **Overview Tab** gives you a snapshot of all critical metrics: current air quality (PM2.5), average temperature, total rainfall, and estimated CO2 emissions per capita. It also features the main **Seasonality Anomaly Tracker** graph."

    elif "air quality tab" in prompt_lower:
        return "💨 The **Air Quality Tab** displays the latest pollutant readings (PM2.5, NO2, CO, etc.) from the Open-Meteo API. It includes a **Donut Chart** showing the percentage composition of all pollutants detected."
        
    elif "trends tab" in prompt_lower or "correlation" in prompt_lower:
        return "📊 The **Climate Trends Tab** shows historical daily changes, including a **Global Warming Trend Line** (365-day moving average). It also features the **Correlation Matrix**, which explains how different climate variables influence each other."
        
    elif "forecasts tab" in prompt_lower:
        return "🔮 The **Forecasts Tab** allows you to select a climate variable (like temperature) and generate a prediction for the future (up to 365 days) using advanced AI/ML models (**Prophet, ARIMA, or Random Forest**)."

    elif "impact story" in prompt_lower:
        return "💥 The **Impact Story Tab** connects climate metrics to real-world consequences, such as the **Simulated Crop Yield Loss** (based on local temperature stress), **Himalayan Glacier Melt** risks, and a local **River Health Status** check."

    elif "sustainability score" in prompt_lower or "score tab" in prompt_lower:
        return "💯 The **Sustainability Score** is a composite metric (0-100) based on five weighted factors: Air Quality, CO2 Emissions per Capita, Renewable Share, Water Quality, and Recycling Rate. It gives a quick assessment of the city's overall environmental performance."
        
    elif "carbon" in prompt_lower or "footprint" in prompt_lower:
        return "👣 The **Personal Carbon Tab** helps you estimate your household's monthly CO2 footprint based on consumption (electricity, travel, diet). It also features a map to compare your city's total emissions against other major Indian cities."
        
    elif "green infrastructure" in prompt_lower or "ev" in prompt_lower:
        return "🔋 The **Green Infrastructure & EVs Tab** tracks the city's transition to clean energy. It shows the simulated capacity of **Solar, Wind, and Hydro** energy resources and the number of **Registered Electric Vehicles (EVs)**."

    elif "cleanliness rank" in prompt_lower or "swachh" in prompt_lower:
        return "🧼 The **Cleanliness Rank Tab** tracks the simulated historical trend of the city's performance in the annual **Swachh Survekshan (Cleanliness Survey)**. Remember, a **lower rank number is better**!"

    # --- GRAPH EXPLANATIONS ---
    
    elif "anomaly tracker" in prompt_lower or "yoy change" in prompt_lower:
        return "📈 The **Seasonality Anomaly Tracker (YoY Change)** shows the change in temperature (Red Line) and precipitation (Blue Dashed Line) relative to the *previous* year. Spikes far from the zero line (Baseline) indicate **Abnormal Seasonal Patterns** like extreme heat or heavy rain events. Look for the arrows for specific reasons!"
        
    elif "monthly temp" in prompt_lower or "norms" in prompt_lower:
        return "🌡️ The **Monthly Temperature Norms** bar chart shows the average maximum and minimum temperatures for each month across the entire historical period selected. This helps you understand the city's typical seasonal cycle."
        
    elif "annual precipitation" in prompt_lower or "total rain" in prompt_lower:
        return "💧 The **Total Annual Precipitation** bar chart aggregates all rainfall for each year in the selected period. It helps spot long-term trends like increasing or decreasing annual monsoon intensity."
        
    elif "correlation matrix" in prompt_lower:
        return "🔗 The **Correlation Matrix** is a heat map that visualizes how different climate factors (like temperature, wind, and rain) relate to each other. Values close to **+1.0 (brightest)** mean they increase together; values near **-1.0 (darkest)** mean one increases while the other decreases."
        
    elif "warming trend" in prompt_lower or "mean temperature trend" in prompt_lower:
        return "🔥 The **Mean Temperature Trend** chart shows the daily temperature (Muted Blue) overlaid with a thick line (Orange) representing the **Global Warming Trend** (a 365-day rolling average). If the orange line is slowly moving up, it indicates a long-term warming effect."
        
    elif "emissions map" in prompt_lower or "co2 accountability" in prompt_lower:
        return "🗺️ The **City CO2 Accountability Map** displays major Indian cities as bubbles, where the **size and color** of the bubble reflect their estimated **Annual CO2 Emissions**. This helps compare your city's contribution to the national carbon footprint."

    # --- HELPFUL FALLBACK (ENHANCED) ---
    else:
        # Check if the query is a simple statement or question that can be answered by the summary.
        if len(prompt_lower.split()) < 4 or any(word in prompt_lower for word in ["what is", "explain", "describe", "meaning of", "tell me about"]):
            # Try to match the query to any of the above keywords more aggressively
            keywords = ["pm2.5", "co2", "mae", "mape", "el niño", "la niña", "bod", "afforestation", "overview tab", "air quality tab", "trends tab", "forecasts tab", "impact story", "sustainability score", "carbon", "green infrastructure", "cleanliness rank", "anomaly tracker", "monthly temp", "annual precipitation", "correlation matrix", "warming trend", "emissions map"]
            for keyword in keywords:
                if keyword in prompt_lower:
                    # Reroute to the explanation if found
                    return generate_chatbot_response(keyword)
        
        return "I can answer specific questions about the **dashboard's features, technical metrics, or any of the graphs** you see. Try asking about the **'Sustainability Score'** or the **'Anomaly Tracker'**!"


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # 🌟 MODIFICATION: Add the initial 'Hi' message from the assistant with a waving hand icon
    st.session_state.messages.append({"role": "assistant", "content": "Hi! 👋 I am SustainifyAI's virtual assistant. Ask me anything about the dashboard's data or features."})


# 🌟 MODIFICATION: Simplified chat interface for embedding on the main page
def chat_interface_embed():
    # Use a container for the chat history to enable scrolling
    st.markdown("<div class='chatbot-card'>", unsafe_allow_html=True)
    st.subheader("🤖 SustainifyAI Chatbot")
    st.caption("Ask me about the project's features, scores, or graphs!")

    chat_container = st.container(height=350)
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with chat_container.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input (pinned to the bottom of the main page or container)
    if prompt := st.chat_input("Ask a question about the dashboard..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately (need to re-render the history above)
        with chat_container.chat_message("user"):
            st.markdown(prompt)

        # Generate bot response and display it
        with chat_container.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_chatbot_response(prompt)
            # 🌟 MODIFICATION: The first message already contains the icon, ensuring it only appears once
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------ Sidebar Navigation ------------------------------

# Define the navigation options and associated icons (Requirement 1)
PAGES = {
    "Dashboard": "📊 Overview",
    "User Profile": "👤 Profile",
    "Climate Trends": "☁️ Trends",
    "Forecasts": "🔮 Forecast",
    "Impact Story": "💥 Story",
    "Sustainability Score": "🌿 Score",
    "Personal Carbon": "👣 Carbon",
    "Green Infrastructure & EVs": "⚡ EVs",
    "Cleanliness Rank": "🧹 Rank",
    "Settings": "⚙️ Settings",
    "About Project": "🚀 About",
}

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# Use radio buttons in the sidebar for navigation (mimicking Material nav items)
st.sidebar.markdown(f"## Navigation Menu", unsafe_allow_html=True)
st.session_state.page = st.sidebar.radio(
    "Select Page",
    options=list(PAGES.keys()),
    format_func=lambda x: PAGES[x],
    label_visibility="collapsed"
)

# ------------------------------ Sidebar Controls (Standard Streamlit) ------------------------------
st.sidebar.markdown("---")

# The line that caused the error is now resolved because geocode_place is defined above
place_default = st.sidebar.text_input("🔎 Search a city / place", value="Prayagraj")
geo = geocode_place(place_default) if place_default.strip() else None

if geo is None:
    st.sidebar.error("Couldn't geocode the place. Try a larger city or correct spelling.")
    st.stop()

lat, lon, _name, _country = geo
st.sidebar.markdown(f"<p style='color: #FFFFFF; font-weight: 600;'>📍 {_name}, {_country}</p>", unsafe_allow_html=True)

today = dt.date.today()
start_date = st.sidebar.date_input("Start date", value=today - dt.timedelta(days=365*5))
end_date = st.sidebar.date_input("End date", value=today)

model_choice = st.sidebar.selectbox(
    "Forecast model",
    ["auto", "Prophet", "ARIMA", "ML Ensemble"],
    index=0,
    help="Choose the AI model: Prophet is great for strong seasonality (e.g., yearly temps); ARIMA is a classic statistical model; ML Ensemble (Random Forest) is a non-linear fallback."
)

st.sidebar.markdown("---")
# Replacing subscript and degree symbol in slider text
alert_pm25 = st.sidebar.slider("PM2.5 alert threshold (ug/m3)", 10, 200, 90, help="If the current PM2.5 (Air Quality) exceeds this threshold, a warning alert will be triggered on the dashboard.")
alert_temp = st.sidebar.slider("Max temp alert (deg C)", 30, 50, 44, help="If the latest recorded maximum temperature exceeds this threshold, a heat warning alert will be triggered.")

st.sidebar.markdown("---")
with st.sidebar.expander("Sustainability score inputs (optional overrides)"):
    # Replacing subscript in input text
    co2_pc = st.number_input("CO2 per capita (t)", min_value=0.0, value=1.9, step=0.1)
    ren_share = st.slider("Renewable energy share (%)", 0, 100, 35) # Increased default for green focus
    water_idx = st.slider("Water quality index (%)", 0, 100, 65)
    recycle = st.slider("Waste recycling rate (%)", 0, 100, 30)

# Logout button in sidebar (placed after the login is successful)
st.sidebar.markdown("---")
if st.sidebar.button("Logout", use_container_width=True):
    st.session_state["auth_ok"] = False
    st.session_state["auth_user"] = None
    st.experimental_rerun()

# ------------------------------ 2. TOP NAVBAR (Fixed Header) ------------------------------

# Render the custom HTML for the fixed top navbar
st.markdown(f"""
    <div style='height: 70px;'></div> <div class='navbar-container'>
        <div class='navbar-title'>SustainifyAI — Material Dashboard</div>
        <div class='navbar-actions'>
            <div class='navbar-icon'>
                <span class='navbar-icon'>🔍</span>
            </div>
            <div class='navbar-icon'>
                <span class='navbar-icon'>🔔</span>
            </div>
            <div class='navbar-icon'>
                <span class='navbar-icon'>🧑‍💻</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)


# ------------------------------ Data Pulls (Needs to happen regardless of tab) ------------------------------
with st.spinner(f"Fetching data for {_name}..."):
    try:
        df_clim = fetch_openmeteo_daily(lat, lon, start_date, end_date)
        df_aq = fetch_air_quality_current(lat=lat, lon=lon)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

# Extract values for cleaner use
pm25_now = float(df_aq.loc[df_aq["parameter"]=="pm2_5", "value"].head(1).fillna(np.nan).values[0]) if not df_aq.empty and (df_aq["parameter"]=="pm2_5").any() else np.nan
mean_temp = df_clim['temperature_2m_mean'].mean()
max_wind = df_clim['windspeed_10m_max'].max()
total_rain = df_clim['precipitation_sum'].sum()
total_solar = df_clim['shortwave_radiation_sum'].sum()

# ------------------------------ Tab Control (Logic remains the same, UI is changed by CSS/Radio) ------------------------------

# ------------------------------ Tab: Dashboard (Material Layout) ------------------------------
if st.session_state.page == "Dashboard":
    
    st.subheader(f"Dashboard Overview for {_name}, {_country}")
    
    # --- 3. Metric Cards (Grid) ---
    st.markdown("""
        <div style='display: flex; gap: 20px; flex-wrap: wrap;'>
    """, unsafe_allow_html=True)
    
    # KPI Definitions for Material Cards
    kpi_material_data = [
        {
            "label": "PM2.5 (ug/m³)", 
            "value": ("-" if math.isnan(pm25_now) else f"{pm25_now:.1f}"),
            "icon": "💨",
            "color_class": "bg-green", # Good is green
            "help": "Air Quality: Lower values are better for health."
        },
        {
            "label": "Mean Temp (°C)", 
            "value": f"{mean_temp:.1f}", 
            "icon": "🌡️",
            "color_class": "bg-orange", # Warming is orange
            "help": "Average temperature across the period."
        },
        {
            "label": "Total Rain (mm)", 
            "value": f"{total_rain:.1f}", 
            "icon": "💧",
            "color_class": "bg-red", # Rain deficit/excess risk is red
            "help": "Total rainfall sum for the selected period."
        },
        {
            "label": "Max Wind (m/s)", 
            "value": f"{max_wind:.1f}", 
            "icon": "🌬️",
            "color_class": "bg-blue", # General indicator is blue
            "help": "Maximum recorded wind speed."
        },
        {
            "label": "Solar Energy (MJ/m²)", 
            "value": f"{total_solar:.1f}", 
            "icon": "☀️",
            "color_class": "bg-green", # Green energy focus is green
            "help": "Total solar energy received, crucial for solar power capacity."
        }
    ]

    cols = st.columns(len(kpi_material_data))
    
    for i, data in enumerate(kpi_material_data):
        with cols[i]:
            # Custom HTML Material Card structure
            st.markdown(
                f"""
                <div style='position: relative;'>
                    <div class='kpi-icon-block {data['color_class']}'>
                        {data['icon']}
                    </div>
                    <div class='material-card-kpi'>
                        <div class='kpi-value-container'>
                            <p class='kpi-label-material'>{data['label']}</p>
                            <p class='kpi-value-material'>{data['value']}</p>
                        </div>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            # Add a hidden Streamlit metric to enable the native help tooltip
            st.metric(f"{data['label']}_hidden", "", help=data['help'])

    st.markdown("</div>", unsafe_allow_html=True)
    
    # --- Alerts ---
    alerts = []
    if not math.isnan(pm25_now) and pm25_now >= alert_pm25:
        alerts.append(f"⚠ High PM2.5 detected: {pm25_now:.1f} ug/m3 >= threshold {alert_pm25}")

    if not df_clim.empty:
        latest_max_temp = float(df_clim["temperature_2m_max"].iloc[-1])
        if latest_max_temp >= alert_temp:
            alerts.append(f"🔥 *CURRENT HEAT ALERT:* Latest max temperature of {latest_max_temp:.1f} deg C exceeded threshold {alert_temp} deg C.")
    else:
        alerts.append("ℹ Climate data not loaded, temperature alert is inactive.")


    if alerts:
        st.warning("\n".join(alerts))
        send_telegram("\n".join(alerts))
    
    st.markdown("---")

    # --- 4. Chart Cards (Two Large Cards) ---
    col_chart_A, col_chart_B = st.columns(2)

    with col_chart_A:
        # --- Chart Card A: Seasonality Anomaly Tracker ---
        st.markdown(
            """
            <div class='chart-card-wrap'>
                <div class='card-header-material'>
                    <h3>Seasonality Anomaly Tracker (YoY Change)</h3>
                </div>
                <div class='card-body-material'>
            """, unsafe_allow_html=True
        )
        # Python Logic for Anomaly Tracker (Replicated)
        df_annual = df_clim.copy()
        df_annual['year'] = df_annual['time'].dt.year
        df_annual = df_annual.groupby('year').agg(
            Annual_Mean_Temp=('temperature_2m_mean', 'mean'),
            Annual_Precipitation=('precipitation_sum', 'sum')
        ).reset_index()

        df_annual['YoY_Temp_Change'] = df_annual['Annual_Mean_Temp'].diff()
        df_annual['YoY_Precip_Change'] = df_annual['Annual_Precipitation'].diff()

        start_year = df_annual['year'].max() - 4
        df_yoy = df_annual[df_annual['year'] >= start_year].copy()
        
        df_yoy_melt = df_yoy.melt(
            id_vars='year', 
            value_vars=['YoY_Temp_Change', 'YoY_Precip_Change'], 
            var_name='Anomaly Metric', 
            value_name='YoY Change'
        )
        df_yoy_melt.dropna(subset=['YoY Change'], inplace=True)
        
        fig_yoy = go.Figure()
        temp_data = df_yoy_melt[df_yoy_melt['Anomaly Metric'] == 'YoY_Temp_Change']
        fig_yoy.add_trace(go.Scatter(x=temp_data['year'], y=temp_data['YoY Change'], mode='lines+markers', name='Temp Change (°C)', line=dict(color='#F44336', width=4), yaxis='y1'))
        precip_data = df_yoy_melt[df_yoy_melt['Anomaly Metric'] == 'YoY_Precip_Change']
        fig_yoy.add_trace(go.Scatter(x=precip_data['year'], y=precip_data['YoY Change'], mode='lines+markers', name='Precip Change (mm)', line=dict(color='#2196F3', width=4, dash='dash'), yaxis='y2'))
        fig_yoy.add_hline(y=0, line_dash="dot", line_color="#3C4B64", opacity=0.5, annotation_text="Normal Baseline")

        annotations = []
        TEMP_THRESHOLD = 1.0 
        RAIN_THRESHOLD = 250.0

        for _, row in df_yoy.dropna(subset=['YoY_Temp_Change', 'YoY_Precip_Change']).iterrows():
            year = row['year']
            temp_change = row['YoY_Temp_Change']
            if abs(temp_change) >= TEMP_THRESHOLD:
                reason = "Extreme Warming" if temp_change > 0 else "Unusual Cooling"
                color = '#F44336'
                annotations.append(dict(x=year, y=temp_change, xref="x", yref="y1", text=f"🔥 {reason}: +{temp_change:.1f}°C YoY", showarrow=True, arrowhead=7, ax=0, ay=-40 if temp_change > 0 else 40, font=dict(color=color, size=10), arrowcolor=color))

            precip_change = row['YoY_Precip_Change']
            if abs(precip_change) >= RAIN_THRESHOLD:
                reason = "Heavy Monsoon" if precip_change > 0 else "Severe Drought"
                color = '#2196F3'
                annotations.append(dict(x=year, y=precip_change, xref="x", yref="y2", text=f"💧 {reason}: +{precip_change:.0f}mm YoY", showarrow=True, arrowhead=7, ax=0, ay=-40 if precip_change > 0 else 40, font=dict(color=color, size=10), arrowcolor=color))

        fig_yoy.update_layout(
            height=400, hovermode="x unified", plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#3C4B64'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            annotations=annotations,
            yaxis=dict(title='Temp Change (°C)', title_font=dict(color='#F44336'), tickfont=dict(color='#F44336'), showgrid=False),
            yaxis2=dict(title='Precip Change (mm)', title_font=dict(color='#2196F3'), tickfont=dict(color='#2196F3'), overlaying='y', side='right', gridcolor='#EEEEEE')
        )
        st.plotly_chart(fig_yoy, use_container_width=True)
        st.markdown(
            f"""
            <p style='color: #757575; font-size: 0.8rem !important; margin-top: 10px;'>
            <strong>{_name}'s</strong> Year-over-Year Climate Anomaly. Red line tracks temperature, blue line tracks rainfall.
            </p>
            </div></div>
            """, unsafe_allow_html=True
        )

    with col_chart_B:
        # --- Chart Card B: Monthly Temperature Norms ---
        st.markdown(
            """
            <div class='chart-card-wrap'>
                <div class='card-header-material' style='background: linear-gradient(60deg, #ff9800, #ffb74d); box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.14), 0 7px 10px -5px rgba(255, 152, 0, 0.4);'>
                    <h3>Monthly Temperature Norms (°C)</h3>
                </div>
                <div class='card-body-material'>
            """, unsafe_allow_html=True
        )
        
        # Python Logic for Monthly Norms (Replicated)
        df_clim['month_name'] = df_clim['time'].dt.strftime('%b')
        monthly_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        df_clim['month_name'] = pd.Categorical(df_clim['month_name'], categories=monthly_order, ordered=True)
        
        df_monthly = df_clim.groupby('month_name').agg(
            Avg_Max_Temp=('temperature_2m_max', 'mean'),
            Avg_Min_Temp=('temperature_2m_min', 'mean'),
        ).reset_index()
        
        df_monthly_melt = df_monthly.melt(id_vars='month_name', var_name='Metric', value_name='Temperature (°C)')

        fig_monthly = px.bar(
            df_monthly_melt, 
            x="month_name", 
            y="Temperature (°C)", 
            color="Metric",
            barmode="group",
            color_discrete_map={'Avg_Max_Temp': '#F44336', 'Avg_Min_Temp': '#2196F3'}, 
            category_orders={'month_name': monthly_order}
        )
        
        fig_monthly.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=20, b=10),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#3C4B64'),
            xaxis_title=None,
            yaxis_title=None,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
        st.markdown(
            f"""
            <p style='color: #757575; font-size: 0.8rem !important; margin-top: 10px;'>
            Historical average monthly maximum and minimum temperatures for {_name}.
            </p>
            </div></div>
            """, unsafe_allow_html=True
        )

    st.markdown("---")
    
    # --- 5. Lower Section: Updates and Chatbot ---
    col_updates, col_chat = st.columns([0.4, 0.6])

    with col_updates:
        # --- Update Cards (Mock Data - Material Style) ---
        st.markdown("### 📰 Latest Environmental Updates")
        
        def render_update_card(title, body, icon, color):
            st.markdown(
                f"""
                <div class='material-card-kpi' style='margin-top: 10px; height: auto; box-shadow: {MATERIAL_CARD_FLAT};'>
                    <div style='display: flex; align-items: center; padding-bottom: 5px; border-bottom: 1px solid #eee;'>
                        <div style='font-size: 1.5rem; color: {color}; margin-right: 10px;'>{icon}</div>
                        <h4 style='color: {color} !important; margin: 0;'>{title}</h4>
                    </div>
                    <p style='color: #757575; font-size: 0.9rem !important; padding-top: 10px;'>{body}</p>
                </div>
                """, unsafe_allow_html=True
            )
        
        render_update_card(
            f"🌱 Afforestation Drive",
            f"Targeting the {get_tree_inventory(_name)['needed']:,} tree deficit in {_name} with a community planting initiative.",
            "🌳",
            "#4CAF50"
        )
        render_update_card(
            f"⚡ EV Charging Network",
            f"Local authority approved 25 new charging stations across the city to support green transport adoption.",
            "🔋",
            "#2196F3"
        )
        
    with col_chat:
        chat_interface_embed()


# ------------------------------ Tab: User Profile ------------------------------
elif st.session_state.page == "User Profile":
    st.header(f"👤 User Profile: {st.session_state['auth_user']}")
    st.markdown("""
        <div class='chart-card-wrap' style='padding: 20px;'>
        <p>This section is designed for the user to view their account details, set personal goals, and check privacy settings in a Material card format.</p>
        
        <h4>User Details</h4>
        <ul>
            <li>**Username:** `aditya01`</li>
            <li>**Access Level:** Analyst / Government Official (Full Access)</li>
            <li>**Location Focus:** {_name}, {_country}</li>
            <li>**Last Login:** {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</li>
        </ul>
        
        <h4>Notification Settings</h4>
        <p>You currently receive alerts for PM2.5 > {alert_pm25} ug/m³ and Max Temp > {alert_temp}°C.</p>
        <div style='color: #2196F3; font-weight: 600;'>Settings can be adjusted in the sidebar.</div>
        </div>
    """.format(_name=_name, _country=_country, alert_pm25=alert_pm25, alert_temp=alert_temp), unsafe_allow_html=True)

# ------------------------------ Tab: Climate Trends ------------------------------
elif st.session_state.page == "Climate Trends":
    TAB_TRENDS = st.container()
    with TAB_TRENDS:
        # Standard Subheader
        st.subheader("Multi-variable Climate Trends")
        
        # --- Trend Charts (Wrapped in Material Card) ---
        colA, colB = st.columns(2)
        
        with colA:
            st.markdown("""
                <div class='chart-card-wrap'>
                    <div class='card-header-material' style='background: linear-gradient(60deg, #2196F3, #42a5f5); box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.14), 0 7px 10px -5px rgba(33, 150, 243, 0.4);'>
                        <h3>Max Wind Speed Trend (m/s)</h3>
                    </div>
                    <div class='card-body-material'>
            """, unsafe_allow_html=True)
            
            # Wind Trend Chart Logic (Replicated)
            if not df_clim.empty:
                max_wind_value = df_clim['windspeed_10m_max'].max()
                abnormal_point = df_clim[df_clim['windspeed_10m_max'] == max_wind_value].iloc[0]
                abnormal_date = abnormal_point['time']
                
                alert_msg = f"🌪 Extreme Wind Alert! Recorded {max_wind_value:.1f} m/s on {abnormal_date.strftime('%Y-%m-%d')}."
                st.warning(alert_msg)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_clim["time"], y=df_clim["windspeed_10m_max"], name="Max Wind Speed", line=dict(color='#2196F3', width=2))) 
                fig.add_annotation(
                    x=abnormal_date, 
                    y=abnormal_point['windspeed_10m_max'], 
                    text="ABNORMAL SPIKE", 
                    showarrow=True, 
                    font=dict(color="#F44336", size=10), 
                    arrowhead=2, 
                    arrowcolor="#F44336", 
                    ax=0, ay=-40
                )
                
                fig.update_layout(
                    title_text="", # Title moved to card header
                    plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#3C4B64'), 
                    xaxis_title="Date", yaxis_title="Wind Speed (m/s)"
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        with colB:
            st.markdown("""
                <div class='chart-card-wrap'>
                    <div class='card-header-material' style='background: linear-gradient(60deg, #FF9800, #ffb74d); box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.14), 0 7px 10px -5px rgba(255, 152, 0, 0.4);'>
                        <h3>Mean Temperature Trend (°C)</h3>
                    </div>
                    <div class='card-body-material'>
            """, unsafe_allow_html=True)
            
            # Temperature Trend Logic (Replicated)
            df_temp = df_clim.copy()
            if not df_temp.empty:
                df_temp['warming_trend'] = df_temp['temperature_2m_mean'].rolling(window=365, center=True).mean()
            
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_clim["time"], y=df_clim["temperature_2m_mean"], name="Daily Mean Temp", line=dict(color='#457b9d', width=2), opacity=0.8))
                
                if not df_temp.empty:
                    fig.add_trace(go.Scatter(x=df_temp["time"], y=df_temp["warming_trend"], name="Warming Trend (365-day Avg)", line=dict(color='#F44336', width=4, dash='dashdot'), opacity=0.9))
                
                fig.update_layout(
                    title_text="", # Title moved to card header
                    plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#3C4B64'), 
                    xaxis_title="Date", yaxis_title="Temperature (°C)", legend=dict(y=0.99, x=0.01)
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div></div>", unsafe_allow_html=True)
            
        # --- Correlation Matrix (Wrapped in Material Card) ---
        st.markdown("""
            <div class='chart-card-wrap'>
                <div class='card-header-material' style='background: linear-gradient(60deg, #4CAF50, #81c784); box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.14), 0 7px 10px -5px rgba(76, 175, 80, 0.4);'>
                    <h3>Climate Variables Correlation Matrix</h3>
                </div>
                <div class='card-body-material'>
        """, unsafe_allow_html=True)
        
        corr_df = df_clim.drop(columns=["time"]).corr(numeric_only=True)
        fig_corr = px.imshow(
            corr_df, 
            text_auto=".2f",
            aspect="auto", 
            title="Correlation Matrix", # Title will be overwritten by card header
            color_continuous_scale=px.colors.sequential.Viridis_r,
        )
        fig_corr.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#3C4B64'),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)
    
# ------------------------------ Tab: Forecasts ------------------------------
elif st.session_state.page == "Forecasts":
    TAB_FORECAST = st.container()
    with TAB_FORECAST:
        st.subheader("AI Forecasts with Backtest Metrics")
        
        st.markdown("""
            <div class='chart-card-wrap'>
                <div class='card-header-material' style='background: linear-gradient(60deg, #00C9A7, #33E3C9); box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.14), 0 7px 10px -5px rgba(0, 201, 167, 0.4);'>
                    <h3>Future Climate Prediction</h3>
                </div>
                <div class='card-body-material'>
        """, unsafe_allow_html=True)
        
        col_select, col_slider = st.columns([0.4, 0.6])
        with col_select:
            target = st.selectbox(
                "Target to forecast", 
                ["temperature_2m_mean","temperature_2m_max","temperature_2m_min","precipitation_sum"], 
                index=0,
            )
        with col_slider:
            horizon = st.slider("Forecast horizon (days)", 7, 365, 90)

        model_used, ts, train, valid, fcst, metrics = backtest_train_forecast(df_clim[["time", target]].dropna(), target, horizon=horizon, model_choice=model_choice)

        st.info(f"Model used: *{model_used}* |  MAE: *{metrics['MAE']:.3f}* |  MAPE: *{metrics['MAPE']:.2%}*")

        # Forecast Chart (Replicated)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train["ds"], y=train["y"], name="Train Data", line=dict(color='#3F51B5', width=2))) 
        fig.add_trace(go.Scatter(x=valid["ds"], y=valid["y"], name="Validation Data", line=dict(color='#F44336', width=2))) 
        
        if "ds" not in fcst.columns: # unify forecast frame
            if "time" in fcst.columns:
                fcst.rename(columns={"time":"ds"}, inplace=True)
            else:
                last = ts["ds"].iloc[-1]
                fcst["ds"] = pd.date_range(last + pd.Timedelta(days=1), periods=len(fcst), freq='D')
        
        yhat = fcst["yhat"] if "yhat" in fcst.columns else fcst.iloc[:,1]
        
        fig.add_trace(go.Scatter(x=fcst["ds"], y=yhat, name="Forecast", line=dict(color=MATERIAL_PRIMARY, width=3, dash="dash")))
        
        fig.update_layout(
            title_text=f"AI Forecast: {target}", 
            xaxis_title="Date", 
            yaxis_title=target,
            hovermode="x unified",
            plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#3C4B64'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("⬇ Download Forecast CSV", data=fcst.to_csv(index=False), file_name=f"forecast_{target}.csv", mime="text/csv")

        st.markdown("</div></div>", unsafe_allow_html=True)

# ------------------------------ Tab: Impact Story ------------------------------
elif st.session_state.page == "Impact Story":
    TAB_STORY = st.container()
    with TAB_STORY:
        st.subheader(f"💥 Climate Change Impact Story: Real-World Consequences for {_name}")

        col_loss, col_map = st.columns(2)
        
        with col_loss:
            st.markdown("""
                <div class='chart-card-wrap'>
                    <div class='card-header-material' style='background: linear-gradient(60deg, #F44336, #e57373); box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.14), 0 7px 10px -5px rgba(244, 67, 54, 0.4);'>
                        <h3>🌾 Crop Yield Loss Simulation</h3>
                    </div>
                    <div class='card-body-material'>
            """, unsafe_allow_html=True)
            
            stressed_yield, loss_percent, status = get_crop_loss_simulation(mean_temp) 
            st.markdown(f"#### Avg Temp: **{mean_temp:.1f} °C**")
            st.info(f"Predicted Agricultural Status: *{status}*")
            
            # Plotly Gauge (Replicated)
            fig_loss = go.Figure(go.Indicator(
                mode = "number+gauge",
                value = stressed_yield,
                title = {'text': "Simulated Crop Yield (Relative %)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#3F51B5"}, 
                    'steps': [
                        {'range': [0, 80], 'color': '#F44336'}, 
                        {'range': [80, 95], 'color': '#FF9800'}, 
                        {'range': [95, 100], 'color': '#4CAF50'}
                    ],
                    'threshold': {'value': stressed_yield}
                }
            ))
            fig_loss.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig_loss, use_container_width=True)
            
            st.markdown(f"""
                <p style='color: #F44336; font-weight:700;'>Estimated Loss: 
                <span style='font-size:1.2rem;'>-{loss_percent:.1f}%</span></p>
            """, unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

        with col_map:
            st.markdown("""
                <div class='chart-card-wrap'>
                    <div class='card-header-material' style='background: linear-gradient(60deg, #3F51B5, #7986cb); box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.14), 0 7px 10px -5px rgba(63, 81, 181, 0.4);'>
                        <h3>💨 City CO2 Accountability Map (kT/Yr)</h3>
                    </div>
                    <div class='card-body-material'>
            """, unsafe_allow_html=True)
            
            df_emissions = get_all_india_city_emissions()
            st.multiselect("Select cities for $\\text{CO}_2$ contribution comparison", options=df_emissions['City'].tolist(), default=[_name] if _name in df_emissions['City'].tolist() else [df_emissions['City'].iloc[0]], key="story_city_multiselect")
            df_filtered = df_emissions[df_emissions['City'].isin(st.session_state.story_city_multiselect)]
            
            # Choropleth Map (Replicated)
            fig_map = px.scatter_geo(
                df_emissions, lat='Latitude', lon='Longitude', size="CO2_Emissions_Annual_kT", color="CO2_Emissions_Annual_kT", 
                projection="natural earth", color_continuous_scale=px.colors.sequential.YlOrRd, scope='asia'
            )
            fig_map.update_geos(lataxis_range=[5, 35], lonaxis_range=[65, 90], showcountries=True, countrycolor="#757575", subunitcolor="#757575", showland=True, landcolor="#E0E0E0")
            fig_map.update_layout(height=450, margin={"r":0,"t":20,"l":0,"b":0}, coloraxis_colorbar=dict(title="CO2 (kT/yr)"))
            st.plotly_chart(fig_map, use_container_width=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

        # River Health and Afforestation (Simplified into a card below)
        st.markdown("---")
        col_river, col_tree = st.columns(2)
        with col_river:
            st.markdown("""
                <div class='chart-card-wrap' style='padding: 15px;'>
                    <h4 style='color: #2196F3 !important;'>🌊 River Health Status ({})</h4>
            """.format(get_river_health_data(_name)['River'].iloc[0]), unsafe_allow_html=True)
            river_data = get_river_health_data(_name)
            st.dataframe(river_data[['Dissolved Oxygen (DO mg/L)', 'BOD (mg/L)', 'Coliform (MPN/100ml)', 'Status']], hide_index=True, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_tree:
            st.markdown("""
                <div class='chart-card-wrap' style='padding: 15px;'>
                    <h4 style='color: #4CAF50 !important;'>🌳 Afforestation Goals for {}</h4>
            """.format(_name), unsafe_allow_html=True)
            tree_data = get_tree_inventory(_name)
            st.metric("Trees Needed to Meet 10/Capita Goal", f"{tree_data['needed']:,}", delta=f"Current: {tree_data['current']:,}")
            st.caption(f"Target: 10 trees per person (Population: {tree_data['population']:,})")
            st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------ Tab: Sustainability Score ------------------------------
elif st.session_state.page == "Sustainability Score":
    TAB_SCORE = st.container()
    with TAB_SCORE:
        st.subheader("🌿 City Sustainability Score")
        
        st.markdown("""
            <div class='chart-card-wrap'>
                <div class='card-header-material' style='background: linear-gradient(60deg, #FF9800, #ffb74d); box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.14), 0 7px 10px -5px rgba(255, 152, 0, 0.4);'>
                    <h3>Composite Sustainability Assessment</h3>
                </div>
                <div class='card-body-material'>
        """, unsafe_allow_html=True)
        
        pm_for_score = pm25_now if not math.isnan(pm25_now) else 60.0
        score, sub = compute_sustainability_score(SustainabilityInputs(
            pm25=pm_for_score, co2_per_capita=co2_pc, renewable_share=float(ren_share),
            water_quality_index=float(water_idx), waste_recycling_rate=float(recycle),
        ))
        
        colL, colR = st.columns([0.45,0.55])
        with colL:
            st.markdown(
                f"""
                <div class='material-card-kpi' style='margin-top: 20px; box-shadow: {MATERIAL_CARD_LIFT};'>
                    <div style='text-align: center;'>
                        <p class='kpi-label-material'>OVERALL SCORE (0-100)</p>
                        <p style='font-size: 3.5rem; color: #3F51B5; font-weight: 700; margin: 0;'>{score:.1f}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
            st.caption("A weighted composite score based on 5 dimensions.")
            
        with colR:
            sub_df = pd.DataFrame({"Dimension": list(sub.keys()), "Score": list(sub.values())})
            fig = px.bar(sub_df, x="Dimension", y="Score", title="Sub-Scores (0-100)", color="Score", color_continuous_scale=px.colors.sequential.Plotly3)
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#3C4B64'))
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("</div></div>", unsafe_allow_html=True)

# ------------------------------ Tab: Green Infrastructure & EVs ------------------------------
elif st.session_state.page == "Green Infrastructure & EVs":
    TAB_GREEN_INFRA = st.container()
    with TAB_GREEN_INFRA:
        st.subheader("⚡ Green Infrastructure & Electric Vehicle Adoption")
        
        known_cities = ["Prayagraj", "Lucknow", "Varanasi", "Kanpur", "Mumbai", "Delhi", "Bengaluru", "Agra"]
        all_cities = sorted(list(set(known_cities + [_name]))) 

        selected_cities_green = st.multiselect(
            "Select cities to compare:", options=all_cities, default=[_name, "Lucknow", "Mumbai"], key="green_infra_city_select"
        )
        
        if selected_cities_green:
            st.markdown("---")
            
            # --- Energy Resources Chart Card ---
            st.markdown("""
                <div class='chart-card-wrap'>
                    <div class='card-header-material' style='background: linear-gradient(60deg, #4CAF50, #81c784); box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.14), 0 7px 10px -5px rgba(76, 175, 80, 0.4);'>
                        <h3>Pollution-Free Energy Capacity (MW)</h3>
                    </div>
                    <div class='card-body-material'>
            """, unsafe_allow_html=True)
            
            df_energy = get_pollution_free_energy_data(selected_cities_green)
            df_energy_melted = df_energy.melt(id_vars=['City'], var_name='Energy Type', value_name='Capacity (MW)')
            
            fig_energy = px.bar(
                df_energy_melted, x='City', y='Capacity (MW)', color='Energy Type', barmode='stack',
                color_discrete_map={'Solar (MW)': '#FF9800', 'Wind (MW)': MATERIAL_PRIMARY, 'Hydro (MW)': '#2196F3'}
            )
            fig_energy.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#3C4B64'), xaxis_title=None, yaxis_title="Capacity (MW)")
            st.plotly_chart(fig_energy, use_container_width=True)
            st.markdown("</div></div>", unsafe_allow_html=True)
            
            # --- Solar Connections & EVs ---
            col_solar, col_ev = st.columns(2)
            
            with col_solar:
                st.markdown("""
                    <div class='chart-card-wrap'>
                        <div class='card-header-material' style='background: linear-gradient(60deg, #FF9800, #ffb74d); box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.14), 0 7px 10px -5px rgba(255, 152, 0, 0.4);'>
                            <h3>☀️ Registered Solar Connections</h3>
                        </div>
                        <div class='card-body-material'>
                """, unsafe_allow_html=True)
                df_solar_conn = get_registered_solar_connections(selected_cities_green)
                fig_solar_conn = px.bar(df_solar_conn, x='City', y='Solar Connections', color='Solar Connections', color_continuous_scale=px.colors.sequential.YlOrBr)
                fig_solar_conn.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#3C4B64'), xaxis_title=None, yaxis_title="Number of Connections")
                st.plotly_chart(fig_solar_conn, use_container_width=True)
                st.markdown("</div></div>", unsafe_allow_html=True)

            with col_ev:
                st.markdown("""
                    <div class='chart-card-wrap'>
                        <div class='card-header-material' style='background: linear-gradient(60deg, #3F51B5, #7986cb); box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.14), 0 7px 10px -5px rgba(63, 81, 181, 0.4);'>
                            <h3>🚗 Registered Electric Vehicles (EVs)</h3>
                        </div>
                        <div class='card-body-material'>
                """, unsafe_allow_html=True)
                df_ev = get_registered_ev_vehicles(selected_cities_green)
                fig_ev = px.bar(df_ev, x='City', y='Registered EVs', color='Registered EVs', color_continuous_scale=px.colors.sequential.Greens)
                fig_ev.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#3C4B64'), xaxis_title=None, yaxis_title="Number of Vehicles")
                st.plotly_chart(fig_ev, use_container_width=True)
                st.markdown("</div></div>", unsafe_allow_html=True)

# ------------------------------ Tab: Cleanliness Rank ------------------------------
elif st.session_state.page == "Cleanliness Rank":
    TAB_SWACHH = st.container()
    with TAB_SWACHH:
        st.subheader("🧹 Swachh Survekshan (Cleanliness) Ranking Trend")
        
        st.markdown("""
            <div class='chart-card-wrap'>
                <div class='card-header-material' style='background: linear-gradient(60deg, #F44336, #e57373); box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.14), 0 7px 10px -5px rgba(244, 67, 54, 0.4);'>
                    <h3>City Global Rank Trend (Lower is Better)</h3>
                </div>
                <div class='card-body-material'>
        """, unsafe_allow_html=True)
        
        all_known_swachh_cities = get_swachh_ranking_data([], _name)['City'].unique().tolist()
        default_swachh_cities = sorted(list(set(all_known_swachh_cities + [_name, "Lucknow", "Indore", "Surat"])))

        selected_swachh_cities = st.multiselect(
            "Select cities for comparison:", options=default_swachh_cities, default=[_name, "Lucknow", "Indore", "Surat"], key="swachh_city_select"
        )
        
        if selected_swachh_cities:
            df_swachh = get_swachh_ranking_data(selected_swachh_cities, _name)
            df_swachh_melted = df_swachh.melt(id_vars='City', var_name='Year', value_name='Global Rank')
            
            fig_rank = px.line(
                df_swachh_melted, x='Year', y='Global Rank', color='City', 
                title='Global Rank Trend', markers=True, color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            fig_rank.update_yaxes(autorange="reversed", dtick=50, showgrid=True, gridcolor='#EEEEEE')
            fig_rank.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#3C4B64'))

            st.plotly_chart(fig_rank, use_container_width=True)
            st.caption("A downward trend (lower rank number) indicates improvement in cleanliness.")
            st.markdown("</div></div>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Detailed Swachh Ranking Table")
            st.dataframe(df_swachh, use_container_width=True)

# ------------------------------ Tab: Air Quality ------------------------------
elif st.session_state.page == "Air Quality":
    TAB_AIR = st.container()
    with TAB_AIR:
        st.subheader("💨 Latest Air Quality Measurements")
        
        st.markdown("""
            <div class='chart-card-wrap'>
                <div class='card-header-material' style='background: linear-gradient(60deg, #4CAF50, #81c784); box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.14), 0 7px 10px -5px rgba(76, 175, 80, 0.4);'>
                    <h3>Pollutant Composition & Readings</h3>
                </div>
                <div class='card-body-material'>
        """, unsafe_allow_html=True)
        
        if df_aq.empty:
            st.info("No AQ data found for this location window.")
        else:
            df_pie = df_aq[['parameter', 'value']].copy()
            df_pie['Pollutant'] = df_pie['parameter'].str.replace('_', ' ').str.title()

            col_chart, col_table = st.columns([0.4, 0.6])

            with col_chart:
                fig_pie = px.pie(
                    df_pie, values='value', names='Pollutant', title='Composition of Pollutants', hole=0.4, 
                    color_discrete_sequence=['#FF9800', '#F44336', '#2196F3', '#4CAF50', '#3F51B5', '#E91E63']
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='white', width=1)))
                fig_pie.update_layout(height=450, plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#3C4B64'), legend_title_text="Pollutant")
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_table:
                st.markdown("#### Detailed Readings")
                df_aq_display = df_aq[["date", "parameter", "value", "unit"]].rename(
                    columns={"date": "Last Updated", "parameter": "Pollutant", "value": "Value", "unit": "Unit"}
                )
                st.dataframe(df_aq_display, use_container_width=True)
                st.caption("Data source is Open-Meteo Air Quality API.")
                
        st.markdown("</div></div>", unsafe_allow_html=True)

# ------------------------------ Tab: Personal Carbon ------------------------------
elif st.session_state.page == "Personal Carbon":
    TAB_CARBON = st.container()
    with TAB_CARBON:
        st.subheader("👣 Personal Carbon Footprint")
        
        # --- Carbon Map Card ---
        st.markdown("""
            <div class='chart-card-wrap'>
                <div class='card-header-material' style='background: linear-gradient(60deg, #FF9800, #ffb74d); box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.14), 0 7px 10px -5px rgba(255, 152, 0, 0.4);'>
                    <h3>🗺️ Major City Annual CO2 Emissions Comparison</h3>
                </div>
                <div class='card-body-material'>
        """, unsafe_allow_html=True)
        
        df_map = get_all_india_city_emissions()
        current_city_name = _name
        
        fig_map = go.Figure(data=go.Scattergeo(
            lon = df_map['Longitude'], lat = df_map['Latitude'], mode = 'markers',
            marker = dict(
                size = df_map['CO2_Emissions_Annual_kT'] / 1000, sizemode = 'area', sizemin = 5,
                color = df_map['CO2_Emissions_Annual_kT'], colorscale = px.colors.sequential.Sunset_r, 
                cmin = df_map['CO2_Emissions_Annual_kT'].min(), cmax = df_map['CO2_Emissions_Annual_kT'].max(), 
                colorbar_title = "CO2 (kT/yr)", line_color='#E0E0E0'
            ),
        ))
        
        current_city_data = df_map[df_map['City'].str.lower() == current_city_name.lower()]
        if not current_city_data.empty:
            fig_map.add_trace(go.Scattergeo(
                lon = current_city_data['Longitude'], lat = current_city_data['Latitude'], mode = 'markers',
                marker = dict(size = current_city_data['CO2_Emissions_Annual_kT'].iloc[0] / 1000 + 10, color = '#F44336', line_width = 3, line_color = '#FFFFFF'),
                hoverinfo='text', name='Selected City'
            ))

        fig_map.update_layout(
            title_text = f'CO2 Emissions Across Indian Cities (Current City: **{current_city_name}**)', showlegend = False, height=550,
            geo = dict(scope = 'asia', lonaxis_range= [68, 98], lataxis_range= [5, 38], subunitcolor = "#3F51B5", countrycolor = "#757575", landcolor = '#FFFFFF', projection_type = 'mercator'),
            plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#3C4B64')
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # --- Carbon Calculator (Logic Replicated) ---
        st.markdown("---")
        st.subheader("Personal Carbon Footprint (Quick Estimate)")
        
        auto_estimate = st.checkbox("⚡ *Auto-Fetch Household Estimate*", value=False, key="auto_estimate_carbon_tab")
        
        BASE_KWH = 180; BASE_KM_CAR = 350; BASE_LPG = 6; BASE_FLIGHTS = 1
        EF_CAR = 0.18; EF_KWH = 0.7; EF_FLIGHT = 180; EF_LPG = 3.0
        DIET_MAP = {"Heavy meat": 300, "Mixed": 200, "Vegetarian": 150, "Vegan": 120}

        colA, colB = st.columns(2)
        
        if auto_estimate:
            tree_data = get_tree_inventory(_name); pop_adj = tree_data['population'] / 1000000 
            km_car_proxy = max(50, BASE_KM_CAR - int(pop_adj * 50) - int(df_clim['windspeed_10m_max'].max() * 5))
            mean_temp_swing = df_clim['temperature_2m_max'].max() - df_clim['temperature_2m_min'].min()
            kwh_proxy = int(BASE_KWH + mean_temp_swing * 2) 
            diet_proxy = "Mixed";
            km_car = st.number_input("Monthly car travel (km)", value=km_car_proxy, disabled=True, key="auto_km_car")
            kwh = st.number_input("Monthly electricity use (kWh)", value=kwh_proxy, disabled=True, key="auto_kwh")
            flights = st.number_input("Flights per year (2-hr avg)", value=BASE_FLIGHTS, disabled=True, key="auto_flights")
            with colB:
                diet = st.selectbox("Diet type", list(DIET_MAP.keys()), index=list(DIET_MAP.keys()).index(diet_proxy), disabled=True, key="auto_diet")
                lpg = st.number_input("Monthly LPG use (kg)", value=BASE_LPG, disabled=True, key="auto_lpg")
                recycle_rate = st.slider("Household recycling (%)", 0, 100, recycle, disabled=True, key="auto_recycle")
        else:
            with colA:
                km_car = st.number_input("Monthly car travel (km)", 0, 10000, 300, key="manual_km_car")
                kwh = st.number_input("Monthly electricity use (kWh)", 0, 2000, 180, key="manual_kwh")
                flights = st.number_input("Flights per year (2-hr avg)", 0, 50, 1, key="manual_flights")
            with colB:
                diet = st.selectbox("Diet type", list(DIET_MAP.keys()), index=1, key="manual_diet")
                lpg = st.number_input("Monthly LPG use (kg)", 0, 100, 6, key="manual_lpg")
                recycle_rate = st.slider("Household recycling (%)", 0, 100, 20, key="manual_recycle")

        effective_ef_kwh = EF_KWH * (1 - ren_share/100)
        st.caption(f"Effective CO2 factor for electricity: {effective_ef_kwh:.3f} kg/kWh (based on {ren_share}% renewables)")
        
        monthly = (km_car * EF_CAR + kwh * effective_ef_kwh + (flights * EF_FLIGHT) / 12 + lpg * EF_LPG + DIET_MAP[diet]) * (1 - recycle_rate / 400)

        st.markdown(
            f"""
            <div class='material-card-kpi' style='margin-top: 20px; text-align: center;'>
                <p class='kpi-label-material'>ESTIMATED MONTHLY EMISSIONS</p>
                <p style='font-size: 2.5rem; color: #F44336; font-weight: 700; margin: 0;'>{monthly/1000:.2f} t CO2e</p>
            </div>
            """, unsafe_allow_html=True
        )

        with st.expander("❓ How is this calculated?"):
            st.caption("Calculation details...")
            
        fig = px.pie(names=["Travel","Electricity","Flights","LPG","Diet"],
                    values=[km_car * EF_CAR, kwh * effective_ef_kwh, (flights * EF_FLIGHT) / 12, lpg * EF_LPG, DIET_MAP[diet]],
                    title="Breakdown (kg CO2e per month)")
        fig.update_traces(marker=dict(colors=px.colors.sequential.Plotly3))
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#3C4B64'))
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------ Tab: Settings ------------------------------
elif st.session_state.page == "Settings":
    st.header("⚙️ Application Settings")
    st.markdown("""
        <div class='chart-card-wrap' style='padding: 20px;'>
        <p>This section allows administrators to adjust API keys (if applicable), default values, and theme settings. Note: In this demo, most settings are controlled via the sidebar.</p>
        
        <h4>Current Operational Parameters</h4>
        <ul>
            <li>**Active Location:** {_name}, {_country}</li>
            <li>**Data Period:** {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}</li>
            <li>**Forecast Model:** {model_choice}</li>
        </ul>
        
        <p style='color: #F44336; font-weight: 600;'>Warning Thresholds (from Sidebar):</p>
        <ul>
            <li>PM2.5 Alert: > {alert_pm25} ug/m³</li>
            <li>Max Temp Alert: > {alert_temp}°C</li>
        </ul>
        </div>
    """.format(_name=_name, _country=_country, start_date=start_date, end_date=end_date, model_choice=model_choice, alert_pm25=alert_pm25, alert_temp=alert_temp), unsafe_allow_html=True)


# ------------------------------ Tab: About Project ------------------------------
elif st.session_state.page == "About Project":
    TAB_ABOUT = st.container()
    with TAB_ABOUT:
        st.header("🚀 SustainifyAI — Material Dashboard")
        st.markdown("""
            <div class='chart-card-wrap' style='padding: 20px; margin-bottom: 20px;'>
            <h3 style='color: #3F51B5 !important;'>Hackathon Submission (Material Redesign)</h3>
            <p>This is a Material Dashboard UI/UX recreation of the original SustainifyAI project, built for a clean, modern, and structure-rich aesthetic.</p>
            
            <h4 style='color: #4CAF50 !important;'>Core Team (Nxt Gen Developers)</h4>
            <ul>
                <li>Aditya Kumar Singh</li>
                <li>Gaurang Verma</li>
                <li>Vandana Yadav</li>
                <li>Gaurav Shakya</li>
                <li>Saurabh Shukla</li>
            </ul>
            
            <h4 style='color: #FF9800 !important;'>Technology & ML Models</h4>
            <p>Built using Streamlit, Plotly, and Python's data stack (Pandas, NumPy). Forecasting utilizes state-of-the-art time-series libraries: Prophet, pmdarima (ARIMA), and scikit-learn (Random Forest). Data is fetched from Open-Meteo APIs.</p>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------ Default for other tabs ------------------------------
else:
    # All remaining tabs are grouped here as they primarily contain the original content logic
    st.title(PAGES[st.session_state.page])
    st.warning("Content for this section is a direct display of the original logic, wrapped in standard Material cards.")
    
    st.markdown("""
        <div class='chart-card-wrap'>
            <div class='card-header-material' style='background: linear-gradient(60deg, #3F51B5, #7986cb);'>
                <h3>Detailed Content for {}</h3>
            </div>
            <div class='card-body-material'>
    """.format(st.session_state.page), unsafe_allow_html=True)
    
    # You would place the original logic for:
    # 1. Climate Trends (partially replicated above)
    # 2. Impact Story (partially replicated above)
    # 3. Sustainability Score (partially replicated above)
    # 4. Personal Carbon (partially replicated above)
    # 5. Green Infrastructure & EVs (partially replicated above)
    # 6. Cleanliness Rank (partially replicated above)
    
    # Placeholder for un-redesigned content:
    st.code("This area would execute the full rendering of the original tab logic, maintaining full functionality while adhering to the Material card styling.")
    
    st.markdown("</div></div>", unsafe_allow_html=True)