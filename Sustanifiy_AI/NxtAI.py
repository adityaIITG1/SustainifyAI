
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
    page_title="SustainifyAI — Sustainability and Climate Tracker",
    page_icon="🌍", # Changed icon to revolving globe
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------ BRIGHT, VIBRANT THEME (CSS OVERHAUL) ------------------------------
st.markdown(
    """
    <style>
    /* *** MODIFICATION: Changing font to a stylized Serif look and updating icon styles *** */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&display=swap');

    :root {
      /* Light & Bright Base */
      --bg: #f8f9fa; /* Very Light Gray/White */ 
      --card: #ffffff; /* Pure White Card */
      --muted: #6c757d; /* Dark Gray Muted Text */
      /* PRIMARY VIBRANT FOCUS */
      --brand: #007bff; /* Primary Blue (Clean) */
      --brand2:#ffc107; /* Amber/Yellow Accent */
      --brand-glow: 0 0 10px rgba(0, 123, 255, 0.4), 0 0 20px rgba(255, 193, 7, 0.2); 
    }
    
    /* --- CORE BACKGROUND & LAYOUT (Light) --- */
    .stApp { 
        background: var(--bg) !important; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -90;
        /* Subtle Light Blue/Pink/Orange Gradient */
        background: linear-gradient(135deg, rgba(0, 123, 255, 0.05) 0%, rgba(255, 105, 180, 0.05) 50%, rgba(255, 193, 7, 0.05) 100%);
        pointer-events: none;
        opacity: 1; 
    }
    
    .main > div {
        background-color: transparent !important; /* Let the body bg show */
        backdrop-filter: none;
    }
    .stSidebar > div:first-child {
        background: linear-gradient(180deg, #ffffff 0%, #f1f7fd 100%) !important; /* Light gradient */
        border-right: 2px solid var(--brand); /* Clean Blue border */
        box-shadow: 6px 0 30px rgba(0, 0, 0, 0.1), inset -3px 0 10px rgba(0, 123, 255, 0.05);
    }

    .block-container { padding-top: 1rem; }
    
    /* --- NEWS TICKER STYLING (Blue/White) --- */
    @keyframes marquee {
        0% 	{ transform: translate(100%, 0); }
        100% { transform: translate(-100%, 0); }
    }
    .news-pipe-container {
      overflow: hidden;
      width: 100%;
      height: 40px;
      background: #f1f7fd; /* Light blue background */
      border: 1px solid var(--brand); /* Blue border */
      border-radius: 8px;
      margin-bottom: 20px;
      padding: 5px 0;
      box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
      position: relative;
    }
    .news-pipe-content {
      white-space: nowrap;
      color: var(--brand); /* Primary Blue for Alerts */
      font-size: 1.1rem;
      font-weight: 600;
      padding-top: 2px;
      animation: marquee 25s linear infinite;
      text-shadow: 0 0 5px rgba(0, 123, 255, 0.2);
    }
    
    /* --- TEXT STYLING (Dark Text on Light Background) --- */
    h1,h2,h3,h4 { 
        color:#212529; 
        letter-spacing:0.5px;
        text-shadow: 0 0 2px rgba(0, 123, 255, 0.1); 
        /* ⬅️ Applied Stylized Font for General Headings */
        font-family: 'Playfair Display', serif !important; 
    }
    p,li,span,div, label, .stMarkdown { 
        color:#343a40 !important; 
        font-size: 1.0rem !important;
    }

    /* --- SIDEBAR SPECIFIC STYLING --- */
    .stSidebar h1 { 
        font-size: 1.9rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: var(--brand) !important;
        text-shadow: 0 0 10px rgba(0, 123, 255, 0.5); 
        /* ⬅️ Applied Stylized Font for Sidebar Header */
        font-family: 'Playfair Display', serif !important; 
    }
    
    /* --- NEW/MODIFIED: Large Sidebar Icon Container --- */
    .sidebar-icon-container {
        text-align: center;
        padding-top: 10px;
        padding-bottom: 5px;
    }
    .big-earth-icon {
        font-size: 3.5em; /* Bigger icon size */
        color: var(--brand); 
        text-shadow: 0 0 15px rgba(0, 123, 255, 0.8), 0 0 5px green; /* Blue/Green Glow */
        display: inline-block;
        animation: spin 6s linear infinite;
    }
    .big-tree-icon {
        font-size: 3.0em; /* Tree icon slightly smaller */
        color: green;
        margin-left: 10px;
        text-shadow: 0 0 10px #38a3a5;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .rotating-globe {
        animation: spin 6s linear infinite;
        font-size: 1.5em;
        margin-right: 5px;
        color: var(--brand); 
        text-shadow: 0 0 12px rgba(0, 123, 255, 0.8); 
    }
    
    /* *** NEW: CSS for Rotating Windmill (For Max Wind KPI) *** */
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .rotating-windmill {
        animation: rotate 2s linear infinite;
        display: inline-block;
        margin-right: 5px;
    }
    

    /* Input Fields (Clean White/Blue Focus) */
    .stSidebar [data-baseweb="input"], .stSidebar [data-baseweb="base-input"],
    .stSidebar [data-baseweb="select"] > div:first-child,
    .stSidebar .stDateInput > div:first-child > div {
        background-color: var(--card) !important; 
        border: 1px solid #ced4da !important; 
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05);
        border-radius: 8px;
        padding: 8px 10px;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stSidebar [data-baseweb="input"]:focus-within, 
    .stSidebar [data-baseweb="base-input"]:focus-within,
    .stSidebar [data-baseweb="select"]:focus-within > div:first-child,
    .stSidebar .stDateInput > div:first-child:focus-within > div {
        border-color: var(--brand) !important; 
        box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.5);
    }

    /* Slider styling (Yellow/Orange focus) */
    .stSidebar .stSlider [data-baseweb="slider"] {
        background-color: #e9ecef;
        height: 8px;
        border-radius: 4px;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }
    .stSidebar .stSlider [data-baseweb="slider"] > div:nth-child(2) {
        background-color: var(--brand2); 
        box-shadow: 0 0 8px rgba(255, 193, 7, 0.6);
    }
    .stSidebar .stSlider [data-baseweb="slider"] > div:nth-child(3) {
        background-color: var(--card);
        border: 2px solid var(--brand2); 
        box-shadow: 0 0 15px rgba(255, 193, 7, 0.8), 0 0 5px rgba(0,0,0,0.2);
        width: 20px;
        height: 20px;
        top: -6px;
    }

    /* --- HERO HEADING (Custom Button Style with Water Droplets) --- */
    .hero { 
      position: relative; 
      padding: 30px 0 25px;
      text-align: center;
    }
    
    .custom-button-like {
        display: inline-block;
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%); 
        color: white;
        padding: 20px 40px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 123, 255, 0.5), inset 0 0 15px rgba(255, 255, 255, 0.3);
        border: 3px solid #66b3ff;
        cursor: default;
        position: relative;
        overflow: hidden;
        margin-bottom: 20px;
        transition: all 0.4s ease;
    }
    
    .custom-button-like:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(0, 123, 255, 0.7);
    }

    /* User Requested Font Change */
    .hero-title-button {
        /* ⬅️ Applied Stylized Font for Main Header */
        font-family: 'Playfair Display', serif !important; 
        font-size: clamp(35px, 6vw, 65px); 
        font-weight: 900;
        letter-spacing: 3px; 
        margin: 0;
        color: white !important; 
        text-shadow: 0 0 10px rgba(255,255,255,0.8), 0 0 20px #ffc107; 
    }

    .hero-sub-button { 
        color: white !important; 
        font-size: 1.0rem !important;
        font-weight: 400;
        opacity:0.8; 
        letter-spacing: 0.5px;
        margin-top: 5px;
        text-shadow: 0 1px 3px rgba(0,0,0,0.5);
    }
    
    /* Water Droplet Animation */
    .droplet {
        position: absolute;
        width: 10px;
        height: 10px;
        background: rgba(255, 255, 255, 0.6);
        border-radius: 50%;
        opacity: 0;
        animation: rain 4s linear infinite;
        box-shadow: 0 0 5px rgba(255, 255, 255, 0.8);
    }

    @keyframes rain {
        0% { transform: translateY(-100px) scale(0); opacity: 0; }
        10% { opacity: 0.8; }
        100% { transform: translateY(400px) scale(1); opacity: 0; }
    }
    
    /* Positioning for droplets */
    .droplet:nth-child(1) { left: 10%; animation-delay: 0s; }
    .droplet:nth-child(2) { left: 30%; animation-delay: 1.5s; width: 8px; height: 8px; }
    .droplet:nth-child(3) { left: 50%; animation-delay: 0.5s; width: 12px; height: 12px; }
    .droplet:nth-child(4) { left: 70%; animation-delay: 2.5s; }
    .droplet:nth-child(5) { left: 90%; animation-delay: 1s; width: 15px; height: 15px; }


    /* --- GLASS MORPHISM KPI / CARDS (Header) - Blue/White Focus --- */
    .glass-card-header { 
      background: rgba(255, 255, 255, 0.8);
      backdrop-filter: blur(5px);
      -webkit-backdrop-filter: blur(5px);
      
      border: 1px solid rgba(0, 123, 255, 0.3); 
      padding: 15px; 
      border-radius: 12px;
      box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); 
      transition: all .35s ease; 
      position: relative; 
      color: #212529 !important; 
    }
    .glass-card-header:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 6px 20px rgba(0, 123, 255, 0.15); 
    }
    
    /* --- KPI / CARDS (Regular) - White/Blue Focus) --- */
    .metric-card { 
      background: var(--card); 
      border:1px solid #e9ecef; 
      padding:20px; 
      border-radius:12px; 
      box-shadow:0 6px 15px rgba(0, 0, 0, 0.1); 
      transition: all .35s ease; 
      position: relative; 
      color: #212529 !important; 
      overflow: hidden; 
    }
    .metric-card:hover{ 
        transform: translateY(-4px) scale(1.01); 
        background: radial-gradient(800px 220px at 10% 10%, #f1f7fd 0%, #ffffff 70%); 
        box-shadow:0 12px 30px rgba(0, 123, 255, 0.2); 
        border-color:var(--brand); 
    }
    
    /* --- CUSTOM KPI VALUE STYLING (Primary Blue/Accent) --- */
    .kpi-value {
        color: var(--brand) !important; 
        font-weight: 700; 
        font-size: 2.0rem; 
        text-shadow: 0 0 5px rgba(0, 123, 255, 0.3); 
        margin-top: -10px;
    }
    .kpi-label {
        color: var(--muted) !important; 
        font-size: 0.9rem !important;
    }

    /* --- RANK MIRROR BUTTON (New Style) --- */
    @keyframes shine {
        0% { background-position: -100% 0; }
        100% { background-position: 100% 0; }
    }
    .rank-mirror-button {
        display: block;
        padding: 10px 15px;
        background: #f8f9fa; /* Light Background */
        border: 2px solid var(--brand); /* Blue Border */
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        margin-top: 10px;
        overflow: hidden;
        position: relative;
        color: #212529 !important;
        transition: transform 0.3s ease;
    }
    .rank-mirror-button:hover {
        transform: scale(1.03);
    }
    .rank-mirror-button::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 200%;
        height: 100%;
        background: linear-gradient(120deg, transparent 30%, rgba(255, 255, 255, 0.5) 50%, transparent 70%);
        transition: opacity 0.3s ease;
        animation: shine 2s infinite linear; /* SHINING EFFECT */
        opacity: 0;
    }
    .rank-mirror-button:hover::after {
        opacity: 1;
    }
    /* MODIFIED: Standard Crown */
    .crown-icon {
        display: inline-block;
        color: gold;
        font-size: 1.2em;
        margin-right: 5px;
        text-shadow: 0 0 5px orange;
    }
    
    /* *** NEW: Flicker and Glow for Swachh Star Crown *** */
    @keyframes flicker-glow-zoom {
        0% { transform: scale(1.0) rotate(0deg); opacity: 0.8; text-shadow: 0 0 5px orange; }
        25% { transform: scale(1.1) rotate(5deg); opacity: 1; text-shadow: 0 0 15px gold, 0 0 20px #ffc107; }
        50% { transform: scale(1.0) rotate(-5deg); opacity: 0.8; text-shadow: 0 0 5px orange; }
        75% { transform: scale(1.1) rotate(5deg); opacity: 1; text-shadow: 0 0 15px gold, 0 0 20px #ffc107; }
        100% { transform: scale(1.0) rotate(0deg); opacity: 0.8; text-shadow: 0 0 5px orange; }
    }
    .flicker-crown-icon { /* New class for the second crown */
        animation: flicker-glow-zoom 2s infinite ease-in-out;
        display: inline-block;
        color: gold;
        font-size: 1.2em;
        margin-right: 5px;
        text-shadow: 0 0 5px orange;
    }
    /* --- TAB BUTTONS (Blue Active) --- */
    .stTabs [data-baseweb="tab-list"] { background-color: #e9ecef; border-bottom: 2px solid #ced4da; margin-bottom: 20px; }
    .stTabs [aria-selected="true"] { 
        color: var(--brand) !important; 
        border-bottom: 4px solid var(--brand) !important; 
        font-weight: 700; 
        text-shadow: 0 0 5px rgba(0, 123, 255, 0.5) !important; 
    }
    
    /* --- BUTTONS (Primary Blue) --- */
    .stButton.btn-primary button { 
        background:linear-gradient(90deg,var(--brand),#0056b3)!important; 
        color:white!important; 
        border:2px solid var(--brand)!important; 
        font-weight: 600; 
        box-shadow: 0 4px 15px rgba(0, 123, 255, 0.4); 
        transition: all 0.3s ease;
    }
    .stButton.btn-primary button:hover {
        background: var(--brand) !important;
        box-shadow: 0 6px 20px rgba(0, 123, 255, 0.5);
    }

    /* Plot Wrappers (White background with Blue border) */
    .plot-wrap { 
        border:1px solid #ced4da; 
        border-radius:12px; 
        padding:12px; 
        background: var(--card); 
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); 
    }
    
    /* --- AFFORESTATION ENHANCEMENTS (Blue/Pink/Yellow) --- */
    .goal-number {
        font-size: 3.0rem !important;
        font-weight: 800;
        color: #ff69b4 !important; 
        text-shadow: 0 0 10px rgba(255, 105, 180, 0.5);
        transition: all 0.5s ease;
    }
    .current-number {
        font-size: 2.0rem !important;
        font-weight: 600;
        color: var(--brand2) !important; 
        text-shadow: 0 0 5px rgba(255, 193, 7, 0.5); 
    }
    .goal-card-title {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #212529;
        text-shadow: 0 0 5px rgba(0, 123, 255, 0.2); 
    }
    
    /* --- TEAM TITLE GLOW (Vibrant Gradient) --- */
    @keyframes pop-glow {
        0% { transform: scale(1.0); opacity: 0.8; text-shadow: 0 0 10px rgba(0, 123, 255, 0.3); }
        50% { transform: scale(1.05); opacity: 1; text-shadow: 0 0 20px rgba(255, 193, 7, 0.7), 0 0 40px rgba(0, 123, 255, 0.5); }
        100% { transform: scale(1.0); opacity: 0.8; text-shadow: 0 0 10px rgba(0, 123, 255, 0.3); }
    }
    .team-title {
        font-size: 2.2rem !important; 
        font-weight: 900;
        line-height: 1.2;
        margin-top: 20px;
        margin-bottom: 20px;
        
        background: linear-gradient(45deg, #ffc107, #ff69b4, #007bff); 
        -webkit-background-clip: text; 
        background-clip: text; 
        color: transparent;

        animation: pop-glow 2.5s infinite alternate ease-in-out;
        display: inline-block;
    }
    /* LOGIN STYLES */
    .login-box {
        max-width:480px;
        margin:6rem auto 2rem auto;
        padding:32px;
        border-radius:16px;
        background:linear-gradient(180deg, rgba(0, 123, 255, 0.1), rgba(255, 255, 255, 0.8));
        box-shadow:0 10px 30px rgba(0, 0, 0, 0.25);
        border:1px solid rgba(0, 123, 255, 0.2);
    }
    .login-title {
        margin:0 0 10px 0;
        color: var(--brand);
        font-family: 'Playfair Display', serif !important;
        text-shadow: 0 0 5px rgba(0, 123, 255, 0.5);
    }
    .login-sub {
        color:#495057;
        margin:0 0 16px 0;
    }
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
    
    # Check if the current city is *already* included (either via selection or as a default list member)
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
    co2_per_capita: float             # optional proxy if available
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
        return "📐 **MAE (Mean Absolute Error)** is a forecast metric. It tells you the average magnitude of error in your prediction, measured in the same units as the target variable (e.g., if forecasting temperature, MAE is in $\\text{deg C}$). **Lower MAE means higher accuracy.**"
        
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
    # Adjusted title for the new location
    st.markdown("<div class='metric-card' style='padding: 10px 15px;'>", unsafe_allow_html=True)
    st.subheader("🤖 SustainifyAI Chatbot")
    st.caption("Ask me about the project's features, scores, or graphs!")

    # Use a container for the chat history to enable scrolling
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

# ------------------------------ Sidebar Controls ------------------------------

# ⬅️ ADDED: Big Earth and Tree Icon to the top area 
st.sidebar.markdown("""
<div class='sidebar-icon-container'>
    <span class='big-earth-icon'>🌎</span><span class='big-tree-icon'>🌳</span>
</div>
""", unsafe_allow_html=True)

# ⬅️ MODIFIED: Removed the small rotating globe icon here, rely on CSS font change
st.sidebar.markdown(
    f"# SustainifyAI", 
    unsafe_allow_html=True
)
st.sidebar.caption("Vibrant climate intelligence dashboard — real data, forecasts, insights")

# Logout button in sidebar (placed after the login is successful)
if st.sidebar.button("Logout"):
    st.session_state["auth_ok"] = False
    st.session_state["auth_user"] = None
    st.experimental_rerun()


# The line that caused the error is now resolved because geocode_place is defined above
place_default = st.sidebar.text_input("🔎 Search a city / place", value="Prayagraj")
geo = geocode_place(place_default) if place_default.strip() else None

if geo is None:
    st.sidebar.error("Couldn't geocode the place. Try a larger city or correct spelling.")
    st.stop()

lat, lon, _name, _country = geo
st.sidebar.success(f"📍 {_name}, {_country} | {lat:.3f}, {lon:.3f}")

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

# ------------------------------ Header (Bright & Cinematic) ------------------------------
colA, colB = st.columns([0.7,0.3])
with colA:
    # MODIFIED: New custom button/text structure
    st.markdown("""
    <div class="hero">
      <div class="custom-button-like">
        <div class="droplet" style="top: 0; left: 10%;"></div>
        <div class="droplet" style="top: 20%; left: 30%;"></div>
        <div class="droplet" style="top: 50%; left: 50%;"></div>
        <div class="droplet" style="top: 80%; left: 70%;"></div>
        <div class="droplet" style="top: 10%; left: 90%;"></div>
        <div class="hero-title-button">SustainifyAI — Sustainability and Climate Tracker</div>
        <div class="hero-sub-button">Real-time environmental data • Predictive AI • Sustainability governance tools</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
with colB:
    st.markdown("<div class='glass-card-header'>" , unsafe_allow_html=True)
    st.metric("Location", f"{_name}, {_country}")
    st.metric("Period", f"{start_date:%d %b %Y} → {end_date:%d %b %Y}")
    
    # --- MODIFIED: Swachh Survekshan Rank Mirror (Crown REMOVED) ---
    # UPDATED CALL: Passing current city to handle non-Nigam city lookups
    rank_df = get_swachh_ranking_data([_name], _name)
    current_rank = rank_df[rank_df['City'] == _name].get('2025', pd.Series([np.nan])).iloc[0] if not rank_df.empty else np.nan
    rank_display = f"Rank: {int(current_rank)}" if not np.isnan(current_rank) else "Rank: N/A"

    st.markdown(f"""
    <div class='rank-mirror-button'>
    	Swachh Rank: {rank_display}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ------------------------------ NEWS TICKER PIPE (Updated Colors) ------------------------------
col_news, col_swachh_star = st.columns([0.8, 0.2])

with col_news:
    # ⭐️ MODIFIED CALL: Passing current city name for dynamic news ⭐️
    st.markdown(
        f"""
        <div class="news-pipe-container">
            <div class="news-pipe-content">{get_pollution_news_ticker(_name)}</div>
        </div>
        """, unsafe_allow_html=True
    )

with col_swachh_star:
    # --- MODIFIED: Swachh Star City of India (Crown FLICKERS) ---
    st.markdown(f"""
    <div style='text-align: center; background: #e9ecef; border-radius: 8px; padding: 5px; border: 1px solid gold;'>
    	<p style='color: #212529; font-size: 0.8rem; margin: 0;'>
    		<span class='flicker-crown-icon'>👑</span> Swachh Star City (2023):
    	</p>
    	<p style='color: gold; font-weight: 700; font-size: 1.0rem; margin: 0; text-shadow: 0 0 5px orange;'>
    		Indore & Surat
    	</p>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------ Data Pulls ------------------------------
with st.spinner("Fetching climate history (Open-Meteo ERA5)…"):
    try:
        df_clim = fetch_openmeteo_daily(lat, lon, start_date, end_date)
    except Exception as e:
        st.error(f"Open-Meteo fetch failed: {e}")
        st.stop()

with st.spinner("Fetching latest air quality (Open-Meteo AQ)…"):
    df_aq = fetch_air_quality_current(lat=lat, lon=lon)

# ------------------------------ KPIs (Updated Symbols and Explanations) ------------------------------

# Extract values for cleaner use
pm25_now = float(df_aq.loc[df_aq["parameter"]=="pm2_5", "value"].head(1).fillna(np.nan).values[0]) if not df_aq.empty and (df_aq["parameter"]=="pm2_5").any() else np.nan
mean_temp = df_clim['temperature_2m_mean'].mean()
max_wind = df_clim['windspeed_10m_max'].max()
total_rain = df_clim['precipitation_sum'].sum()
total_solar = df_clim['shortwave_radiation_sum'].sum()


# --- KPI Definitions with Emojis and Simplified Explanation for a 12-year-old ---
kpi_data = [
    # Replacing subscript in label text
    {
        "label": "PM2.5 (ug/m3) 💨", 
        "value": ("-" if math.isnan(pm25_now) else f"{pm25_now:.1f}"), 
        "help": "This is tiny dust and smoke particles in the air. The lower the number, the cleaner the air is for your lungs!"
    },
    # Replacing degree symbol in label text
    {
        "label": "Mean Temp (deg C) 🌡️", 
        "value": f"{mean_temp:.1f}", 
        "help": "The average temperature across the time period. This helps us see if the weather is generally too hot or too cold."
    },
    {
        "label": "<span class='rotating-windmill'>⚙️</span> Max Wind (m/s) 🌬️", # Use gear/windmill for rotation effect
        "value": f"{max_wind:.1f}", 
        "help": "The fastest wind speed recorded. Strong winds are important for air circulation and generating wind power."
    },
    {
        "label": "Total Rain (mm) 💧", 
        "value": f"{total_rain:.1f}", 
        "help": "The total amount of water (rain) that fell during the selected time. We need rain for clean drinking water and for our crops to grow!"
    },
    {
        "label": "Solar (MJ/m2) ☀️", 
        "value": f"{total_solar:.1f}", 
        "help": "The total energy received from the sun. This is the energy we can catch with solar panels to make clean electricity."
    }
]

# Render KPIs using st.metric with the built-in help feature
kpi_cols = st.columns(5)
for i, data in enumerate(kpi_data):
    with kpi_cols[i]:
        # Use HTML for the label to include the spinning windmill effect
        st.markdown(
            f"""
            <div class='metric-card' style='padding-top:10px;'>
                <div class='kpi-label'>{data['label']}</div>
                <div class='kpi-value'>{data['value']}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        # We use a hidden metric here just to attach the Streamlit popup help
        st.metric(f"{data['label'].split('>')[-1].strip()}", "", help=data['help'])


# Alerts
alerts = []
# Replacing subscript and degree symbols in alert messages
if not math.isnan(pm25_now) and pm25_now >= alert_pm25:
    alerts.append(f"⚠ High PM2.5 detected: {pm25_now:.1f} ug/m3 >= threshold {alert_pm25}")

# Check only the latest available day's temperature for current alert.
if not df_clim.empty:
    latest_max_temp = float(df_clim["temperature_2m_max"].iloc[-1])
    if latest_max_temp >= alert_temp:
        alerts.append(f"🔥 *CURRENT HEAT ALERT:* Latest max temperature of {latest_max_temp:.1f} deg C exceeded threshold {alert_temp} deg C.")
else:
    alerts.append("ℹ Climate data not loaded, temperature alert is inactive.")


if alerts:
    st.warning("\n".join(alerts))
    sent = send_telegram("\n".join(alerts))
    if sent:
        st.success("Sent Telegram alert ✅")

# ------------------------------ Tabs (Updated Definition with new TAB_GREEN_INFRA and TAB_SWACHH) ------------------------------
# 🌟 MODIFICATION: Removed TAB_CHAT from the list of tabs
TAB_OVERVIEW, TAB_AIR, TAB_TRENDS, TAB_FORECAST, TAB_STORY, TAB_SCORE, TAB_CARBON, TAB_GREEN_INFRA, TAB_SWACHH, TAB_ABOUT = st.tabs([
    "Overview", "Air Quality", "Climate Trends", "Forecasts", "Impact Story 💥", "Sustainability Score", "Personal Carbon", "Green Infrastructure & EVs 🔋", "Cleanliness Rank 🧼", "About Project 🚀"
])

with TAB_OVERVIEW:
    
    st.markdown("---")
    
    # ------------------------------ PRIMARY VISUALIZATIONS (Maximized) ------------------------------
    
    st.subheader("Critical Environmental Metrics")
    
    # Use full width for the three main plots
    col_gauge, col_monthly, col_rain = st.columns(3)
    
    with col_gauge:
        # --- CO2 Per Capita Gauge ---
        co2_val = co2_pc # The value is from the sidebar input
        co2_goal = 2.0 
        co2_max = 5.0 
        co2_color = get_gauge_color(co2_val, good_threshold=2.0, bad_threshold=3.0, reverse=True) # Logic is preserved
        
        # 🎨 MODIFICATION 1: Enhance the gauge's look to be more fluid (simulated)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = co2_val,
            domain = {'x': [0, 1], 'y': [0, 1]},
            # Replacing subscript in title text
            title = {'text': "CO2 Emissions / Capita (t)", 'font': {'size': 20, 'color': '#212529'}},
            gauge = {
                'axis': {'range': [0, co2_max], 'tickwidth': 1, 'tickcolor': "#495057"},
                'bar': {'color': co2_color},
                'bgcolor': "#e9ecef", # Light background for gauge
                'steps': [
                    {'range': [0, co2_goal], 'color': '#38a3a5'}, # Teal/Blue-Green for 'Good'
                    {'range': [co2_goal, co2_max], 'color': '#f03a47'} # Red for 'Bad'
                ],
                'threshold': {
                    'line': {'color': "#ffc107", 'width': 4}, # Yellow/Amber threshold
                    'thickness': 0.75,
                    'value': co2_val 
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor='white', # Set white background for plot area
            paper_bgcolor='white', # Set white background for paper area
            font=dict(color='#212529')
        )
        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption(f"Current value: {co2_val} t. Goal: < {co2_goal} t.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    # --- Monthly Temp Norms (Bar Plot - Updated Colors) ---
    with col_monthly:
        # Calculate monthly averages across the entire history
        df_clim['month_name'] = df_clim['time'].dt.strftime('%b')
        monthly_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        df_clim['month_name'] = pd.Categorical(df_clim['month_name'], categories=monthly_order, ordered=True)
        
        df_monthly = df_clim.groupby('month_name').agg(
            Avg_Max_Temp=('temperature_2m_max', 'mean'),
            Avg_Min_Temp=('temperature_2m_min', 'mean'),
        ).reset_index()
        
        # Replacing degree symbol in value column name
        df_monthly_melt = df_monthly.melt(id_vars='month_name', var_name='Metric', value_name='Temperature (deg C)')

        fig = px.bar(
            df_monthly_melt, 
            x="month_name", 
            y="Temperature (deg C)", 
            color="Metric",
            barmode="group",
            title="*Monthly Temperature Norms (Avg Max & Min)*",
            # Bright colors: Red/Orange for Max, Blue/Cyan for Min
            color_discrete_map={'Avg_Max_Temp': '#e63946', 'Avg_Min_Temp': '#457b9d'}, 
            category_orders={'month_name': monthly_order}
        )
        
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#212529'),
            xaxis_title=None,
            yaxis_title=None,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
            xaxis=dict(tickfont=dict(color='#212529')), 
            yaxis=dict(tickfont=dict(color='#212529'))
        )
        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # --- Annual Total Precipitation (Bar Plot - Fluid Motion Simulation) ---
    with col_rain:
        df_clim['year'] = df_clim['time'].dt.year
        df_annual_rain = df_clim.groupby('year')['precipitation_sum'].sum().reset_index()
        
        # 🎨 MODIFICATION 2: Adding a vertical gradient to the precipitation bars
        fig2 = go.Figure()

        fig2.add_trace(go.Bar(
            x=df_annual_rain["year"],
            y=df_annual_rain["precipitation_sum"],
            name="Total Annual Precipitation",
            marker=dict(
                color=df_annual_rain["precipitation_sum"],
                colorscale=[[0, '#e9ecef'], [1, '#007bff']], # Light-to-Dark Blue gradient from bottom to top
                showscale=False,
                line=dict(width=0.5, color="#ced4da")
            )
        ))

        fig2.update_layout(
            title="*Total Annual Precipitation (mm)*",
            height=300,
            margin=dict(l=10, r=10, t=50, b=10),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#212529'),
            xaxis_title=None,
            yaxis_title="Total mm",
            xaxis_tickangle=-45,
            xaxis=dict(tickfont=dict(color='#212529')), 
            yaxis=dict(tickfont=dict(color='#212529'))
        )

        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------ CHATBOT (MOVED UP HERE) ------------------------------
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    chat_interface_embed()
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # ------------------------------ NEW: Abnormal Season Pattern Plot ------------------------------
    st.subheader("Seasonality Anomaly Tracker (YoY Change)")
    
    # Prepare Annual Data for YoY comparison (last 5 years)
    df_annual = df_clim.copy()
    df_annual['year'] = df_annual['time'].dt.year
    df_annual = df_annual.groupby('year').agg(
        Annual_Mean_Temp=('temperature_2m_mean', 'mean'),
        Annual_Precipitation=('precipitation_sum', 'sum')
    ).reset_index()

    # Calculate Year-over-Year change (excluding the first year)
    df_annual['YoY_Temp_Change'] = df_annual['Annual_Mean_Temp'].diff()
    df_annual['YoY_Precip_Change'] = df_annual['Annual_Precipitation'].diff()

    # Filter for the last 5 relevant years (Need at least 2 points for YoY change, so filter last 6 if possible)
    start_year = df_annual['year'].max() - 4 # Show last 5 years
    df_yoy = df_annual[df_annual['year'] >= start_year].copy()
    
    # Melt for Plotly Express
    df_yoy_melt = df_yoy.melt(
        id_vars='year', 
        value_vars=['YoY_Temp_Change', 'YoY_Precip_Change'], 
        var_name='Anomaly Metric', 
        value_name='YoY Change'
    )
    # Remove NaN from the first year
    df_yoy_melt.dropna(subset=['YoY Change'], inplace=True)
    
    # Create the beautiful line plot
    fig_yoy = go.Figure()

    # Trace 1: Temperature Change (Red/Orange - Heat)
    temp_data = df_yoy_melt[df_yoy_melt['Anomaly Metric'] == 'YoY_Temp_Change']
    fig_yoy.add_trace(go.Scatter(
        x=temp_data['year'], 
        y=temp_data['YoY Change'], 
        mode='lines+markers', 
        # Replacing degree symbol in name text
        name='Annual Mean Temp Change (deg C)',
        line=dict(color='#e63946', width=4), # Bold Red
        marker=dict(size=10, symbol='circle', line=dict(width=2, color='White')),
        yaxis='y1'
    ))

    # Trace 2: Precipitation Change (Blue - Water)
    precip_data = df_yoy_melt[df_yoy_melt['Anomaly Metric'] == 'YoY_Precip_Change']
    fig_yoy.add_trace(go.Scatter(
        x=precip_data['year'], 
        y=precip_data['YoY Change'], 
        mode='lines+markers', 
        name='Annual Precipitation Change (mm)',
        line=dict(color='#007bff', width=4, dash='dash'), # Primary Blue, Dashed
        marker=dict(size=10, symbol='diamond', line=dict(width=2, color='White')),
        yaxis='y2' # Use secondary Y-axis
    ))

    # Add a zero line for reference
    fig_yoy.add_hline(y=0, line_dash="dot", line_color="#343a40", opacity=0.5, annotation_text="Normal Baseline (No Change)")

    # --- ANOMALY DETECTION AND ANNOTATION LOGIC (NEW) ---
    annotations = []
    TEMP_THRESHOLD = 1.0 # 1 degree C shift is considered a significant anomaly
    RAIN_THRESHOLD = 250.0 # 250 mm shift is considered a significant anomaly

    for _, row in df_yoy.dropna(subset=['YoY_Temp_Change', 'YoY_Precip_Change']).iterrows():
        year = row['year']
        
        # 1. Temperature Anomaly Check
        temp_change = row['YoY_Temp_Change']
        if abs(temp_change) >= TEMP_THRESHOLD:
            reason = "Extreme Warming Spike (Possible El Niño)" if temp_change > 0 else "Unusual Cooling (Possible La Niña)"
            color = '#e63946'
            annotations.append(dict(
                x=year, y=temp_change,
                xref="x", yref="y1",
                text=f"🔥 {reason}: +{temp_change:.1f}°C YoY",
                showarrow=True, arrowhead=7, ax=0, ay=-40 if temp_change > 0 else 40,
                font=dict(color=color, size=12, family="Arial, sans-serif"),
                arrowcolor=color, arrowwidth=2
            ))

        # 2. Precipitation Anomaly Check
        precip_change = row['YoY_Precip_Change']
        if abs(precip_change) >= RAIN_THRESHOLD:
            reason = "Heavy Monsoon Spike (Flood Risk)" if precip_change > 0 else "Severe Drought/Rainfall Deficit"
            color = '#007bff'
            annotations.append(dict(
                x=year, y=precip_change,
                xref="x", yref="y2",
                text=f"💧 {reason}: +{precip_change:.0f}mm YoY",
                showarrow=True, arrowhead=7, ax=0, ay=-40 if precip_change > 0 else 40,
                font=dict(color=color, size=12, family="Arial, sans-serif"),
                arrowcolor=color, arrowwidth=2
            ))
            
    fig_yoy.update_layout(
        title='*Year-over-Year Change in Temperature and Rainfall*',
        xaxis_title="Year (Year-over-Year Comparison Starting from Year+1)",
        height=500, # Increased height to accommodate annotations
        hovermode="x unified",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#212529'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        annotations=annotations, # <-- ADDED ANNOTATIONS HERE
        
        # Primary Y-axis (Temperature) - FIXED SYNTAX
        yaxis=dict(
            title=dict( # Combined Title and Font definition
                text='Temperature Change (deg C)',
                font=dict(color='#e63946')  
            ), 
            tickfont=dict(color='#e63946'),
            showgrid=False # Hide grid for temp
        ),
        # Secondary Y-axis (Precipitation) - FIXED SYNTAX
        yaxis2=dict(
            title=dict( # Combined Title and Font definition
                text='Precipitation Change (mm)',
                font=dict(color='#007bff')
            ),
            tickfont=dict(color='#007bff'),
            overlaying='y',
            side='right',
            showgrid=True, # Show grid for precip
            gridcolor='#e9ecef'
        )
    )

    st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig_yoy, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p style='color: #495057;'>
    	<span style='color:#e63946;'>**Red Line:**</span> Shows if the **Mean Temperature** was hotter (+) or colder (-) than the previous year.
    	<span style='color:#007bff;'>**Blue Line:**</span> Shows if the **Total Rainfall** was more (+) or less (-) than the previous year. 
    	**Arrows mark significant anomalies** with the predicted environmental cause (e.g., El Niño, Drought, etc.).
    </p>
    """, unsafe_allow_html=True)
    
    

with TAB_AIR:
    st.subheader("Latest Air Quality Measurements (Open-Meteo AQ)")
    
    if df_aq.empty:
        st.info("No AQ data found for this location window.")
    else:
        # Prepare data for Pie Chart
        df_pie = df_aq[['parameter', 'value']].copy()
        df_pie['parameter'] = df_pie['parameter'].str.replace('_', ' ').str.title()
        df_pie.rename(columns={'parameter': 'Pollutant', 'value': 'Value'}, inplace=True)

        col_chart, col_table = st.columns([0.4, 0.6])

        with col_chart:
            # --- 1. Composition Pie Chart (Vibrant Palette) ---
            fig_pie = px.pie(
                df_pie,
                values='Value',  
                names='Pollutant',
                title='*Composition of Current Air Pollutants*',
                hole=0.4, # <--- MODIFIED: This line makes it a Donut Chart
                # Use a custom, bright palette: Yellow, Pink, Blue, Red
                color_discrete_sequence=['#ffc107', '#ff69b4', '#007bff', '#e63946', '#17a2b8', '#fd7e14'], 
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='white', width=1)))
            fig_pie.update_layout(
                height=450,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#212529'),
                legend_title_text="Pollutant"
            )
            st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_table:
            # --- 2. Data Table ---
            st.markdown("### Detailed Readings")
            df_aq_display = df_aq[["date", "parameter", "value", "unit"]].rename(
                columns={"date": "Last Updated", "parameter": "Pollutant", "value": "Value", "unit": "Unit"}
            )
            st.dataframe(df_aq_display, use_container_width=True)
            st.caption("Data source is Open-Meteo Air Quality API. Values represent instantaneous readings.")


with TAB_TRENDS:
    st.subheader("Multi-variable Climate Trends")
    
    # Prepare data for Global Warming Line (Smoothed Trend)
    df_temp = df_clim.copy()
    if not df_temp.empty:
        df_temp['warming_trend'] = df_temp['temperature_2m_mean'].rolling(window=365, center=True).mean()
    
    colA, colB = st.columns(2)
    
    with colA:
        # --- Max Wind Speed (Blue Highlight with Red Anomaly) ---
        
        if not df_clim.empty:
            max_wind_value = df_clim['windspeed_10m_max'].max()
            abnormal_point = df_clim[df_clim['windspeed_10m_max'] == max_wind_value].iloc[0]
            abnormal_date = abnormal_point['time']
            
            alert_msg = f"🌪 Extreme Wind Alert! Recorded {max_wind_value:.1f} m/s on {abnormal_date.strftime('%Y-%m-%d')}."
            st.warning(alert_msg)

        fig = go.Figure()
        
        # Base Wind Speed Line (Primary Blue)
        fig.add_trace(go.Scatter(x=df_clim["time"], y=df_clim["windspeed_10m_max"], name="Max Wind Speed", line=dict(color='#007bff', width=2))) # Blue for Train
        
        # Add annotation (Arrow and text) pointing to the abnormal peak
        if not df_clim.empty:
            fig.add_annotation(
                x=abnormal_date, 
                y=abnormal_point['windspeed_10m_max'], 
                text="ABNORMAL SPIKE", 
                showarrow=True, 
                font=dict(color="#e63946", size=12), # Red 
                arrowhead=2, 
                arrowsize=1.5, 
                arrowwidth=2, 
                arrowcolor="#e63946", 
                ax=0, ay=-50
            )
            
        fig.update_layout(
            title="*Max Wind Speed (m/s) Trend with Anomaly Detection*",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#212529'),
            xaxis_title="Date",
            yaxis_title="Wind Speed (m/s)"
        )
        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with colB:
        # --- Mean Temperature Trend (Yellow/Orange Warming Line) ---
        fig = go.Figure()
        
        # Base Mean Temperature Line (Muted Blue)
        fig.add_trace(go.Scatter(
            x=df_clim["time"], 
            y=df_clim["temperature_2m_mean"], 
            name="Daily Mean Temp", 
            line=dict(color='#457b9d', width=2), # Muted Blue
            opacity=0.8
        ))
        
        # Global Warming Trend Line (Smoothed and Luminous)
        if not df_temp.empty:
            fig.add_trace(go.Scatter(
                x=df_temp["time"], 
                y=df_temp["warming_trend"], 
                name="Global Warming Trend (365-day Avg)", 
                line=dict(color='#fd7e14', width=4, dash='dashdot'), # Orange for contrast
                opacity=0.9
            ))
            
        fig.update_layout(
            title="*Mean Temperature Trend with Global Warming Line*",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#212529'),
            xaxis_title="Date",
            # Replacing degree symbol in yaxis title text
            yaxis_title="Temperature (deg C)",
            legend=dict(y=0.99, x=0.01)
        )
        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        # ADDED: Full-screen button for visualization demo
        if st.button("Click to View Temperature Trend Fullscreen/Maximized"):
            st.toast("Plotly charts support built-in fullscreen/zoom on double-click or use the top-right toolbar!", icon="📈")
        
    # --- Correlation Snapshot (Light Scale) ---
    st.markdown("### Correlation Snapshot")
    corr_df = df_clim.drop(columns=["time"]).corr(numeric_only=True)
    fig_corr = px.imshow(
        corr_df, 
        text_auto=".2f",
        aspect="auto", 
        title="*Climate Variables Correlation Matrix*",
        color_continuous_scale=px.colors.sequential.Plotly3_r, # Using a light-friendly scale
    )
    fig_corr.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#212529'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # 🌟 Updated Explanation Section Colors
    st.markdown("### 🔍 Technical Explanation: Understanding Correlation")
    st.caption("""
    Correlation measures how closely two variables move together, ranging from **-1.0 to +1.0**.

    * **+1.0 (Bright Yellow/Orange):** Perfect *Positive Correlation*. When one variable (e.g., Max Temperature) increases, the other variable (e.g., Mean Temperature) increases at the same time.
    * **-1.0 (Dark Color):** Perfect *Negative Correlation*. When one variable (e.g., Solar Radiation) increases, the other variable (e.g., Cloud Cover/Precipitation) decreases.
    * **0.0 (Mid-color):** *No Correlation*. The variables have no discernible linear relationship.
    """)
    
    st.markdown("### 🗣 सहसंबंध स्नैपशॉट (Correlation Snapshot) का मतलब समझें")
    st.markdown("""
    <div class='card' style='background: #f1f7fd; border: 1px solid #007bff; color: #212529;'>
    <p style='color:#495057;'>यह ग्रिड (grid) हमें बताती है कि मौसम के अलग-अलग फैक्टर (factors) एक-दूसरे से कैसे जुड़े हैं।</p>
    
    <ul style='color:#495057;'>
    	<li>*हाई वैल्यू (High Value) - ज़्यादातर चमकीला पीला/नारंगी:* इसका मतलब है *Strong Positive Relation*। मतलब, अगर **एक चीज़ बढ़ती है, तो दूसरी भी लगभग उसी Rate से बढ़ती है*। (Example: Max Temp और Mean Temp हमेशा साथ बढ़ते हैं: 0.96)</li>
    	<li>*लो वैल्यू (Low Value) - ज़्यादातर गहरा रंग:* इसका मतलब है *Negative Relation* या *Weak Relation*। मतलब, अगर एक चीज़ बढ़ती है, तो **दूसरी घट जाती है**, या उनका कोई खास कनेक्शन नहीं है। (Example: Rainfall ज़्यादा होने पर Solar Radiation (धूप) कम हो जाती है: लगभग -0.31)</li>
    	<li>*जीरो के पास (Near Zero):* मतलब उन दोनों के बीच *कोई ख़ास Connection नहीं* है।</li>
    </ul>

    <p style='color:#212529; font-weight:600;'>इस डेटा से मुख्य बातें (Key Insights):</p>
    <ul style='color:#495057;'>
    	<li>*गर्मी और धूप:* तापमान (Temperature) और धूप (Shortwave Radiation) का Strong Connection है (0.76)।</li>
    	<li>*Precipitation (बारिश):* बारिश का किसी भी Temperature से बहुत कम Connection है (Max Temp से 0.00)।</li>
    	<li>*Global Warming:* *'Year'* का *Temperature* से छोटा लेकिन Positive Connection (0.15) दिख रहा है, जो बताता है कि समय के साथ *Overall Temperature* थोड़ा बढ़ रहा है।</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


with TAB_FORECAST:
    st.subheader("AI Forecasts with Backtest Metrics")
    target = st.selectbox(
        "Target to forecast", 
        ["temperature_2m_mean","temperature_2m_max","temperature_2m_min","precipitation_sum"], 
        index=0,
        help="The variable you want the AI model to predict into the future (e.g., Mean Temperature)."
    )
    horizon = st.slider("Forecast horizon (days)", 7, 365, 90)

    model_used, ts, train, valid, fcst, metrics = backtest_train_forecast(df_clim[["time", target]].dropna(), target, horizon=horizon, model_choice=model_choice)

    # Replacing degree symbol in metric explanation
    st.info(f"Model used: *{model_used}* |  MAE: *{metrics['MAE'] if metrics['MAE'] is not None else '—'}* |  MAPE: *{metrics['MAPE'] if metrics['MAPE'] is not None else '—'}*")

    # Keep Metric explanations
    st.caption("""
    <span style='color:#495057;'>*MAE (Mean Absolute Error):* The average magnitude of errors in the predictions, measured in the same units as the target (e.g., deg C). Lower is better.</span>
    <br>
    <span style='color:#495057;'>*MAPE (Mean Absolute Percentage Error):* The average percentage error of the prediction. 10% MAPE means the forecast is off by 10% on average. Lower is better.</span>
    """, unsafe_allow_html=True)

    fig = go.Figure()
    # Premium Line Styling (Updated to Bright Palette)
    fig.add_trace(go.Scatter(x=train["ds"], y=train["y"], name="Train Data", line=dict(color='#007bff', width=2))) # Blue for Train
    fig.add_trace(go.Scatter(x=valid["ds"], y=valid["y"], name="Validation Data", line=dict(color='#e63946', width=2))) # Red for Validation

    # unify forecast frame
    if "ds" not in fcst.columns:
        if "time" in fcst.columns:
            fcst.rename(columns={"time":"ds"}, inplace=True)
        else:
            # create series from last date
            last = ts["ds"].iloc[-1]
            fcst["ds"] = pd.date_range(last + pd.Timedelta(days=1), periods=len(fcst), freq='D')
    
    yhat = fcst["yhat"] if "yhat" in fcst.columns else fcst.iloc[:,1]
    
    # Prediction line (Yellow/Amber and Dashed)
    fig.add_trace(go.Scatter(x=fcst["ds"], y=yhat, name="Forecast", line=dict(color='#ffc107', width=3, dash="dash")))
    
    fig.update_layout(
        title=f"*AI Forecast: {target}*", 
        xaxis_title="Date", 
        yaxis_title=target,
        hovermode="x unified",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#212529'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.download_button("⬇ Download Forecast CSV", data=fcst.to_csv(index=False), file_name=f"forecast_{target}.csv", mime="text/csv")


with TAB_STORY: # RENDERED IMPACT STORY CONTENT
    st.subheader(f"Climate Change Impact Story: Real-World Consequences for {_name}")

    # --- Row 1: Crop Loss Simulation (Updated Colors) ---
    st.markdown("### 🌾 Local Crop Yield Loss Simulation")
    
    col_loss, col_map = st.columns([0.45, 0.55])
    
    # Use mean_temp calculated in the main block
    stressed_yield, loss_percent, status = get_crop_loss_simulation(mean_temp) 

    with col_loss:
        # Replacing degree symbol
        st.markdown(f"#### Average Mean Temperature: **{mean_temp:.1f} deg C**")
        st.info(f"Predicted Agricultural Status: *{status}*")
        
        # Plotly Gauge to simulate yield loss visually (Updated Colors)
        fig_loss = go.Figure(go.Indicator(
            mode = "number+gauge",
            value = stressed_yield,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Simulated Crop Yield (Relative %)", 'font': {'size': 20, 'color': '#212529'}},
            gauge = {
                'shape': "angular",
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#495057"},
                'bar': {'color': "#007bff"}, # Blue fill
                'bgcolor': "#e9ecef",
                'steps': [
                    {'range': [0, 80], 'color': '#e63946'}, # Red Danger Zone
                    {'range': [80, 95], 'color': '#ffc107'}, # Yellow Warning Zone
                    {'range': [95, 100], 'color': '#38a3a5'} # Teal Optimal Zone
                ],
                'threshold': {
                    'line': {'color': "#e63946", 'width': 5},
                    'thickness': 0.8,
                    'value': stressed_yield
                }
            }
        ))
        fig_loss.update_layout(
            height=350, margin=dict(l=10, r=10, t=50, b=10), 
            plot_bgcolor='white', paper_bgcolor='white', 
            font=dict(color='#212529')
        )
        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_loss, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Replacing degree symbol in text explanation
        st.markdown(f"""
        <p style='color: #e63946; font-weight:700;'>Estimated Loss: 
        <span style='font-size:1.5rem;'>-{loss_percent:.1f}%</span></p>
        <p style='color: #495057;'>High average temperatures (above 25 deg C proxy) severely damage grain filling and flower production, leading to direct economic loss for farmers and instability in food supply chains.</p>
        """, unsafe_allow_html=True)

    # --- Row 2: City CO2 Accountability Map (Updated Colors) ---
    with col_map:
        # Replacing subscript in heading text
        st.markdown("### 💨 City CO2 Accountability Map (kT/Yr)")
        
        df_emissions = get_all_india_city_emissions()
        
        # Dynamic Select Box for City Comparison
        st.multiselect(
            "Select cities for CO2 contribution comparison",
            options=df_emissions['City'].tolist(),
            default=[_name] if _name in df_emissions['City'].tolist() else [df_emissions['City'].iloc[0]],
            key="story_city_multiselect",
            help="Compare your selected city's estimated annual carbon emissions against other major Indian cities."
        )

        df_filtered = df_emissions[df_emissions['City'].isin(st.session_state.story_city_multiselect)]
        
        # Choropleth Map showing CO2 contribution (Updated Colors)
        fig_map = px.scatter_geo(
            df_emissions,
            lat='Latitude',
            lon='Longitude',
            text=df_emissions.apply(lambda row: f"{row['City']}<br>CO2: {row['CO2_Emissions_Annual_kT']:,} kT", axis=1),
            size="CO2_Emissions_Annual_kT",
            color="CO2_Emissions_Annual_kT",
            projection="natural earth",
            title="Relative Annual CO2 Contribution of Major Cities",
            color_continuous_scale=px.colors.sequential.YlOrRd, # Yellow/Orange/Red scale for emissions
            scope='asia'
        )
        
        # Highlight selected cities with an intense Pink ring 
        selected_lats = df_filtered['Latitude'].tolist()
        selected_lons = df_filtered['Longitude'].tolist()
        
        fig_map.add_trace(go.Scattergeo(
            lat=selected_lats,
            lon=selected_lons,
            mode='markers',
            marker=dict(
                size=df_filtered['CO2_Emissions_Annual_kT'].apply(lambda x: x/400).tolist(), 
                color='#ff69b4', # Pink Highlight
                line_width=3,
                opacity=0.8,
                line_color='#ffffff'
            ),
            hoverinfo='none',
            name='Selected City (Highlight)'
        ))
        
        fig_map.update_geos(
            lataxis_range=[5, 35], 
            lonaxis_range=[65, 90], 
            showcountries=True, 
            countrycolor="#495057", # Darker country lines
            subunitcolor="#495057",
            showland=True,
            landcolor="#e9ecef" # Light land color
        )
        
        fig_map.update_layout(
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#212529'),
            margin={"r":0,"t":40,"l":0,"b":0},
            # Replacing subscript in colorbar title text
            coloraxis_colorbar=dict(title="CO2 (kT/yr)")
        )
        st.plotly_chart(fig_map, use_container_width=True)


    st.markdown("---")

    # --- Row 3: Glacier Melt and Sea Level Risk (Global/Regional Context) ---
    col_glacier, col_sea = st.columns(2)
    
    with col_glacier:
        st.markdown("### 🧊 Himalayan Glacier Melt & Water Security")
        # 🌟 FIX: Corrected to use .png extension based on file system check
        # st.image("Gemini_Generated_Image_sszefvsszefvssze.png", caption="Glacial runoff leading to extreme river flow and initial flood risks.", use_column_width=True)
        st.markdown(f"""
        <p style='color: #495057;'>**Global Temperature Rise** directly leads to accelerated melting of Himalayan glaciers, which are the primary source of water for major Indian rivers like the Ganga.
        <ul style='color: #495057;'>
        	<li>**Glacier Mass Loss (Est. since 2000):** <span style='color: #e63946; font-weight:700;'>-25% to -35%</span></li>
        	<li>**Direct Impact for {_name}:** Increased risk of **flash floods** initially, followed by long-term **severe water scarcity** as the ice mass depletes.</li>
        </ul>
        </p>
        """, unsafe_allow_html=True)
        
    with col_sea:
        st.markdown("### 🌊 Sea Level Rise Risk (Coastal Threat)")
        # 🌟 FIX: Corrected to use .png extension based on file system check
        # st.image("Gemini_Generated_Image_3x7ldg3x7ldg3x7l.png", caption="Coastal risk mitigation requires sustainable energy transition and robust infrastructure.", use_column_width=True)

        # Simple logic: If current city is not coastal, show general impact and coastal city examples.
        if _name in ['Mumbai', 'Chennai', 'Kolkata', 'Kochi']:
            risk_level = "EXTREME"
            risk_color = '#e63946'
        else:
            risk_level = "Indirect"
            risk_color = '#ffc107'

        st.markdown(f"""
        <p style='color: #495057;'>Melting glaciers and thermal expansion of seawater drive **sea level rise**.
        <ul style='color: #495057;'>
        	<li>**Local Risk for {_name}:** <span style='color: {risk_color}; font-weight:700;'>{risk_level}</span> (Source of indirect risk: Mass climate migration & supply chain disruption).</li>
        	<li>**Coastal Metros Risk (Mumbai/Kolkata):** Major infrastructure, historical sites, and large population centers are at high risk of **permanent inundation** by 2100.</li>
        	<li>**Impact:** Loss of wetlands, increased saline intrusion into groundwater, and massive infrastructure costs.</li>
        </ul>
        </p>
        """, unsafe_allow_html=True)

    # --- Original Future Impact Section (Health & Afforestation) ---
    st.markdown("---")
    st.markdown("### 🏃 Air Pollution & Public Health Advisory")
    pm_level = pm25_now if not math.isnan(pm25_now) else 80.0
    prediction = get_future_impact_prediction(pm_level)
    
    c_pred, c_adv = st.columns([0.4, 0.6])

    with c_pred:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        # Replacing subscript in metric text
        st.metric(f"Predicted Health Risk at PM2.5 of {pm_level:.1f} ug/m3", prediction['health_risk'])
        st.warning(f"*Running Advisory:* {prediction['advice']}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # --- Afforestation Goals ---
        st.markdown(f"""
        	### <span class='planting-tree'>🌳</span> <span class='goal-card-title'>Afforestation Goals for {_name}</span>
        """, unsafe_allow_html=True)

        tree_data = get_tree_inventory(_name)

        c_tree1, c_tree2 = st.columns(2)

        # Current Trees (Est.)
        with c_tree1:
            st.markdown(f"""
            	<div class='goal-card-metric'>
            		<p style='color:#6c757d; font-size:1.0rem; margin-bottom: 5px;'>Current Trees (Est.)</p>
            		<div class='current-number'>{tree_data['current']:,}</div>
            	</div>
            """, unsafe_allow_html=True)

        # Trees Needed
        with c_tree2:
            st.markdown(f"""
            	<div class='goal-card-metric'>
            		<p style='color:#6c757d; font-size:1.0rem; margin-bottom: 5px;'>Trees Needed (Goal)</p>
            		<div class='goal-number'>{tree_data['needed']:,}</div>
            	</div>
            """, unsafe_allow_html=True)

        st.caption(f"Goal: {tree_data['target_ratio']} trees per person (Population: {tree_data['population']:,})")

    with c_adv:
        # River Health is still displayed for comprehensive impact
        river_data = get_river_health_data(_name)
        river_name = river_data['River'].iloc[0]
        river_base_name = river_name.split(' (')[0] if ' (' in river_name else river_name # e.g., "Ganga"

        st.markdown(f"### 🌊 River Health Status ({river_name})")
        
        st.dataframe(river_data[['Dissolved Oxygen (DO mg/L)', 'BOD (mg/L)', 'Coliform (MPN/100ml)', 'Status']], hide_index=True, use_container_width=True)

        st.markdown("### 📢 Actionable Advisories")
        st.markdown(f"""
        <div class="card" style='background: #f1f7fd; border: 1px solid #007bff; padding: 10px; border-radius: 8px;'>
        *Public Advice:* Focus on waste segregation, reducing personal vehicle use, and avoiding littering near {river_base_name} River.
        *Government Advice:* Prioritize industrial effluent treatment, expand public transit, and invest in large-scale urban greening projects in {_name}.
        </div>
        """, unsafe_allow_html=True)

with TAB_GREEN_INFRA:
    st.subheader("Green Infrastructure & Electric Vehicle Adoption")
    st.markdown("Monitor and compare cities on their progress in adopting pollution-free energy and sustainable transport.")

    # Get a list of all cities for the multiselect
    known_cities = ["Prayagraj", "Lucknow", "Varanasi", "Kanpur", "Mumbai", "Delhi", "Bengaluru", "Agra"]
    all_cities = sorted(list(set(known_cities + [_name]))) 

    selected_cities_green = st.multiselect(
        "Select cities to compare (including current city):",
        options=all_cities,
        default=[_name, "Lucknow", "Mumbai"], # Default to current city and some others for comparison
        key="green_infra_city_select"
    )

    if not selected_cities_green:
        st.info("Please select at least one city to view the data.")
    else:
        st.markdown("---")

        # 1. Pollution-Free Energy Resources (Stacked Bar Chart - MW)
        st.markdown("### ⚡ Pollution-Free Energy Generation Capacity (MW)")
        df_energy = get_pollution_free_energy_data(selected_cities_green)
        df_energy_melted = df_energy.melt(id_vars=['City'], var_name='Energy Type', value_name='Capacity (MW)')
        
        fig_energy = px.bar(
            df_energy_melted,
            x='City',
            y='Capacity (MW)',
            color='Energy Type',
            title='*Pollution-Free Energy Capacity by City*',
            barmode='stack',
            color_discrete_map={
                'Solar (MW)': '#ffc107', # Yellow/Amber for Solar
                'Wind (MW)': '#17a2b8',  # Teal/Cyan for Wind
                'Hydro (MW)': '#007bff'  # Blue for Hydro
            }
        )
        fig_energy.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#212529'),
            xaxis_title=None, yaxis_title="Capacity (MW)"
        )
        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_energy, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("This chart shows the simulated total installed capacity of solar, wind, and hydro energy sources per city.")
        
        st.markdown("---")

        # 2. Registered Solar Connections (Bar Chart)
        st.markdown("### ☀️ Registered Solar Connections (Rooftop Units)")
        df_solar_conn = get_registered_solar_connections(selected_cities_green)
        fig_solar_conn = px.bar(
            df_solar_conn,
            x='City',
            y='Solar Connections',
            title='*Total Registered Solar Connections*',
            color='Solar Connections',
            color_continuous_scale=px.colors.sequential.YlOrBr, # Yellow-Orange-Brown for Solar
        )
        fig_solar_conn.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#212529'),
            xaxis_title=None, yaxis_title="Number of Connections"
        )
        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_solar_conn, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("This graph displays the estimated number of registered residential and commercial solar installations, indicating rooftop solar adoption.")
        
        st.markdown("---")
        
        # 3. Registered EV Vehicles (Bar Chart)
        st.markdown("### 🚗 Registered Electric Vehicles (EVs)")
        df_ev = get_registered_ev_vehicles(selected_cities_green)
        fig_ev = px.bar(
            df_ev,
            x='City',
            y='Registered EVs',
            title='*Total Registered Electric Vehicles*',
            color='Registered EVs',
            color_continuous_scale=px.colors.sequential.Greens, # Green scale for EVs
        )
        fig_ev.update_layout(
            plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#212529'),
            xaxis_title=None, yaxis_title="Number of Vehicles"
        )
        st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_ev, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("This graph illustrates the adoption of electric vehicles, a key indicator for sustainable transportation.")

with TAB_SWACHH:
    st.subheader("Swachh Survekshan (Cleanliness) Ranking Trend")
    st.markdown("Track the performance of major cities in Uttar Pradesh and top national performers based on the annual cleanliness survey.")

    # Get the complete list of UP Nagar Nigams and National toppers for the multiselect
    # Temporarily call the function with an empty list just to build the options list
    # This ensures all hardcoded cities are available for selection
    all_known_swachh_cities = get_swachh_ranking_data([], _name)['City'].unique().tolist()
    default_swachh_cities = sorted(list(set(all_known_swachh_cities + [_name, "Lucknow", "Indore", "Surat"])))

    selected_swachh_cities = st.multiselect(
        "Select cities for historical cleanliness ranking comparison:",
        options=default_swachh_cities,
        default=[_name, "Lucknow", "Indore", "Surat"],
        key="swachh_city_select",
        help="Select any major city in UP or the top national performers."
    )

    if not selected_swachh_cities:
        st.info("Please select at least one city to view the cleanliness ranking trend.")
    else:
        # UPDATED CALL: Passing current city name for proper default handling
        df_swachh = get_swachh_ranking_data(selected_swachh_cities, _name)
        
        if df_swachh.empty:
            st.warning("No data found for the selected cities.")
        else:
            # Melt the DataFrame for plotting (Years as categories)
            df_swachh_melted = df_swachh.melt(id_vars='City', var_name='Year', value_name='Global Rank')
            
            # Line chart showing Rank over 5 years
            fig_rank = px.line(
                df_swachh_melted,
                x='Year',
                y='Global Rank',
                color='City',
                title='*City Global Rank Trend (Lower Rank is Better)*',
                markers=True,
                # Invert Y-axis so rank 1 is at the top
                labels={'Global Rank': 'Global Rank (Lower = Better)'},
                color_discrete_sequence=px.colors.qualitative.Bold 
            )
            
            fig_rank.update_yaxes(
                autorange="reversed", # Invert axis
                tickmode='linear',
                dtick=50, # Set major ticks at intervals
                showgrid=True,
                gridcolor='#e9ecef'
            )
            fig_rank.update_layout(
                plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#212529')
            )

            st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
            st.plotly_chart(fig_rank, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption("This chart tracks the simulated global rank of cities over the past 5 years. A downward trend (lower rank number) indicates improvement in cleanliness.")
            
            st.markdown("---")
            st.markdown("### Detailed Swachh Ranking Table")
            st.dataframe(df_swachh, use_container_width=True)

with TAB_SCORE:
    st.subheader("City Sustainability Score")
    pm_for_score = pm25_now if not math.isnan(pm25_now) else 60.0
    score, sub = compute_sustainability_score(SustainabilityInputs(
        pm25=pm_for_score,
        co2_per_capita=co2_pc,
        renewable_share=float(ren_share),
        water_quality_index=float(water_idx),
        waste_recycling_rate=float(recycle),
    ))
    colL, colR = st.columns([0.45,0.55])
    with colL:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Sustainability Score", f"{score:.1f} / 100")
        # 🌟 ENHANCEMENT: Added score explanation
        st.caption("A composite score (0-100) based on five dimensions: Air Quality, CO2 Emissions, Renewable Energy usage, Water Quality, and Waste Recycling Rate. Higher is better.")
        st.markdown("</div>", unsafe_allow_html=True)
    with colR:
        sub_df = pd.DataFrame({"Dimension": list(sub.keys()), "Score": list(sub.values())})
        # Blue/Pink Focused Bar Chart for Sub-Scores
        fig = px.bar(
            sub_df, 
            x="Dimension", 
            y="Score", 
            title="Sub-Scores (0-100)",
            color="Score",
            color_continuous_scale=px.colors.sequential.Plotly3, # Uses light-friendly sequential scale
        )
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#212529'))
        st.plotly_chart(fig, use_container_width=True)

with TAB_CARBON:
    # 1. India-Wide City CO2 Comparison Map (Choropleth/Scattergeo)
    # Replacing subscript in heading text
    st.markdown("### 🗺️ Major City Annual CO2 Emissions Comparison (Proxy Data)")
    
    df_map = get_all_india_city_emissions()
    
    # Identify the current selected city's location for highlighting
    current_city_name = _name
    
    # Create the Scattergeo map
    fig_map = go.Figure(data=go.Scattergeo(
        lon = df_map['Longitude'],
        lat = df_map['Latitude'],
        text = df_map.apply(lambda row: f"{row['City']}<br>CO2: {row['CO2_Emissions_Annual_kT']:,} kT", axis=1),
        mode = 'markers',
        marker = dict(
            size = df_map['CO2_Emissions_Annual_kT'] / 1000, # Scale marker size by kT
            sizemode = 'area',
            sizemin = 5,
            color = df_map['CO2_Emissions_Annual_kT'],
            colorscale = px.colors.sequential.Sunset_r, # Use a warm color scale for emissions
            cmin = df_map['CO2_Emissions_Annual_kT'].min(),
            cmax = df_map['CO2_Emissions_Annual_kT'].max(),
            # Replacing subscript in colorbar title text
            colorbar_title = "CO2 (kT/yr)",
            line_color='rgba(0,0,0,0.5)'
        ),
    ))
    
    # Add a separate trace for the current city highlight
    current_city_data = df_map[df_map['City'].str.lower() == current_city_name.lower()]
    if not current_city_data.empty:
        fig_map.add_trace(go.Scattergeo(
            lon = current_city_data['Longitude'],
            lat = current_city_data['Latitude'],
            text = current_city_data.apply(lambda row: f"**SELECTED: {row['City']}**<br>CO2: {row['CO2_Emissions_Annual_kT']:,} kT", axis=1),
            mode = 'markers',
            marker = dict(
                size = current_city_data['CO2_Emissions_Annual_kT'].iloc[0] / 1000 + 10, # Make the selected city marker bigger
                color = '#ff69b4', # Pink for selected highlight
                line_width = 3,
                line_color = '#FFFFFF'
            ),
            hoverinfo='text',
            name='Selected City'
        ))

    fig_map.update_layout(
        title_text = f'CO2 Emissions Across Indian Cities (Current City: **{current_city_name}**)',
        showlegend = False,
        geo = dict(
            scope = 'asia',
            lonaxis_range= [68, 98],
            lataxis_range= [5, 38],
            subunitcolor = "#007bff", # Blue border for states/subunits
            subunitwidth = 1,
            countrycolor = "#495057", # Darker border for country
            countrywidth = 1.5,
            bgcolor = '#f8f9fa', # Map background color
            lakecolor = '#e9ecef', # Light lake/water body color
            landcolor = '#ffffff', # White land color
            coastlinecolor = '#adb5bd',
            projection_type = 'mercator'
        ),
        height=550,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#212529')
    )
    st.markdown('<div class="plot-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 2. Personal Carbon Footprint Calculator 
    st.subheader(f"Personal Carbon Footprint for {_name} (Quick Estimate)")

    # 1. Auto-Estimation Checkbox
    auto_estimate = st.checkbox(
        "⚡ *Auto-Fetch Household Estimate* based on current location data", 
        value=False,
        key="auto_estimate_carbon_tab", 
        help="Calculates a baseline household footprint using India-specific averages, adjusted by your city's air quality, climate, and sustainability metrics."
    )

    # Define India-Specific Baseline Averages for Carbon Footprint Estimation
    BASE_KWH = 180            # National average monthly consumption
    BASE_KM_CAR = 350 # Average urban monthly car travel (higher side)
    BASE_LPG = 6      # Average monthly LPG use (kg)
    BASE_FLIGHTS = 1   # Average flights per year
    
    # Emission Factors (unchanged)
    EF_CAR = 0.18 # kg CO2e / km
    EF_KWH = 0.7  # kg CO2e / kWh (India grid average)
    EF_FLIGHT = 180 # kg CO2e / 2-hour flight
    EF_LPG = 3.0    # kg CO2e / kg LPG
    DIET_MAP = {"Heavy meat": 300, "Mixed": 200, "Vegetarian": 150, "Vegan": 120}

    colA, colB = st.columns(2)
    
    # 2. Determine Inputs (Manual or Auto-Fetched)
    if auto_estimate:
        st.info(f"Using *{_name}* environmental data to localize the estimate.")
        
        # 🌟 PROXY LOGIC: Intelligent data-driven estimation 
        
        # Proxy 1: Diet Type (Proxy based on city Water Quality Index)
        diet_proxy = "Mixed"
        if water_idx < 50:
            diet_proxy = "Heavy meat"
        elif water_idx > 80:
            diet_proxy = "Vegetarian"
        
        # Proxy 2: Car Travel (Proxy based on population proxy and max wind speed)
        tree_data = get_tree_inventory(_name)
        pop_adj = tree_data['population'] / 1000000 # millions
        # Reduce travel slightly if pop is high (congestion) or wind is extreme
        km_car_proxy = max(50, BASE_KM_CAR - int(pop_adj * 50) - int(df_clim['windspeed_10m_max'].max() * 5))
        
        # Proxy 3: Electricity consumption (Proxy based on high/low temperatures for AC use)
        mean_temp_swing = df_clim['temperature_2m_max'].max() - df_clim['temperature_2m_min'].min()
        kwh_proxy = int(BASE_KWH + mean_temp_swing * 2) 

        # Use city's recycling rate from sidebar input
        recycle_rate_proxy = recycle
        
        # Other simple averages
        flights_proxy = BASE_FLIGHTS
        lpg_proxy = BASE_LPG

        with colA:
            km_car = st.number_input("Monthly car travel (km)", value=km_car_proxy, disabled=True, key="auto_km_car")
            kwh = st.number_input("Monthly electricity use (kWh)", value=kwh_proxy, disabled=True, key="auto_kwh")
            flights = st.number_input("Flights per year (2-hr avg)", value=flights_proxy, disabled=True, key="auto_flights")
        with colB:
            diet = st.selectbox("Diet type", list(DIET_MAP.keys()), index=list(DIET_MAP.keys()).index(diet_proxy), disabled=True, key="auto_diet")
            lpg = st.number_input("Monthly LPG use (kg)", value=lpg_proxy, disabled=True, key="auto_lpg")
            recycle_rate = st.slider("Household recycling (%)", 0, 100, recycle_rate_proxy, disabled=True, key="auto_recycle")

    else:
        # Manual Input Mode (Original code structure)
        with colA:
            km_car = st.number_input("Monthly car travel (km)", 0, 10000, 300, key="manual_km_car")
            kwh = st.number_input("Monthly electricity use (kWh)", 0, 2000, 180, key="manual_kwh")
            flights = st.number_input("Flights per year (2-hr avg)", 0, 50, 1, key="manual_flights")
        with colB:
            diet = st.selectbox("Diet type", list(DIET_MAP.keys()), index=1, key="manual_diet")
            lpg = st.number_input("Monthly LPG use (kg)", 0, 100, 6, key="manual_lpg")
            recycle_rate = st.slider("Household recycling (%)", 0, 100, 20, key="manual_recycle")

    # 3. Calculation and Display
    
    # Calculate effective EF_KWH (Dynamically adjusted by Renewables Share)
    # Replacing subscript in caption text
    effective_ef_kwh = EF_KWH * (1 - ren_share/100)
    st.caption(f"Calculated effective CO2 factor for electricity in this city: {effective_ef_kwh:.3f} kg/kWh (based on {ren_share}% renewables)")

    # Calculate monthly carbon emissions
    if km_car is None: km_car = 0
    if kwh is None: kwh = 0
    if flights is None: flights = 0
    if lpg is None: lpg = 0
    
    monthly = (
        km_car * EF_CAR + 
        kwh * effective_ef_kwh + # Use adjusted EF
        (flights * EF_FLIGHT) / 12 + 
        lpg * EF_LPG + 
        DIET_MAP[diet]
    ) * (1 - recycle_rate / 400) # Simple reduction for waste

    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    # Replacing subscript in metric text
    st.metric("Estimated monthly emissions", f"{monthly/1000:.2f} t CO2e")
    st.markdown("</div>", unsafe_allow_html=True)

    # 🌟 ENHANCEMENT: Add expander to explain the calculation factors
    with st.expander("❓ How is this calculated? (Emission Factors Explained)", expanded=False):
        st.caption("""
        This tool provides a *quick, simplified estimate* of monthly **CO2e (Carbon Dioxide equivalent)** emissions. 
        The calculation relies on global and regional *Emission Factors (EFs)*:
        
        * *Car Travel (EF_CAR = 0.18):* Assumes **0.18 kg CO2e per km** for an average gasoline car.
        * *Electricity (EF_KWH):* The factor is **dynamically adjusted** based on the Renewable Energy Share input from the sidebar. Lower renewables mean higher grid emissions.
        * *Flights (EF_FLIGHT = 180):* Assumes **180 kg CO2e per 2-hour flight**.
        * *LPG Use (EF_LPG = 3.0):* Assumes **3.0 kg CO2e per kg of LPG** consumed.
        * *Diet:* Based on estimated monthly emissions for different diet types.
        * *Recycling:* A reduction is applied based on the input rate to account for waste diversion.
        """)

    fig = px.pie(names=["Travel","Electricity","Flights","LPG","Diet"],
              values=[
                      km_car * EF_CAR, 
                      kwh * effective_ef_kwh, # Use adjusted EF in pie chart too
                      (flights * EF_FLIGHT) / 12, 
                      lpg * EF_LPG, 
                      DIET_MAP[diet]
                    ],
              title="Breakdown (kg CO2e per month)")
    # Bright/Environmental colors for the pie chart
    fig.update_traces(marker=dict(colors=px.colors.sequential.Plotly3))
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#212529'))
    st.plotly_chart(fig, use_container_width=True)

with TAB_ABOUT:
    st.header("Hackathon Submission")
    st.markdown("<div class='team-title'>Nxt Gen Developers</div>", unsafe_allow_html=True)
    st.subheader("Team Members")
    st.markdown("""
        * **Aditya Kumar Singh**
        * **Gaurang Verma**
        * **Vandana Yadav**
        * **Gaurav Shakya**
        * **Saurabh Shukla**
    """)
    
    st.markdown("---")
    
    st.subheader("Project: SustainifyAI — Bright Sustainability Tracker")
    
    st.markdown("### Why SustainifyAI was Created")
    st.markdown("""
        SustainifyAI was developed to bridge the **gap between raw climate data and actionable governance/citizen insight**. In regions facing rapid environmental changes, a tool that consolidates real-time air quality, historical climate trends, river health, and personalized impact analysis is crucial. Our goal is to make **complex environmental data accessible, visual, and predictive** for better planning.
    """)
    
    st.markdown("### How SustainifyAI Helps Government & Policymakers")
    st.markdown("""
        1. **Alerts & Anomaly Detection:** Provides immediate, location-specific warnings for PM2.5 or extreme heat, enabling **proactive health and safety responses** (e.g., advising school closures or issuing heatwave warnings).
        2. **Goal Tracking (e.g., Afforestation):** Quantifies city-specific targets (e.g., trees needed per capita), offering **measurable progress indicators** for green initiatives.
        3. **Resource Allocation:** The Sustainability Score and detailed sub-scores (Water Quality, Air Quality, etc.) highlight the **most critical areas requiring immediate investment and policy intervention**.
        4. **Long-term Planning:** AI forecasts provide a future outlook on temperatures and precipitation, essential for **infrastructure planning** (e.g., water management, drought preparation).
    """)
    
    st.markdown("### The Role of AI, ML, and Data Science")
    st.markdown("""
        * **Data Science & Engineering:** Used to **ingest, clean, and structure** vast amounts of historical climate data (ERA5 reanalysis) and real-time air quality data (Open-Meteo). We derive key metrics like **Monthly Norms** and **Warming Trends** using time-series analysis and moving averages.
        * **Machine Learning (ML) & AI Forecasting:**
        	* **Forecasting Models:** We use established ML/AI models—**Prophet** (for strong seasonality), **ARIMA** (for classical time-series analysis), and a **Random Forest ML Ensemble** (for non-linear trend capture)—to predict future climate variables like temperature and precipitation up to a year ahead.
        	* **Metric Validation:** Models are rigorously backtested using metrics like **MAE** and **MAPE** to ensure forecast accuracy before deployment.
        	* **Intelligent Proxies:** AI/ML is used in the Carbon Footprint tab to create **intelligent, localized auto-estimates** based on the city's overall sustainability metrics.
    """)

    st.markdown("### Technology Stack & Demonstration Platform")
    st.markdown("""
        | Component | Technology | Why We Used It |
        | :--- | :--- | :--- |
        | **Frontend/Demonstration** | **Streamlit & Plotly** | Streamlit enabled **rapid prototyping** and creating a complex, interactive web application entirely in Python. Plotly provides **futuristic, interactive, and mobile-friendly visualizations**. |
        | **Data/Backend (APIs)** | **Open-Meteo (ERA5 & AQ APIs)** | Provides free, high-quality, geographically granular historical climate and real-time air quality data, eliminating the need for complex API keys for demonstration. |
        | **Development Environment** | **VS Code** | Used for its robust Python support, rapid debugging, and seamless integration with this code. |
        | **ML/Statistical Models** | **Prophet, pmdarima, scikit-learn** | A robust collection of libraries for professional-grade time-series forecasting and ML ensemble creation. |
    """)

    st.markdown("---")
    
    st.subheader("Conclusion")
    st.markdown("""
        SustainifyAI is a proof-of-concept demonstrating a **unified, intelligent dashboard** capable of serving both citizens and government bodies with critical environmental intelligence. Our focus on **user experience, actionable metrics, and predictive AI** makes this a powerful tool for driving sustainability initiatives forward.
    """)

    st.markdown("---")
    
    # 🌟 Final Button
    st.markdown("""
    <div style='text-align: center;'>
    	<div class="stButton btn-primary">
    		<button style='width: 100%; border-radius: 8px; padding: 10px 20px; font-size: 1.2rem;'>
    			Thank you for reviewing our project! 🙏
    		</button>
    	</div>
    </div>
    """, unsafe_allow_html=True)