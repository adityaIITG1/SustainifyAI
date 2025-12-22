# 🌍 SustainifyAI - Comprehensive Sustainability & Climate Intelligence Platform

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> **GreenTech Hackathon Submission** | Making Environmental Data Accessible, Visual, and Actionable

---

## 🎯 Project Summary

**SustainifyAI** is a comprehensive web-based sustainability intelligence platform that bridges the gap between raw environmental data and actionable insights for citizens, policymakers, and environmental advocates. By consolidating real-time air quality monitoring, historical climate trends, satellite imagery analysis, and personalized carbon footprint tracking, we empower users to make data-driven decisions for a greener future.

### 🌱 Environmental Problem We Solve

Climate change and environmental degradation are accelerating, yet critical environmental data remains scattered, complex, and inaccessible to the average citizen. Decision-makers lack unified platforms to visualize threats, track progress, and implement evidence-based sustainability policies.

### 💡 Our Solution

A **unified, interactive dashboard** that:
- **Visualizes** real-time air quality and climate data for 100+ Indian cities
- **Predicts** future climate scenarios using advanced forecasting models
- **Monitors** critical ecosystems through satellite imagery analysis
- **Tracks** personal and city-level carbon footprints
- **Recommends** specific, actionable interventions for sustainability

### 👥 Target Users

1. **Citizens** - Track local air quality, understand climate impacts, reduce carbon footprint
2. **Government Officials** - Data-driven policy decisions, resource allocation, progress tracking
3. **Environmental NGOs** - Evidence for advocacy, awareness campaigns, community engagement
4. **Researchers** - Historical data analysis, trend identification, impact assessment

### 🌟 Anticipated Impact

- **Awareness**: 10,000+ users accessing real-time environmental data monthly
- **Action**: 25% reduction in personal carbon footprints through our calculator
- **Policy**: Evidence-based interventions saving 100,000 tons CO2e annually
- **Conservation**: Early threat detection protecting 7 critical ecosystems

---

## ✨ Key Features

### 1. 🌬️ Real-Time Air Quality Intelligence
- **Live AQI Monitoring** for 100+ cities across India
- **Dynamic Solution Engine** - AI-powered recommendations based on pollution patterns
  - Emergency protocols for hazardous AQI (300+)
  - NCR crisis response strategies
  - Industrial belt pollution control measures
- **NASA EONET Integration** - Global environmental event tracking (wildfires, dust storms)
- **Severity Classification** - Color-coded alerts (Good → Hazardous)

### 2. 📈 Climate Trends & Historical Analysis
- **30-Year Historical Data** from ERA5 Climate Reanalysis
- **Interactive Visualizations**:
  - Temperature trends with warming indicators
  - Precipitation patterns and drought analysis
  - Extreme weather event detection
  - Monthly climate norms vs. current conditions
- **Anomaly Detection** - Automated identification of unusual climate patterns
- **Correlation Matrix** - Understand relationships between climate variables

### 3. 🔮 AI-Powered Climate Forecasting
- **Multi-Model Ensemble** combining:
  - Prophet (seasonal patterns)
  - ARIMA (time-series analysis)
  - Random Forest (non-linear trends)
- **12-Month Predictions** for temperature and precipitation
- **Model Validation** with MAE and MAPE metrics
- **Confidence Intervals** for forecast reliability

### 4. 🛰️ Satellite Ecosystem Monitoring (NEW!)
- **Real Satellite Imagery** from ArcGIS World Imagery
- **AI Vision Analysis** detecting:
  - Deforestation percentage
  - Illegal mining sites
  - Urban encroachment
  - Vegetation health assessment
- **4 Interactive Threat Graphs**:
  - Threat distribution donut chart
  - Vegetation health gauge (0-100)
  - Severity metrics bar chart
  - AI confidence indicator
- **7 Critical Ecosystems Tracked**:
  - Aravali Range (desertification barrier)
  - Western Ghats (biodiversity hotspot)
  - Sundarbans (mangrove forest)
  - Hasdeo Arand (coal mining threat)
  - Jim Corbett, Kaziranga, Dehing Patkai

### 5. 💯 Sustainability Scoring System
- **Composite Score** (0-100) based on:
  - Air Quality Index (30%)
  - Water Quality Index (25%)
  - Green Cover Percentage (20%)
  - Waste Management Efficiency (15%)
  - Renewable Energy Share (10%)
- **City Rankings** - Compare sustainability across regions
- **Progress Tracking** - Monitor improvements over time
- **Actionable Recommendations** - Specific steps to improve scores

### 6. 🧍 Personal Carbon Footprint Calculator
- **India-Specific Baselines** adjusted by city environmental metrics
- **Intelligent Auto-Estimation** using:
  - City air quality for diet proxy
  - Population density for travel patterns
  - Temperature swings for electricity use
- **Breakdown Analysis**:
  - Travel, electricity, flights, LPG, diet
  - Recycling impact calculation
  - Renewable energy adjustment
- **Interactive Pie Chart** showing emission sources

### 7. 🔋 Green Infrastructure Dashboard
- **Clean Energy Mix** visualization (Solar, Wind, Hydro)
- **EV Adoption Tracking** across cities
- **Solar Connection Growth** monitoring
- **Comparative Analysis** - Multi-city donut charts

### 8. 🗑️ Waste Management Intelligence
- **3D Geographic Visualization** of waste generation (TPD)
- **4R Waste Model** recommendations (Reduce, Reuse, Recycle, Recover)
- **Plastic Waste Tracking**
- **Predicted Trend Analysis**

### 9. 🧼 Swachh Survekshan Ranking
- **5-Year Historical Trends** for cleanliness rankings
- **Multi-City Comparison** (UP cities + national toppers)
- **Progress Visualization** with inverted axis (lower rank = better)

---

## 🛠️ Tech Stack

### Frontend & Visualization
- **Streamlit** - Interactive web application framework
- **Plotly** - Dynamic, responsive charts and graphs
- **PyDeck** - 3D geospatial visualizations
- **ECharts** - Advanced map visualizations

### Data Processing & Analysis
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Prophet** - Time-series forecasting (seasonal patterns)
- **pmdarima** - ARIMA model implementation
- **scikit-learn** - Machine learning (Random Forest ensemble)

### APIs & Data Sources
- **Open-Meteo** - Historical climate data (ERA5) & real-time AQI
- **ArcGIS World Imagery** - Satellite imagery
- **Gemini Vision API** - AI-powered image analysis
- **NASA EONET** - Global environmental events
- **WAQI** - Air quality data

### Database & Caching
- **SQLite** - Local data persistence
- **Streamlit Cache** - Performance optimization

### Development Tools
- **Python 3.10+**
- **Git** - Version control
- **VS Code** - Development environment

---

## 🚀 Installation & Setup

### Prerequisites
```bash
- Python 3.10 or higher
- pip (Python package manager)
- Git
```

### Step 1: Clone Repository
```bash
git clone https://github.com/adityaIITG1/sustainify-AI.git
cd sustainify-AI
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure API Keys
Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
SAMBANOVA_API_KEY = "your-sambanova-key-here"
```

**Get API Keys:**
- Gemini: [Google AI Studio](https://aistudio.google.com/app/apikey)
- SambaNova: [SambaNova Cloud](https://cloud.sambanova.ai/)

### Step 4: Run Application
```bash
streamlit run SustainifyAI.py
```

Access at: `http://localhost:8501`

---

## 📊 Data Sources & Attribution

| Data Type | Source | License | Update Frequency |
|-----------|--------|---------|------------------|
| Historical Climate | ERA5 (via Open-Meteo) | CC BY 4.0 | Daily |
| Real-time AQI | WAQI / Open-Meteo | Public | Hourly |
| Satellite Imagery | ArcGIS World Imagery | Esri Terms | On-demand |
| Environmental Events | NASA EONET | Public Domain | Real-time |
| City Data | Open Government Data | OGL | Static |

---

## 🎥 Demo & Screenshots

### Dashboard Overview
![Dashboard](https://via.placeholder.com/800x400?text=SustainifyAI+Dashboard)

### Air Quality Intelligence
![AQI](https://via.placeholder.com/800x400?text=Real-Time+AQI+Monitoring)

### Satellite Analysis
![Satellite](https://via.placeholder.com/800x400?text=AI+Vision+Satellite+Analysis)

### Climate Forecasting
![Forecast](https://via.placeholder.com/800x400?text=12-Month+Climate+Predictions)

**📹 Full Demo Video**: [Watch on YouTube](#) *(Coming Soon)*

---

## 💻 Usage Guide

### For Citizens
1. **Check Local Air Quality**: Navigate to "Air Quality" tab, search your city
2. **Track Climate Trends**: View 30-year historical data in "Climate Trends"
3. **Calculate Carbon Footprint**: Use "Personal Carbon" tab with auto-estimation
4. **Monitor Ecosystems**: Explore "Nature Guardian" for satellite analysis

### For Policymakers
1. **Sustainability Scoring**: Compare cities in "Sustainability Score" tab
2. **Waste Management**: Analyze TPD data and 4R recommendations
3. **Green Infrastructure**: Track renewable energy and EV adoption
4. **Forecasting**: Use 12-month predictions for infrastructure planning

### For Researchers
1. **Historical Analysis**: Download climate data from "Climate Trends"
2. **Anomaly Detection**: Identify unusual patterns in temperature/precipitation
3. **Correlation Studies**: Use correlation matrix for variable relationships
4. **Satellite Monitoring**: Track ecosystem changes over time

---

## 🏆 Hackathon Alignment: GreenTech Theme

### ✅ Environmental Impact
- **Energy Conservation**: Tracks renewable energy share, promotes efficiency
- **Waste Reduction**: 4R model recommendations, waste-to-energy insights
- **Environmental Awareness**: Real-time AQI, climate trends, ecosystem threats
- **Resource Tracking**: Carbon footprint calculator, water quality monitoring
- **Renewable Energy**: Solar/wind capacity visualization, EV adoption tracking

### ✅ Innovation
- **Multi-Model Forecasting**: Combines 3 ML models for accurate predictions
- **Satellite AI Vision**: Detects threats from real imagery (unique approach)
- **Dynamic Solution Engine**: Context-aware recommendations based on pollution patterns
- **Intelligent Auto-Estimation**: Uses city metrics to estimate carbon footprints

### ✅ Technical Quality
- **Robust Architecture**: Modular design, error handling, caching
- **Performance Optimized**: Lazy loading, 24-hour cache, efficient queries
- **Data Validation**: MAE/MAPE metrics for forecast accuracy
- **Scalable**: Supports 100+ cities, extensible to more regions

### ✅ User Experience
- **Intuitive Navigation**: 12 organized tabs, clear information hierarchy
- **Visual Excellence**: Glassmorphism design, animated elements, color-coded alerts
- **Accessibility**: High-contrast colors, readable fonts, responsive layout
- **Interactive**: Hover tooltips, clickable maps, dynamic graphs

### ✅ Feasibility
- **Free APIs**: No cost barriers for deployment
- **Open Data**: All sources publicly available and legal
- **Lightweight**: Runs on standard hardware, minimal server requirements
- **Deployment Ready**: Streamlit Cloud compatible, Docker support

---

## 🌟 What Makes SustainifyAI Stand Out

### 1. **Comprehensive Integration**
Unlike single-purpose tools, we unify air quality, climate, satellite, and carbon tracking in one platform.

### 2. **Actionable Intelligence**
Not just data visualization - we provide specific, context-aware recommendations for improvement.

### 3. **Real-World Impact**
- **Aravali Range**: Detects illegal mining threatening Delhi's air quality
- **NCR Crisis**: Emergency protocols for hazardous pollution events
- **Carbon Reduction**: Personalized footprint calculator with India-specific baselines

### 4. **Scalable & Extensible**
- Easy to add new cities, data sources, or features
- Modular codebase for team collaboration
- API-first design for future integrations

### 5. **Evidence-Based Design**
- All metrics validated against scientific standards
- Transparent data sources and methodologies
- Confidence scores for AI predictions

---

## 📈 Future Roadmap

### Phase 1 (Current)
- [x] Real-time AQI monitoring
- [x] Climate forecasting
- [x] Satellite ecosystem analysis
- [x] Carbon footprint calculator

### Phase 2 (Next 3 Months)
- [ ] Mobile app (React Native)
- [ ] Community reporting (citizen science)
- [ ] Gamification (eco-challenges, leaderboards)
- [ ] Multi-language support (Hindi, Bengali, Tamil)

### Phase 3 (6-12 Months)
- [ ] IoT sensor integration (air quality monitors)
- [ ] Blockchain-based carbon credits
- [ ] Corporate sustainability dashboards
- [ ] Government API for policy integration

---

## 👥 Team: Nxt Gen Developers

- **Aditya Kumar Singh** - Lead Developer, Backend Architecture
- **Gaurang Verma** - Data Science, Forecasting Models
- **Ujjwal Singh** - Frontend Design, Visualization
- **Prakriti Jaiswal** - API Integration, Testing
- **Saurabh Shukla** - Documentation, Deployment

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Open-Meteo** for free climate data API
- **NASA EONET** for environmental event tracking
- **Esri** for ArcGIS World Imagery
- **Google** for Gemini Vision API
- **Streamlit** for the amazing framework
- **Open-source community** for libraries and tools

---

## 📞 Contact & Support

- **GitHub**: [adityaIITG1/sustainify-AI](https://github.com/adityaIITG1/sustainify-AI)
- **Email**: [Your Email]
- **LinkedIn**: [Your LinkedIn]
- **Demo**: [Live Demo Link]

---

## 🌍 Join the Green Revolution

**SustainifyAI** is more than a dashboard - it's a movement towards data-driven environmental action. Every insight generated, every alert raised, and every carbon footprint reduced brings us closer to a sustainable future.

### 🚀 Try it now: [Live Demo](#)

---

<div align="center">

**Made with 💚 for a Greener Planet**

⭐ Star this repo if you believe in sustainable technology!

</div>

