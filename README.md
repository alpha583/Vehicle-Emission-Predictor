Vehicle Emission Prediction App using Streamlit and Random Forest
# 🏎️ Automotive Data Intelligence & Emission Predictor (1984-2024)
Live Demo
https://vehicle-emission-predictor.streamlit.app/

## 📌 Project Overview
An end-to-end Data Science and Business Intelligence platform designed to analyze 40 years of automotive evolution (46,000+ vehicle records). This tool combines **Predictive Analytics (Machine Learning)** with **Market Intelligence** to help users understand brand growth, efficiency trends, and environmental impact.

## 🚀 Key Features

### 1. 🤖 ML-Powered Emission Predictor
- **Personalized Predictions:** Uses a tuned **Random Forest Regressor** to predict CO2 emissions based on user-defined engine specs (Displacement, Cylinders, Fuel Type, etc.).
- **Data Integrity:** Implementation of professional Scikit-Learn pipelines for robust preprocessing and handling categorical variables.

### 2. 📊 Market & Manufacturer Insights
- **Top-10 Leaderboard:** Dynamic analysis of top manufacturers filtered by fuel segments (Diesel, Electric, Gasoline).
- **Multi-Brand Comparison:** Interactive tool for head-to-head comparison of brand performance and fuel efficiency.
- **Year-Based Toggles:** Full control over temporal data analysis ranging from 1984 to 2024.

### 3. 📉 Brand Evolution & Scorecards
- **Portfolio Growth:** Visualization of brand expansion over decades.
- **Efficiency Improvement Trends:** Analyzing how manufacturers reduced emissions while maintaining performance.
- **Brand Growth Scorecard:** Comparative metrics for brand dominance and innovation speed.

### 4. 💡 Recommendation Engine
- Smart suggestions for brands and segments based on historical performance and efficiency benchmarks.

## 🛠️ Tech Stack
- **Languages:** Python
- **Libraries:** Pandas, Scikit-Learn, NumPy, Matplotlib, Seaborn
- **Deployment:** Streamlit Cloud
- **VCS:** GitHub

## 📊 Methodology & Best Practices
## 🛠️ Development & Methodology
- **AI-Assisted Development:** Leveraged **GitHub Copilot** as a pair-programmer to accelerate boilerplate coding, unit testing, and UI structure implementation.
- **Human-in-the-Loop:** While AI assisted with syntax and structure, the core logic for **Data Leakage Prevention**, **Feature Engineering**, and **Market Intelligence metrics** was manually designed and verified to ensure professional standards and data accuracy.
- **Data Cleaning:** Automated imputation and handling of missing values (especially for older vehicle records).
- **Feature Engineering:** Developed custom "Efficiency Scores" to track technological advancements.
- **Leakage Prevention:** Removed performance-dependent variables (like MPG) during training to ensure the model focuses purely on engine architecture.

---
**Developed by Rahul Kumar** [LinkedIn](https://www.linkedin.com/in/rahul-kumar-64b2151ba?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3B9sGtkcboS%2FqT8U9ViyKYZw%3D%3D)) 
