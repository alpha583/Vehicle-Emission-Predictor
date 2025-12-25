import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')
import os
import joblib

# Configure page
st.set_page_config(
    page_title="Vehicle Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ============ LOAD & PREPARE DATA ============
@st.cache_data
def load_data():
    path = r'C:\Users\user\Downloads\Only Python\Vehicle Fuel Economy Data.csv'
    if not os.path.exists(path):
        st.error(f"Data file not found: {path}")
        st.stop()
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()
    return df

@st.cache_data
def prepare_ml_data(df):
    """Prepare features and target for ML models using engine/vehicle attributes.
    Returns X (DataFrame), y (Series) and cleaned dataframe copy.
    """
    ml_df = df.copy()
    # Target: CO2 emissions (use 'co2' column)
    target_col = 'co2'

    # Features to use (engine and vehicle attributes) - include Year
    feature_cols = ['Year', 'displ', 'cylinders', 'fuelType', 'trany', 'VClass']

    # Remove mileage / derived mpg columns to avoid leakage
    drop_patterns = ('city', 'highway', 'comb', 'mpg')
    drop_cols = [c for c in ml_df.columns if c.lower().startswith(drop_patterns) or 'mpg' in c.lower()]
    if drop_cols:
        ml_df = ml_df.drop(columns=drop_cols, errors='ignore')

    # Coerce numeric columns
    ml_df['displ'] = pd.to_numeric(ml_df.get('displ'), errors='coerce')
    ml_df['cylinders'] = pd.to_numeric(ml_df.get('cylinders'), errors='coerce')
    ml_df['Year'] = pd.to_numeric(ml_df.get('Year'), errors='coerce')

    # Fill categorical NAs
    for c in ['fuelType', 'trany', 'VClass']:
        if c in ml_df.columns:
            ml_df[c] = ml_df[c].fillna('Unknown')
        else:
            ml_df[c] = 'Unknown'

    # Drop rows missing target
    ml_df = ml_df.dropna(subset=[target_col])

    # Keep only rows with at least numeric engine info (imputation handled later)
    X = ml_df[feature_cols].copy()
    y = ml_df[target_col].astype(float).copy()

    return X, y, ml_df

# Load data
df = load_data()

# Identify EV segment
def categorize_vehicle_segment(df):
    """Categorize vehicles into segments"""
    df['Segment'] = 'Other'
    
    # EV/Electric
    df.loc[(df['fuelType'].str.contains('Electricity', na=False)) | 
           (df['fuelType1'].str.contains('Electricity', na=False)), 'Segment'] = 'Electric'
    
    # Hybrid
    df.loc[(df['fuelType'].str.contains('Hybrid', na=False)) | 
           (df['fuelType1'].str.contains('Hybrid', na=False)), 'Segment'] = 'Hybrid'
    
    # Diesel
    df.loc[(df['fuelType'].str.contains('Diesel', na=False)) | 
           (df['fuelType1'].str.contains('Diesel', na=False)), 'Segment'] = 'Diesel'
    
    # Premium Gasoline
    df.loc[(df['fuelType'].str.contains('Premium', na=False)) | 
           (df['fuelType1'].str.contains('Premium', na=False)), 'Segment'] = 'Premium Gasoline'
    
    # Regular Gasoline (default for remaining)
    df.loc[df['Segment'] == 'Other', 'Segment'] = 'Regular Gasoline'
    
    return df

df = categorize_vehicle_segment(df)

# ============ SIDEBAR FILTERS ============
st.sidebar.title("🎛️ Filters & Options")
st.sidebar.divider()

# Filter sections
with st.sidebar:
    st.subheader("Data Filters")
    
    # Year range filter
    min_year, max_year = st.slider(
        "Select Year Range",
        int(df['Year'].min()),
        int(df['Year'].max()),
        (int(df['Year'].min()), int(df['Year'].max())),
        key="year_slider"
    )
    
    # Segment filter
    segments = ['All'] + sorted(df['Segment'].unique().tolist())
    selected_segment = st.selectbox("Vehicle Segment", segments, key="segment_select")
    
    # Manufacturer filter
    manufacturers = ['All'] + sorted(df['Manufacturer'].unique().tolist())
    selected_manufacturer = st.selectbox("Manufacturer", manufacturers, key="mfg_select")
    
    # Fuel Economy range filter
    if 'comb08' in df.columns:
        min_mpg = st.number_input("Min Fuel Economy (MPG)", 
                                   value=float(df['comb08'].min()), 
                                   key="min_mpg")
        max_mpg = st.number_input("Max Fuel Economy (MPG)", 
                                   value=float(df['comb08'].max()), 
                                   key="max_mpg")
    
    # CO2 filter
    if 'co2' in df.columns:
        max_co2 = st.number_input("Max CO2 Emissions (g/km)", 
                                   value=float(df['co2'].max()), 
                                   key="max_co2")
    
    st.divider()
    
    # Analysis Options
    st.subheader("Analysis Options")
    
    analysis_tabs_select = st.multiselect(
        "Select Analysis Views",
        ["Market Overview", "Competitive Analysis", "Brand Comparison", 
         "Segment Analysis", "Growth Trends", "ML Predictions"],
        default=["Market Overview"],
        key="analysis_select"
    )

# ============ APPLY FILTERS ============
filtered_df = df.copy()

# Apply year filter
filtered_df = filtered_df[(filtered_df['Year'] >= min_year) & (filtered_df['Year'] <= max_year)]

# Apply segment filter
if selected_segment != 'All':
    filtered_df = filtered_df[filtered_df['Segment'] == selected_segment]

# Apply manufacturer filter
if selected_manufacturer != 'All':
    filtered_df = filtered_df[filtered_df['Manufacturer'] == selected_manufacturer]

# Apply fuel economy filter
if 'comb08' in filtered_df.columns:
    filtered_df = filtered_df[(filtered_df['comb08'] >= min_mpg) & (filtered_df['comb08'] <= max_mpg)]

# Apply CO2 filter
if 'co2' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['co2'] <= max_co2]

# ============ HEADER WITH KEY METRICS ============
st.title("🚗 Vehicle Fuel Economy & Competitive Analytics")
st.divider()

# Display metrics based on filtered data
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Vehicles Analyzed", len(filtered_df), f"of {len(df)} total")

with col2:
    st.metric("Manufacturers", filtered_df['Manufacturer'].nunique())

with col3:
    avg_mpg = filtered_df['comb08'].mean() if 'comb08' in filtered_df.columns else 0
    st.metric("Avg Fuel Economy", f"{avg_mpg:.1f} MPG")

with col4:
    avg_co2 = filtered_df['co2'].mean() if 'co2' in filtered_df.columns else 0
    st.metric("Avg CO2", f"{avg_co2:.0f} g/km")

with col5:
    models_count = filtered_df['Model'].nunique()
    st.metric("Models", models_count)

st.divider()

# ============ MARKET OVERVIEW TAB ============
if "Market Overview" in analysis_tabs_select:
    st.header("📊 Market Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Manufacturers by Volume")
        top_mfg = filtered_df['Manufacturer'].value_counts().head(10)
        if top_mfg.empty:
            st.info("No manufacturers match the current filters.")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            top_mfg.plot(kind='barh', ax=ax, color='steelblue')
            ax.set_xlabel('Number of Models')
            ax.set_ylabel('Manufacturer')
            ax.set_title('Market Share by Volume')
            st.pyplot(fig)
            plt.close()
    
    with col2:
        st.subheader("Vehicles by Segment")
        segment_dist = filtered_df['Segment'].value_counts()
        if segment_dist.empty:
            st.info("No segment data available for the selected filters.")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
            ax.pie(segment_dist.values, labels=segment_dist.index, autopct='%1.1f%%', 
                   colors=colors[:len(segment_dist)], startangle=90)
            ax.set_title('Market Distribution by Segment')
            st.pyplot(fig)
            plt.close()
    
    st.divider()
    
    # Year-over-year trend
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fuel Economy Trend Over Years")
        yearly_mpg = filtered_df.groupby('Year')['comb08'].agg(['mean', 'min', 'max']) if 'comb08' in filtered_df.columns else pd.DataFrame()
        if yearly_mpg.empty:
            st.info('No fuel economy data available for the selected filters.')
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(yearly_mpg.index, yearly_mpg['mean'], marker='o', linewidth=2.5, 
                markersize=8, color='green', label='Average')
            ax.fill_between(yearly_mpg.index, yearly_mpg['min'], yearly_mpg['max'], 
                    alpha=0.2, color='green', label='Range')
            ax.set_xlabel('Year')
            ax.set_ylabel('Combined Fuel Economy (MPG)')
            ax.set_title('Fuel Economy Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
    
    with col2:
        st.subheader("CO2 Emissions Trend Over Years")
        yearly_co2 = filtered_df.groupby('Year')['co2'].agg(['mean', 'min', 'max']) if 'co2' in filtered_df.columns else pd.DataFrame()
        if yearly_co2.empty:
            st.info('No CO2 data available for the selected filters.')
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(yearly_co2.index, yearly_co2['mean'], marker='s', linewidth=2.5, 
                markersize=8, color='red', label='Average')
            ax.fill_between(yearly_co2.index, yearly_co2['min'], yearly_co2['max'], 
                    alpha=0.2, color='red', label='Range')
            ax.set_xlabel('Year')
            ax.set_ylabel('CO2 Emissions (g/km)')
            ax.set_title('Emissions Reduction Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

# ============ COMPETITIVE ANALYSIS TAB ============
if "Competitive Analysis" in analysis_tabs_select:
    st.header("🏆 Competitive Analysis")
    
    # EV Segment Competitive Analysis
    if 'Segment' in filtered_df.columns and 'Electric' in filtered_df['Segment'].unique():
        st.subheader("⚡ Electric Vehicle (EV) Segment - Competitive Landscape")
        
        ev_df = filtered_df[filtered_df['Segment'] == 'Electric']
        
        if len(ev_df) > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("EV Models Available", len(ev_df))
            with col2:
                st.metric("EV Manufacturers", ev_df['Manufacturer'].nunique())
            with col3:
                avg_ev_mpg = ev_df['comb08'].mean()
                st.metric("Avg EV Efficiency", f"{avg_ev_mpg:.1f} MPG")
            with col4:
                avg_ev_co2 = ev_df['co2'].mean()
                st.metric("Avg EV CO2", f"{avg_ev_co2:.0f} g/km")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top EV Manufacturers")
                top_ev_mfg = ev_df['Manufacturer'].value_counts().head(8)
                fig, ax = plt.subplots(figsize=(10, 6))
                top_ev_mfg.plot(kind='barh', ax=ax, color='#4ECDC4')
                ax.set_xlabel('Number of Models')
                ax.set_title('EV Market Share by Manufacturer')
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.subheader("EV Efficiency Leaders")
                top_ev_eff = ev_df.groupby('Manufacturer')['comb08'].mean().nlargest(8)
                fig, ax = plt.subplots(figsize=(10, 6))
                top_ev_eff.plot(kind='barh', ax=ax, color='#45B7D1')
                ax.set_xlabel('Average MPG (Efficiency)')
                ax.set_title('Most Efficient EV Manufacturers')
                st.pyplot(fig)
                plt.close()
        else:
            st.info("No Electric vehicles in selected filter range")
    
    st.divider()
    
    # Hybrid Segment Analysis
    if 'Segment' in filtered_df.columns and 'Hybrid' in filtered_df['Segment'].unique():
        st.subheader("🔄 Hybrid Vehicle Segment Analysis")
        
        hybrid_df = filtered_df[filtered_df['Segment'] == 'Hybrid']
        
        if len(hybrid_df) > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Hybrid Models", len(hybrid_df))
            with col2:
                st.metric("Hybrid Manufacturers", hybrid_df['Manufacturer'].nunique())
            with col3:
                avg_hybrid_mpg = hybrid_df['comb08'].mean()
                st.metric("Avg Hybrid MPG", f"{avg_hybrid_mpg:.1f}")
            with col4:
                avg_hybrid_co2 = hybrid_df['co2'].mean()
                st.metric("Avg Hybrid CO2", f"{avg_hybrid_co2:.0f}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top Hybrid Manufacturers")
                top_hybrid = hybrid_df['Manufacturer'].value_counts().head(8)
                fig, ax = plt.subplots(figsize=(10, 6))
                top_hybrid.plot(kind='barh', ax=ax, color='#FFA07A')
                ax.set_xlabel('Number of Models')
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.subheader("Hybrid Efficiency Ranking")
                hybrid_eff = hybrid_df.groupby('Manufacturer')['comb08'].mean().nlargest(8)
                fig, ax = plt.subplots(figsize=(10, 6))
                hybrid_eff.plot(kind='barh', ax=ax, color='#98D8C8')
                ax.set_xlabel('Average MPG')
                st.pyplot(fig)
                plt.close()
        else:
            st.info("No Hybrid vehicles in selected filter range")

# ============ BRAND COMPARISON TAB ============
if "Brand Comparison" in analysis_tabs_select:
    st.header("🔄 Multi-Brand Comparison Analysis")
    
    # Select brands to compare
    available_brands = sorted(filtered_df['Manufacturer'].unique().tolist())
    
    if len(available_brands) > 0:
        selected_brands = st.multiselect(
            "Select 2-5 Brands to Compare",
            available_brands,
            default=available_brands[:3] if len(available_brands) >= 3 else available_brands,
            max_selections=5,
            key="brand_compare"
        )
        
        if len(selected_brands) > 0:
            comparison_df = filtered_df[filtered_df['Manufacturer'].isin(selected_brands)]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Models Per Brand")
                models_per_brand = comparison_df['Manufacturer'].value_counts()
                fig, ax = plt.subplots(figsize=(10, 6))
                models_per_brand.plot(kind='bar', ax=ax, color='steelblue')
                ax.set_ylabel('Number of Models')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.subheader("Fuel Economy Comparison")
                mpg_by_brand = comparison_df.groupby('Manufacturer')['comb08'].mean().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                mpg_by_brand.plot(kind='bar', ax=ax, color='green')
                ax.set_ylabel('Average MPG')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                st.pyplot(fig)
                plt.close()
            
            with col3:
                st.subheader("CO2 Emissions Comparison")
                co2_by_brand = comparison_df.groupby('Manufacturer')['co2'].mean().sort_values()
                fig, ax = plt.subplots(figsize=(10, 6))
                co2_by_brand.plot(kind='bar', ax=ax, color='red')
                ax.set_ylabel('Average CO2 (g/km)')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                st.pyplot(fig)
                plt.close()
            
            st.divider()
            
            # Detailed Brand Performance Table
            st.subheader("📋 Brand Performance Metrics")
            
            brand_metrics = []
            for brand in selected_brands:
                brand_data = comparison_df[comparison_df['Manufacturer'] == brand]
                
                brand_metrics.append({
                    'Brand': brand,
                    'Models': len(brand_data),
                    'Avg MPG': f"{brand_data['comb08'].mean():.1f}",
                    'Best MPG': f"{brand_data['comb08'].max():.1f}",
                    'Worst MPG': f"{brand_data['comb08'].min():.1f}",
                    'Avg CO2': f"{brand_data['co2'].mean():.0f}",
                    'Best CO2': f"{brand_data['co2'].min():.0f}",
                    'Avg Year': f"{brand_data['Year'].mean():.0f}"
                })
            
            metrics_table = pd.DataFrame(brand_metrics)
            st.dataframe(metrics_table, width='stretch')
            
            st.divider()
            
            # Brand Positioning & Strengths
            st.subheader("💡 Brand Analysis & Recommendations")
            
            for brand in selected_brands:
                brand_data = comparison_df[comparison_df['Manufacturer'] == brand]
                
                with st.expander(f"📌 {brand} - Detailed Analysis"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(f"{brand} - Efficiency Rank", 
                                  f"{len([b for b in selected_brands if comparison_df[comparison_df['Manufacturer']==b]['comb08'].mean() >= brand_data['comb08'].mean()])} / {len(selected_brands)}")
                        
                        st.metric(f"{brand} - Emissions Rank", 
                                  f"{len([b for b in selected_brands if comparison_df[comparison_df['Manufacturer']==b]['co2'].mean() <= brand_data['co2'].mean()])} / {len(selected_brands)}")
                    
                    with col2:
                        st.metric(f"{brand} - Model Diversity", 
                                  f"{len(brand_data)} models")
                        
                        avg_year = brand_data['Year'].mean()
                        st.metric(f"{brand} - Portfolio Age", 
                                  f"{avg_year:.0f}")
                    
                    # Segment distribution
                    segment_dist = brand_data['Segment'].value_counts()
                    st.write(f"**Segment Distribution:**")
                    for seg, count in segment_dist.items():
                        st.write(f"  • {seg}: {count} models ({count/len(brand_data)*100:.1f}%)")
                    
                    # Growth analysis
                    if len(brand_data['Year'].unique()) > 1:
                        yearly_data = brand_data.groupby('Year')['comb08'].mean()
                        growth = yearly_data.iloc[-1] - yearly_data.iloc[0] if len(yearly_data) > 1 else 0
                        st.write(f"**Fuel Economy Improvement:** {growth:+.1f} MPG")
                    
                    # Areas for enhancement
                    overall_best_mpg = comparison_df['comb08'].max()
                    brand_avg_mpg = brand_data['comb08'].mean()
                    gap = overall_best_mpg - brand_avg_mpg
                    
                    st.write(f"**Enhancement Opportunity:** Need to improve by ~{gap:.1f} MPG to match leader")

# ============ SEGMENT ANALYSIS TAB ============
if "Segment Analysis" in analysis_tabs_select:
    st.header("📊 Vehicle Segment Deep Dive")
    
    segments = sorted(filtered_df['Segment'].unique().tolist())
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Segments", len(segments))
    
    with col2:
        largest_segment = filtered_df['Segment'].value_counts().idxmax()
        st.metric("Largest Segment", largest_segment)
    
    with col3:
        most_efficient_segment = filtered_df.groupby('Segment')['comb08'].mean().idxmax()
        st.metric("Most Efficient", most_efficient_segment)
    
    with col4:
        lowest_co2_segment = filtered_df.groupby('Segment')['co2'].mean().idxmin()
        st.metric("Lowest Emissions", lowest_co2_segment)
    
    st.divider()
    
    # Segment comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Segment Size & Growth")
        segment_size = filtered_df['Segment'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        segment_size.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_ylabel('Number of Models')
        ax.set_xlabel('Segment')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Fuel Economy by Segment")
        segment_mpg = filtered_df.groupby('Segment')['comb08'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        segment_mpg.plot(kind='bar', ax=ax, color='green')
        ax.set_ylabel('Average MPG')
        ax.set_xlabel('Segment')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)
        plt.close()
    
    st.divider()
    
    # Detailed segment analysis
    for segment in segments:
        segment_data = filtered_df[filtered_df['Segment'] == segment]
        
        with st.expander(f"🔍 {segment} Segment Analysis"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Models", len(segment_data))
            with col2:
                st.metric("Manufacturers", segment_data['Manufacturer'].nunique())
            with col3:
                st.metric("Avg MPG", f"{segment_data['comb08'].mean():.1f}")
            with col4:
                st.metric("Avg CO2", f"{segment_data['co2'].mean():.0f} g/km")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top Manufacturers in this Segment:**")
                top_mfg = segment_data['Manufacturer'].value_counts().head(5)
                for mfg, count in top_mfg.items():
                    st.write(f"  • {mfg}: {count} models")
            
            with col2:
                st.write("**Top Performers:**")
                top_perf = segment_data.nlargest(5, 'comb08')[['Manufacturer', 'Model', 'Year', 'comb08']]
                for idx, row in top_perf.iterrows():
                    st.write(f"  • {row['Manufacturer']} {row['Model']} ({row['Year']:.0f}): {row['comb08']:.1f} MPG")

# ============ GROWTH TRENDS TAB ============
if "Growth Trends" in analysis_tabs_select:
    st.header("📈 Growth & Evolution Analysis")
    
    # Overall market growth
    col1, col2, col3 = st.columns(3)
    
    yearly_summary = filtered_df.groupby('Year').agg({
        'Manufacturer': 'count',
        'comb08': 'mean',
        'co2': 'mean'
    }).rename(columns={'Manufacturer': 'Total_Models'})
    
    if len(yearly_summary) > 1:
        growth_models = yearly_summary['Total_Models'].iloc[-1] - yearly_summary['Total_Models'].iloc[0]
        growth_mpg = yearly_summary['comb08'].iloc[-1] - yearly_summary['comb08'].iloc[0]
        reduction_co2 = yearly_summary['co2'].iloc[0] - yearly_summary['co2'].iloc[-1]
        
        with col1:
            st.metric("Portfolio Growth", f"{growth_models:+.0f} models", 
                     f"{growth_models/yearly_summary['Total_Models'].iloc[0]*100:+.1f}%")
        
        with col2:
            st.metric("Efficiency Improvement", f"{growth_mpg:+.1f} MPG", 
                     f"{growth_mpg/yearly_summary['comb08'].iloc[0]*100:+.1f}%")
        
        with col3:
            st.metric("CO2 Reduction", f"{reduction_co2:+.0f} g/km", 
                     f"{reduction_co2/yearly_summary['co2'].iloc[0]*100:+.1f}%")
    
    st.divider()
    
    # Brand growth trends
    st.subheader("Brand Evolution Over Time")
    
    top_brands_overall = filtered_df['Manufacturer'].value_counts().head(8).index.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Portfolio Size Growth")
        fig, ax = plt.subplots(figsize=(12, 6))
        for brand in top_brands_overall:
            brand_yearly = filtered_df[filtered_df['Manufacturer'] == brand].groupby('Year').size()
            ax.plot(brand_yearly.index, brand_yearly.values, marker='o', label=brand, linewidth=2)
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Models')
        ax.set_title('Brand Portfolio Expansion')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Efficiency Improvement Trends")
        fig, ax = plt.subplots(figsize=(12, 6))
        for brand in top_brands_overall:
            brand_yearly = filtered_df[filtered_df['Manufacturer'] == brand].groupby('Year')['comb08'].mean()
            ax.plot(brand_yearly.index, brand_yearly.values, marker='s', label=brand, linewidth=2)
        ax.set_xlabel('Year')
        ax.set_ylabel('Average Fuel Economy (MPG)')
        ax.set_title('Brand Efficiency Evolution')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    st.divider()
    
    # Individual brand growth metrics
    st.subheader("📊 Brand Growth Scorecard")
    
    growth_metrics = []
    for brand in top_brands_overall:
        brand_data = filtered_df[filtered_df['Manufacturer'] == brand]
        
        if len(brand_data['Year'].unique()) > 1:
            yearly_data = brand_data.groupby('Year').agg({
                'Model': 'count',
                'comb08': 'mean',
                'co2': 'mean'
            })
            
            portfolio_growth = (yearly_data['Model'].iloc[-1] - yearly_data['Model'].iloc[0]) / yearly_data['Model'].iloc[0] * 100
            efficiency_growth = (yearly_data['comb08'].iloc[-1] - yearly_data['comb08'].iloc[0]) / yearly_data['comb08'].iloc[0] * 100
            emission_reduction = (yearly_data['co2'].iloc[0] - yearly_data['co2'].iloc[-1]) / yearly_data['co2'].iloc[0] * 100
            
            growth_metrics.append({
                'Brand': brand,
                'Portfolio Growth %': f"{portfolio_growth:+.1f}%",
                'Efficiency Gain %': f"{efficiency_growth:+.1f}%",
                'Emission Reduction %': f"{emission_reduction:+.1f}%"
            })
    
    if growth_metrics:
        growth_df = pd.DataFrame(growth_metrics)
        st.dataframe(growth_df, width='stretch')

# ============ ML PREDICTIONS TAB ============
if "ML Predictions" in analysis_tabs_select:
    st.header("🤖 CO2 Emissions Prediction (Engine & Vehicle Features)")

    # Prepare data using engine/vehicle features
    X, y, ml_df = prepare_ml_data(filtered_df)

    if len(X) < 20:
        st.warning("⚠️ Not enough data points for robust ML training (need >= 20 rows after filters)")
    else:
        # Define feature groups
        numeric_features = ['Year', 'displ', 'cylinders']
        categorical_features = ['fuelType', 'trany', 'VClass']

        # Preprocessor
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='drop')

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build pipelines for RandomForest and XGBoost
        rf_pipeline = Pipeline(steps=[('pre', preprocessor), ('model', RandomForestRegressor(random_state=42))])
        xgb_pipeline = Pipeline(steps=[('pre', preprocessor), ('model', XGBRegressor(random_state=42, verbosity=0))])

        # Hyperparameter search spaces
        rf_param_dist = {
            'model__n_estimators': [100, 200, 400, 800],
            'model__max_depth': [None, 5, 10, 20],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }

        xgb_param_dist = {
            'model__n_estimators': [100, 200, 400],
            'model__max_depth': [3, 5, 8, 12],
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'model__subsample': [0.6, 0.8, 1.0]
        }

        # Streamlit option: run quick randomized search or use defaults
        tune = st.checkbox('Run hyperparameter tuning (RandomizedSearchCV, may be slow)', value=False)
        n_iter = st.number_input('RandomizedSearchCV iterations', min_value=10, max_value=200, value=30)

        if tune:
            st.info('Running RandomizedSearchCV for RandomForest and XGBoost...')
            rf_search = RandomizedSearchCV(rf_pipeline, rf_param_dist, n_iter=int(n_iter), cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
            rf_search.fit(X_train, y_train)

            xgb_search = RandomizedSearchCV(xgb_pipeline, xgb_param_dist, n_iter=min(25, int(n_iter)), cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
            xgb_search.fit(X_train, y_train)

            best_rf = rf_search.best_estimator_
            best_xgb = xgb_search.best_estimator_
        else:
            # Fit default models for quick results
            best_rf = rf_pipeline.set_params(model__n_estimators=200)
            best_rf.fit(X_train, y_train)

            best_xgb = xgb_pipeline.set_params(model__n_estimators=200)
            best_xgb.fit(X_train, y_train)

        # Evaluate models
        models = {'RandomForest': best_rf, 'XGBoost': best_xgb}
        eval_rows = []
        preds = {}
        for name, mdl in models.items():
            y_pred = mdl.predict(X_test)
            preds[name] = y_pred
            eval_rows.append({
                'Model': name,
                'R2': r2_score(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred)
            })

        # Baseline linear regression on processed features
        lr_pipeline = Pipeline(steps=[('pre', preprocessor), ('model', LinearRegression())])
        lr_pipeline.fit(X_train, y_train)
        y_pred_lr = lr_pipeline.predict(X_test)
        eval_rows.insert(0, {'Model': 'LinearRegression', 'R2': r2_score(y_test, y_pred_lr), 'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)), 'MAE': mean_absolute_error(y_test, y_pred_lr)})

        comparison_df = pd.DataFrame(eval_rows)
        st.subheader('Model Performance Comparison (higher R2 better, lower RMSE/MAE better)')
        st.dataframe(comparison_df.sort_values('R2', ascending=False), width='stretch')

        # Show actual vs predicted for best model (by R2)
        best_model_name = comparison_df.sort_values('R2', ascending=False).iloc[0]['Model']
        best_model = {'LinearRegression': lr_pipeline, 'RandomForest': best_rf, 'XGBoost': best_xgb}[best_model_name]

        st.markdown(f"**Best model:** {best_model_name}")

        # Persist best model pipeline
        save_path = os.path.join(os.getcwd(), 'best_model_pipeline.joblib')
        try:
            joblib.dump(best_model, save_path)
            st.success(f"Best model pipeline saved to: {save_path}")
        except Exception as e:
            st.warning(f"Could not save model pipeline: {e}")

        # Prediction options: allow selecting which trained model to use or upload a pipeline
        st.subheader('Prediction Options')
        model_options = ['Best model (auto)', 'RandomForest', 'XGBoost', 'LinearRegression', 'Load from file']
        predict_choice = st.selectbox('Model to use for custom prediction', model_options, index=0)

        uploaded_model = None
        if predict_choice == 'Load from file':
            uploaded = st.file_uploader('Upload joblib model pipeline', type=['joblib', 'pkl'])
            if uploaded is not None:
                try:
                    uploaded_model = joblib.load(uploaded)
                    st.success('Uploaded model pipeline loaded for predictions.')
                except Exception as e:
                    st.error(f'Failed to load uploaded model: {e}')

        # determine which model to use for prediction
        model_for_prediction = best_model
        if uploaded_model is not None:
            model_for_prediction = uploaded_model
        else:
            if predict_choice == 'RandomForest':
                model_for_prediction = best_rf
            elif predict_choice == 'XGBoost':
                model_for_prediction = best_xgb
            elif predict_choice == 'LinearRegression':
                model_for_prediction = lr_pipeline

        st.subheader('Actual vs Predicted (Selected Model)')
        y_pred_best = model_for_prediction.predict(X_test)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred_best, alpha=0.4)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel('Actual CO2')
        ax.set_ylabel('Predicted CO2')
        st.pyplot(fig)
        plt.close()

        st.divider()

        # Feature importance: aggregate importance for original features
        st.subheader('Feature Importance (aggregated)')
        # Use the tree-based best model if available, else permutation importance
        if best_model_name in ['RandomForest', 'XGBoost']:
            tree = best_model.named_steps['model']
            # get preprocessor to extract feature names
            pre = best_model.named_steps['pre']
            # fit preprocessor on full training data to get feature names
            pre.fit(X_train)
            # numeric names
            num_names = numeric_features
            # categorical one-hot names
            cat_ohe = pre.named_transformers_['cat'].named_steps['onehot']
            try:
                cat_names = list(cat_ohe.get_feature_names_out(categorical_features))
            except Exception:
                # fallback
                cat_names = []
            feature_names = num_names + cat_names

            importances = tree.feature_importances_
            # aggregate to original feature level
            agg = {}
            for fname, imp in zip(feature_names, importances):
                base = fname
                for orig in numeric_features + categorical_features:
                    if fname.startswith(orig):
                        base = orig
                        break
                agg[base] = agg.get(base, 0) + imp

            agg_df = pd.DataFrame([{'Feature': k, 'Importance': v} for k, v in agg.items()])
            agg_df = agg_df.sort_values('Importance', ascending=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(agg_df['Feature'], agg_df['Importance'], color='steelblue')
            ax.set_xlabel('Aggregated Importance')
            st.pyplot(fig)
            plt.close()

            st.write('**Reasoning:** Higher importance values indicate stronger model contribution to CO2 prediction; engine size (displ) and cylinders are often primary drivers for higher CO2 due to larger displacement and more cylinders leading to higher fuel consumption and emissions.')
        else:
            # Permutation importance fallback
            st.info('Computing permutation importance (model-agnostic)')
            r = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
            feat_names = numeric_features + categorical_features
            perm_df = pd.DataFrame({'Feature': feat_names, 'Importance': r.importances_mean})
            perm_df = perm_df.sort_values('Importance', ascending=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(perm_df['Feature'], perm_df['Importance'], color='steelblue')
            st.pyplot(fig)
            plt.close()

        st.divider()

        # Prediction form for users to input vehicle specs
        st.subheader('Predict CO2 for a Custom Vehicle')
        with st.form('predict_form'):
            d_year = st.number_input('Year', value=int(X['Year'].median()), step=1)
            d_displ = st.number_input('Engine displacement (displ)', value=float(X['displ'].median()), step=0.1)
            d_cyl = st.number_input('Cylinders', value=int(X['cylinders'].median()), step=1)
            fuel_opts = sorted(ml_df['fuelType'].fillna('Unknown').unique().tolist())
            trany_opts = sorted(ml_df['trany'].fillna('Unknown').unique().tolist())
            vclass_opts = sorted(ml_df['VClass'].fillna('Unknown').unique().tolist())
            d_fuel = st.selectbox('Fuel Type', fuel_opts)
            d_trany = st.selectbox('Transmission (trany)', trany_opts)
            d_vclass = st.selectbox('Vehicle Class (VClass)', vclass_opts)
            submit = st.form_submit_button('Predict CO2')

        if submit:
            input_df = pd.DataFrame([{ 'Year': d_year, 'displ': d_displ, 'cylinders': d_cyl, 'fuelType': d_fuel, 'trany': d_trany, 'VClass': d_vclass }])
            try:
                pred_co2 = model_for_prediction.predict(input_df)[0]
                st.success(f'Predicted CO2 emissions: {pred_co2:.1f} g/km')
            except Exception as e:
                st.error(f'Prediction failed: {e}')
            st.info('This prediction is based on the selected trained model and the features: displ, cylinders, fuelType, trany, VClass.')

# ============ FOOTER ============
st.divider()
st.write("*© 2024 Vehicle Fuel Economy Analytics Dashboard | Data-Driven Insights for the Automotive Industry*")
