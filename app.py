import streamlit as st
import pandas as pd
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Global Emission Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium design
st.markdown("""
    <style>
    /* Main background and font styling */
    .reportview-container {
        background: #f0f2f6;
    }
    
    /* Header card */
    .header-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .header-box h1 {
        margin: 0;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        color: white;
    }
    .header-box p {
        margin: 5px 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
        color: #e0e0e0;
    }
    
    /* Prediction output card */
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.05);
        border-left: 6px solid #4CAF50;
        margin-bottom: 20px;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        color: #2e7d32;
        line-height: 1;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Quick stats cards */
    .stat-card {
        background: white;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.02);
        border: 1px solid #eaeaea;
        margin-bottom: 15px;
    }
    .stat-num {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e3c72;
    }
    .stat-desc {
        font-size: 0.85rem;
        color: #777;
    }
    </style>
""", unsafe_allow_html=True)

# Load data helper
@st.cache_data
def load_data():
    df = pd.read_csv('emissions.csv')
    # Generate mapping from code/name
    mapping = df[['CountryCode', 'CountryName']].drop_duplicates().sort_values('CountryName')
    return df, mapping

# Load model helper
@st.cache_resource
def load_model():
    if not os.path.exists('model_pipeline.joblib') and os.path.exists('model_pipeline.zip'):
        import zipfile
        with zipfile.ZipFile('model_pipeline.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
            
    if os.path.exists('model_pipeline.joblib'):
        return joblib.load('model_pipeline.joblib')
    return None

df, mapping = load_data()
model = load_model()

# Header banner
st.markdown("""
    <div class="header-box">
        <h1>🌍 Global Emission Predictor</h1>
        <p>Analyze historical greenhouse gas emissions and predict future CO2 equivalent output using Machine Learning.</p>
    </div>
""", unsafe_allow_html=True)

# Layout division
col_sidebar, col_main = st.columns([1, 2.5], gap="large")

with col_sidebar:
    st.subheader("🔮 Prediction Inputs")
    
    # Country Dropdown mapping Name to Code
    country_options = [f"{row['CountryName']} ({row['CountryCode']})" for _, row in mapping.iterrows()]
    selected_country_str = st.selectbox(
        "Select Country",
        options=country_options,
        index=0
    )
    # Extract selected code
    selected_code = selected_country_str.split('(')[-1].strip(')')
    selected_name = selected_country_str.split('(')[0].strip()
    
    # Year Selection Slider
    selected_year = st.slider(
        "Target Year",
        min_value=1990,
        max_value=2035,
        value=2026,
        step=1
    )
    
    # Emission Level Selection
    selected_level = st.selectbox(
        "Desired Emission Level Target",
        options=['Low', 'Medium', 'High'],
        index=0
    )
    
    st.divider()
    
    # Quick info sidebar
    st.markdown("### 📋 Instructions")
    st.info("""
    1. Adjust variables to define target scenario.
    2. Model estimates **Total Greenhouse Gas Emissions (ktCO2e)** dynamically.
    3. Check prediction against the country's historical timeline in the graph.
    """)

with col_main:
    if model is None:
        st.warning("⚠️ Model file `model_pipeline.joblib` not found. Please train the model using `train_model.py` first.")
    else:
        # Run prediction
        input_df = pd.DataFrame([{
            'CountryCode': selected_code,
            'Year': selected_year,
            'EmissionLevel': selected_level
        }])
        
        prediction = model.predict(input_df)[0]
        
        # Display prediction results
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Greenhouse Gas Emissions</div>
                <div class="metric-value">{prediction:,.2f} <span style="font-size:1.5rem; font-weight:500; color:#555;">ktCO2e</span></div>
                <p style="margin: 5px 0 0 0; color:#555;">Estimated scenario for <b>{selected_name}</b> in <b>{selected_year}</b> with <b>{selected_level}</b> target.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Historical context & visualizations
        st.subheader("📈 Historical Trends & Projection Context")
        
        # Filter historical data for selected country
        country_hist = df[df['CountryCode'] == selected_code].sort_values('Year')
        
        if not country_hist.empty:
            # Create a line chart of historical emissions
            chart_data = country_hist[['Year', 'Emissions_ktCO2e']].copy()
            chart_data = chart_data.set_index('Year')
            
            # Show historical stats
            avg_emission = country_hist['Emissions_ktCO2e'].mean()
            max_emission = country_hist['Emissions_ktCO2e'].max()
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-desc">Historical Average</div>
                        <div class="stat-num">{avg_emission:,.1f} kt</div>
                    </div>
                """, unsafe_allow_html=True)
            with col_stat2:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-desc">Highest Recorded</div>
                        <div class="stat-num">{max_emission:,.1f} kt</div>
                    </div>
                """, unsafe_allow_html=True)
            with col_stat3:
                # Calculate change compared to history average
                pct_change = ((prediction - avg_emission) / avg_emission) * 100 if avg_emission > 0 else 0
                arrow = "🔺" if pct_change > 0 else "🔻"
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-desc">Comparison to Average</div>
                        <div class="stat-num">{arrow} {abs(pct_change):,.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
                
            # Plot
            st.markdown(f"**Historical Emissions for {selected_name} (1990 - 2028)**")
            st.line_chart(chart_data)
            
            # Detailed data table in expander
            with st.expander("📄 View Historical Data Table"):
                st.dataframe(country_hist[['Year', 'Emissions_ktCO2e', 'EmissionLevel']].reset_index(drop=True), use_container_width=True)
        else:
            st.info(f"No historical emissions records found for {selected_name}.")
