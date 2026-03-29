import streamlit as st
import requests
import pandas as pd
import numpy as np
from typing import Optional

# Page configuration
st.set_page_config(
    page_title=" Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B6B;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
# CHANGE THIS TO YOUR RENDER API URL AFTER DEPLOYMENT
# Local: http://localhost:8000
# Render: https://housing-price-api.onrender.com
API_URL = "http://localhost:8000"

# ============================================================================
# TITLE & INTRO
# ============================================================================
st.title("Housing Price Prediction")
st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    st.write("""
    **Predict California housing prices** using a trained Machine Learning model.
    
    This model was trained on the California Housing Dataset with:
    - 📊 9 features (location, age, rooms, income, etc.)
    - 🎯 HistGradientBoosting algorithm
    - 📈 R² Score: 0.57 (Test Set Performance)
    - 📉 RMSE: ~$73,000
    """)
with col2:
    st.metric("Model Type", "HistGradientBoosting", "Tuned")
    st.metric("R² Score", "0.57", "+0.15 from baseline")

st.markdown("---")


st.sidebar.title("⚙️ Navigation")
prediction_type = st.sidebar.radio(
    "Select Mode:",
    ["Single Prediction", "Batch Prediction", "Feature Information", "API Status"]
)


if prediction_type == "Single Prediction":
    st.subheader("📝 Enter House Features")
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Location & Age**")
        longitude = st.number_input(
            "Longitude", 
            value=-122.230,
            min_value=-125.0,
            max_value=-114.0,
            step=0.001,
            help="X coordinate (East-West), Range: -125 to -114"
        )
        latitude = st.number_input(
            "Latitude",
            value=37.880,
            min_value=32.0,
            max_value=42.0,
            step=0.001,
            help="Y coordinate (North-South), Range: 32 to 42"
        )
        housing_median_age = st.number_input(
            "Housing Median Age (years)",
            value=41,
            min_value=1,
            max_value=52,
            step=1,
            help="Age of housing block in years"
        )
    
    with col2:
        st.write("**Building Details**")
        total_rooms = st.number_input(
            "Total Rooms",
            value=880,
            min_value=1,
            step=10,
            help="Total number of rooms in the block"
        )
        total_bedrooms = st.number_input(
            "Total Bedrooms",
            value=129,
            min_value=0,
            step=10,
            help="Total number of bedrooms (0 if unknown)"
        )
        households = st.number_input(
            "Number of Households",
            value=126,
            min_value=1,
            step=10,
            help="Number of households in the block"
        )
    
    st.divider()
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("**Demographics & Income**")
        population = st.number_input(
            "Population",
            value=322,
            min_value=1,
            step=10,
            help="Number of residents"
        )
        median_income = st.number_input(
            "Median Income",
            value=8.3252,
            min_value=0.5,
            max_value=15.0,
            step=0.1,
            help="Median income in tens of thousands (e.g., 8.3 = $83k)"
        )
    
    with col4:
        st.write("**Location Type**")
        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
            index=3,
            help="Categorical feature: distance/type relative to ocean"
        )
        st.info(" Different ocean proximities have different price impacts")
    
    st.divider()
    
    # Prediction button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        predict_clicked = st.button(" Predict Price", use_container_width=True, type="primary")
    
    with col_btn2:
        clear_clicked = st.button("🔄 Clear", use_container_width=True)
    
    if clear_clicked:
        st.rerun()
    
    if predict_clicked:
        with st.spinner("🔄 Making prediction... This may take a few seconds"):
            try:
                # Prepare request payload
                payload = {
                    "longitude": float(longitude),
                    "latitude": float(latitude),
                    "housing_median_age": float(housing_median_age),
                    "total_rooms": float(total_rooms),
                    "total_bedrooms": float(total_bedrooms) if total_bedrooms > 0 else None,
                    "population": float(population),
                    "households": float(households),
                    "median_income": float(median_income),
                    "ocean_proximity": ocean_proximity
                }
                
                # Call API
                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display result
                    st.success("Prediction Complete!", icon="✅")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            " Predicted Price (USD)",
                            f"${result['predicted_price']:,.0f}",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            "💰 In Lakhs (INR)",
                            f"₹{result['price_in_lakhs']:.2f} L",
                            delta=None
                        )
                    
                    with col3:
                        st.metric(
                            "📍 Location (Lat, Long)",
                            f"{latitude:.3f}°, {longitude:.3f}°",
                            delta=None
                        )
                    
                    st.divider()
                    
                    # Display input features used
                    st.subheader("📊 Input Features Summary:")
                    
                    features_df = pd.DataFrame({
                        "Feature": [
                            "Longitude",
                            "Latitude",
                            "Housing Age",
                            "Total Rooms",
                            "Total Bedrooms",
                            "Population",
                            "Households",
                            "Median Income",
                            "Ocean Proximity"
                        ],
                        "Value": [
                            f"{longitude:.3f}",
                            f"{latitude:.3f}",
                            f"{housing_median_age} years",
                            f"{int(total_rooms)} rooms",
                            f"{int(total_bedrooms)} rooms",
                            f"{int(population)} people",
                            f"{int(households)} households",
                            f"${median_income*10:.1f}k",
                            ocean_proximity
                        ]
                    })
                    
                    st.dataframe(
                        features_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Feature": st.column_config.TextColumn(width="medium"),
                            "Value": st.column_config.TextColumn(width="medium"),
                        }
                    )
                
                else:
                    error_detail = response.json().get('detail', 'Unknown error')
                    st.error(f"API Error: {error_detail}", icon="❌")
            
            except requests.exceptions.ConnectionError:
                st.error(
                    f" **Connection Error**: Cannot reach API at {API_URL}\n\n"
                    "**Make sure to:**\n"
                    "1. Run FastAPI backend locally: `python backend/app.py`\n"
                    "2. OR update API_URL to your Render deployment URL\n"
                    f"3. Current API_URL: `{API_URL}`",
                    icon=""
                )
            except requests.exceptions.Timeout:
                st.error("❌ API request timed out. The server took too long to respond.", icon="⏱️")
            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}", icon="❌")


elif prediction_type == "Batch Prediction":
    st.subheader("📂 Bulk Prediction from CSV")
    
    st.markdown("""
    Upload a CSV file with the following columns (in any order):
    - `longitude` - East-West location (-125 to -114)
    - `latitude` - North-South location (32 to 42)
    - `housing_median_age` - Age in years (1 to 52)
    - `total_rooms` - Total rooms
    - `total_bedrooms` - Total bedrooms (optional)
    - `population` - Number of residents
    - `households` - Number of households
    - `median_income` - Income in tens of thousands
    - `ocean_proximity` - One of: <1H OCEAN, INLAND, ISLAND, NEAR BAY, NEAR OCEAN
    """)
    
    
    uploaded_file = st.file_uploader(
        "📤 Upload CSV file",
        type=["csv"],
        help="Select a CSV file with housing data"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Display file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(" Rows", df.shape[0])
            with col2:
                st.metric(" Columns", df.shape[1])
            with col3:
                st.metric("✓ Valid", "Yes" if df.shape[0] > 0 else "No")
            
            st.write("**Preview (first 5 rows):**")
            st.dataframe(df.head(), use_container_width=True)
            
            # Prediction button
            col_pred, col_space = st.columns([1, 3])
            
            with col_pred:
                predict_batch_clicked = st.button(
                    "🔮 Predict All",
                    use_container_width=True,
                    type="primary"
                )
            
            if predict_batch_clicked:
                with st.spinner(f" Making {df.shape[0]} predictions... Please wait"):
                    try:
                        # Convert to list of dicts
                        requests_list = df.to_dict('records')
                        
                        # Call batch API
                        response = requests.post(
                            f"{API_URL}/predict-batch",
                            json=requests_list,
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Add predictions to dataframe
                            df['Predicted_Price_USD'] = result['predictions']
                            df['Predicted_Price_Lakhs'] = df['Predicted_Price_USD'] / 100000
                            
                            st.success(f" Successfully predicted {result['count']} houses!", icon="✅")
                            
                            # Display results
                            st.subheader(" Predictions Results")
                            st.dataframe(df, use_container_width=True)
                            
                            # Download button
                            csv_data = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions as CSV",
                                data=csv_data,
                                file_name="housing_predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                            # Summary stats
                            st.subheader("📈 Prediction Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric(
                                    "Average Price",
                                    f"${df['Predicted_Price_USD'].mean():,.0f}"
                                )
                            with col2:
                                st.metric(
                                    "Min Price",
                                    f"${df['Predicted_Price_USD'].min():,.0f}"
                                )
                            with col3:
                                st.metric(
                                    "Max Price",
                                    f"${df['Predicted_Price_USD'].max():,.0f}"
                                )
                            with col4:
                                st.metric(
                                    "Std Dev",
                                    f"${df['Predicted_Price_USD'].std():,.0f}"
                                )
                        
                        else:
                            error_detail = response.json().get('detail', 'Unknown error')
                            st.error(f"❌ API Error: {error_detail}", icon="❌")
                    
                    except requests.exceptions.ConnectionError:
                        st.error(
                            f"❌ Cannot connect to API at {API_URL}\n\n"
                            "Make sure FastAPI is running or update API_URL",
                            icon="❌"
                        )
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}", icon="❌")
        
        except Exception as e:
            st.error(f"❌ Error reading CSV: {str(e)}", icon="❌")


elif prediction_type == "Feature Information":
    st.subheader("📚 Understanding the Model")
    
    tab1, tab2, tab3 = st.tabs(["Features", "Model Info", "Tips"])
    
    with tab1:
        st.markdown("""
        ### 🏡 Input Features
        
        | Feature | Type | Range | Description |
        |---------|------|-------|-------------|
        | **Longitude** | Numeric | -125 to -114 | East-West location (California) |
        | **Latitude** | Numeric | 32 to 42 | North-South location (California) |
        | **Housing Median Age** | Numeric | 1 to 52 years | Age of housing block (newer = higher value) |
        | **Total Rooms** | Numeric | 6 to 39,320 | Sum of all room types in the block |
        | **Total Bedrooms** | Numeric | 0 to 6,445 | Number of bedrooms (optional, can be missing) |
        | **Population** | Numeric | 3 to 35,682 | Number of residents in the block |
        | **Households** | Numeric | 1 to 6,082 | Number of households |
        | **Median Income** | Numeric | 0.5 to 15.0 | Scaled median income (in tens of thousands) |
        | **Ocean Proximity** | Categorical | 5 values | Distance/type relative to ocean |
        
        ### 🏷️ Ocean Proximity Categories
        - **<1H OCEAN**: Less than 1 hour to ocean (highest value)
        - **NEAR OCEAN**: Near the ocean
        - **NEAR BAY**: Near the bay
        - **INLAND**: Inland areas
        - **ISLAND**: Island locations
        """)
    
    with tab2:
        st.markdown("""
        ### 🎯 Model Performance
        
        **Algorithm**: HistGradientBoosting Regressor (Tuned)
        
        **Training Performance**:
        - RMSE: ~$73,000
        - MAE: ~$52,000
        - R² Score: 0.58
        
        **Test Performance**:
        - RMSE: ~$73,000
        - MAE: ~$52,000
        - R² Score: 0.57
        
        **What does R² = 0.57 mean?**
        - The model explains **57%** of the variance in house prices
        - Remaining 43% is due to other factors not in the dataset
        - Good performance for real estate prediction
        
        ### 🔧 Model Architecture
        
        ```
        Input Data (9 features)
                ↓
        Preprocessing Pipeline:
        - Numerical features: Imputation (median) → Scaling
        - Categorical features: Imputation (mode) → One-hot encoding
                ↓
        HistGradientBoosting Regressor
        - Learning rate: 0.1
        - Max depth: None (unbounded)
        - Max leaf nodes: 63
        - Min samples per leaf: 20
        - L2 regularization: 0.1
                ↓
        Predicted House Price
        ```
        """)
    
    with tab3:
        st.markdown("""
        ### 💡 How to Get Better Predictions
        
        1. **Enter accurate location data**
           - Longitude and latitude are the strongest predictors
           - Use precise coordinates if possible
        
        2. **Include median income**
           - Income has the highest correlation with price
           - Even rough estimates help
        
        3. **Don't leave bedrooms blank**
           - If unknown, estimate based on total rooms
           - The model will impute if missing, but estimates are better
        
        4. **Consider ocean proximity carefully**
           - Proximity to ocean is a strong price driver
           - Choose the most accurate category
        
        5. **Note: Model Limitations**
           - Trained on California housing data only
           - May not work for other regions
           - Doesn't account for recent market changes
           - No information about: condition, amenities, views, school quality
        
        ### ⚠️ Important Notes
        
        - **This is a demonstration model** for educational purposes
        - Actual house prices depend on many unmeasured factors
        - Use predictions as a rough estimate, not final valuation
        - Consult real estate professionals for actual valuations
        """)


elif prediction_type == "API Status":
    st.subheader("🔍 API Status Check")
    
    st.write(f"**Current API URL**: `{API_URL}`")
    st.write("---")
    
    # Check API health
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🔄 Check Health", use_container_width=True, type="secondary"):
            with st.spinner("Checking..."):
                try:
                    response = requests.get(
                        f"{API_URL}/health",
                        timeout=10
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ API is healthy!", icon="✅")
                        st.json(result)
                    else:
                        st.error(f"❌ API returned status {response.status_code}", icon="❌")
                except Exception as e:
                    st.error(f"❌ Cannot reach API: {str(e)}", icon="❌")
    
    st.divider()
    
    
    st.subheader("📡 Available Endpoints")
    
    endpoints_data = {
        "Endpoint": [
            "GET /health",
            "GET /",
            "POST /predict",
            "POST /predict-batch",
            "GET /docs"
        ],
        "Description": [
            "Health check",
            "Root endpoint with info",
            "Single house prediction",
            "Batch prediction",
            "Interactive API documentation"
        ],
        "Example URL": [
            f"{API_URL}/health",
            f"{API_URL}/",
            f"{API_URL}/predict",
            f"{API_URL}/predict-batch",
            f"{API_URL}/docs"
        ]
    }
    
    endpoints_df = pd.DataFrame(endpoints_data)
    st.dataframe(endpoints_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.subheader("🛠️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        ### 🔗 To Use Different API URL:
        
        1. Edit `streamlit_app.py`
        2. Find this line:
           ```python
           API_URL = "http://localhost:8000"
           ```
        3. Change to:
           ```python
           API_URL = "https://your-render-url.onrender.com"
           ```
        4. Save and restart Streamlit
        """)
    
    with col2:
        st.info("""
        ### 📚 Learn More:
        
        - [FastAPI Documentation](https://fastapi.tiangolo.com)
        - [Streamlit Documentation](https://docs.streamlit.io)
        - [Render Documentation](https://render.com/docs)
        - [scikit-learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
        """)


st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px; padding: 1rem;'>
    🔧 Built with Streamlit |  Powered by HistGradientBoosting | 
    🚀 Deployed on Render & Streamlit Cloud |  California Housing Dataset
    <br><br>
    <em>This is a demonstration project for educational purposes only.</em>
</div>
""", unsafe_allow_html=True)
