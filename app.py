import streamlit as st
import pandas as pd
from pathlib import Path
from streamlit_folium import st_folium

from preprocessing.clean_data import clean_tourism_data
from models.train_model import train_tourism_model, predict_visitors
from visualization.charts import (
    visitors_by_state_chart, monthly_trend_chart,
    visitor_type_chart, top_destinations_chart
)
from visualization.tourism_map import create_tourism_map

st.set_page_config(
    page_title="Smart Tourism Analytics",
    page_icon="🧭",
    layout="wide"
)

st.title("🧭 Smart Tourism Analytics System")
st.caption("Explore tourism trends, analyse visitor data and estimate future visits.")

DEFAULT_DATA = Path(__file__).parent / "data" / "tourism_data.csv"

@st.cache_data
def load_default_data():
    return pd.read_csv(DEFAULT_DATA)

uploaded_file = st.sidebar.file_uploader("Upload your tourism CSV (optional)", type=["csv"])

try:
    raw_data = pd.read_csv(uploaded_file) if uploaded_file else load_default_data()
    cleaned_data = clean_tourism_data(raw_data)
except Exception as error:
    st.error(f"Could not load or clean the dataset: {error}")
    st.info("Use the sample data/tourism_data.csv file as a template for column names.")
    st.stop()

st.sidebar.header("Dashboard Filters")
states = sorted(cleaned_data["state"].dropna().unique().tolist())
selected_states = st.sidebar.multiselect("Select state(s)", states, default=states)
filtered_data = cleaned_data[cleaned_data["state"].isin(selected_states)]

if filtered_data.empty:
    st.warning("No records match the selected filters. Select at least one state.")
    st.stop()

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Visitors", f"{int(filtered_data['total_visitors'].sum()):,}")
col2.metric("Destinations", f"{filtered_data['destination'].nunique():,}")
col3.metric("States", f"{filtered_data['state'].nunique():,}")
col4.metric("Dataset Records", f"{len(filtered_data):,}")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "🧹 Data Preprocessing", "🤖 ML Prediction", "🗺️ Tourism Map"
])

with tab1:
    st.subheader("Tourism Insights")
    left, right = st.columns(2)
    left.plotly_chart(visitors_by_state_chart(filtered_data), use_container_width=True)
    right.plotly_chart(monthly_trend_chart(filtered_data), use_container_width=True)
    left, right = st.columns(2)
    left.plotly_chart(visitor_type_chart(filtered_data), use_container_width=True)
    right.plotly_chart(top_destinations_chart(filtered_data), use_container_width=True)
    st.subheader("Tourism Dataset Preview")
    st.dataframe(filtered_data, use_container_width=True)

with tab2:
    st.subheader("Data Cleaning and Preprocessing")
    st.write("The application standardizes column names, converts numeric fields, removes duplicate rows, fills missing values and creates total_visitors.")
    a, b, c = st.columns(3)
    a.metric("Rows in uploaded/original data", len(raw_data))
    b.metric("Rows after cleaning", len(cleaned_data))
    c.metric("Duplicate rows removed", max(0, len(raw_data) - len(cleaned_data)))
    st.write("Cleaned dataset")
    st.dataframe(cleaned_data, use_container_width=True)
    csv_bytes = cleaned_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download cleaned CSV",
        data=csv_bytes,
        file_name="cleaned_tourism_data.csv",
        mime="text/csv"
    )

with tab3:
    st.subheader("Tourist Visit Prediction")
    st.warning("This is a student project demonstration. Predictions depend on the quality and size of the supplied dataset; they are not official tourism forecasts.")
    try:
        model, metrics = train_tourism_model(cleaned_data)
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{metrics['MAE']:,.0f}")
        m2.metric("RMSE", f"{metrics['RMSE']:,.0f}")
        m3.metric("R² score", f"{metrics['R2']:.3f}" if pd.notna(metrics["R2"]) else "N/A")
        st.caption(f"Evaluation used {metrics['test_rows']} test records. MAE/RMSE are visitor-count errors; R² can be unstable on small datasets.")

        available_destinations = sorted(cleaned_data["destination"].unique())
        available_states = sorted(cleaned_data["state"].unique())
        with st.form("prediction_form"):
            year = st.number_input("Year", min_value=2020, max_value=2040, value=2027, step=1)
            month = st.selectbox("Month", list(range(1, 13)), index=0)
            state = st.selectbox("State", available_states)
            destinations_for_state = sorted(
                cleaned_data.loc[cleaned_data["state"] == state, "destination"].unique()
            )
            destination = st.selectbox("Destination", destinations_for_state)
            submitted = st.form_submit_button("Predict visitors")
        if submitted:
            estimate = predict_visitors(model, year, month, state, destination)
            st.success(f"Estimated visitors: {estimate:,}")
    except Exception as error:
        st.error(f"Model could not be trained: {error}")
        st.info("Add at least 8 valid records and check that all required columns are present.")

with tab4:
    st.subheader("Interactive India Tourism Map")
    tourism_map = create_tourism_map(filtered_data)
    st_folium(tourism_map, width=None, height=550, returned_objects=[])

st.sidebar.markdown("---")
st.sidebar.caption("Academic demonstration project • Smart Tourism Analytics")
