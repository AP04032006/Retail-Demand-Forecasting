
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

BASE_PATH = Path(__file__).parent

# -----------------------------------
# CUSTOM DASHBOARD DESIGN
# -----------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #0E1726;
    color: #FFFFFF;
}

[data-testid="stSidebar"] {
    background-color: #172337;
}

.metric-card {
    background-color: #1B2A41;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #30445F;
}

h1, h2, h3 {
    color: #FFFFFF;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# LOAD PROJECT FILES
# -----------------------------------

@st.cache_resource
def load_model():
    return joblib.load(
        BASE_PATH / "demand_forecasting_model.pkl"
    )

@st.cache_data
def load_history():
    data = pd.read_csv(BASE_PATH / "history.csv")
    data["Date"] = pd.to_datetime(data["Date"])
    return data

@st.cache_data
def load_results():
    return pd.read_csv(BASE_PATH / "model_results.csv")

model_bundle = load_model()
history_df = load_history()
results_df = load_results()

model = model_bundle["model"]
feature_columns = model_bundle["feature_columns"]

# -----------------------------------
# FORECASTING FUNCTION
# -----------------------------------

def generate_forecast(selected_sku, forecast_days):

    product_history = history_df[
        history_df["SKU"].astype(str) == str(selected_sku)
    ].sort_values("Date")

    demand_history = (
        product_history["Daily_Demand"]
        .astype(float)
        .tolist()
    )

    last_date = history_df["Date"].max()

    forecast_results = []

    for day in range(1, forecast_days + 1):

        forecast_date = last_date + pd.Timedelta(days=day)

        row = {
            column: 0
            for column in feature_columns
        }

        row["Day"] = forecast_date.day
        row["Month"] = forecast_date.month
        row["DayOfWeek"] = forecast_date.dayofweek
        row["IsWeekend"] = int(forecast_date.dayofweek >= 5)

        row["Lag_1"] = demand_history[-1]
        row["Lag_7"] = demand_history[-7]
        row["Lag_14"] = demand_history[-14]

        row["Rolling_7"] = np.mean(demand_history[-7:])
        row["Rolling_30"] = np.mean(demand_history[-30:])

        sku_column = "SKU_" + str(selected_sku)

        if sku_column in feature_columns:
            row[sku_column] = 1

        input_data = pd.DataFrame(
            [row],
            columns=feature_columns
        )

        prediction = model.predict(input_data)[0]

        prediction = max(0, float(prediction))

        demand_history.append(prediction)

        forecast_results.append({
            "Date": forecast_date,
            "Predicted_Demand": round(prediction, 2)
        })

    return pd.DataFrame(forecast_results)

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("📊 Demand Forecasting")

st.sidebar.markdown("---")

st.sidebar.subheader("Forecast Settings")

sku_list = sorted(
    history_df["SKU"].dropna().unique().tolist(),
    key=str
)

selected_sku = st.sidebar.selectbox(
    "Select Product (SKU)",
    sku_list
)

forecast_days = st.sidebar.selectbox(
    "Forecast Period",
    [7, 14, 30]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Machine Learning powered retail demand forecasting."
)

# -----------------------------------
# MAIN DASHBOARD
# -----------------------------------

st.title("📈 Retail Demand Forecasting Dashboard")

st.markdown(
    "### Machine Learning Based Sales & Demand Prediction"
)

st.markdown("---")

# -----------------------------------
# GENERATE FORECAST
# -----------------------------------

forecast_df = generate_forecast(
    selected_sku,
    forecast_days
)

# -----------------------------------
# KPI METRICS
# -----------------------------------

total_products = history_df["SKU"].nunique()

total_demand = history_df["Daily_Demand"].sum()

average_forecast = forecast_df[
    "Predicted_Demand"
].mean()

total_forecast = forecast_df[
    "Predicted_Demand"
].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Products",
    f"{total_products:,}"
)

col2.metric(
    "Historical Units",
    f"{total_demand:,.0f}"
)

col3.metric(
    "Forecast Demand",
    f"{total_forecast:,.1f}"
)

col4.metric(
    "Average Daily Forecast",
    f"{average_forecast:,.2f}"
)

st.markdown("---")

# -----------------------------------
# FORECAST SECTION
# -----------------------------------

st.subheader("🔮 Product Demand Forecast")

st.write(
    f"Showing {forecast_days}-day forecast for SKU: {selected_sku}"
)

chart_df = forecast_df.rename(
    columns={"Predicted_Demand": "Demand"}
)

chart_df["Type"] = "Predicted Demand"

fig = px.line(
    chart_df,
    x="Date",
    y="Demand",
    markers=True,
    title="Future Product Demand Forecast",
    template="plotly_dark"
)

fig.update_traces(
    line=dict(width=3)
)

fig.update_layout(
    paper_bgcolor="#0E1726",
    plot_bgcolor="#172337",
    font_color="white",
    xaxis_title="Forecast Date",
    yaxis_title="Predicted Quantity"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------------
# FORECAST TABLE
# -----------------------------------

st.subheader("📋 Forecast Details")

display_forecast = forecast_df.copy()

display_forecast["Date"] = (
    display_forecast["Date"].dt.strftime("%d-%m-%Y")
)

st.dataframe(
    display_forecast,
    use_container_width=True,
    hide_index=True
)

# -----------------------------------
# DOWNLOAD FORECAST
# -----------------------------------

csv_data = forecast_df.to_csv(index=False)

st.download_button(
    label="⬇️ Download Forecast Report",
    data=csv_data,
    file_name="demand_forecast.csv",
    mime="text/csv"
)

st.markdown("---")

# -----------------------------------
# MODEL PERFORMANCE
# -----------------------------------

st.subheader("🏆 Machine Learning Model Performance")

st.dataframe(
    results_df.round(3),
    use_container_width=True,
    hide_index=True
)

best_model_name = model_bundle["model_name"]

st.success(
    f"Selected forecasting model: {best_model_name}"
)

# -----------------------------------
# HISTORICAL DEMAND
# -----------------------------------

st.subheader("📊 Historical Product Demand")

product_history = history_df[
    history_df["SKU"].astype(str) == str(selected_sku)
].sort_values("Date").tail(30)

historical_fig = px.line(
    product_history,
    x="Date",
    y="Daily_Demand",
    markers=True,
    title="Last 30 Days of Historical Demand",
    template="plotly_dark"
)

historical_fig.update_layout(
    paper_bgcolor="#0E1726",
    plot_bgcolor="#172337",
    font_color="white"
)

st.plotly_chart(
    historical_fig,
    use_container_width=True
)

# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown("---")

st.markdown(
    "<center>Retail Demand Forecasting | "
    "Machine Learning Project</center>",
    unsafe_allow_html=True
)
