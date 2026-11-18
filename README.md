
# Retail Demand Forecasting Using Machine Learning

## Project Overview

This project predicts future retail product demand using
Machine Learning algorithms and historical Amazon sales data.

The project includes an interactive Streamlit dashboard
for exploring products and viewing future demand forecasts.

## Objectives

- Analyze historical retail sales data.
- Clean and preprocess sales records.
- Identify daily product demand patterns.
- Train and compare Machine Learning models.
- Generate future product demand forecasts.
- Display predictions through an interactive dashboard.

## Machine Learning Algorithms

1. Decision Tree Regressor
2. Random Forest Regressor
3. Extra Trees Regressor

The best-performing model is selected using evaluation metrics.

## Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared Score (R2)

## Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Plotly
- Streamlit
- GitHub
- Render

## Dashboard Features

- Interactive product selection.
- 7-day, 14-day and 30-day demand forecasting.
- Machine Learning based predictions.
- Historical demand visualization.
- Model performance comparison.
- Forecast report download.

## Project Structure

```text
deployment/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- demand_forecasting_model.pkl
|-- history.csv
|-- model_results.csv
```

## How to Run Locally

1. Clone the GitHub repository.
2. Install the required libraries:

   pip install -r requirements.txt

3. Start the Streamlit application:

   streamlit run app.py

4. Open the local URL displayed in the terminal.

## Dataset

The project uses an Amazon Sale Report dataset
containing historical retail order information.

## Disclaimer

Forecasts are experimental estimates based on the available
historical dataset. Results may vary with product and time period.

## Author

Machine Learning Academic Project
