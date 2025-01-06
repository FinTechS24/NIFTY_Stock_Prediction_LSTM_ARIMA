import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


def load_data():
    """
    Load and preprocess the dataset.
    """
    df = pd.read_csv("NIFTY 50 - 3 minute_with_indicators_.csv")
    df = df[['date', 'close', 'high', 'low', 'open']]
    df.columns = ["Date", "Close", "High", "Low", "Open"]
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def create_correlation_plots(df):
    """
    Create correlation plots and heatmaps.
    """
    st.subheader("Correlation Analysis")
    st.write(
        "Understanding the relationships between different features in the dataset is crucial. "
        "Correlation plots help us identify how different variables are related."
    )

    # Correlation heatmap
    corr = df.drop('Date', axis=1).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation between variables')
    st.pyplot(fig)

    # Pair plot
    st.write("### Pair Plot")
    st.write("Visualizing pairwise relationships between variables using a scatterplot matrix.")
    fig = sns.pairplot(df.drop('Date', axis=1), palette='rainbow')
    st.pyplot(fig)


def plot_stock_distributions(df):
    """
    Create box plots for stock price distributions.
    """
    st.subheader("Stock Price Distributions")
    st.write(
        "Box plots provide a quick summary of the distribution of stock prices, "
        "highlighting key statistics like median, quartiles, and potential outliers."
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(
        [df.Close, df.Open, df.High, df.Low],
        patch_artist=True,
        medianprops=dict(color='black'),
    )
    colors = ['lightblue', 'lightgreen', 'pink', 'lightgray']
    for box, color in zip(bp['boxes'], colors):
        box.set_facecolor(color)
    ax.set_title('Stock Prices Distributions')
    ax.set_xticklabels(['Close', 'Open', 'High', 'Low'])
    st.pyplot(fig)


def plot_price_trends(df):
    """
    Plot multi-line price trends for close, open, high, and low prices.
    """
    st.subheader("Price Trends")
    st.write(
        "Below is a multi-line graph representing stock price trends over time, "
        "which helps identify general trends, anomalies, and patterns."
    )
    fig = go.Figure()
    for col in ['Close', 'Open', 'High', 'Low']:
        fig.add_trace(go.Scatter(x=df['Date'], y=df[col], name=col))
    fig.update_layout(
        title='NIFTY Stock Prices From 2015 until 2023',
        xaxis_title='Date',
        yaxis_title='Prices',
        template='plotly_white',
    )
    st.plotly_chart(fig)

    # Observations about price trends
    st.markdown("""
    **Key Observations**:

    - There is a significant drop in stock prices during the **2015-2016 recession** period.
    - A noticeable dip in stock prices occurred in **2016**, attributed to the **demonetization drive** in India.
    - Another sharp decline happened in **2020**, driven by the global **COVID-19 pandemic**.
    - By the end of 2020 through 2023, stock prices showed a steady upward trend, reflecting economic recovery.
    """)


def train_lstm_model(df):
    """
    Train and evaluate an LSTM model for stock price prediction.
    """
    st.subheader("LSTM Model Training")
    st.write(
        "LSTM (Long Short-Term Memory) is a specialized recurrent neural network (RNN) architecture, "
        "known for its ability to capture long-term dependencies in sequential data."
    )

    # Preprocessing
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])
    sequence_length = 60  # Using 60 days of historical data to predict the next value
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM input format

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)

    # Predictions
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    # Actual values
    actual = scaler.inverse_transform(y.reshape(-1, 1))

    # Metrics
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)
    mape = mean_absolute_percentage_error(actual, predictions)
    st.write(f"**Performance Metrics**: RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

    # Plot predictions vs actual values
    st.write("### LSTM Predictions vs Actual")
    fig, ax = plt.subplots()
    ax.plot(actual, label='Actual')
    ax.plot(predictions, label='Predicted')
    ax.legend()
    st.pyplot(fig)

    return rmse, mae, mape


def train_arima_model(df):
    """
    Train and evaluate an ARIMA model for stock price prediction.
    """
    st.subheader("ARIMA Model Training")
    st.write(
        "ARIMA (Autoregressive Integrated Moving Average) is a popular statistical model for time-series analysis, "
        "used for capturing linear dependencies in data."
    )

    # Resample data to daily frequency for ARIMA modeling
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    daily_data = df['Close'].resample('D').mean().dropna()

    # Fit ARIMA model
    model = ARIMA(daily_data, order=(3, 1, 2))  # Example ARIMA(p, d, q) configuration
    results = model.fit()
    st.write("### ARIMA Model Summary")
    st.text(results.summary())

    # Forecast the next 30 days
    forecast = results.forecast(steps=30)
    forecast_index = pd.date_range(start=daily_data.index[-1], periods=30, freq='D')

    # Plot actual vs forecasted data
    st.write("### ARIMA Predictions")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_data, label="Actual")
    ax.plot(forecast_index, forecast, label="Forecast", color="orange")
    ax.legend()
    st.pyplot(fig)

    # Metrics
    st.write("### ARIMA Performance Metrics")
    rmse = np.sqrt(mean_squared_error(daily_data[-30:], forecast[:len(daily_data[-30:])]))
    mae = mean_absolute_error(daily_data[-30:], forecast[:len(daily_data[-30:])])
    mape = mean_absolute_percentage_error(daily_data[-30:], forecast[:len(daily_data[-30:])])
    st.write(f"**Performance Metrics**: RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

    return rmse, mae, mape


def main():
    """
    Main Streamlit application function.
    """
    st.title("Indian Stock Market Analysis and Prediction")
    st.write(
        "This application provides insights into stock market trends and predictions using LSTM and ARIMA models. "
        "Explore interactive visualizations and evaluate model performance."
    )

    # Load dataset
    df = load_data()
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Visualizations
    create_correlation_plots(df)
    plot_stock_distributions(df)
    plot_price_trends(df)

    # Model training buttons
    if st.button("Train LSTM Model"):
        train_lstm_model(df)

    if st.button("Train ARIMA Model"):
        train_arima_model(df)


if __name__ == "__main__":
    main()
