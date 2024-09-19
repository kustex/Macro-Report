# Macro Report App

## Overview

The **Macro Report App** provides a daily overview of changes in macro financial markets across different time horizons. It tracks performance metrics and analyzes correlations between major asset classes. This tool is ideal for financial analysts, investors, and anyone interested in macroeconomic trends.

### Key Features:
- **Performance Monitoring**: Track the performance of various sectors, industries, and macroeconomic indicators.
- **Correlation Analysis**: Analyze changes in correlations between major asset classes over different time horizons.
- **Daily Updates**: Data is updated daily for fresh insights.

## Project Structure

```plaintext
Macro_Report_App/
│
├── app_performance.py        # Main Python script to run the app and display dashboard
├── data_performance.py       # Script to establish connection with TWS API and retrieve data
├── requirements.txt          # Python dependencies
├── tickers/                  # Directory containing contact details for performance tickers grouped by sector in csv files
├── tickers_corr/             # Directory containing contract details for correlation tickers in csv files
└── README.md                 # This README file
```

## Installation

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/kustex/Macro_Report_App.git
cd Macro_Report_App
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Running the App
1. Prepare Data Files: Ensure that the necessary data files (tickers/*.csv and tickers_corr/correlations.csv) are available.
2. Have an instance of Interactive Brokers' TWS running. Make sure you are subscribed to their data tier & change the socket port for establishing API connection to '7496' in Configure/API/Settings. 
3. Run the App: Execute the main Python script to start the app:
```bash
python app_performance.py
```
### 4. Access the App: Open your web browser and go to:
```arduino
http://127.0.0.1:8050/
```



