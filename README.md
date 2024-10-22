# Macro Report App

[![CI Pipeline](https://github.com/kustex/Macro-Report/actions/workflows/ci.yml/badge.svg)](https://github.com/kustex/Macro-Report/actions/workflows/ci.yml)
[![CD Pipeline](https://github.com/kustex/Macro-Report/actions/workflows/cd.yml/badge.svg)](https://github.com/kustex/Macro-Report/actions/workflows/cd.yml)

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
├── src        # Main Python scripts to run the app and display dashboard
├── res        # Folder that contains csv files to retrieve necessary tickers.
├── test       # Python script to test functionalities for functions inside classes
├── .gitignore
├── Dockerfile  
├── requirements.txt
└── README.md                 
```

## Installation

### 1. Clone the Repository

Clone the repository to your local machine & make it the current directory in terminal:

```bash
git clone https://github.com/kustex/Macro_Report_App.git
cd Macro_Report_App
```

### 3. Create & activate a virtual environment 
```bash
python -m venv venv
source venv/bin/activate 
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Running the App
1. Prepare Data Files: Ensure that the necessary data files (tickers/*.csv and tickers_corr/correlations.csv) are available.
2. Have an instance of Interactive Brokers' TWS running. Make sure you are subscribed to their data tier & change the socket port for establishing API connection to '7496' in Configure/API/Settings. 
3. Run the App: Execute the main Python script to start the app:
```bash
python src/main.py
```
### 6. Access the App: Open your web browser and go to:
```arduino
http://localhost:5000/
```



