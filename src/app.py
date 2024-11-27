import dash_bootstrap_components as dbc
import dash
import logging
import os
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, html, dcc, State
from dash.dependencies import Input, Output
from dash.dash_table import DataTable
from calculations import StockCalculations 
from stock_data_service import StockDataService
from layout import *
from database_client import DatabaseClient
from datetime import datetime

logging.basicConfig(level=logging.DEBUG)

# Initialize global instances for database and services
# db_client = DatabaseClient(mongo_uri="mongodb://ip-172-31-87-70.ec2.internal:27017", db_name="macro_report")
db_client = DatabaseClient(db_name="macro_report")
ap = StockDataService(db_client)
calc = StockCalculations()

# Set date range
start_date = (datetime.today() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')

# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Macro Report"

# Define the Content Area
content = html.Div(id="page-content", style={"margin-left": "18rem", "padding": "2rem 1rem"})

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    sidebar,
    content,
    dcc.Store(id='data_store'),  
    dcc.Store(id='data_correlation_store'),  
    dcc.Store(id='rates_spreads_store')  
])

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    # Default to /performance for root route
    if pathname in ['/', '/performance']:
        return performance_layout
    elif pathname == '/correlations':
        return correlations_layout
    elif pathname == '/risk-metrics':
        return risk_metrics_layout
    else:
        return performance_layout

@app.callback(
    [Output('performance-link', 'active'),
     Output('correlations-link', 'active'),
     Output('risk-metrics-link', 'active')],
    Input('url', 'pathname')
)
def update_active_links(pathname):
    return (
        pathname in ['/', '/performance'],  
        pathname == '/correlations',
        pathname == '/risk-metrics'
    )

@app.callback(
    Output('returns_graph', 'figure'),
    [Input('lookback_dropdown', 'value'),  
     Input('returns_table', 'selected_rows'),  
     Input('returns_table', 'data')]  
)
def update_returns_graph(lookback_period, selected_rows, stored_data):
    """
    Updates the returns graph based on the selected row and lookback period.
    """
    if not selected_rows or not stored_data:
        return go.Figure()  
    df = pd.DataFrame(stored_data)
    selected_ticker = df.iloc[selected_rows[0]]['Ticker']

    today = datetime.today()
    if lookback_period == '1m':
        start_date = today - pd.DateOffset(weeks=4)
    elif lookback_period == '3m':
        start_date = today - pd.DateOffset(weeks=12)
    elif lookback_period == '6m':
        start_date = today - pd.DateOffset(weeks=24)
    elif lookback_period == '1y':
        start_date = today - pd.DateOffset(years=1)
    elif lookback_period == '3y':
        start_date = today - pd.DateOffset(years=3)
    elif lookback_period == 'all':
        start_date = None 

    if start_date:
        start_date = start_date.strftime('%Y-%m-%d')

    return calc.generate_returns_graph(selected_ticker, start_date, today.strftime('%Y-%m-%d'))

@app.callback(
    Output('volume_graph', 'figure'),
    [Input('lookback_dropdown_volume', 'value'),  
     Input('volume_table', 'selected_rows'), 
     Input('volume_table', 'data')]  
)
def update_volume_graph(lookback_period, selected_rows, stored_data):
    """
    Updates the volume and rolling average graph based on the selected row and lookback period.
    """
    if not selected_rows or not stored_data:
        return go.Figure()  
    df = pd.DataFrame(stored_data)
    selected_ticker = df.iloc[selected_rows[0]]['Ticker']
    today = datetime.today()
    if lookback_period == '1m':
        start_date = today - pd.DateOffset(weeks=4)
    elif lookback_period == '3m':
        start_date = today - pd.DateOffset(weeks=12)
    elif lookback_period == '6m':
        start_date = today - pd.DateOffset(weeks=24)
    elif lookback_period == '1y':
        start_date = today - pd.DateOffset(years=1)
    elif lookback_period == '3y':
        start_date = today - pd.DateOffset(years=3)
    elif lookback_period == 'all':
        start_date = None  

    if start_date:
        start_date = start_date.strftime('%Y-%m-%d')
    return calc.create_volume_and_rolling_avg_graph(selected_ticker, start_date, today.strftime('%Y-%m-%d'))

@app.callback(
    Output('data_store', 'data'),
    Input('ticker_dropdown', 'value')
)
def update_performance_store(value):
    logging.debug(f"Fetching data for {value}")
    dir = 'res/tickers/'
    tickers = ap.get_tickers(dir, f'{value}.csv')
    data = ap.fetch_prices_from_db(tickers, start_date, end_date) 
    return data  

@app.callback(
    [Output('returns_table', 'data'),
     Output('returns_table', 'columns')],
    Input('data_store', 'data')
)
def update_returns_table(data):
    if data is None:
        return [], []

    df = calc.get_performance(data)
    df.reset_index(inplace=True)
    columns = [{'name': col, 'id': col} for col in df.columns]
    return df.round(2).to_dict('records'), columns

@app.callback(
    [Output('volume_table', 'data'),
     Output('volume_table', 'columns')],
    Input('data_store', 'data')
)
def update_volume_table(data):
    if data is None:
        return [], []

    df = calc.get_performance_vs_rolling_mean(data)
    df.reset_index(inplace=True)
    columns = [{'name': col, 'id': col} for col in df.columns]
    return df.round(2).to_dict('records'), columns

@app.callback(
    Output('data_correlation_store', 'data'),
    Input('ticker_dropdown_correlations', 'value')
)
def update_correlation_store(value):
    logging.debug(f"Fetching data for {value}")
    dir = 'res/tickers_corr/'
    filename = 'correlations_etfs.csv'
    tickers = ap.get_tickers(dir, filename) 
    data = ap.fetch_prices_from_db(tickers, start_date, end_date) 
    return data 

@app.callback(
    Output('dd_output_container_correlations', 'children'),
    [Input('ticker_dropdown_correlations', 'value'),
    Input('data_correlation_store', 'data')]
)
def update_correlation_table(value, stored_data):
    """
    Update the correlation table for the selected ticker.
    """
    if stored_data is None:
        logging.warning("No stored data available for the correlation table")
        return html.Div("No data available", style={'textAlign': 'center'})
    
    # Fetch data for the selected ticker from the stored data
    try:
        data = pd.DataFrame(stored_data)
        df_correlation, _ = calc.get_correlation_table_window_x(data, value)
        correlation_table = dbc.Table.from_dataframe(df_correlation, bordered=True)
        return correlation_table
    except Exception as e:
        logging.error(f"Error updating correlation table: {e}")
        return html.Div("Error generating correlation table", style={'textAlign': 'center'})

@app.callback(
    Output('dd_output_container_correlation_graphs', 'figure'),
    [Input('ticker_dropdown_correlations', 'value')],
    [Input('data_correlation_store', 'data')]
)
def update_correlation_graph(value, stored_data):
    """
    Update the correlation graph for the selected ticker.
    """
    if stored_data is None:
        logging.warning("No stored data available for the correlation graph")
        # Return an empty figure if no data is available
        return go.Figure()

    try:
        # Convert stored data into a DataFrame
        dataframe = pd.DataFrame(stored_data)
        ticker_list = list(dataframe.columns)
        
        # Generate the correlation graph
        fig = calc.create_correlation_graph(dataframe, ticker_list, value)
        return fig

    except Exception as e:
        logging.error(f"Error updating correlation graph: {e}")
        # Return an empty figure in case of errors
        return go.Figure()


@app.callback(
    Output('rates_spreads_store', 'data'),
    Input('url', 'pathname')
)
def fetch_and_store_fred_data(pathname):
    """
    Fetch and store FRED data when the user navigates to the app.
    """
    if pathname == '/risk-metrics':  # Only fetch when viewing risk metrics
        df = ap.get_rates_spreads_data()
        return df
    return dash.no_update

@app.callback(
    [Output('rates_table', 'data'),
     Output('rates_table', 'columns')],
    Input('rates_spreads_store', 'data')
)
def update_rates_table(stored_data):
    """
    Update the Rates Table based on the stored FRED data.
    """
    if not stored_data:
        return [], []

    df = pd.DataFrame(stored_data)
    performance_data = calc.df_performance_rates_spreads(df)

    columns = [{'name': col, 'id': col} for col in performance_data.columns]
    return performance_data.round(2).to_dict('records'), columns

@app.callback(
    Output('rates_chart', 'figure'),
    [Input('lookback_dropdown_rates', 'value'),
     Input('rates_table', 'selected_rows'),
     Input('rates_spreads_store', 'data')]
)
def update_rates_chart(lookback_period, selected_rows, store_data):
    """
    Updates the Rates chart based on the selected row and lookback period.
    """
    
    df_stored = pd.DataFrame.from_dict(store_data)
    tickers = [col for col in df_stored.columns if col != 'date']  # All tickers
    selected_ticker = tickers[selected_rows[0]] if selected_rows else None

    df = pd.DataFrame()
    df.index = pd.to_datetime(df_stored[selected_ticker].loc['date'])
    df['close'] = df_stored[selected_ticker].loc['close']

    # Apply the lookback filter
    today = datetime.today()
    if lookback_period == '1m':
        start_date = today - pd.DateOffset(weeks=4)
    elif lookback_period == '3m':
        start_date = today - pd.DateOffset(weeks=12)
    elif lookback_period == '6m':
        start_date = today - pd.DateOffset(weeks=24)
    elif lookback_period == '1y':
        start_date = today - pd.DateOffset(years=1)
    elif lookback_period == '3y':
        start_date = today - pd.DateOffset(years=3)
    elif lookback_period == 'all':
        start_date = None
    else:
        raise ValueError(f"Invalid lookback period: {lookback_period}")

    # Filter data based on lookback period if applicable
    df_filtered = df if start_date is None else df[df.index >= start_date]
    return calc.chart_rates_spreads(df_filtered)


