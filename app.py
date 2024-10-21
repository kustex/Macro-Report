import asyncio
import dash_bootstrap_components as dbc
import os
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from data_performance import app_performance
from calculations import StockCalculations 
from stock_data_service import StockDataService
from database_client import DatabaseClient
from datetime import datetime

# Initialize global instances for database and services
db_client = DatabaseClient('stock_data.db')  
ap = StockDataService(db_client)
calc = StockCalculations()

# Set date range
start_date = ap.time_delta(2)
end_date = datetime.today().strftime('%Y-%m-%d')

# Define lists for dropdown options
extension = 'csv'
files = os.listdir('tickers')
files = sorted([i[:-4] for i in files])
corr_tickers = pd.read_csv('tickers_corr/correlations_etfs.csv')['Ticker'].tolist()

rates_spreads_tickers = ['2Y-10Y Spread', '5Y Breakeven', 'HY-OAS', 'IG Spread', 'High Yield', '3M t-bill', '2Y t-note', '5Y t-note', '10Y t-note', '30Y t-note']
timeframes = ['1Y', '5Y', '10Y', 'MAX']

# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Macro Report"

# Define the Sidebar
sidebar = html.Div(
    [
        html.H2("Macro Report", className="display-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Performance", href="/performance", id="performance-link", active="exact"),
                dbc.NavLink("Correlations", href="/correlations", id="correlations-link", active="exact"),
                dbc.NavLink("Risk Metrics", href="/risk-metrics", id="risk-metrics-link", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    },
)

# Define the Content Area
content = html.Div(id="page-content", style={"margin-left": "18rem", "padding": "2rem 1rem"})

# Define the app layout
app.layout = html.Div([dcc.Location(id='url', refresh=False), sidebar, content])

# Define the layouts for each page
# Performance Layout
performance_layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Br(),
                html.H2(children='Performance', style={'textAlign': 'center'}),
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            options=[{'label': x, 'value': x} for x in files],
                            value='sectors',
                            id='ticker_dropdown'
                        ),
                        html.Br(),
                    ], style={'textAlign': 'center', 'marginLeft': 'auto', 'marginRight': 'auto'}),
                ]),
                dbc.Col([
                    html.Div(id='dd_output_container')
                ], style={'textAlign': 'center'})

            ], align='center')
        ])
    ])
])

# Risk Metrics Layout
risk_metrics_layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Br(),
                html.H2('Rates and Spreads', style={'textAlign': 'center'}),
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id='input_rates_spreads',
                            options=[{'label': x, 'value': x} for x in rates_spreads_tickers],
                            value='2Y-10Y Spread'
                        ),
                    ], style={'textAlign': 'center','marginLeft':'auto', 'marginRight': 'auto'}),
                ]),
                html.Br(),
                dcc.Graph(id='chart_rates_spreads'),
                html.Div(id='rates_spreads_performance')
            ], style={'textAlign': 'center', 'marginLeft':'auto', 'marginRight': 'auto'})
        ], align='center')
    ])
])

# Correlations Layout
correlations_layout = html.Div([
    html.Br(),
    html.H2(children='Correlations', style={'textAlign': 'center'}),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                options=[{'label': x, 'value': x} for x in corr_tickers],
                value='UUP',
                id='ticker_dropdown_correlations'
            ),
            html.Br(),
            html.Div(id='dd_output_container_correlations')
        ], style={'textAlign': 'center','marginLeft':'auto', 'marginRight': 'auto'}, width={"size": 7}),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='dd_output_container_correlation_graphs', style={'textAlign': 'center', 'marginLeft':'auto', 'marginRight': 'auto'}),
        ])
    ], align='center')
])

# Callback to control page navigation
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/correlations':
        return correlations_layout
    elif pathname == '/risk-metrics':
        return risk_metrics_layout
    else:
        return performance_layout  # Default page is Performance

# Async function to fetch performance data
async def fetch_performance_data(value):
    dir = 'tickers/'
    tickers = ap.get_tickers(dir, f'{value}.csv')
    df, _ = await ap.get_closing_prices_for_tickers(tickers, start_date, end_date)
    df_performance = calc.get_performance(df)
    return df_performance

@app.callback(
    Output('dd_output_container', 'children'),
    [Input('ticker_dropdown', 'value')]
)
def update_performance(value):
    performance_data = asyncio.run(fetch_performance_data(value))
    performance_data.reset_index(inplace=True)
    return dbc.Table.from_dataframe(performance_data.round(2), bordered=True, striped=True, hover=True)

async def fetch_performance_rates():
    df = await ap.df_rates_spreads()
    return df

@app.callback(
    Output('chart_rates_spreads', 'figure'),
    [Input('input_rates_spreads', 'value')]
)
def chart_rates_spreads(value):
    df = asyncio.run(fetch_performance_rates())
    df = df.loc[:, value]
    fig = calc.chart_rates_spreads(df, value)
    return fig

@app.callback(
    Output('rates_spreads_performance', 'children'),
    [Input('input_rates_spreads', 'value')]
)
def update_rates_spreads_performance(value):
    df = asyncio.run(fetch_performance_rates())
    data = calc.df_performance_rates_spreads(df)
    return dbc.Table.from_dataframe(data.round(2), bordered=True)

async def fetch_correlation_data(value):
    dir = 'tickers_corr/'
    tickers = ap.get_tickers(dir, 'correlations_etfs.csv')
    print(tickers)
    df, ticker_list = await ap.get_closing_prices_for_tickers(tickers, start_date, end_date)

    if df.empty:
        return dbc.Alert("No data available for the selected ticker.", color="warning"), {}

    df_correlation, dataframe = calc.get_correlation_table_window_x(df, value)
    correlation_table = dbc.Table.from_dataframe(df_correlation, bordered=True)
    fig = calc.create_correlation_graph(dataframe, ticker_list, value)
    return correlation_table, fig

# Async callback for correlation data
@app.callback(
    [Output('dd_output_container_correlations', 'children'),
     Output('dd_output_container_correlation_graphs', 'figure')],
    [Input('ticker_dropdown_correlations', 'value')]
)
def update_correlations(value):
    correlation_data = asyncio.run(fetch_correlation_data(value))
    return correlation_data

# Highlight the active link
@app.callback(
    [Output(f"{link}-link", "active") for link in ["performance", "correlations", "risk-metrics"]],
    [Input("url", "pathname")]
)
def toggle_active_links(pathname):
    return [pathname == f"/{link}" for link in ["performance", "correlations", "risk-metrics"]]

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)
