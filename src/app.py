import asyncio
import dash_bootstrap_components as dbc
import os
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, html, dcc, State
from dash.dependencies import Input, Output
from dash.dash_table import DataTable
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
files = os.listdir('res/tickers')
files = sorted([i[:-4] for i in files])
corr_tickers = pd.read_csv('res/tickers_corr/correlations_etfs.csv')['Ticker'].tolist()

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

performance_layout = html.Div([
    dbc.Container([
        # Dropdown for selecting stock group
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    options=[{'label': x, 'value': x} for x in files],
                    value='sectors',  # Default value
                    id='ticker_dropdown',
                    clearable=False,
                ),
                html.Br(),
            ], width=12, style={'margin': '0 auto', 'textAlign': 'center'}),
        ]),
        html.Br(),
        # Row for Returns Table and Returns Graph
        dbc.Row([
            # First Column: Returns Table
            dbc.Col([
                html.H4("Returns Table", style={'textAlign': 'center'}),
                html.Br(),
                dcc.Loading(
                    type='circle',
                    id='loading-performance-table',
                    children=[
                        DataTable(
                            id='returns_table',
                            columns=[],
                            data=[],
                            row_selectable='single',
                            selected_rows=[0],
                            style_table={'overflowX': 'auto', 'height': '400px'},  # Ensure full height visibility
                            style_cell={'textAlign': 'center'}
                        )
                    ],
                ),
            ], width=6),  # Fixed width for Returns Table
            # Second Column: Returns Graph
            dbc.Col([
                html.H4("Returns Graph", style={'textAlign': 'center'}),
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            options=[
                                {'label': '1 Month', 'value': '1m'},
                                {'label': '3 Months', 'value': '3m'},
                                {'label': '6 Months', 'value': '6m'},
                                {'label': '1 Year', 'value': '1y'},
                                {'label': '3 Years', 'value': '3y'},
                                {'label': 'All History', 'value': 'all'}
                            ],
                            value='1y',  # Default lookback period
                            id='lookback_dropdown',
                            placeholder="Select Lookback Period",
                            clearable=False
                        ),
                    ], width=12, style={'margin': '0 auto', 'textAlign': 'center'}),
                ]),
                dcc.Loading(
                    type='circle',
                    id='loading-performance-graph',
                    children=[dcc.Graph(id='returns_graph', style={'height': '400px'})],
                ),
            ], width=6),  # Fixed width for Returns Graph
        ]),
        html.Br(),
        # Row for Volume Table and Volume Graph
        dbc.Row([
            # Volume Table
            dbc.Col([
                html.H4("Volume Table", style={'textAlign': 'center'}),
                html.Br(),
                dcc.Loading(
                    type='circle',
                    id='loading-performance-volume-table',
                    children=[
                        DataTable(
                            id='volume_table',
                            columns=[],
                            data=[],
                            row_selectable='single',
                            selected_rows=[0],
                            style_table={'overflowX': 'auto', 'height': '400px'},  # Ensure full height visibility
                            style_cell={'textAlign': 'center'}
                        )
                    ],
                ),
            ], width=6),  # Fixed width for Volume Table
            # Volume Graph
            dbc.Col([
                html.H4("Volume Graph", style={'textAlign': 'center'}),
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            options=[
                                {'label': '1 Month', 'value': '1m'},
                                {'label': '3 Months', 'value': '3m'},
                                {'label': '6 Months', 'value': '6m'},
                                {'label': '1 Year', 'value': '1y'},
                                {'label': '3 Years', 'value': '3y'},
                                {'label': 'All History', 'value': 'all'}
                            ],
                            value='1y',  # Default lookback period
                            id='lookback_dropdown_volume',
                            placeholder="Select Lookback Period",
                            clearable=False
                        ),
                    ], width=12, style={'margin': '0 auto', 'textAlign': 'center'}),
                ]),
                dcc.Loading(
                    type='circle',
                    id='loading-graph-volume-graph',
                    children=[dcc.Graph(id='volume_graph', style={'height': '400px'})],
                ),
            ], width=6),  # Fixed width for Volume Graph
        ])
    ], fluid=True)  # Ensures container spans full width
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
                    ], style={'textAlign': 'center', 'marginLeft':'auto', 'marginRight': 'auto'}),
                ]),
                html.Br(),
                html.Br(),
                dcc.Loading(
                    type='circle',
                    id='loading-rates-spreads-chart',
                    children=[dcc.Graph(id='chart_rates_spreads')],
                ),
                html.Br(),
                dcc.Loading(
                    type='circle',
                    id='loading-rates-spreads-performance',
                    children=[html.Div(id='rates_spreads_performance')],
                )
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
            html.Br(),
            dcc.Loading(
                    type='circle',
                id='loading-correlation-table',
                children=[html.Div(id='dd_output_container_correlations')],
            )
        ], style={'textAlign': 'center', 'marginLeft':'auto', 'marginRight': 'auto'}, width={"size": 7}),
    ]),
    dbc.Row([
        dbc.Col([
            html.Br(),
            dcc.Loading(
                type='circle',
                id='loading-correlation-graphs',
                children=[dcc.Graph(id='dd_output_container_correlation_graphs')],
            )
        ])
    ], align='center')
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
    # Treat `/` as `/performance` for highlighting
    return (
        pathname in ['/', '/performance'],  # Highlight performance for `/` or `/performance`
        pathname == '/correlations',
        pathname == '/risk-metrics'
    )

@app.callback(
    Output('returns_graph', 'figure'),
    [Input('lookback_dropdown', 'value'),  # Lookback period
     Input('returns_table', 'selected_rows'),  # Row selection
     Input('returns_table', 'data')]  # Table data
)
def update_returns_graph(lookback_period, selected_rows, table_data):
    """
    Updates the returns graph based on the selected row and lookback period.
    """
    if not selected_rows or not table_data:
        return go.Figure()  # Return an empty figure if no row is selected

    # Get the selected ticker
    selected_ticker = table_data[selected_rows[0]]['Ticker']

    # Determine the start date based on the lookback period
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
        start_date = None  # No filtering for all history

    # Convert start_date to string if not None
    if start_date is not None:
        start_date = start_date.strftime('%Y-%m-%d')  # Format as 'YYYY-MM-DD'

    # Generate the graph
    return calc.generate_returns_graph(selected_ticker, start_date, today.strftime('%Y-%m-%d'))

@app.callback(
    Output('volume_graph', 'figure'),
    [Input('lookback_dropdown_volume', 'value'),
     Input('volume_table', 'selected_rows'),
     Input('volume_table', 'data')]
)
def update_volume_graph(lookback_period, selected_rows, table_data):
    """
    Updates the volume and rolling average graph based on the selected row and lookback period.
    """
    if not selected_rows or not table_data:
        return go.Figure()  # Return an empty figure if no row is selected

    # Get the selected ticker
    selected_ticker = table_data[selected_rows[0]]['Ticker']

    # Determine the start date based on the lookback period
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
        start_date = None  # No filtering for all history

    if start_date is not None:
        start_date = start_date.strftime('%Y-%m-%d')  # Format as 'YYYY-MM-DD'

    return calc.create_volume_and_rolling_avg_graph(selected_ticker, start_date, today.strftime('%Y-%m-%d'))

async def fetch_volume_data(value):
    dir = 'res/tickers/'  
    tickers = ap.get_tickers(dir, f'{value}.csv')
    data, _ = await ap.get_prices_for_tickers(tickers, start_date, end_date) 
    df = calc.get_performance_vs_rolling_mean(data)
    return df

@app.callback(
    [Output('volume_table', 'data'),
     Output('volume_table', 'columns')],
    [Input('ticker_dropdown', 'value')]
)
def update_volume_table(value):
    """
    Updates the volume table data and columns based on the selected stock group.
    """
    volume_data = asyncio.run(fetch_volume_data(value))
    volume_data.reset_index(inplace=True)
    columns = [{'name': col, 'id': col} for col in volume_data.columns]
    return volume_data.round(2).to_dict('records'), columns

async def fetch_performance_data(value):
    dir = 'res/tickers/'
    tickers = ap.get_tickers(dir, f'{value}.csv')
    df, _ = await ap.get_prices_for_tickers(tickers, start_date, end_date)
    df_performance = calc.get_performance(df)
    return df_performance

@app.callback(
    [Output('returns_table', 'data'),
     Output('returns_table', 'columns')],
    [Input('ticker_dropdown', 'value')]
)
def update_performance_table(value):
    performance_data = asyncio.run(fetch_performance_data(value))
    performance_data.reset_index(inplace=True)
    columns = [{'name': col, 'id': col} for col in performance_data.columns]
    return performance_data.round(2).to_dict('records'), columns

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
    dir = 'res/tickers_corr/'
    tickers = ap.get_tickers(dir, 'correlations_etfs.csv')
    df, ticker_list = await ap.get_prices_for_tickers(tickers, start_date, end_date)
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


