import dash_bootstrap_components as dbc
import pandas as pd
import os

from dash import Dash, html, dcc, State
from dash.dash_table import DataTable

files = os.listdir('res/tickers')
files = sorted([i[:-4] for i in files])
corr_tickers = pd.read_csv('res/tickers_corr/correlations_etfs.csv')['Ticker'].tolist()
rates_spreads_tickers = ['2Y-10Y Spread', '5Y Breakeven', 'HY-OAS', 'IG Spread', 'High Yield', '3M t-bill', '2Y t-note', '5Y t-note', '10Y t-note', '30Y t-note']

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

performance_layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    options=[{'label': x, 'value': x} for x in files],
                    value='sectors',  
                    id='ticker_dropdown',
                    clearable=False,
                ),
                html.Br(),
            ], width=12, style={'margin': '0 auto', 'textAlign': 'center'}),
        ]),
        html.Br(),
        dbc.Row([
            # First Column: Returns Table
            dbc.Col([
                html.Div([
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
                                sort_action='native',
                                style_table={'overflowX': 'auto', 'height': '400px'},  
                                style_cell={'textAlign': 'center'}
                            )
                        ],
                    ),
                ], style={'height': '100%'})  
            ], width=6),
            # Second Column: Returns Graph
            dbc.Col([
                html.Div([
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
                                value='1y',  
                                id='lookback_dropdown',
                                placeholder="Select Lookback Period",
                                clearable=False
                            ),
                        ], width=12, style={'margin': '0 auto', 'textAlign': 'center'}),
                    ]),
                    dcc.Loading(
                        type='circle',
                        id='loading-performance-graphh',
                        children=[dcc.Graph(
                            id='returns_graph',
                            style={'height': '400px', 'width': '100%'}  
                        )],
                    ),
                ], style={'height': '100%'})
            ], width=6),
        ]),
        html.Br(),
        # Row for Volume Table and Volume Graph
        dbc.Row([
            # Volume Table
            dbc.Col([
                html.Div([
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
                                sort_action='native',
                                style_table={'overflowX': 'auto', 'height': '400px'},  
                                style_cell={'textAlign': 'center'}
                            )
                        ],
                    ),
                ], style={'height': '100%'})
            ], width=6),
            # Volume Graph
            dbc.Col([
                html.Div([
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
                                value='1y',  
                                id='lookback_dropdown_volume',
                                placeholder="Select Lookback Period",
                                clearable=False
                            ),
                        ], width=12, style={'margin': '0 auto', 'textAlign': 'center'}),
                    ]),
                    dcc.Loading(
                        type='circle',
                        id='loading-graph-volume-graph',
                        children=[dcc.Graph(
                            id='volume_graph',
                            style={'height': '400px', 'width': '100%'}  # Fixed graph height
                        )],
                    ),
                ], style={'height': '100%'})
            ], width=6),
        ])
    ], fluid=True)  # Ensures container spans full width
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
            )
        ], style={'textAlign':'center'})
    ]),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Br(),
                html.Br(),
                dcc.Loading(
                        type='circle',
                    id='loading-correlation-table',
                    children=[html.Div(id='dd_output_container_correlations')],
                )
            ], style={'textAlign': 'center', 'height':'100%'}),
        ], width=6),
        dbc.Col([
            html.Div([
                html.Br(),
                dcc.Loading(
                    type='circle',
                    id='loading-correlation-graphs',
                    children=[dcc.Graph(id='dd_output_container_correlation_graphs')],
                )
            ], style={'height':'100%'})
        ], width=6)
    ])
])

risk_metrics_layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Br(),
                html.H2('Rates and Spreads', style={'textAlign': 'center'}),
                html.Br(),
            ], width=12, style={'textAlign': 'center'}),
        ]),
        html.Br(),
        dbc.Row([
            # Rates Table
            dbc.Col([
                html.Div([
                    html.H4("Rates Table", style={'textAlign': 'center'}),
                    html.Br(),
                    dcc.Loading(
                        type='circle',
                        id='loading-rates-table',
                        children=[
                            DataTable(
                                id='rates_table',
                                columns=[],
                                data=[],
                                row_selectable='single',
                                selected_rows=[0],
                                sort_action='native',
                                style_table={'overflowX': 'auto', 'height': '400px'},
                                style_cell={'textAlign': 'center'}
                            )
                        ],
                    ),
                ], style={'height': '100%'})
            ], width=6),
            # Rates Chart
            dbc.Col([
                html.Div([
                    html.H4("Rates Chart", style={'textAlign': 'center'}),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='lookback_dropdown_rates',
                                options=[
                                    {'label': '1 Month', 'value': '1m'},
                                    {'label': '3 Months', 'value': '3m'},
                                    {'label': '6 Months', 'value': '6m'},
                                    {'label': '1 Year', 'value': '1y'},
                                    {'label': '3 Years', 'value': '3y'},
                                    {'label': 'All History', 'value': 'all'}
                                ],
                                value='1y',
                                placeholder="Select Lookback Period",
                                clearable=False
                            ),
                        ], width=12, style={'textAlign': 'center'}),
                    ]),
                    dcc.Loading(
                        type='circle',
                        id='loading-rates-chart',
                        children=[dcc.Graph(
                            id='rates_chart',
                            style={'height': '400px', 'width': '100%'}
                        )],
                    ),
                ], style={'height': '100%'})
            ], width=6),
        ])
    ])
])
