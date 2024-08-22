import dash_bootstrap_components as dbc
import os
import re
import pandas as pd

from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from data_performance import app_performance

extension = 'csv'
files = os.listdir('tickers')
files = sorted([i[:-4] for i in files])
corr_tickers = pd.read_csv('tickers_corr/correlations.csv')
corr_tickers = list(corr_tickers['Ticker'])


ap = app_performance()

# Initialize the app with suppress_callback_exceptions=True
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Macro Report"

# Define the Sidebar
sidebar = html.Div(
    [
        html.H2("Macro Report", className="display-4"),
        html.Hr(),
        #html.P("Navigation", className="lead"),
        dbc.Nav(
            [
                dbc.NavLink("Performance", href="/performance", id="performance-link", active="exact"),
                dbc.NavLink("Correlations", href="/correlations", id="correlations-link", active="exact"),
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
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    sidebar,
    content
])

# Callback to control page navigation
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/correlations':
        return html.Div([
            html.Br(),
            html.H2(children='Correlations', style={'textAlign': 'center'}),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(options=corr_tickers,
                                 value='UUP',
                                 id='ticker_dropdown_correlations'),
                    html.Br(),
                    html.Div(id='dd_output_container_correlations')
                ], width={"size": 6, "offset": 3}),
            ]),
        ])
    else:
        return html.Div([
            html.Br(),
            html.H2(children='Performance', style={'textAlign': 'center'}),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(options=files,
                                 value='sectors',
                                 id='ticker_dropdown'),
                    html.Br(),
                    html.Div(id='dd_output_container')
                ], width={"size": 6, "offset": 3}),
            ]),
        ])  # Default page is Performance

# Original callbacks
@app.callback(
    Output(component_id='dd_output_container', component_property='children'),
    Input(component_id='ticker_dropdown', component_property='value')
)
def update_performance(value):
    df, ticker_list = ap.get_df_all_data(f'tickers/{value}.csv')
    df_performance = ap.get_performance(df)
    return dbc.Table.from_dataframe(df_performance.round(2), bordered=True)

@app.callback(
    Output(component_id='dd_output_container_correlations', component_property='children'),
    Input(component_id='ticker_dropdown_correlations', component_property='value')
)
def update_correlations(value):
    df, ticker_list = ap.get_df_all_data('tickers_corr/correlations.csv')
    df_correlation, dataframe = ap.get_correlation_table_window_x(df, value)
    return dbc.Table.from_dataframe(df_correlation, bordered=True)

# Highlight the active link
@app.callback(
    [Output(f"{link}-link", "active") for link in ["performance", "correlations"]],
    [Input("url", "pathname")]
)
def toggle_active_links(pathname):
    return [pathname == f"/{link}" for link in ["performance", "correlations"]]

if __name__ == '__main__':
    app.run_server(debug=True)









