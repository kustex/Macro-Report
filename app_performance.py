import dash_bootstrap_components as dbc
import os
import re

import pandas as pd
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from data_performance import app_performance

extension = 'csv'
files = os.listdir('tickers')
files = [i[:-4] for i in files]

corr_tickers = pd.read_csv('/tickers_corr/correlations.csv')
corr_tickers = list(corr_tickers['Ticker'])


ap = app_performance()
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([
    html.H2(children='Performance', style={'textAlign': 'center'}),
    dbc.Row([
        dbc.Col([
            dbc.Row(
            ),
            dcc.Dropdown(options=files,
                         value='sectors',
                         id='ticker_dropdown'),
            html.Br(),
            html.Div(id='dd_output_container')
        ], width={"size": 6, "offset": 3}),
    ]),
    html.Br(),
    html.H2(children='Correlations', style={'textAlign': 'center'}),
    dbc.Row([
        dbc.Col([
            dbc.Row(
            ),
            dcc.Dropdown(options=corr_tickers,
                         value='EURUSD',
                         id='ticker_dropdown_correlations'),
            html.Br(),
            html.Div(id='dd_output_container_correlations')
        ], width={"size": 6, "offset": 3})
    ])
])


@app.callback(
    Output(component_id='dd_output_container', component_property='children'),
    Input(component_id='ticker_dropdown', component_property='value')
)
def dropdown_filename(value):
    df, ticker_list = ap.get_df_all_data('/tickers/{}.csv'.format(value))
    df_performance = ap.get_performance(df)
    return dbc.Table.from_dataframe(
        df_performance.round(2),
        bordered=True)

@app.callback(
    Output(component_id='dd_output_container_correlations', component_property='children'),
    Input(component_id='ticker_dropdown_correlations', component_property='value')
)
def dropdown_filename(value):
    df, ticker_list = ap.get_df_all_data('/tickers_corr/correlations.csv')
    df_correlation, dataframe = ap.get_correlation_table_window_x(df, value)
    return dbc.Table.from_dataframe(
        df_correlation,
        bordered=True)


if __name__ == '__main__':
    app.run_server(debug=True)