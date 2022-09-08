import datetime
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc

from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from data_performance import app_performance

rates_spreads_tickers = ['2Y-10Y Spread', '5Y Breakeven', 'HY-OAS', 'IG Spread', 'High Yield', '3M t-bill', '2Y t-note', '5Y t-note', '10Y t-note', '30Y t-note']
timeframes = ['1Y', '5Y', '10Y', 'MAX']
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
i_app = app_performance()
data = i_app.df_rates_spreads()


def rates_spreads_performance():
    data = i_app.df_performance_rates_spreads()
    return dbc.Table.from_dataframe(
        data.round(2),
        bordered=True)


# @app.callback(
#     Output('chart_rates_spreads', 'figure'),
#     [Input('submit_rates_spreads', 'n_clicks'),
#      Input('submit_timeframe', 'n_clicks')],
#     [State('input_rates_spreads', 'value'),
#      State('input_timeframe', 'value')]
# )
# def chart_rates_spreads(n_clicks, n_clicks_timeframe, TICKER, TIME):
    # df = da   ta.loc[:, TICKER]
    # fig = i_app.chart_rates_spreads(df, TICKER)
    # return fig

@app.callback(
    Output('chart_rates_spreads', 'figure'),
    [Input('submit_rates_spreads', 'n_clicks'),
     Input('submit_timeframe', 'n_clicks')],
    [State('input_rates_spreads', 'value'),
     State('input_timeframe', 'value')]
)
def chart_rates_spreads(n_clicks, n_clicks_timeframe, TICKER, TIME):
    df = data.loc[:, TICKER]
    fig = i_app.chart_rates_spreads(df, TICKER)
    return fig

app.layout = html.Div(children=[
    dbc.Container([
        dbc.Col([
            html.H1(children='Rates and spreads'),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='input_rates_spreads',
                        options=[{'label': x, 'value': x} for x in rates_spreads_tickers],
                        multi=False,
                        placeholder='Select ticker',
                        value='2Y-10Y Spread'
                    ),
                    dbc.Button(
                        'Submit ticker(s)',
                        id='submit_rates_spreads'
                    )
                ]),
                dbc.Col([
                    dcc.Dropdown(
                        id='input_timeframe',
                        options=[{'label': x, 'value': x} for x in timeframes],
                        multi=False,
                        placeholder='Select timeframe',
                        value='5Y'
                    ),
                    dbc.Button(
                        'Submit timeframe',
                        id='submit_timeframe'
                    )
                ])
            ]),
            html.Br(),
            dcc.Graph(
                id='chart_rates_spreads'
            ),
            html.Div(
                rates_spreads_performance()
            )
        ])
    ])
])


if __name__ == '__main__':
    app.run_server(debug=True)
