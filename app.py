import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from data import app_yf


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
i_app = app_yf()
df, tickers, latest_date = i_app.get_df_all_data()
tickers = i_app.sort(tickers)
window_list = [15, 30, 90, 120, 180]

@app.callback(
    Output('performance_table', 'children'),
    [Input('submit_val', 'n_clicks'),
     Input('submit_val_rel_p', 'n_clicks')],
    [State('input_on_submit', 'value'),
     State('input_on_submit_rel_p', 'value')]
)
def get_performance_table(n_clicks, n_clicks_rel_val, TICKER, MULTP_TICKERS):
    data = i_app.get_performance(df, TICKER, MULTP_TICKERS)
    return dbc.Table.from_dataframe(
        data.round(2),
        bordered=True)

@app.callback(
    Output('relative_performance_table', 'children'),
    [Input('submit_val', 'n_clicks'),
     Input('submit_val_rel_p', 'n_clicks')],
    [State('input_on_submit', 'value'),
     State('input_on_submit_rel_p', 'value')]
)
def relative_performance(n_clicks, n_clicks_rel_p, TICKER, MULTP_TICKERS):
    data = i_app.relative_performance(df, TICKER, MULTP_TICKERS)
    return dbc.Table.from_dataframe(
        data.round(2),
        bordered=True)

@app.callback(
    Output('corr_table_window_x', 'children'),
    [Input('submit_val_corr', 'n_clicks'),
     Input('submit_val_corr_drop', 'n_clicks')],
    [State('input_on_submit_corr', 'value'),
     State('input_on_submit_corr_drop', 'value')]
)
def correlation_window(n_clicks, n_clicks_corr, TICKER, MULTP_TICKERS):
    data = i_app.get_correlation_table_window_x(df, TICKER, MULTP_TICKERS)[0]
    return dbc.Table.from_dataframe(
        data.round(2),
        bordered=True)

@app.callback(
    Output('corr_chart', 'figure'),
    [Input('submit_val_corr', 'n_clicks'),
     Input('submit_val_corr_drop', 'n_clicks'),
     Input('submit_val_window', 'n_clicks')],
    [State('input_on_submit_corr', 'value'),
     State('input_on_submit_corr_drop', 'value'),
     State('input_corr_window', 'value')]
)
def correlation_charts(n_clicks, n_clicks_corr, n_clicks_window, TICKER, MULTP_TICKERS, WINDOW):
    data = i_app.get_correlation_table_window_x(df, TICKER, MULTP_TICKERS)[1]
    data = data.filter(regex=f'{WINDOW}')
    fig = i_app.get_correlation_chart(data, TICKER, WINDOW)
    return fig

# @app.callback(
#     Output('rates_spread_performance', 'children'),
#     [Input('submit_val', 'n_clicks'),
#      Input('submit_val_rel_p', 'n_clicks')],
#     [State('input_on_submit', 'value'),
#      State('input_on_submit_rel_p', 'value')]
# )
# def rates_spreads_performance(n_clicks, n_clicks_rel_val, TICKER, MULTP_TICKERS):
#     data = i_app.df_performance_rates_spreads()
#     return dbc.Table.from_dataframe(
#         data.round(2),
#         bordered=True)


app.layout = html.Div(children=[
    dbc.Container([
        dbc.Col([
            dbc.Row(
                html.H1(children='Performance')
            ),
            dbc.Row(
                html.H3(children=f'{latest_date.date()}')
            ),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Br(),
                    html.H4('Base Ticker'),
                    dcc.Dropdown(
                        id='input_on_submit',
                        options=[{'label': x, 'value': x} for x in tickers],
                        value='SPY',
                        placeholder='Select security',
                        multi=False
                    ),
                    html.Br(),
                    dbc.Button(
                        'Submit ticker',
                        id='submit_val')
                ]),

                dbc.Col([
                    html.H4('Comp Tickers'),
                    dcc.Dropdown(
                        id='input_on_submit_rel_p',
                        options=[{'label': x, 'value': x} for x in tickers],
                        multi=True,
                        placeholder='Select ticker(s)',
                        value=['XLY', 'XLF', 'XLV', 'XLK', 'XLP', 'XLI', 'XLB', 'XLE', 'XLU', 'XLRE', 'XLC', 'XRT']
                    ),
                    html.Br(),
                    dbc.Button(
                        'Submit ticker(s)',
                        id='submit_val_rel_p'
                    )
                ])
            ]),

            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Br(),
                    html.Br(),
                    html.Div(
                        id='performance_table'
                    ),
                    html.Br(),
                    html.Div(
                        id='relative_performance_table'
                    )
                ])

            ]),
            html.Br(),
        ]),

        dbc.Col([
            dbc.Row(
                html.H1(children='Correlations')
            ),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.H4('Base Ticker'),
                    dcc.Dropdown(
                        id='input_on_submit_corr',
                        options=[{'label': x, 'value': x} for x in tickers],
                        value='EURUSD=X',
                        placeholder='Select security',
                        multi=False
                    ),
                    html.Br(),
                    dbc.Button(
                        'Submit ticker',
                        id='submit_val_corr')
                ]),

                dbc.Col([
                    html.H4('Comp Tickers'),
                    dcc.Dropdown(
                        id='input_on_submit_corr_drop',
                        options=[{'label': x, 'value': x} for x in tickers],
                        multi=True,
                        placeholder='Select ticker(s)',
                        value=['SPY', 'USO','^FVX', '^TNX', '^TYX', 'GLD', 'BITO', 'DBC']
                    ),
                    html.Br(),
                    dbc.Button(
                        'Submit ticker(s)',
                        id='submit_val_corr_drop'
                    )
                ]),

                dbc.Col([
                    html.H4('Window'),
                    dcc.Dropdown(
                        id='input_corr_window',
                        options=[{'label': x, 'value': x} for x in window_list],
                        multi=False,
                        placeholder='Select ticker(s)',
                        value=30
                    ),
                    html.Br(),
                    dbc.Button(
                        'Submit ticker(s)',
                        id='submit_val_window'
                    )
                ])
            ])
        ]),
        html.Br(),
        html.Br(),
        html.Div(
            id='corr_table_window_x'
        ),
        dcc.Graph(
            id='corr_chart'
        ),
        html.Br(),
        html.Br()
    ])
])



if __name__ == '__main__':
    app.run_server(debug=True)
