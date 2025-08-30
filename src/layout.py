import os
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dash_table import DataTable

# ---------- data for pickers ----------
files = sorted([f[:-4] for f in os.listdir("res/tickers") if f.endswith(".csv")])
corr_tickers = pd.read_csv("res/tickers_corr/correlations_etfs.csv")["Ticker"].tolist()
rates_spreads_tickers = [
    "2Y-10Y Spread", "5Y Breakeven", "HY-OAS", "IG Spread", "High Yield",
    "3M t-bill", "2Y t-note", "5Y t-note", "10Y t-note", "30Y t-note"
]
lookback_options = [
    {"label": "1 Month", "value": "1m"},
    {"label": "3 Months", "value": "3m"},
    {"label": "6 Months", "value": "6m"},
    {"label": "1 Year", "value": "1y"},
    {"label": "3 Years", "value": "3y"},
    {"label": "10 Years", "value": "10y"},
    {"label": "All History", "value": "all"},
]

# ---------- sidebar (fixed on desktop, collapses visually on mobile via CSS) ----------
sidebar = html.Div(
    [
        html.H2("Macro Report"),
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
    className="sidebar",
)

# ---------- PERFORMANCE ----------
performance_layout = html.Div(
    dbc.Container(
        [
            # Ticker picker row
            dbc.Row(
                dbc.Col(
                    dcc.Dropdown(
                        options=[{"label": x, "value": x} for x in files],
                        value="sectors",
                        id="ticker_dropdown",
                        clearable=False,
                    ),
                    xs=12,
                    className="mb-2",
                ),
                className="g-2",
            ),

            # Returns table + Returns graph (top half of viewport)
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H4("Returns Table", className="text-center mb-2"),
                                dcc.Loading(
                                    id="loading-performance-table",
                                    type="circle",
                                    children=[
                                        DataTable(
                                            id="returns_table",
                                            columns=[], data=[],
                                            row_selectable="single", selected_rows=[0],
                                            sort_action="native",
                                            style_table={"height": "40vh", "overflowY": "auto"},  # fits half
                                            style_cell={"textAlign": "center"},
                                        )
                                    ],
                                ),
                            ],
                            className="card-body-grow",
                        ),
                        xs=12, lg=6, className="min-h-0",
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.H4("Returns Graph", className="text-center mb-2"),
                                dbc.Row(
                                    dbc.Col(
                                        dcc.Dropdown(
                                            options=lookback_options,
                                            value="1y",
                                            id="lookback_dropdown",
                                            placeholder="Select Lookback Period",
                                            clearable=False,
                                        ),
                                        xs=12,
                                    )
                                ),
                                dcc.Loading(
                                    id="loading-performance-graphh",
                                    type="circle",
                                    children=[dcc.Graph(id="returns_graph", style={"height": "40vh", "width": "100%"})],
                                ),
                            ],
                            className="card-body-grow",
                        ),
                        xs=12, lg=6, className="min-h-0",
                    ),
                ],
                className="g-2",
            ),

            # Volume table + Volume graph (bottom half of viewport)
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H4("Volume Table", className="text-center mb-2"),
                                dcc.Loading(
                                    id="loading-performance-volume-table",
                                    type="circle",
                                    children=[
                                        DataTable(
                                            id="volume_table",
                                            columns=[], data=[],
                                            row_selectable="single", selected_rows=[0],
                                            sort_action="native",
                                            style_table={"height": "40vh", "overflowY": "auto"},
                                            style_cell={"textAlign": "center"},
                                        )
                                    ],
                                ),
                            ],
                            className="card-body-grow",
                        ),
                        xs=12, lg=6, className="min-h-0",
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.H4("Volume Graph", className="text-center mb-2"),
                                dbc.Row(
                                    dbc.Col(
                                        dcc.Dropdown(
                                            options=lookback_options,
                                            value="1y",
                                            id="lookback_dropdown_volume",  # keep your original ID
                                            placeholder="Select Lookback Period",
                                            clearable=False,
                                        ),
                                        xs=12,
                                    )
                                ),
                                dcc.Loading(
                                    id="loading-graph-volume-graph",
                                    type="circle",
                                    children=[dcc.Graph(id="volume_graph", style={"height": "40vh", "width": "100%"})],
                                ),
                            ],
                            className="card-body-grow",
                        ),
                        xs=12, lg=6, className="min-h-0",
                    ),
                ],
                className="g-2",
            ),
        ],
        fluid=True,
        className="page-wrap",
    ),
    className="content",
)

# ---------- CORRELATIONS ----------
correlations_layout = html.Div(
    dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    html.H2("Correlations", className="text-center my-2"),
                    xs=12,
                ),
                className="g-2",
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Dropdown(
                        options=[{"label": x, "value": x} for x in corr_tickers],
                        value="UUP",
                        id="ticker_dropdown_correlations",
                        clearable=False,
                    ),
                    xs=12,
                ),
                className="g-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                dcc.Loading(
                                    id="loading-correlation-table",
                                    type="circle",
                                    children=[html.Div(id="dd_output_container_correlations")],
                                )
                            ],
                            className="card-body-grow",
                        ),
                        xs=12, lg=6, className="min-h-0",
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                dcc.Loading(
                                    id="loading-correlation-graphs",
                                    type="circle",
                                    children=[dcc.Graph(id="dd_output_container_correlation_graphs", style={"height": "80vh"})],
                                )
                            ],
                            className="card-body-grow",
                        ),
                        xs=12, lg=6, className="min-h-0",
                    ),
                ],
                className="g-2",
            ),
        ],
        fluid=True,
        className="page-wrap",
    ),
    className="content",
)

# ---------- RISK METRICS ----------
risk_metrics_layout = html.Div(
    dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    html.H2("Rates and Spreads", className="text-center my-2"),
                    xs=12,
                ),
                className="g-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H4("Rates Table", className="text-center mb-2"),
                                dcc.Loading(
                                    id="loading-rates-table",
                                    type="circle",
                                    children=[
                                        DataTable(
                                            id="rates_table",
                                            columns=[], data=[],
                                            row_selectable="single", selected_rows=[0],
                                            sort_action="native",
                                            style_table={"height": "40vh", "overflowY": "auto"},
                                            style_cell={"textAlign": "center"},
                                        ),
                                    ],
                                ),
                            ],
                            className="card-body-grow",
                        ),
                        xs=12, lg=6, className="min-h-0",
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.H4("Rates Chart", className="text-center mb-2"),
                                dbc.Row(
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id="lookback_dropdown_rates",   # keep your original ID
                                            options=lookback_options,
                                            value="1y",
                                            placeholder="Select Lookback Period",
                                            clearable=False,
                                        ),
                                        xs=12,
                                    )
                                ),
                                dcc.Loading(
                                    id="loading-rates-chart",
                                    type="circle",
                                    children=[dcc.Graph(id="rates_chart", style={"height": "40vh", "width": "100%"})],
                                ),
                            ],
                            className="card-body-grow",
                        ),
                        xs=12, lg=6, className="min-h-0",
                    ),
                ],
                className="g-2",
            ),
        ],
        fluid=True,
        className="page-wrap",
    ),
    className="content",
)
