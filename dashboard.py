#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Crypto Trader Dashboard

A modern, interactive dashboard for the AI Crypto Trader system.
Provides real-time visualization of trading data, signals, portfolio performance,
and social sentiment analysis.

Author: zd87pl
Version: 1.1.0
"""

import os
import json
import redis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from threading import Thread
import time
from collections import deque
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Redis connection
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_password = os.getenv('REDIS_PASSWORD', None)
r = redis.Redis(
    host=redis_host,
    port=redis_port,
    password=redis_password,
    decode_responses=True
)

# Initialize data storage
class DataStore:
    def __init__(self, max_length=1000):
        self.price_data = {}
        self.social_data = {}
        self.signals = []
        self.trades = []
        self.portfolio = {}
        self.ai_models = {}
        self.risk_metrics = {}
        self.max_length = max_length
        
    def add_price_update(self, symbol, data):
        if symbol not in self.price_data:
            self.price_data[symbol] = deque(maxlen=self.max_length)
        self.price_data[symbol].append(data)
        
    def add_social_update(self, symbol, data):
        self.social_data[symbol] = data
        
    def add_signal(self, signal):
        self.signals.append(signal)
        if len(self.signals) > self.max_length:
            self.signals.pop(0)
            
    def add_trade(self, trade):
        self.trades.append(trade)
        if len(self.trades) > self.max_length:
            self.trades.pop(0)
            
    def update_portfolio(self, portfolio):
        self.portfolio = portfolio
        
    def update_ai_models(self, models_data):
        self.ai_models = models_data
        
    def update_risk_metrics(self, risk_data):
        self.risk_metrics = risk_data

# Create a data store instance
data_store = DataStore()

# Thread to listen to Redis updates
def redis_listener():
    pubsub = r.pubsub()
    pubsub.subscribe(
        'market_updates', 
        'social_updates', 
        'trading_signals', 
        'trade_executions', 
        'portfolio_updates', 
        'ai_model_updates', 
        'risk_metrics_updates'
    )
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                channel = message['channel']
                data = json.loads(message['data'])
                
                if channel == 'market_updates':
                    symbol = data.get('symbol')
                    if symbol:
                        data_store.add_price_update(symbol, data)
                
                elif channel == 'social_updates':
                    symbol = data.get('symbol')
                    if symbol:
                        data_store.add_social_update(symbol, data)
                
                elif channel == 'trading_signals':
                    data_store.add_signal(data)
                
                elif channel == 'trade_executions':
                    data_store.add_trade(data)
                
                elif channel == 'portfolio_updates':
                    data_store.update_portfolio(data)
                    
                elif channel == 'ai_model_updates':
                    data_store.update_ai_models(data)
                    
                elif channel == 'risk_metrics_updates':
                    data_store.update_risk_metrics(data)
                    
            except Exception as e:
                print(f"Error processing message: {e}")

# Start the Redis listener thread
redis_thread = Thread(target=redis_listener, daemon=True)
redis_thread.start()

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no"}
    ],
)
server = app.server
app.title = "AI Crypto Trader Dashboard"

# Define the navbar
navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(
                            # This would be replaced with your actual logo
                            src="https://via.placeholder.com/40",
                            height="40px",
                            className="d-inline-block align-top"
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        html.Span(
                            "AI Crypto Trader",
                            className="ms-2 navbar-brand"
                        ),
                        width="auto",
                    ),
                ],
                align="center",
                className="g-0",
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                dbc.Nav(
                    [
                        dbc.NavItem(dbc.NavLink("Trading", href="#trading")),
                        dbc.NavItem(dbc.NavLink("Portfolio", href="#portfolio")),
                        dbc.NavItem(dbc.NavLink("AI Models", href="#ai-models")),
                        dbc.NavItem(dbc.NavLink("Risk", href="#risk")),
                    ],
                    className="ms-auto",
                    navbar=True,
                ),
                id="navbar-collapse",
                navbar=True,
                is_open=False,
            ),
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
    className="sticky-top mb-3",
)

# Define the layout
app.layout = html.Div([
    navbar,
    dbc.Container(
        [
            # Main header section
            dbc.Row([
                dbc.Col(html.H1("AI Crypto Trader Dashboard", className="text-center mt-4 mb-4 d-none d-md-block"), width=12)
            ]),
            
            # Portfolio Overview
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4("Portfolio Overview", className="text-center"),
                            html.Div([
                                dbc.Button(
                                    html.I(className="fas fa-sync-alt"),
                                    id="refresh-portfolio",
                                    color="link",
                                    size="sm",
                                    className="position-absolute top-0 end-0 mt-2 me-2"
                                )
                            ])
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id="portfolio-value", className="d-flex justify-content-center align-items-center"),
                                    html.Div(id="portfolio-change", className="d-flex justify-content-center align-items-center mt-2")
                                ], xs=12, md=6),
                                dbc.Col([
                                    html.Div(id="portfolio-risk", className="d-flex justify-content-center align-items-center flex-column")
                                ], xs=12, md=6, className="mt-3 mt-md-0")
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id="portfolio-assets", className="mt-3")
                                ], width=12)
                            ])
                        ])
                    ], className="shadow mb-4")
                ], width=12)
            ], id="portfolio"),
            
            # Price Charts and Social Data
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            dbc.Row([
                                dbc.Col(html.H4("Market Data", className="text-center"), xs=12, md=8),
                                dbc.Col([
                                    dcc.Dropdown(
                                        id="symbol-selector",
                                        options=[],
                                        value=None,
                                        placeholder="Select a trading pair",
                                        className="mr-2"
                                    ),
                                ], xs=12, md=4, className="mt-2 mt-md-0"),
                            ])
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id="price-chart", style={"height": "500px"}),
                        ])
                    ], className="shadow mb-4")
                ], xs=12, lg=8, className="mb-4 mb-lg-0"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4("Social Sentiment", className="text-center"),
                            dbc.Button(
                                "Details",
                                id="sentiment-details-btn",
                                color="primary",
                                size="sm",
                                outline=True,
                                className="position-absolute top-0 end-0 mt-2 me-2"
                            )
                        ]),
                        dbc.CardBody([
                            html.Div(id="social-metrics"),
                            dcc.Graph(id="sentiment-chart", style={"height": "220px"}),
                        ])
                    ], className="shadow mb-4"),
                    
                    dbc.Card([
                        dbc.CardHeader(html.H4("Latest News", className="text-center")),
                        dbc.CardBody([
                            html.Div(id="news-feed", style={"overflow-y": "scroll", "height": "180px"})
                        ])
                    ], className="shadow")
                ], xs=12, lg=4)
            ], id="trading"),
            
            # Trading Signals and History
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4("Recent Trading Signals", className="text-center"),
                            dbc.Button(
                                "AI Explanation",
                                id="ai-explanation-btn",
                                color="success",
                                size="sm",
                                outline=True,
                                className="position-absolute top-0 end-0 mt-2 me-2"
                            )
                        ]),
                        dbc.CardBody([
                            html.Div(id="signals-table", style={"height": "300px", "overflow-y": "scroll"})
                        ])
                    ], className="shadow mb-4")
                ], xs=12, lg=6, className="mb-4 mb-lg-0"),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Recent Trades", className="text-center")),
                        dbc.CardBody([
                            html.Div(id="trades-table", style={"height": "300px", "overflow-y": "scroll"})
                        ])
                    ], className="shadow mb-4")
                ], xs=12, lg=6)
            ]),
            
            # AI Model Performance and Comparison
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("AI Model Performance", className="text-center")),
                        dbc.CardBody([
                            dbc.Tabs([
                                dbc.Tab([
                                    dcc.Graph(id="ai-model-performance", style={"height": "300px"})
                                ], label="Performance"),
                                dbc.Tab([
                                    dcc.Graph(id="ai-model-comparison", style={"height": "300px"})
                                ], label="Comparison"),
                                dbc.Tab([
                                    html.Div(id="ai-model-details", className="mt-3")
                                ], label="Details"),
                            ])
                        ])
                    ], className="shadow mb-4")
                ], width=12)
            ], id="ai-models"),
            
            # Risk Management Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Portfolio Risk Management", className="text-center")),
                        dbc.CardBody([
                            dbc.Tabs([
                                dbc.Tab([
                                    dbc.Row([
                                        dbc.Col([
                                            dcc.Graph(id="var-chart", style={"height": "300px"})
                                        ], xs=12, lg=6),
                                        dbc.Col([
                                            dcc.Graph(id="stop-loss-chart", style={"height": "300px"})
                                        ], xs=12, lg=6, className="mt-4 mt-lg-0")
                                    ])
                                ], label="VaR & Stop-Loss"),
                                dbc.Tab([
                                    dcc.Graph(id="correlation-heatmap", style={"height": "300px"})
                                ], label="Correlations"),
                                dbc.Tab([
                                    html.Div(id="position-sizing", className="mt-3")
                                ], label="Position Sizing"),
                            ])
                        ])
                    ], className="shadow mb-4")
                ], width=12)
            ], id="risk"),
            
            # Performance Metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Performance Metrics", className="text-center")),
                        dbc.CardBody([
                            dcc.Graph(id="performance-chart", style={"height": "300px"}),
                        ])
                    ], className="shadow mb-4")
                ], width=12)
            ]),
            
            # Modals for detailed views
            dbc.Modal(
                [
                    dbc.ModalHeader("AI Decision Explanation"),
                    dbc.ModalBody([
                        html.Div(id="ai-explanation-content")
                    ]),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="close-ai-explanation", className="ms-auto")
                    ),
                ],
                id="ai-explanation-modal",
                size="xl",
                is_open=False,
            ),
            
            dbc.Modal(
                [
                    dbc.ModalHeader("Social Sentiment Details"),
                    dbc.ModalBody([
                        html.Div(id="sentiment-details-content")
                    ]),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="close-sentiment-details", className="ms-auto")
                    ),
                ],
                id="sentiment-details-modal",
                size="lg",
                is_open=False,
            ),
            
            # Refresh interval
            dcc.Interval(id="refresh-interval", interval=5000, n_intervals=0),
            
            # Store current symbols and other data
            dcc.Store(id="available-symbols"),
            dcc.Store(id="current-signal")
        ],
        fluid=True,
        className="bg-dark text-light",
    )
], className="bg-dark text-light min-vh-100")

# Callback to update symbol selector
@app.callback(
    Output("symbol-selector", "options"),
    Output("symbol-selector", "value"),
    Output("available-symbols", "data"),
    Input("refresh-interval", "n_intervals")
)
def update_symbol_selector(n):
    symbols = list(data_store.price_data.keys())
    options = [{'label': symbol, 'value': symbol} for symbol in symbols]
    value = symbols[0] if symbols else None
    return options, value, symbols

# Callback to update portfolio overview
@app.callback(
    Output("portfolio-value", "children"),
    Output("portfolio-change", "children"),
    Output("portfolio-assets", "children"),
    Input("refresh-interval", "n_intervals")
)
def update_portfolio_overview(n):
    portfolio = data_store.portfolio
    if not portfolio:
        return "No portfolio data", "", ""
    
    total_value = portfolio.get('total_value', 0)
    daily_change = portfolio.get('daily_change', 0)
    assets = portfolio.get('assets', {})
    
    value_display = html.H2(f"${total_value:,.2f}", className="mb-0")
    
    change_class = "text-success" if daily_change >= 0 else "text-danger"
    change_display = html.H5(
        f"{daily_change:+.2f}% today", 
        className=f"{change_class} mb-0"
    )
    
    asset_rows = []
    for asset, data in assets.items():
        amount = data.get('amount', 0)
        value = data.get('value', 0)
        asset_pct = (value / total_value * 100) if total_value > 0 else 0
        
        asset_rows.append(
            dbc.Row([
                dbc.Col(html.Span(asset), width=2),
                dbc.Col(html.Span(f"{amount:.6f}"), width=3),
                dbc.Col(html.Span(f"${value:,.2f}"), width=3),
                dbc.Col([
                    dbc.Progress(value=asset_pct, color="info", className="mb-1"),
                    html.Span(f"{asset_pct:.1f}%", style={"fontSize": "0.8rem"})
                ], width=4)
            ], className="mb-2")
        )
    
    assets_section = [
        dbc.Row([
            dbc.Col(html.Strong("Asset"), width=2),
            dbc.Col(html.Strong("Amount"), width=3),
            dbc.Col(html.Strong("Value"), width=3),
            dbc.Col(html.Strong("Allocation"), width=4)
        ], className="mb-2"),
        html.Hr(className="my-1"),
        *asset_rows
    ]
    
    return value_display, change_display, assets_section

# Callback to update price chart
@app.callback(
    Output("price-chart", "figure"),
    Input("refresh-interval", "n_intervals"),
    Input("symbol-selector", "value")
)
def update_price_chart(n, symbol):
    if not symbol or symbol not in data_store.price_data or not data_store.price_data[symbol]:
        # Empty chart
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin={"t": 10, "r": 10, "l": 10, "b": 10},
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
        )
        return fig
    
    # Convert deque to list for processing
    data_list = list(data_store.price_data[symbol])
    
    # Extract data
    timestamps = [item.get('timestamp') for item in data_list]
    opens = [item.get('open') for item in data_list]
    highs = [item.get('high') for item in data_list]
    lows = [item.get('low') for item in data_list]
    closes = [item.get('close') for item in data_list]
    volumes = [item.get('volume') for item in data_list]
    
    # Check if we have technical indicators
    has_rsi = 'rsi' in data_list[0] if data_list else False
    has_macd = 'macd' in data_list[0] if data_list else False
    has_bb = 'bb_upper' in data_list[0] if data_list else False
    
    # Create subplots
    rows = 2 + (1 if has_rsi else 0) + (1 if has_macd else 0)
    specs = [[{"secondary_y": True}]] + [[{}] for _ in range(rows - 1)]
    
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        specs=specs,
        row_heights=[0.5] + [0.15] * (rows - 1)
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=timestamps,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name=symbol,
            increasing_line_color='rgba(0, 255, 0, 0.7)',
            decreasing_line_color='rgba(255, 0, 0, 0.7)',
            increasing_fillcolor='rgba(0, 255, 0, 0.3)',
            decreasing_fillcolor='rgba(255, 0, 0, 0.3)'
        ),
        row=1, col=1
    )
    
    # Add Bollinger Bands if available
    if has_bb:
        bb_upper = [item.get('bb_upper') for item in data_list]
        bb_middle = [item.get('bb_middle') for item in data_list]
        bb_lower = [item.get('bb_lower') for item in data_list]
        
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=bb_upper, name="BB Upper",
                line=dict(color="rgba(173, 216, 230, 0.7)", width=1),
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=bb_middle, name="BB Middle",
                line=dict(color="rgba(255, 255, 255, 0.7)", width=1),
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=bb_lower, name="BB Lower",
                line=dict(color="rgba(173, 216, 230, 0.7)", width=1),
                showlegend=True,
                fill='tonexty',
                fillcolor='rgba(173, 216, 230, 0.05)'
            ),
            row=1, col=1
        )
    
    # Add trading signals as markers
    signal_data = [signal for signal in data_store.signals if signal.get('symbol') == symbol]
    if signal_data:
        buy_x = []
        buy_y = []
        sell_x = []
        sell_y = []
        
        for signal in signal_data:
            timestamp = signal.get('timestamp')
            action = signal.get('action')
            price = signal.get('price')
            
            if action == 'BUY':
                buy_x.append(timestamp)
                buy_y.append(price)
            elif action == 'SELL':
                sell_x.append(timestamp)
                sell_y.append(price)
        
        fig.add_trace(
            go.Scatter(
                x=buy_x, y=buy_y, name="Buy Signal",
                mode="markers",
                marker=dict(symbol="triangle-up", size=15, color="green"),
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=sell_x, y=sell_y, name="Sell Signal",
                mode="markers",
                marker=dict(symbol="triangle-down", size=15, color="red"),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=timestamps, y=volumes, name="Volume",
            marker=dict(color="rgba(100, 100, 255, 0.5)")
        ),
        row=2, col=1
    )
    
    current_row = 3
    
    # RSI indicator
    if has_rsi:
        rsi_values = [item.get('rsi') for item in data_list]
        
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=rsi_values, name="RSI",
                line=dict(color="orange", width=1.5)
            ),
            row=current_row, col=1
        )
        
        # Add RSI reference lines
        fig.add_shape(
            type="line", line=dict(dash="dash", color="red", width=1),
            y0=70, y1=70, x0=timestamps[0], x1=timestamps[-1],
            row=current_row, col=1
        )
        
        fig.add_shape(
            type="line", line=dict(dash="dash", color="green", width=1),
            y0=30, y1=30, x0=timestamps[0], x1=timestamps[-1],
            row=current_row, col=1
        )
        
        current_row += 1
    
    # MACD indicator
    if has_macd:
        macd_values = [item.get('macd') for item in data_list]
        macd_signal = [item.get('macd_signal') for item in data_list]
        macd_hist = [item.get('macd_hist') for item in data_list]
        
        # MACD line
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=macd_values, name="MACD",
                line=dict(color="blue", width=1.5)
            ),
            row=current_row, col=1
        )
        
        # Signal line
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=macd_signal, name="Signal",
                line=dict(color="red", width=1.5)
            ),
            row=current_row, col=1
        )
        
        # Histogram
        colors = ["green" if val >= 0 else "red" for val in macd_hist]
        
        fig.add_trace(
            go.Bar(
                x=timestamps, y=macd_hist, name="Histogram",
                marker=dict(color=colors, opacity=0.5)
            ),
            row=current_row, col=1
        )
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 10, "r": 10, "l": 10, "b": 10},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.1)",
        rangeslider_visible=False,
        showticklabels=True,
        row=rows, col=1  # Only show timestamps on the bottom chart
    )
    
    # Hide x-axis labels for all rows except the last one
    for row in range(1, rows):
        fig.update_xaxes(showticklabels=False, row=row, col=1)
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.1)"
    )
    
    if has_rsi:
        rsi_row = 3
        fig.update_yaxes(range=[0, 100], row=rsi_row, col=1, title_text="RSI")
    
    return fig

# Callback to update social metrics
@app.callback(
    Output("social-metrics", "children"),
    Output("sentiment-chart", "figure"),
    Output("news-feed", "children"),
    Input("refresh-interval", "n_intervals"),
    Input("symbol-selector", "value")
)
def update_social_data(n, symbol):
    if not symbol or symbol not in data_store.social_data:
        return "No social data available", {}, "No news available"
    
    social_data = data_store.social_data.get(symbol, {})
    
    # Extract metrics
    sentiment = social_data.get('sentiment', 0)
    volume = social_data.get('volume', 0)
    contributors = social_data.get('contributors', 0)
    engagement = social_data.get('engagement', 0)
    
    # Format social metrics cards
    metrics_cards = [
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Sentiment", className="text-center mb-1"),
                        html.H3(
                            f"{sentiment:.1f}", 
                            className=f"text-center {'text-success' if sentiment >= 60 else 'text-danger' if sentiment <= 40 else 'text-warning'}"
                        )
                    ])
                ], className="bg-dark border")
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Volume", className="text-center mb-1"),
                        html.H3(f"{volume:,}", className="text-center text-info")
                    ])
                ], className="bg-dark border")
            ], width=6)
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Contributors", className="text-center mb-1"),
                        html.H3(f"{contributors:,}", className="text-center text-primary")
                    ])
                ], className="bg-dark border")
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Engagement", className="text-center mb-1"),
                        html.H3(f"{engagement:,}", className="text-center text-secondary")
                    ])
                ], className="bg-dark border")
            ], width=6)
        ])
    ]
    
    # Create sentiment gauge chart
    sentiment_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Social Sentiment"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "rgba(50,200,50,0.8)" if sentiment >= 60 else "rgba(200,50,50,0.8)" if sentiment <= 40 else "rgba(200,200,50,0.8)"},
            'steps': [
                {'range': [0, 40], 'color': "rgba(200,50,50,0.3)"},
                {'range': [40, 60], 'color': "rgba(200,200,50,0.3)"},
                {'range': [60, 100], 'color': "rgba(50,200,50,0.3)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.8,
                'value': sentiment
            }
        }
    ))
    
    sentiment_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 40, "r": 25, "l": 25, "b": 20},
        height=200
    )
    
    # Extract news
    news_items = social_data.get('news', [])
    news_cards = []
    
    for item in news_items[:5]:  # Show only top 5 news items
        title = item.get('title', 'No title')
        source = item.get('source', 'Unknown')
        url = item.get('url', '#')
        published = item.get('published', 'Unknown date')
        
        news_cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.H6(
                        dcc.Link(title, href=url, target="_blank"),
                        className="mb-1"
                    ),
                    html.Small(f"{source} | {published}", className="text-muted")
                ])
            ], className="mb-2 bg-dark border")
        )
    
    if not news_cards:
        news_cards = [html.P("No recent news available")]
    
    return metrics_cards, sentiment_fig, news_cards

# Callback to update signals table
@app.callback(
    Output("signals-table", "children"),
    Input("refresh-interval", "n_intervals"),
    Input("symbol-selector", "value")
)
def update_signals_table(n, symbol):
    if not data_store.signals:
        return html.P("No trading signals available")
    
    # Filter signals by selected symbol if one is selected
    filtered_signals = data_store.signals
    if symbol:
        filtered_signals = [signal for signal in data_store.signals if signal.get('symbol') == symbol]
    
    # Sort by timestamp (newest first)
    sorted_signals = sorted(filtered_signals, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    # Create table rows
    signal_rows = []
    for signal in sorted_signals[:10]:  # Show only the 10 most recent signals
        timestamp = datetime.fromisoformat(signal.get('timestamp').replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
        symbol = signal.get('symbol')
        action = signal.get('action')
        confidence = signal.get('confidence', 0)
        price = signal.get('price', 0)
        reasoning = signal.get('reasoning', 'N/A')
        
        action_class = "text-success" if action == "BUY" else "text-danger"
        
        signal_rows.append(
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Span(timestamp, className="text-muted small"),
                            html.H6(symbol, className="mb-0 mt-1")
                        ], width=3),
                        
                        dbc.Col([
                            html.H5(action, className=f"{action_class} font-weight-bold"),
                            html.Div([
                                dbc.Progress(value=confidence*100, color="info", className="mb-1", style={"height": "5px"}),
                                html.Small(f"Confidence: {confidence*100:.0f}%")
                            ])
                        ], width=3),
                        
                        dbc.Col([
                            html.H6(f"${price:.2f}", className="mb-2"),
                            html.P(reasoning, className="small text-muted mb-0")
                        ], width=6)
                    ])
                ])
            ], className="mb-2 bg-dark border")
        )
    
    if not signal_rows:
        signal_rows = [html.P("No signals available for the selected symbol")]
    
    return signal_rows

# Callback to update trades table
@app.callback(
    Output("trades-table", "children"),
    Input("refresh-interval", "n_intervals"),
    Input("symbol-selector", "value")
)
def update_trades_table(n, symbol):
    if not data_store.trades:
        return html.P("No trade history available")
    
    # Filter trades by selected symbol if one is selected
    filtered_trades = data_store.trades
    if symbol:
        filtered_trades = [trade for trade in data_store.trades if trade.get('symbol') == symbol]
    
    # Sort by timestamp (newest first)
    sorted_trades = sorted(filtered_trades, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    # Create table rows
    trade_rows = []
    for trade in sorted_trades[:10]:  # Show only the 10 most recent trades
        timestamp = datetime.fromisoformat(trade.get('timestamp').replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
        symbol = trade.get('symbol')
        side = trade.get('side')
        quantity = trade.get('quantity', 0)
        price = trade.get('price', 0)
        value = quantity * price
        pnl = trade.get('pnl')
        
        side_class = "text-success" if side == "BUY" else "text-danger"
        pnl_class = "text-success" if pnl and pnl > 0 else "text-danger" if pnl and pnl < 0 else "text-muted"
        
        trade_rows.append(
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Span(timestamp, className="text-muted small"),
                            html.H6(symbol, className="mb-0 mt-1")
                        ], width=3),
                        
                        dbc.Col([
                            html.H5(side, className=f"{side_class} font-weight-bold"),
                            html.Small(f"{quantity:.6f} @ ${price:.2f}")
                        ], width=3),
                        
                        dbc.Col([
                            html.H6(f"${value:.2f}", className="mb-2"),
                            html.P(f"P&L: ${pnl:.2f}" if pnl is not None else "P&L: N/A", 
                                   className=f"{pnl_class} mb-0 font-weight-bold")
                        ], width=6)
                    ])
                ])
            ], className="mb-2 bg-dark border")
        )
    
    if not trade_rows:
        trade_rows = [html.P("No trades available for the selected symbol")]
    
    return trade_rows

# Callback to update performance chart
@app.callback(
    Output("performance-chart", "figure"),
    Input("refresh-interval", "n_intervals")
)
def update_performance_chart(n):
    # Create mock performance data if real data isn't available yet
    # This would be replaced with actual portfolio history in a real implementation
    if not data_store.portfolio:
        # Empty chart
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin={"t": 10, "r": 10, "l": 10, "b": 10},
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
        )
        return fig
    
    # Use portfolio history if available or create mock data
    portfolio_history = data_store.portfolio.get('history', [])
    
    if not portfolio_history:
        # Create mock data for demo purposes
        end_date = datetime.now()
        days = 30
        dates = [(end_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days, -1, -1)]
        
        # Create a somewhat realistic portfolio growth curve
        np.random.seed(42)  # For reproducibility
        base_value = 10000
        daily_returns = np.random.normal(0.002, 0.01, days+1).cumsum()
        values = [base_value * (1 + ret) for ret in daily_returns]
        
        portfolio_history = [{'date': date, 'value': value} for date, value in zip(dates, values)]
    
    # Extract data for plotting
    dates = [item.get('date') for item in portfolio_history]
    values = [item.get('value') for item in portfolio_history]
    
    # Create the performance chart
    fig = go.Figure()
    
    # Add portfolio value line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode="lines",
            name="Portfolio Value",
            line=dict(color="rgba(0, 255, 0, 0.7)", width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 255, 0, 0.1)"
        )
    )
    
    # Add trade markers
    trade_timestamps = []
    trade_values = []
    trade_texts = []
    trade_colors = []
    
    for trade in data_store.trades:
        timestamp = trade.get('timestamp')
        if timestamp:
            date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime('%Y-%m-%d')
            side = trade.get('side')
            symbol = trade.get('symbol')
            quantity = trade.get('quantity', 0)
            price = trade.get('price', 0)
            value = quantity * price
            
            # Find the portfolio value on that date
            portfolio_value = next((item.get('value') for item in portfolio_history if item.get('date') == date), None)
            
            if portfolio_value:
                trade_timestamps.append(date)
                trade_values.append(portfolio_value)
                trade_texts.append(f"{side} {quantity} {symbol} @ ${price:.2f}")
                trade_colors.append("green" if side == "BUY" else "red")
    
    if trade_timestamps:
        fig.add_trace(
            go.Scatter(
                x=trade_timestamps,
                y=trade_values,
                mode="markers",
                name="Trades",
                marker=dict(size=10, color=trade_colors, symbol="circle"),
                text=trade_texts,
                hoverinfo="text"
            )
        )
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 10, "r": 10, "l": 10, "b": 10},
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)"
        ),
        yaxis=dict(
            title="Portfolio Value (USD)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)"
        )
    )
    
    return fig

# Navbar toggle callback
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# Callback to update portfolio risk metrics
@app.callback(
    Output("portfolio-risk", "children"),
    Input("refresh-interval", "n_intervals")
)
def update_portfolio_risk(n):
    if not data_store.risk_metrics:
        return html.Div([
            html.H5("Risk Metrics", className="text-center mb-2"),
            html.P("No risk data available", className="text-center")
        ])
        
    # Extract risk metrics
    var = data_store.risk_metrics.get('portfolio_var', 0) * 100
    max_drawdown = data_store.risk_metrics.get('max_drawdown', 0) * 100
    correlation_factor = data_store.risk_metrics.get('avg_correlation', 0)
    
    # Create risk cards
    risk_cards = html.Div([
        html.H5("Risk Metrics", className="text-center mb-3"),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H6("VaR (95%)", className="text-center mb-1"),
                    html.H4(f"{var:.2f}%", 
                           className=f"text-center {'text-danger' if var > 5 else 'text-warning' if var > 3 else 'text-success'}")
                ], className="mb-2")
            ], width=6),
            dbc.Col([
                html.Div([
                    html.H6("Max Drawdown", className="text-center mb-1"),
                    html.H4(f"{max_drawdown:.2f}%",
                           className=f"text-center {'text-danger' if max_drawdown > 15 else 'text-warning' if max_drawdown > 10 else 'text-success'}")
                ], className="mb-2")
            ], width=6)
        ]),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H6("Correlation", className="text-center mb-1"),
                    html.H4(f"{correlation_factor:.2f}",
                           className=f"text-center {'text-danger' if correlation_factor > 0.7 else 'text-warning' if correlation_factor > 0.5 else 'text-success'}")
                ], className="mb-2")
            ], width=12)
        ])
    ])
    
    return risk_cards

# Callback to update AI model performance chart
@app.callback(
    Output("ai-model-performance", "figure"),
    Input("refresh-interval", "n_intervals")
)
def update_ai_model_performance(n):
    if not data_store.ai_models:
        # Empty chart
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin={"t": 10, "r": 10, "l": 10, "b": 10},
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
        )
        return fig

    # Extract model performance data
    model_history = data_store.ai_models.get('performance_history', [])
    
    if not model_history:
        # Create mock data for demonstration
        end_date = datetime.now()
        days = 30
        dates = [(end_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days, -1, -1)]
        
        accuracy = [0.65 + (np.sin(i/5) * 0.15) for i in range(days+1)]
        profit_factor = [1.2 + (np.sin(i/4) * 0.4) for i in range(days+1)]
        
        model_history = [
            {'date': date, 'accuracy': acc, 'profit_factor': pf, 'trades': 10 + i % 15} 
            for i, (date, acc, pf) in enumerate(zip(dates, accuracy, profit_factor))
        ]
    
    # Extract data for plotting
    dates = [item.get('date') for item in model_history]
    accuracy = [item.get('accuracy', 0) * 100 for item in model_history]
    profit_factor = [item.get('profit_factor', 0) for item in model_history]
    trades = [item.get('trades', 0) for item in model_history]
    
    # Create the figure with dual y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add accuracy line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=accuracy,
            mode="lines",
            name="Accuracy (%)",
            line=dict(color="rgba(0, 255, 0, 0.7)", width=2)
        ),
        secondary_y=False
    )
    
    # Add profit factor line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=profit_factor,
            mode="lines",
            name="Profit Factor",
            line=dict(color="rgba(255, 165, 0, 0.7)", width=2)
        ),
        secondary_y=True
    )
    
    # Add trade count as bars
    fig.add_trace(
        go.Bar(
            x=dates,
            y=trades,
            name="Trades",
            opacity=0.3,
            marker=dict(color="rgba(100, 149, 237, 0.5)")
        ),
        secondary_y=False
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 10, "r": 10, "l": 10, "b": 10},
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    # Update axes titles
    fig.update_yaxes(title_text="Accuracy (%) / Trades", secondary_y=False)
    fig.update_yaxes(title_text="Profit Factor", secondary_y=True)
    
    return fig

# Callback to update AI model comparison chart
@app.callback(
    Output("ai-model-comparison", "figure"),
    Input("refresh-interval", "n_intervals")
)
def update_ai_model_comparison(n):
    if not data_store.ai_models:
        # Empty chart
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin={"t": 10, "r": 10, "l": 10, "b": 10},
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
        )
        return fig
    
    # Extract model comparison data
    model_versions = data_store.ai_models.get('versions', [])
    
    if not model_versions:
        # Create mock data for demonstration
        model_versions = [
            {
                'version_id': f'v1.{i}',
                'accuracy': 0.6 + (i * 0.03),
                'profit_factor': 1.2 + (i * 0.1),
                'sharpe_ratio': 0.8 + (i * 0.15),
                'trades_count': 100 + (i * 30),
                'win_rate': 0.5 + (i * 0.02),
                'avg_return': 0.015 + (i * 0.002)
            } 
            for i in range(5)
        ]
    
    # Extract data for plotting
    versions = [item.get('version_id') for item in model_versions]
    accuracy = [item.get('accuracy', 0) * 100 for item in model_versions]
    profit_factor = [item.get('profit_factor', 0) for item in model_versions]
    sharpe_ratio = [item.get('sharpe_ratio', 0) for item in model_versions]
    win_rate = [item.get('win_rate', 0) * 100 for item in model_versions]
    avg_return = [item.get('avg_return', 0) * 100 for item in model_versions]
    
    # Create the comparison chart
    fig = go.Figure()
    
    # Add traces for each metric
    fig.add_trace(
        go.Bar(
            x=versions,
            y=accuracy,
            name="Accuracy (%)",
            marker_color="rgba(0, 255, 0, 0.7)"
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=versions,
            y=profit_factor,
            name="Profit Factor",
            marker_color="rgba(255, 165, 0, 0.7)"
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=versions,
            y=sharpe_ratio,
            name="Sharpe Ratio",
            marker_color="rgba(0, 191, 255, 0.7)"
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=versions,
            y=win_rate,
            name="Win Rate (%)",
            marker_color="rgba(255, 99, 71, 0.7)"
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=versions,
            y=avg_return,
            name="Avg Return (%)",
            marker_color="rgba(186, 85, 211, 0.7)"
        )
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 10, "r": 10, "l": 10, "b": 10},
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1
    )
    
    return fig

# Callback to update AI model details
@app.callback(
    Output("ai-model-details", "children"),
    Input("refresh-interval", "n_intervals")
)
def update_ai_model_details(n):
    if not data_store.ai_models:
        return html.P("No AI model data available")
    
    # Extract current model data
    current_model = data_store.ai_models.get('current_model', {})
    
    if not current_model:
        # For demonstration purposes
        current_model = {
            'version_id': 'v1.4',
            'version_name': 'LSTM-Transformer-Hybrid',
            'created_at': '2025-03-12 14:30:45',
            'accuracy': 0.72,
            'profit_factor': 1.6,
            'sharpe_ratio': 1.4,
            'trades_count': 230,
            'win_rate': 0.58,
            'avg_return': 0.021,
            'features': ['price_action', 'technical_indicators', 'social_sentiment', 'volatility'],
            'description': 'Hybrid model combining LSTM for sequential price data with transformer architecture for social metric analysis.'
        }
    
    # Create the model details card
    details_card = [
        dbc.Row([
            dbc.Col([
                html.H5(f"Current Model: {current_model.get('version_name', 'Unknown')}", className="mb-3"),
                html.Div([
                    html.Strong("Version ID: "),
                    html.Span(current_model.get('version_id', 'Unknown'))
                ], className="mb-2"),
                html.Div([
                    html.Strong("Created: "),
                    html.Span(current_model.get('created_at', 'Unknown'))
                ], className="mb-2"),
                html.Div([
                    html.Strong("Description: "),
                    html.Span(current_model.get('description', 'No description available'))
                ], className="mb-2"),
            ], xs=12, md=6),
            
            dbc.Col([
                html.H5("Performance Metrics", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Strong("Accuracy: "),
                            html.Span(f"{current_model.get('accuracy', 0) * 100:.1f}%")
                        ], className="mb-2"),
                        html.Div([
                            html.Strong("Win Rate: "),
                            html.Span(f"{current_model.get('win_rate', 0) * 100:.1f}%")
                        ], className="mb-2"),
                        html.Div([
                            html.Strong("Avg Return: "),
                            html.Span(f"{current_model.get('avg_return', 0) * 100:.2f}%")
                        ], className="mb-2"),
                    ], width=6),
                    
                    dbc.Col([
                        html.Div([
                            html.Strong("Profit Factor: "),
                            html.Span(f"{current_model.get('profit_factor', 0):.2f}")
                        ], className="mb-2"),
                        html.Div([
                            html.Strong("Sharpe Ratio: "),
                            html.Span(f"{current_model.get('sharpe_ratio', 0):.2f}")
                        ], className="mb-2"),
                        html.Div([
                            html.Strong("Trades: "),
                            html.Span(f"{current_model.get('trades_count', 0)}")
                        ], className="mb-2"),
                    ], width=6)
                ])
            ], xs=12, md=6, className="mt-3 mt-md-0")
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H5("Features Used", className="mb-3 mt-4"),
                html.Div([
                    dbc.Badge(feature, color="info", className="me-2 mb-2 p-2") 
                    for feature in current_model.get('features', [])
                ])
            ], width=12)
        ])
    ]
    
    return details_card

# Callback to update VaR chart
@app.callback(
    Output("var-chart", "figure"),
    Input("refresh-interval", "n_intervals")
)
def update_var_chart(n):
    if not data_store.risk_metrics:
        # Empty chart
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin={"t": 10, "r": 10, "l": 10, "b": 10},
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
        )
        return fig
    
    # Extract VaR history
    var_history = data_store.risk_metrics.get('var_history', [])
    
    if not var_history:
        # Create mock data for demonstration
        end_date = datetime.now()
        days = 30
        dates = [(end_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days, -1, -1)]
        
        portfolio_var = [0.03 + (np.sin(i/10) * 0.015) for i in range(days+1)]
        market_vol = [0.015 + (np.sin(i/7) * 0.01) for i in range(days+1)]
        
        var_history = [
            {'date': date, 'portfolio_var': pvar, 'market_volatility': mvol} 
            for date, pvar, mvol in zip(dates, portfolio_var, market_vol)
        ]
    
    # Extract data for plotting
    dates = [item.get('date') for item in var_history]
    portfolio_var = [item.get('portfolio_var', 0) * 100 for item in var_history]
    market_vol = [item.get('market_volatility', 0) * 100 for item in var_history]
    
    # Create the VaR chart
    fig = go.Figure()
    
    # Add Portfolio VaR line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=portfolio_var,
            mode="lines+markers",
            name="Portfolio VaR (95%)",
            line=dict(color="rgba(255, 99, 71, 0.8)", width=2)
        )
    )
    
    # Add Market Volatility line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=market_vol,
            mode="lines",
            name="Market Volatility",
            line=dict(color="rgba(100, 149, 237, 0.6)", width=1.5, dash="dash")
        )
    )
    
    # Add threshold line at 5% VaR
    fig.add_shape(
        type="line",
        x0=dates[0],
        y0=5,
        x1=dates[-1],
        y1=5,
        line=dict(color="rgba(255, 0, 0, 0.5)", width=1, dash="dot")
    )
    
    # Add annotation for the threshold line
    fig.add_annotation(
        x=dates[0],
        y=5.1,
        text="Risk Threshold (5%)",
        showarrow=False,
        font=dict(size=10, color="rgba(255, 0, 0, 0.7)")
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 30, "r": 10, "l": 10, "b": 10},
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text="Value at Risk (VaR) Over Time", x=0.5),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(
            title="Percentage (%)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)"
        ),
        hovermode="x unified"
    )
    
    return fig

# Callback to update stop loss chart
@app.callback(
    Output("stop-loss-chart", "figure"),
    Input("refresh-interval", "n_intervals"),
    Input("symbol-selector", "value")
)
def update_stop_loss_chart(n, symbol):
    if not data_store.risk_metrics or not symbol:
        # Empty chart
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin={"t": 10, "r": 10, "l": 10, "b": 10},
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
        )
        return fig
    
    # Extract stop loss history
    stop_loss_data = data_store.risk_metrics.get('stop_loss_data', {}).get(symbol, [])
    
    if not stop_loss_data:
        # Create mock data for demonstration
        end_date = datetime.now()
        days = 30
        dates = [(end_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days, -1, -1)]
        
        price = [100 * (1 + np.cumsum([np.random.normal(0.001, 0.015) for _ in range(days+1)])[i]) for i in range(days+1)]
        volatility = [0.015 + (np.sin(i/7) * 0.01) for i in range(days+1)]
        adaptive_sl = [p * (1 - (0.02 + v * 2)) for p, v in zip(price, volatility)]
        fixed_sl = [p * 0.97 for p in price]
        
        stop_loss_data = [
            {
                'date': date, 
                'price': p, 
                'adaptive_stop_loss': asl, 
                'fixed_stop_loss': fsl,
                'volatility': v
            } 
            for date, p, asl, fsl, v in zip(dates, price, adaptive_sl, fixed_sl, volatility)
        ]
    
    # Extract data for plotting
    dates = [item.get('date') for item in stop_loss_data]
    price = [item.get('price', 0) for item in stop_loss_data]
    adaptive_sl = [item.get('adaptive_stop_loss', 0) for item in stop_loss_data]
    fixed_sl = [item.get('fixed_stop_loss', 0) for item in stop_loss_data]
    
    # Create the stop loss chart
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=price,
            mode="lines",
            name="Price",
            line=dict(color="rgba(255, 255, 255, 0.8)", width=2)
        )
    )
    
    # Add adaptive stop loss line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=adaptive_sl,
            mode="lines",
            name="Adaptive Stop-Loss",
            line=dict(color="rgba(255, 99, 71, 0.8)", width=2)
        )
    )
    
    # Add fixed stop loss line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=fixed_sl,
            mode="lines",
            name="Fixed Stop-Loss",
            line=dict(color="rgba(100, 149, 237, 0.6)", width=1.5, dash="dash")
        )
    )
    
    # Create filled area between price and adaptive stop loss
    fig.add_trace(
        go.Scatter(
            x=dates+dates[::-1],
            y=price+adaptive_sl[::-1],
            fill='toself',
            fillcolor='rgba(255, 99, 71, 0.1)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            showlegend=False,
            hoverinfo='skip'
        )
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 30, "r": 10, "l": 10, "b": 10},
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text=f"Adaptive Stop-Loss for {symbol}", x=0.5),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(
            title="Price",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)"
        ),
        hovermode="x unified"
    )
    
    return fig

# Callback to update correlation heatmap
@app.callback(
    Output("correlation-heatmap", "figure"),
    Input("refresh-interval", "n_intervals")
)
def update_correlation_heatmap(n):
    if not data_store.risk_metrics:
        # Empty chart
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin={"t": 10, "r": 10, "l": 10, "b": 10},
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
        )
        return fig
    
    # Extract correlation data
    corr_matrix = data_store.risk_metrics.get('correlation_matrix', {})
    
    if not corr_matrix:
        # Create mock data for demonstration
        assets = ['BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'SOL-USDT', 'XRP-USDT', 'ADA-USDT']
        n = len(assets)
        
        # Create a realistic correlation matrix (positive definite)
        np.random.seed(42)
        A = np.random.randn(n, n)
        base_corr = A.dot(A.T)
        # Normalize to correlation matrix
        D = np.diag(np.sqrt(np.diag(base_corr)))
        D_inv = np.linalg.inv(D)
        corr = D_inv.dot(base_corr).dot(D_inv)
        
        # Ensure diagonal is 1
        np.fill_diagonal(corr, 1)
        
        # Convert to dictionary format
        corr_matrix = {
            'assets': assets,
            'matrix': corr.tolist()
        }
    
    # Extract data for plotting
    assets = corr_matrix.get('assets', [])
    matrix = corr_matrix.get('matrix', [])
    
    # Create the correlation heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=assets,
        y=assets,
        colorscale='RdBu_r',  # Red for positive, blue for negative correlations
        zmid=0,
        text=[[f"{val:.2f}" for val in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='%{y} to %{x}: %{z:.3f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 30, "r": 10, "l": 10, "b": 10},
        title=dict(text="Asset Correlation Matrix", x=0.5),
        xaxis=dict(
            title="",
            showgrid=False,
            tickangle=-45
        ),
        yaxis=dict(
            title="",
            showgrid=False
        )
    )
    
    return fig

# Callback to update position sizing data
@app.callback(
    Output("position-sizing", "children"),
    Input("refresh-interval", "n_intervals")
)
def update_position_sizing(n):
    if not data_store.risk_metrics:
        return html.P("No position sizing data available")
    
    # Extract position sizing data
    position_data = data_store.risk_metrics.get('position_sizing', [])
    
    if not position_data:
        # Create mock data for demonstration
        position_data = [
            {
                'symbol': 'BTC-USDT',
                'allocation': 0.30,
                'volatility_factor': 1.8,
                'correlation_adjustment': -0.12,
                'risk_score': 0.85,
                'max_position_size': 0.25,
                'current_position_size': 0.18
            },
            {
                'symbol': 'ETH-USDT',
                'allocation': 0.25,
                'volatility_factor': 1.5,
                'correlation_adjustment': -0.08,
                'risk_score': 0.78,
                'max_position_size': 0.22,
                'current_position_size': 0.20
            },
            {
                'symbol': 'BNB-USDT',
                'allocation': 0.15,
                'volatility_factor': 1.2,
                'correlation_adjustment': 0.05,
                'risk_score': 0.65,
                'max_position_size': 0.18,
                'current_position_size': 0.12
            },
            {
                'symbol': 'SOL-USDT',
                'allocation': 0.12,
                'volatility_factor': 2.1,
                'correlation_adjustment': -0.15,
                'risk_score': 0.92,
                'max_position_size': 0.15,
                'current_position_size': 0.10
            },
            {
                'symbol': 'ADA-USDT',
                'allocation': 0.10,
                'volatility_factor': 1.4,
                'correlation_adjustment': 0.10,
                'risk_score': 0.72,
                'max_position_size': 0.15,
                'current_position_size': 0.08
            },
            {
                'symbol': 'XRP-USDT',
                'allocation': 0.08,
                'volatility_factor': 1.1,
                'correlation_adjustment': 0.12,
                'risk_score': 0.60,
                'max_position_size': 0.12,
                'current_position_size': 0.05
            }
        ]
    
    # Create position sizing table
    position_table = html.Div([
        html.H5("Risk-Optimized Position Sizing", className="mb-3"),
        html.P("Position sizes adjusted for volatility and correlation", className="text-muted mb-3"),
        
        # Header row
        dbc.Row([
            dbc.Col(html.Strong("Asset"), width=2),
            dbc.Col(html.Strong("Allocation"), width=2),
            dbc.Col(html.Strong("Risk Score"), width=2),
            dbc.Col(html.Strong("Vol. Factor"), width=2),
            dbc.Col(html.Strong("Corr. Adj."), width=2),
            dbc.Col(html.Strong("Position Size"), width=2),
        ], className="mb-2"),
        
        # Data rows
        *[
            dbc.Row([
                dbc.Col(html.Span(item.get('symbol', '')), width=2),
                dbc.Col([
                    html.Span(f"{item.get('allocation', 0) * 100:.1f}%"),
                ], width=2),
                dbc.Col([
                    html.Span(f"{item.get('risk_score', 0):.2f}",
                              className=f"{'text-danger' if item.get('risk_score', 0) > 0.8 else 'text-warning' if item.get('risk_score', 0) > 0.6 else 'text-success'}")
                ], width=2),
                dbc.Col(html.Span(f"{item.get('volatility_factor', 0):.2f}"), width=2),
                dbc.Col(html.Span(f"{item.get('correlation_adjustment', 0):.2f}"), width=2),
                dbc.Col([
                    dbc.Progress(
                        value=(item.get('current_position_size', 0) / item.get('max_position_size', 1)) * 100,
                        color="info",
                        className="mb-1",
                        style={"height": "8px"}
                    ),
                    html.Small(f"{item.get('current_position_size', 0) * 100:.1f}% / {item.get('max_position_size', 0) * 100:.1f}%")
                ], width=2),
            ], className="mb-2")
            for item in position_data
        ],
        
        html.Div([
            html.Strong("Position Sizing Method: "),
            html.Span(data_store.risk_metrics.get('position_sizing_method', 'equal_risk'))
        ], className="mt-4 text-muted")
    ])
    
    return position_table

# Callbacks for modal functionality
@app.callback(
    Output("ai-explanation-modal", "is_open"),
    [Input("ai-explanation-btn", "n_clicks"), Input("close-ai-explanation", "n_clicks")],
    [State("ai-explanation-modal", "is_open")],
)
def toggle_ai_explanation_modal(open_clicks, close_clicks, is_open):
    if open_clicks or close_clicks:
        return not is_open
    return is_open

@app.callback(
    Output("sentiment-details-modal", "is_open"),
    [Input("sentiment-details-btn", "n_clicks"), Input("close-sentiment-details", "n_clicks")],
    [State("sentiment-details-modal", "is_open")],
)
def toggle_sentiment_details_modal(open_clicks, close_clicks, is_open):
    if open_clicks or close_clicks:
        return not is_open
    return is_open

# Callback to update AI explanation content
@app.callback(
    Output("ai-explanation-content", "children"),
    Input("ai-explanation-modal", "is_open"),
    Input("current-signal", "data")
)
def update_ai_explanation_content(is_open, signal_data):
    if not is_open or not signal_data:
        return html.P("Select a signal to view explanation")
    
    # Extract explanation data
    explanation = signal_data.get('explanation', {})
    factor_weights = signal_data.get('factor_weights', {})
    
    if not explanation or not factor_weights:
        # For demonstration purposes
        explanation = {
            'summary': 'BUY signal based on strong technical indicators and positive social sentiment.',
            'technical_factors': 'RSI showing oversold conditions at 28.5. MACD showing bullish crossover. Price breaking above upper Bollinger Band suggesting strong momentum.',
            'social_factors': 'Social sentiment is highly positive at 72.3, with increasing social volume and engagement over the past 24 hours. Contributing to a positive sentiment shift.',
            'key_indicators': ['RSI', 'MACD', 'Bollinger Bands', 'Social Sentiment', 'Volume Trend'],
            'risk_assessment': 'Medium risk due to overall market volatility. Suggested position size reduced by 20% from baseline.'
        }
        
        factor_weights = {
            'technical_indicators': {
                'rsi': 0.25,
                'macd': 0.22,
                'bollinger_bands': 0.18,
                'price_action': 0.20,
                'other': 0.15
            },
            'price_action': {
                'momentum': 0.45,
                'volatility': 0.30,
                'volume': 0.25
            },
            'social_metrics': {
                'sentiment': 0.45,
                'volume': 0.30,
                'engagement': 0.25
            },
            'market_context': 0.35
        }
    
    # Create AI explanation content
    content = [
        dbc.Row([
            dbc.Col([
                html.H4("Decision Explanation", className="mb-3"),
                html.Div([
                    html.H6("Summary", className="mb-2"),
                    html.P(explanation.get('summary', 'No summary available'), className="mb-3"),
                    
                    html.H6("Technical Analysis", className="mb-2"),
                    html.P(explanation.get('technical_factors', 'No technical analysis available'), className="mb-3"),
                    
                    html.H6("Social Metrics Analysis", className="mb-2"),
                    html.P(explanation.get('social_factors', 'No social analysis available'), className="mb-3"),
                    
                    html.H6("Risk Assessment", className="mb-2"),
                    html.P(explanation.get('risk_assessment', 'No risk assessment available'), className="mb-3"),
                    
                    html.H6("Key Indicators", className="mb-2"),
                    html.Div([
                        dbc.Badge(indicator, color="success", className="me-2 mb-2 p-2") 
                        for indicator in explanation.get('key_indicators', [])
                    ], className="mb-3"),
                ])
            ], xs=12, lg=6),
            
            dbc.Col([
                html.H4("Factor Weights", className="mb-3"),
                
                # Technical Indicators Weights
                html.H6("Technical Indicators", className="mb-2"),
                *[
                    dbc.Row([
                        dbc.Col(html.Span(indicator.title().replace('_', ' ')), width=4),
                        dbc.Col([
                            dbc.Progress(
                                value=weight * 100,
                                color="info",
                                className="mb-1",
                                style={"height": "15px"}
                            ),
                        ], width=6),
                        dbc.Col(html.Span(f"{weight * 100:.0f}%"), width=2),
                    ], className="mb-2")
                    for indicator, weight in factor_weights.get('technical_indicators', {}).items()
                ],
                
                # Price Action Weights
                html.H6("Price Action", className="mb-2 mt-4"),
                *[
                    dbc.Row([
                        dbc.Col(html.Span(factor.title()), width=4),
                        dbc.Col([
                            dbc.Progress(
                                value=weight * 100,
                                color="success",
                                className="mb-1",
                                style={"height": "15px"}
                            ),
                        ], width=6),
                        dbc.Col(html.Span(f"{weight * 100:.0f}%"), width=2),
                    ], className="mb-2")
                    for factor, weight in factor_weights.get('price_action', {}).items()
                ],
                
                # Social Metrics Weights
                html.H6("Social Metrics", className="mb-2 mt-4"),
                *[
                    dbc.Row([
                        dbc.Col(html.Span(metric.title()), width=4),
                        dbc.Col([
                            dbc.Progress(
                                value=weight * 100,
                                color="warning",
                                className="mb-1",
                                style={"height": "15px"}
                            ),
                        ], width=6),
                        dbc.Col(html.Span(f"{weight * 100:.0f}%"), width=2),
                    ], className="mb-2")
                    for metric, weight in factor_weights.get('social_metrics', {}).items()
                ],
                
                # Market Context Weight
                html.H6("Market Context", className="mb-2 mt-4"),
                dbc.Row([
                    dbc.Col(html.Span("Overall Impact"), width=4),
                    dbc.Col([
                        dbc.Progress(
                            value=factor_weights.get('market_context', 0) * 100,
                            color="danger",
                            className="mb-1",
                            style={"height": "15px"}
                        ),
                    ], width=6),
                    dbc.Col(html.Span(f"{factor_weights.get('market_context', 0) * 100:.0f}%"), width=2),
                ], className="mb-2"),
            ], xs=12, lg=6, className="mt-4 mt-lg-0")
        ])
    ]
    
    return content

# Callback to update sentiment details content
@app.callback(
    Output("sentiment-details-content", "children"),
    Input("sentiment-details-modal", "is_open"),
    Input("symbol-selector", "value")
)
def update_sentiment_details_content(is_open, symbol):
    if not is_open or not symbol or symbol not in data_store.social_data:
        return html.P("No sentiment data available for the selected symbol")
    
    # Extract social data
    social_data = data_store.social_data.get(symbol, {})
    
    # Create detailed content for sentiment modal
    content = [
        html.H4(f"Social Sentiment Analysis for {symbol}", className="mb-4 text-center"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Sentiment Breakdown", className="mb-3"),
                        
                        # Detailed sentiment metrics
                        dbc.Row([
                            dbc.Col([
                                html.Strong("Overall Sentiment:"),
                                html.Span(f" {social_data.get('sentiment', 0):.1f}/100", 
                                       className=f"{'text-success' if social_data.get('sentiment', 0) >= 60 else 'text-danger' if social_data.get('sentiment', 0) <= 40 else 'text-warning'}")
                            ], width=12, className="mb-2"),
                            
                            dbc.Col([
                                html.Strong("Positive Mentions:"),
                                html.Span(f" {social_data.get('positive_mentions', 0):,}")
                            ], width=6, className="mb-2"),
                            
                            dbc.Col([
                                html.Strong("Negative Mentions:"),
                                html.Span(f" {social_data.get('negative_mentions', 0):,}")
                            ], width=6, className="mb-2"),
                            
                            dbc.Col([
                                html.Strong("Neutral Mentions:"),
                                html.Span(f" {social_data.get('neutral_mentions', 0):,}")
                            ], width=6, className="mb-2"),
                            
                            dbc.Col([
                                html.Strong("Sentiment Change:"),
                                html.Span(f" {social_data.get('sentiment_change_24h', 0):+.1f}%", 
                                       className=f"{'text-success' if social_data.get('sentiment_change_24h', 0) > 0 else 'text-danger' if social_data.get('sentiment_change_24h', 0) < 0 else ''}")
                            ], width=6, className="mb-2"),
                        ]),
                        
                        # Sentiment by source
                        html.H6("Sentiment by Source", className="mt-4 mb-3"),
                        *[
                            dbc.Row([
                                dbc.Col(html.Span(source.title()), width=4),
                                dbc.Col([
                                    dbc.Progress(
                                        value=score,
                                        color="success" if score >= 60 else "danger" if score <= 40 else "warning",
                                        className="mb-1",
                                        style={"height": "10px"}
                                    ),
                                ], width=6),
                                dbc.Col(html.Span(f"{score:.1f}"), width=2),
                            ], className="mb-2")
                            for source, score in social_data.get('sentiment_by_source', {
                                'twitter': 65.2,
                                'reddit': 58.7,
                                'news': 62.1,
                                'blogs': 59.3
                            }).items()
                        ],
                    ])
                ], className="mb-3")
            ], xs=12, lg=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Engagement Metrics", className="mb-3"),
                        
                        # Detailed engagement metrics
                        dbc.Row([
                            dbc.Col([
                                html.Strong("Total Social Volume:"),
                                html.Span(f" {social_data.get('volume', 0):,}")
                            ], width=12, className="mb-2"),
                            
                            dbc.Col([
                                html.Strong("Social Engagement:"),
                                html.Span(f" {social_data.get('engagement', 0):,}")
                            ], width=6, className="mb-2"),
                            
                            dbc.Col([
                                html.Strong("Contributors:"),
                                html.Span(f" {social_data.get('contributors', 0):,}")
                            ], width=6, className="mb-2"),
                            
                            dbc.Col([
                                html.Strong("Volume Change:"),
                                html.Span(f" {social_data.get('volume_change_24h', 0):+.1f}%",
                                       className=f"{'text-success' if social_data.get('volume_change_24h', 0) > 0 else 'text-danger' if social_data.get('volume_change_24h', 0) < 0 else ''}")
                            ], width=6, className="mb-2"),
                            
                            dbc.Col([
                                html.Strong("Engagement Change:"),
                                html.Span(f" {social_data.get('engagement_change_24h', 0):+.1f}%",
                                       className=f"{'text-success' if social_data.get('engagement_change_24h', 0) > 0 else 'text-danger' if social_data.get('engagement_change_24h', 0) < 0 else ''}")
                            ], width=6, className="mb-2"),
                        ]),
                        
                        # Volume by source
                        html.H6("Volume by Source", className="mt-4 mb-3"),
                        *[
                            dbc.Row([
                                dbc.Col(html.Span(source.title()), width=4),
                                dbc.Col([
                                    dbc.Progress(
                                        value=(volume / social_data.get('volume', 1)) * 100,
                                        color="info",
                                        className="mb-1",
                                        style={"height": "10px"}
                                    ),
                                ], width=6),
                                dbc.Col(html.Span(f"{volume:,}"), width=2),
                            ], className="mb-2")
                            for source, volume in social_data.get('volume_by_source', {
                                'twitter': 12500,
                                'reddit': 8700,
                                'news': 3200,
                                'blogs': 1800
                            }).items()
                        ],
                    ])
                ], className="mb-3")
            ], xs=12, lg=6, className="mt-3 mt-lg-0"),
        ]),
        
        # Trending keywords/topics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Trending Keywords & Topics", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    dbc.Badge(
                                        topic['keyword'], 
                                        color="primary", 
                                        className="me-2 mb-2 p-2",
                                        style={"fontSize": f"{10 + (topic['weight'] * 10)}px"}
                                    )
                                    for topic in social_data.get('trending_topics', [
                                        {'keyword': 'partnerships', 'weight': 0.95},
                                        {'keyword': 'adoption', 'weight': 0.85},
                                        {'keyword': 'development', 'weight': 0.80},
                                        {'keyword': 'bullish', 'weight': 0.75},
                                        {'keyword': 'update', 'weight': 0.70},
                                        {'keyword': 'technology', 'weight': 0.65},
                                        {'keyword': 'price', 'weight': 0.60},
                                        {'keyword': 'trading', 'weight': 0.55},
                                        {'keyword': 'market', 'weight': 0.50},
                                        {'keyword': 'future', 'weight': 0.45},
                                        {'keyword': 'investment', 'weight': 0.40},
                                        {'keyword': 'wallet', 'weight': 0.35},
                                    ])
                                ])
                            ], width=12)
                        ])
                    ])
                ])
            ], width=12)
        ])
    ]
    
    return content

# Callback to store current signal for explanation modal
@app.callback(
    Output("current-signal", "data"),
    Input("signals-table", "children"),
    State("symbol-selector", "value")
)
def store_current_signal(signals_children, symbol):
    # In a real implementation, this would find the most recent signal
    # for the selected symbol with explanation data
    
    # For demonstration, we'll create mock data
    if not symbol:
        return {}
    
    # Find signal with explanation data for this symbol
    # (this is mock data since we don't have actual signals with explanation)
    mock_signal = {
        'symbol': symbol,
        'action': 'BUY',
        'timestamp': datetime.now().isoformat(),
        'confidence': 0.82,
        'price': 43250.75,
        'explanation': {
            'summary': f'BUY signal for {symbol} based on strong technical indicators and positive social sentiment.',
            'technical_factors': 'RSI showing oversold conditions at 28.5. MACD showing bullish crossover. Price breaking above upper Bollinger Band suggesting strong momentum.',
            'social_factors': 'Social sentiment is highly positive at 72.3, with increasing social volume and engagement over the past 24 hours. Contributing to a positive sentiment shift.',
            'key_indicators': ['RSI', 'MACD', 'Bollinger Bands', 'Social Sentiment', 'Volume Trend'],
            'risk_assessment': 'Medium risk due to overall market volatility. Suggested position size reduced by 20% from baseline.'
        },
        'factor_weights': {
            'technical_indicators': {
                'rsi': 0.25,
                'macd': 0.22,
                'bollinger_bands': 0.18,
                'price_action': 0.20,
                'other': 0.15
            },
            'price_action': {
                'momentum': 0.45,
                'volatility': 0.30,
                'volume': 0.25
            },
            'social_metrics': {
                'sentiment': 0.45,
                'volume': 0.30,
                'engagement': 0.25
            },
            'market_context': 0.35
        }
    }
    
    return mock_signal

# Run the app
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
