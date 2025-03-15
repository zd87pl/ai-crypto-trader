#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Crypto Trader Dashboard

A modern, interactive dashboard for the AI Crypto Trader system.
Provides real-time visualization of trading data, signals, portfolio performance,
and social sentiment analysis.

Author: zd87pl
Version: 1.0.0
"""

import os
import json
import redis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
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

# Create a data store instance
data_store = DataStore()

# Thread to listen to Redis updates
def redis_listener():
    pubsub = r.pubsub()
    pubsub.subscribe('market_updates', 'social_updates', 'trading_signals', 'trade_executions', 'portfolio_updates')
    
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
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
)
server = app.server
app.title = "AI Crypto Trader Dashboard"

# Define the layout
app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col(html.H1("AI Crypto Trader Dashboard", className="text-center mt-4 mb-4"), width=12)
        ]),
        
        # Market Overview
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Portfolio Overview", className="text-center")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div(id="portfolio-value", className="d-flex justify-content-center align-items-center"),
                                html.Div(id="portfolio-change", className="d-flex justify-content-center align-items-center mt-2")
                            ], width=12),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Div(id="portfolio-assets", className="mt-3")
                            ], width=12)
                        ])
                    ])
                ], className="shadow mb-4")
            ], width=12)
        ]),
        
        # Price Charts and Social Data
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Row([
                            dbc.Col(html.H4("Market Data", className="text-center"), width=8),
                            dbc.Col([
                                dcc.Dropdown(
                                    id="symbol-selector",
                                    options=[],
                                    value=None,
                                    placeholder="Select a trading pair",
                                    className="mr-2"
                                ),
                            ], width=4),
                        ])
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="price-chart", style={"height": "500px"}),
                    ])
                ], className="shadow mb-4")
            ], width=8),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Social Sentiment", className="text-center")),
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
            ], width=4)
        ]),
        
        # Trading Signals and History
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Recent Trading Signals", className="text-center")),
                    dbc.CardBody([
                        html.Div(id="signals-table", style={"height": "300px", "overflow-y": "scroll"})
                    ])
                ], className="shadow mb-4")
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Recent Trades", className="text-center")),
                    dbc.CardBody([
                        html.Div(id="trades-table", style={"height": "300px", "overflow-y": "scroll"})
                    ])
                ], className="shadow mb-4")
            ], width=6)
        ]),
        
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
        
        # Refresh interval
        dcc.Interval(id="refresh-interval", interval=5000, n_intervals=0),
        
        # Store current symbols
        dcc.Store(id="available-symbols")
    ],
    fluid=True,
    className="bg-dark text-light",
)

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

# Run the app
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
