# AI Crypto Trader

## ⚠️ Important Disclaimer

This project is an **experimental cryptocurrency trading system** designed for educational and research purposes only. 

**PLEASE NOTE:**
- This is NOT financial advice
- This is NOT investment advice
- The trading signals and analyses generated are NOT recommendations to buy, sell, or trade any assets
- The system's AI-generated insights should NOT be considered as professional trading guidance
- Past performance does not guarantee future results
- Cryptocurrency trading involves substantial risk of loss
- You should never invest money you cannot afford to lose

By using this system, you acknowledge that you are doing so at your own risk and that the creators and contributors of this project cannot be held responsible for any financial losses incurred.

This project serves as a technical demonstration of integrating AI with cryptocurrency markets and should be treated as an educational resource for understanding automated trading systems.

---

## Dashboard

The project includes a modern, interactive dashboard that provides real-time insights into your trading system:

![Dashboard Screenshot](https://placeholder-for-dashboard-screenshot.png)

### Dashboard Features

- **Real-time Price Charts**: Interactive candlestick charts with technical indicators (RSI, MACD, Bollinger Bands)
- **Portfolio Overview**: Track total value, daily change, and asset allocation
- **Trading Signals**: View recent trading signals with confidence levels and reasoning
- **Trade History**: Monitor executed trades and performance
- **Social Sentiment Analysis**: Track social media sentiment, volume, engagement, and contributors
- **News Feed**: Latest crypto news affecting your trading pairs
- **Performance Metrics**: Visual representation of portfolio growth over time

### Accessing the Dashboard

When running the system with Docker Compose, the dashboard is available at:
```
http://localhost:8050
```

For local setup, start the dashboard separately with:
```bash
python dashboard.py
```

## System Components

### 1. Market Monitor Service
- Monitors real-time market data from Binance
- Calculates technical indicators and price changes across multiple timeframes (1m, 3m, 5m, 15m)
- Publishes market updates to Redis
- Logs to: `logs/market_monitor.log`

### 2. Social Monitor Service
- Fetches social metrics and news from LunarCrush API
- Tracks social sentiment, engagement, and volume
- Monitors news impact and social trends
- Caches data to respect API rate limits
- Logs to: `logs/social_monitor.log`

### 3. AI Analyzer Service
- Processes market updates using OpenAI GPT-4
- Incorporates social sentiment and news analysis
- Generates context-aware trading signals
- Publishes trading signals to Redis
- Logs to: `logs/ai_analyzer.log`

### 4. Trade Executor Service
- Executes trades based on AI analysis signals
- Manages positions and risk
- Updates portfolio holdings
- Logs to: `logs/trade_executor.log`

### 5. Strategy Evolution Service
- Self-improves trading strategies based on performance data
- Uses AI to generate and refine trading approaches
- Maintains a history of strategy performance
- Logs to: `logs/strategy_evolution.log`

### 6. Backtesting Framework
- Fetches and manages historical market and social data
- Simulates trading strategy performance on historical data
- Integrates with the AI trading logic for realistic backtests
- Generates performance analytics and visualizations
- Detailed documentation in `backtesting/README.md`

### 7. Model Registry Service
- Tracks AI model versions and performance metrics
- Provides version control for trading models
- Maintains a registry of all model versions
- Enables model comparison and selection
- Logs to: `logs/model_registry.log`

### 8. AI Explainability Service
- Enhances trading decisions with detailed explanations
- Visualizes factor weights influencing decisions
- Provides technical and social factors analysis
- Explains the reasoning behind each trade
- Logs to: `logs/ai_explainability.log`

### 9. Portfolio Risk Management Service
- Implements portfolio-wide risk management
- Calculates Value at Risk (VaR) across the entire portfolio
- Provides adaptive stop-losses based on market volatility
- Optimizes position sizing for better risk-adjusted returns
- Monitors asset correlations for improved diversification
- Logs to: `logs/portfolio_risk.log`

## Environment Setup

1. Create a `.env` file with the following credentials:
```env
# Binance API credentials
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret

# OpenAI API credentials
OPENAI_API_KEY=your_openai_api_key

# LunarCrush API credentials
LUNARCRUSH_API_KEY=your_lunarcrush_api_key

# Service ports
MARKET_MONITOR_PORT=8001
TRADE_EXECUTOR_PORT=8002
AI_ANALYZER_PORT=8003
STRATEGY_EVOLUTION_PORT=8004
SOCIAL_MONITOR_PORT=8005
MODEL_REGISTRY_PORT=8006
AI_EXPLAINABILITY_PORT=8007

# Redis configuration
REDIS_HOST=redis
REDIS_PORT=6379
```

## Getting Started

### With Docker (Recommended)

1. Make sure you have Docker and Docker Compose installed
2. Clone the repository and navigate to the project directory
3. Create your `.env` file with the required credentials
4. Start all services with:

```bash
docker-compose up -d
```

This will start all services, including the dashboard which will be available at `http://localhost:8050`.

### Local Development

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Create your `.env` file with the required credentials
3. Start each service separately:

```bash
# Start the market monitor
python services/market_monitor_service.py

# Start the social monitor
python services/social_monitor_service.py

# Start the AI analyzer
python services/ai_analyzer_service.py

# Start the trade executor
python services/trade_executor_service.py

# Start the strategy evolution service
python services/strategy_evolution_service.py

# Start the model registry service
python run_ai_model_services.py --model-registry

# Start the AI explainability service
python run_ai_model_services.py --explainability

# Start the dashboard
python dashboard.py
```

### Using the Backtesting Framework

The backtesting framework allows you to test trading strategies on historical data:

1. Fetch historical data for Bitcoin (1-hour intervals, past 30 days):
```bash
python run_backtest.py fetch --symbols BTCUSDC --intervals 1h --days 30
```

2. Run a backtest on the fetched data:
```bash
python run_backtest.py backtest --symbols BTCUSDC --intervals 1h --days 30 --balance 10000
```

3. View your available historical data:
```bash
python run_backtest.py list
```

4. Analyze backtest results:
```bash
python run_backtest.py analyze --metric sharpe_ratio
```

For more detailed instructions, see `backtesting/README.md`.

## Data Structures

### Market Data
```json
{
    "symbol": "BTCUSDC",
    "current_price": 42150.75,
    "avg_volume": 1250000.25,
    "timestamp": "2024-11-24T21:33:45.029Z",
    "rsi": 55.32,
    "rsi_3m": 54.18,
    "rsi_5m": 53.95,
    "stoch_k": 65.43,
    "macd": 0.00123,
    "macd_3m": 0.00156,
    "macd_5m": 0.00189,
    "williams_r": -34.57,
    "bb_position": 0.65,
    "trend": "uptrend",
    "trend_strength": 0.75,
    "price_change_1m": 0.12,
    "price_change_3m": 0.25,
    "price_change_5m": 0.38,
    "price_change_15m": 0.52
}
```

### Social Data
```json
{
    "symbol": "BTCUSDC",
    "data": {
        "metrics": {
            "social_volume": 15000,
            "social_engagement": 25000,
            "social_contributors": 1200,
            "social_sentiment": 0.75,
            "twitter_volume": 8000,
            "reddit_volume": 3000,
            "news_volume": 150
        },
        "weighted_sentiment": 0.68,
        "recent_news": [
            {
                "title": "Major Partnership Announcement",
                "sentiment": 0.85,
                "engagement": 12000
            }
        ],
        "timestamp": "2024-11-24T21:33:45.029Z"
    }
}
```

### Trading Signals
```json
{
    "decision": "BUY",
    "confidence": 0.85,
    "reasoning": "Strong technical indicators with positive social sentiment",
    "risk_level": "MEDIUM",
    "key_indicators": [
        "RSI trending up",
        "High social engagement",
        "Positive news sentiment"
    ],
    "social_impact": "High positive sentiment with strong engagement",
    "timestamp": "2024-11-24T21:33:45.033Z",
    "symbol": "BTCUSDC",
    "model_version": "ai_trader_gpt_4o_a1b2c3d4",
    "model_id": "a1b2c3d4",
    "explanation": {
        "summary": "Strong bullish signals with technical indicators and social sentiment aligned",
        "technical_factors": "RSI at 55.32 showing momentum, MACD positive and increasing, price breaking above resistance",
        "social_factors": "High social engagement (25,000) with very positive sentiment (0.75), increasing social volume",
        "key_indicators": ["RSI", "MACD", "Social sentiment", "Price action"],
        "risk_assessment": "Medium risk due to some market volatility, but strong technical and social indicators support the trade"
    },
    "factor_weights": {
        "technical_indicators": {
            "rsi": 0.25,
            "macd": 0.30,
            "bollinger_bands": 0.15,
            "price_action": 0.20,
            "other": 0.10
        },
        "social_metrics": {
            "sentiment": 0.50,
            "volume": 0.30,
            "engagement": 0.20
        },
        "market_context": 0.15
    }
}
```

## Error Handling

### Market Monitor Errors
- Connection issues with Binance: Retries with exponential backoff
- Redis connection failures: Automatic reconnection attempts
- Invalid market data: Logged and skipped

### Social Monitor Errors
- LunarCrush API rate limits: Cached data with configurable duration
- Missing social metrics: Falls back to default neutral values
- News processing errors: Skips problematic items while preserving valid ones

### AI Analyzer Errors
- OpenAI API failures: Logged with full error context
- Missing market/social data: Detailed validation errors
- Redis publishing errors: Automatic retry mechanism

### Trade Executor Errors
- Insufficient balance: Logged and trade skipped
- Order placement failures: Full error context with retry attempts
- Position management errors: Detailed error tracking

## Monitoring and Debugging

### Log Locations
- All logs are stored in the `logs/` directory
- Each service has its own log file with rotation enabled
- Log files are limited to 10MB with 5 backup files

### Common Issues and Solutions

1. LunarCrush API Rate Limits
```
ERROR - [SocialMonitor] Rate limit exceeded
```
Solution: Adjust cache_duration in config.json

2. Missing Social Data
```
ERROR - [AIAnalyzer] Missing social metrics for symbol
```
Solution: Verify LunarCrush API key and symbol support

3. OpenAI API Errors
```
ERROR - [AIAnalyzer] OpenAI API error: Invalid model specified
```
Solution: Check OpenAI configuration and API key validity

## Performance Monitoring

The system tracks:
- Portfolio value updates
- Trade success rates
- API response times
- Social sentiment accuracy
- Error frequencies
- System resource usage

Example monitoring log:
```
2024-11-24 21:33:45,400 - INFO - [TradeExecutor] Updated holdings - Total Portfolio Value: $257.65
2024-11-24 21:33:45,401 - INFO - [TradeExecutor] Available USDC: $1.18
2024-11-24 21:33:45,402 - INFO - [SocialMonitor] Social sentiment accuracy: 82%
```
