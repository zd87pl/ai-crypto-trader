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

#### General Features
- **Mobile Responsive Design**: Optimized for all devices from smartphones to desktop monitors
- **Interactive Navigation**: Quick access to trading, portfolio, AI models, and risk sections
- **Real-time Updates**: All data refreshes automatically at configurable intervals

#### Trading Analytics
- **Real-time Price Charts**: Interactive candlestick charts with technical indicators (RSI, MACD, Bollinger Bands)
- **Trading Signals**: View recent trading signals with confidence levels and reasoning
- **Trade History**: Monitor executed trades and performance
- **Social Sentiment Analysis**: Track social media sentiment, volume, engagement, and contributors
- **News Feed**: Latest crypto news affecting your trading pairs

#### Portfolio Management
- **Portfolio Overview**: Track total value, daily change, and asset allocation
- **Performance Metrics**: Visual representation of portfolio growth over time
- **Asset Allocation**: Monitor distribution of investments across different cryptocurrencies

#### AI Model Insights
- **Model Performance**: Track accuracy, profit factor, and trade counts over time
- **Version Comparison**: Compare metrics across different AI model versions
- **Feature Importance**: Visualize which features have the most influence on trading decisions
- **Explainable AI**: Detailed breakdowns of how the AI reaches trading decisions

#### Risk Management
- **Portfolio Risk Metrics**: Monitor VaR (Value at Risk) and maximum drawdown
- **Correlation Analysis**: Visualize asset correlations with interactive heatmaps
- **Position Sizing**: Risk-optimized position sizing with volatility and correlation adjustments
- **Adaptive Stop-Loss**: Visualize how stop-losses adapt to changing market volatility

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
- Integrates social sentiment data into strategy optimization
- Applies genetic algorithms and GPT-based optimization
- Maintains a history of strategy performance
- Logs to: `logs/strategy_evolution.log`

### 6. Market Regime Service
- Detects market regimes (bull, bear, ranging, volatile)
- Adapts trading strategies to current market conditions
- Tracks strategy performance across different regimes
- Logs to: `logs/market_regime.log`

### 7. Strategy Selection Service
- Automatically selects optimal trading strategies
- Considers multiple factors: market regime, historical performance, risk profile, social sentiment
- Implements time-based adjustments for different market hours
- Uses advanced scoring algorithms with configurable weights
- Adapts to changing market conditions in real-time
- Makes data-driven strategy switching decisions
- Logs to: `logs/strategy_selection.log`

### 8. Social Strategy Integrator
- Analyzes correlation between social sentiment and price movements
- Identifies lead/lag relationships in social metrics
- Generates specialized strategies based on social patterns
- Optimizes trading parameters based on social sentiment
- Creates social-specific strategy variants (trend-following, contrarian, etc.)
- Provides social impact scores for strategy selection
- Logs to: `logs/social_strategy.log`

### 9. Backtesting Framework
- Fetches and manages historical market and social data
- Simulates trading strategy performance on historical data
- Integrates with the AI trading logic for realistic backtests
- Generates performance analytics and visualizations
- Detailed documentation in `backtesting/README.md`

### 10. Model Registry Service
- Tracks AI model versions and performance metrics
- Provides version control for trading models
- Maintains a registry of all model versions
- Enables model comparison and selection
- Logs to: `logs/model_registry.log`

### 11. AI Explainability Service
- Enhances trading decisions with detailed explanations
- Visualizes factor weights influencing decisions
- Provides technical and social factors analysis
- Explains the reasoning behind each trade
- Logs to: `logs/ai_explainability.log`

### 12. Feature Importance Analyzer
- Analyzes the importance of different features in machine learning models
- Uses permutation importance and RandomForest feature analysis
- Identifies which indicators have the most influence on trading decisions
- Categorizes features by group (price action, momentum, volatility, trend, volume, social)
- Generates visualizations and comprehensive reports
- Creates optimized models by pruning low-importance features
- Publishes results to other services via Redis
- Provides real-time feature importance tracking in the dashboard
- Helps focus data collection on high-value indicators
- Logs to: `logs/feature_importance.log`

### 13. Market Regime Detection Service
- Identifies current market conditions using machine learning and rule-based systems
- Classifies markets into four regimes: bull, bear, ranging, and volatile
- Uses multiple detection methods: KMeans, Gaussian Mixture Models, Hidden Markov Models
- Provides regime probabilities and confidence scores
- Adapts trading strategies to current market conditions
- Analyzes historical performance of strategies in different regimes
- Generates market regime visualizations
- Monitors regime changes and triggers strategy switching
- Supports dynamic machine learning model retraining
- Logs to: `logs/market_regime.log`

### 14. Portfolio Risk Management Service
- Implements portfolio-wide risk management
- Calculates Value at Risk (VaR) across the entire portfolio
- Provides adaptive stop-losses based on market volatility
- Optimizes position sizing for better risk-adjusted returns
- Monitors asset correlations for improved diversification
- Logs to: `logs/portfolio_risk.log`

### 14. Social Risk Adjuster Service
- Dynamically adjusts risk parameters based on social sentiment
- Modifies position sizes, stop-losses, and take-profit levels
- Applies time decay to social sentiment influence
- Adjusts correlation thresholds based on social metrics
- Detects volatility patterns in social engagement
- Applies different adjustment strategies for bullish/bearish sentiment
- Includes data quality assessment for reliable adjustments
- Logs to: `logs/social_risk_adjuster.log`

### 15. Monte Carlo Simulation Service
- Generates thousands of price path simulations for future risk projection
- Implements multiple simulation methods (geometric Brownian motion, historical)
- Calculates Value at Risk (VaR) and Conditional VaR metrics
- Scenarios analysis for different market conditions (bull, bear, volatile, etc.)
- Creates probability distributions for future price movements
- Analyzes maximum drawdown scenarios
- Generates visual representations of price path simulations
- Produces comprehensive risk reports for the entire portfolio
- Logs to: `logs/monte_carlo.log`

### 16. Neural Network Price Prediction Service
- Implements deep learning models for cryptocurrency price prediction
- Supports multiple neural network architectures (LSTM, GRU, Bidirectional LSTM, CNN-LSTM, Attention)
- Enables ensemble models combining multiple architectures for improved predictions
- Generates predictions for multiple timeframes (1h, 4h, 24h)
- Provides prediction confidence scores and directional indicators
- Features automatic training and model tuning capabilities
- Creates visualizations for both predictions and training performance
- Supports real-time model evaluation and adaptation
- Integrates with trading strategies for signal optimization
- Logs to: `logs/neural_network.log`

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
SOCIAL_RISK_PORT=8008
MONTE_CARLO_PORT=8009

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

# Start the market regime service
python services/market_regime_service.py

# Start the strategy selection service
python services/strategy_selection_service.py

# Start the social strategy integrator
python services/social_strategy_integrator.py

# Start the model registry service
python run_ai_model_services.py --model-registry

# Start the AI explainability service
python run_ai_model_services.py --explainability

# Start the feature importance analyzer
python services/feature_importance_analyzer.py

# Start the social risk adjuster service
python services/social_risk_adjuster.py

# Start the Monte Carlo simulation service
python services/monte_carlo_service.py

# Start the Neural Network price prediction service
python services/neural_network_service.py

# Start the dashboard
python dashboard.py

# Or use the integrated run_trader.py script to run everything together (recommended)
python run_trader.py
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

### Social Risk Adjustment
```json
{
    "symbol": "BTCUSDC",
    "sentiment_score": 0.75,
    "sentiment_type": "BULLISH",
    "position_size_adj": 0.15,
    "stop_loss_adj": -0.1,
    "take_profit_adj": 0.2,
    "correlation_threshold_adj": 0.1,
    "volatility_adj": 0.08,
    "timestamp": "2024-11-24T21:33:45.029Z",
    "confidence": 0.85
}
```

### Feature Importance Analysis
```json
{
    "timestamp": "2024-11-24T14:30:00.000Z",
    "analysis_type": "feature_importance",
    "model_type": "RandomForest",
    "feature_count": 24,
    "top_features_permutation": {
        "social_sentiment": 0.145,
        "rsi": 0.132,
        "social_volume": 0.118,
        "price_change_5m": 0.097,
        "bb_position": 0.089,
        "macd": 0.082,
        "social_engagement": 0.076,
        "stoch_k": 0.067,
        "williams_r": 0.058,
        "trend_strength": 0.054
    },
    "top_categories": {
        "social": 0.325,
        "momentum": 0.257,
        "price_action": 0.218,
        "volatility": 0.124,
        "trend": 0.076
    },
    "recommendations": {
        "features_to_prioritize": [
            "social_sentiment",
            "rsi",
            "social_volume",
            "price_change_5m",
            "bb_position"
        ],
        "features_to_reconsider": [
            "ichimoku_a",
            "ichimoku_b",
            "vwap",
            "price_change_30m",
            "sma_200"
        ],
        "categories_to_prioritize": [
            "social",
            "momentum"
        ],
        "categories_to_reconsider": [
            "volume"
        ]
    }
}
```

### Market Regime Detection
```json
{
    "regime": "bull",
    "confidence": 0.87,
    "probs": {
        "bull": 0.65,
        "volatile": 0.25,
        "ranging": 0.08,
        "bear": 0.02
    },
    "timestamp": "2024-11-24T15:45:22.134Z",
    "data_end": "2024-11-24T15:45:00.000Z"
}
```

### Neural Network Price Prediction
```json
{
    "symbol": "BTCUSDC",
    "interval": "4h",
    "current_price": 42150.75,
    "predicted_price": 42950.25,
    "change_pct": 1.89,
    "prediction_time": "2024-11-24T21:33:45.029Z",
    "reference_time": "2024-11-24T17:33:45.029Z",
    "confidence": 0.75,
    "model_type": "lstm",
    "status": "success",
    "training_metrics": {
        "val_loss": 0.0023,
        "val_mae": 0.0374,
        "epochs_trained": 87
    },
    "features_used": [
        "close", 
        "volume", 
        "rsi", 
        "macd", 
        "bb_position"
    ]
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
    "selected_strategy": {
        "name": "social_trend_following_v3",
        "type": "trend_following",
        "social_optimization": true,
        "market_regime": "bull",
        "performance_score": 0.87,
        "risk_profile": "medium",
        "social_correlation": 0.72,
        "time_period": "3h",
        "selection_factors": {
            "social_sentiment_score": 0.85,
            "market_regime_score": 0.92,
            "historical_performance": 0.78,
            "risk_score": 0.76
        }
    },
    "explanation": {
        "summary": "Strong bullish signals with technical indicators and social sentiment aligned",
        "technical_factors": "RSI at 55.32 showing momentum, MACD positive and increasing, price breaking above resistance",
        "social_factors": "High social engagement (25,000) with very positive sentiment (0.75), increasing social volume",
        "social_lead_lag": "Social sentiment leads price action by approximately 45 minutes",
        "correlation_strength": "Strong correlation (0.72) between social metrics and price movements",
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
        "market_regime": 0.25,
        "strategy_performance": 0.30,
        "social_correlation": 0.30
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

### Social Strategy Integrator Errors
- Correlation analysis failures: Falls back to conservative defaults
- Strategy generation errors: Retries with alternative parameters
- Social data synchronization issues: Uses time-windowed matching
- Lead/lag detection failures: Defaults to concurrent correlation

### AI Analyzer Errors
- OpenAI API failures: Logged with full error context
- Missing market/social data: Detailed validation errors
- Redis publishing errors: Automatic retry mechanism

### Trade Executor Errors
- Insufficient balance: Logged and trade skipped
- Order placement failures: Full error context with retry attempts
- Position management errors: Detailed error tracking

### Strategy Selection Errors
- Missing factor data: Uses weighted average of available factors
- Strategy switching conflicts: Implements cool-down periods
- Time-based adjustment failures: Reverts to default strategy weights

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

4. Social Correlation Errors
```
ERROR - [SocialStrategyIntegrator] Insufficient data points for correlation analysis
```
Solution: Ensure sufficient historical social data is available for the symbol

5. Strategy Selection Conflicts
```
ERROR - [StrategySelection] Rapid strategy switching detected, enforcing cool-down period
```
Solution: Adjust strategy_switch_threshold in config.json

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
