# Crypto Trader

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

An automated cryptocurrency trading system that uses AI for market analysis and trade execution.

## System Components

### 1. Market Monitor Service
- Monitors real-time market data from Binance
- Calculates technical indicators and price changes across multiple timeframes (1m, 3m, 5m, 15m)
- Publishes market updates to Redis
- Logs to: `logs/market_monitor.log`
- Log format: `timestamp - level - [MarketMonitor] message`

Example market monitor log:
```
2024-11-24 21:33:45,029 - INFO - [MarketMonitor] Market update - BTCUSDC: $42150.75 (24h volume: $1250000.25, RSI: 55.32)
2024-11-24 21:33:45,030 - DEBUG - [MarketMonitor] Publishing market update to Redis: {...}
```

### 2. AI Analyzer Service
- Processes market updates using OpenAI GPT-4
- Generates trading signals based on technical and AI analysis
- Publishes trading signals to Redis
- Logs to: `logs/ai_analyzer.log`
- Log format: `timestamp - level - [AIAnalyzer] message`

Example AI analyzer log:
```
2024-11-24 21:33:45,031 - INFO - [AIAnalyzer] Starting analysis for BTCUSDC
2024-11-24 21:33:45,032 - DEBUG - [AIAnalyzer] Market update data: {...}
2024-11-24 21:33:45,033 - INFO - [AIAnalyzer] AI Analysis for BTCUSDC:
2024-11-24 21:33:45,034 - INFO - [AIAnalyzer] Decision: BUY
2024-11-24 21:33:45,035 - INFO - [AIAnalyzer] Confidence: 0.85
2024-11-24 21:33:45,036 - INFO - [AIAnalyzer] Reasoning: Strong uptrend with positive momentum
```

### 3. Trade Executor Service
- Executes trades based on AI analysis signals
- Manages positions and risk
- Updates portfolio holdings
- Logs to: `logs/trade_executor.log`
- Log format: `timestamp - level - [TradeExecutor] message`

Example trade executor log:
```
2024-11-24 21:33:45,037 - INFO - [TradeExecutor] Processing trading signal for BTCUSDC
2024-11-24 21:33:45,038 - INFO - [TradeExecutor] Opening position:
2024-11-24 21:33:45,039 - INFO - [TradeExecutor] Entry Price: $42150.75
2024-11-24 21:33:45,040 - INFO - [TradeExecutor] Quantity: 0.5
2024-11-24 21:33:45,041 - INFO - [TradeExecutor] Stop Loss: $41308.24
2024-11-24 21:33:45,042 - INFO - [TradeExecutor] Take Profit: $43836.78
```

## Configuration

The system is configured through `config.json` with the following main sections:

### Trading Parameters
```json
{
    "min_volume_usdc": 50000,
    "min_price_change_pct": 0.5,
    "position_size": 0.4,
    "max_positions": 5,
    "stop_loss_pct": 2.0,
    "take_profit_pct": 4.0,
    "min_trade_amount": 40,
    "ai_analysis_interval": 60,
    "ai_confidence_threshold": 0.7
}
```

### OpenAI Configuration
```json
{
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 2000
}
```

## Market Data Structure

Market updates include:
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

## Trading Signals Structure

AI analysis produces:
```json
{
    "decision": "BUY",
    "confidence": 0.85,
    "reasoning": "Strong uptrend with positive momentum",
    "risk_level": "MEDIUM",
    "key_indicators": [
        "RSI trending up",
        "MACD crossover",
        "Strong volume"
    ],
    "timestamp": "2024-11-24T21:33:45.033Z",
    "symbol": "BTCUSDC"
}
```

## Error Handling

### Market Monitor Errors
- Connection issues with Binance: Retries with exponential backoff
- Redis connection failures: Automatic reconnection attempts
- Invalid market data: Logged and skipped
```
2024-11-24 21:33:45,100 - ERROR - [MarketMonitor] Failed to connect to Binance: Network error
2024-11-24 21:33:45,101 - INFO - [MarketMonitor] Retrying connection in 5 seconds...
```

### AI Analyzer Errors
- OpenAI API failures: Logged with full error context
- Missing market data fields: Detailed validation errors
- Redis publishing errors: Automatic retry mechanism
```
2024-11-24 21:33:45,200 - ERROR - [AIAnalyzer] OpenAI API error: Invalid model specified
2024-11-24 21:33:45,201 - ERROR - [AIAnalyzer] Missing required field in market data: price_change_1m
```

### Trade Executor Errors
- Insufficient balance: Logged and trade skipped
- Order placement failures: Full error context with retry attempts
- Position management errors: Detailed error tracking
```
2024-11-24 21:33:45,300 - ERROR - [TradeExecutor] Insufficient balance for trade
2024-11-24 21:33:45,301 - ERROR - [TradeExecutor] Order placement failed: MIN_NOTIONAL
```

## Monitoring and Debugging

### Log Locations
- All logs are stored in the `logs/` directory
- Each service has its own log file with rotation enabled
- Log files are limited to 10MB with 5 backup files

### Common Issues and Solutions

1. Missing Price Changes
```
ERROR - [AIAnalyzer] Error during analysis: 'price_change_1m'
```
Solution: Verify market monitor is calculating all timeframe changes correctly

2. OpenAI API Errors
```
ERROR - [AIAnalyzer] OpenAI API error: Invalid model specified
```
Solution: Check OpenAI configuration and API key validity

3. Trade Execution Failures
```
ERROR - [TradeExecutor] Amount too small to sell: 0.00029202 BNB (minimum: 0.001)
```
Solution: Verify minimum trade amounts in configuration

## Performance Monitoring

The system tracks:
- Portfolio value updates
- Trade success rates
- API response times
- Error frequencies
- System resource usage

Example monitoring log:
```
2024-11-24 21:33:45,400 - INFO - [TradeExecutor] Updated holdings - Total Portfolio Value: $257.65
2024-11-24 21:33:45,401 - INFO - [TradeExecutor] Available USDC: $1.18
2024-11-24 21:33:45,402 - INFO - [TradeExecutor] Trade success rate: 75%
