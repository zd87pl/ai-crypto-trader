# Backtesting Framework

This directory contains the backtesting framework for the AI Crypto Trader system. The framework allows you to:

1. Fetch and manage historical market data from Binance
2. Integrate historical social metrics data from LunarCrush
3. Backtest the current AI trading strategy with various parameters
4. Analyze and visualize backtest results

## Components

The framework consists of several key components:

- **HistoricalDataManager**: Fetches, stores, and manages historical market and social data
- **SocialDataProvider**: Manages social metrics data for backtesting
- **StrategyTester**: Tests trading strategies against historical data
- **ResultAnalyzer**: Analyzes and visualizes backtest results
- **BacktestEngine**: Coordinates the backtesting process

## Directory Structure

```
backtesting/
├── __init__.py              # Package initialization
├── README.md                # This documentation file
├── backtest_engine.py       # Main orchestration engine
├── data_manager.py          # Historical data management
├── result_analyzer.py       # Results analysis and visualization
├── social_data_provider.py  # Social metrics integration
├── strategy_tester.py       # Strategy testing logic
├── data/                    # Storage for historical data
│   ├── market/              # Market data for each symbol
│   └── social/              # Social metrics data for each symbol
├── results/                 # Storage for backtest results
└── plots/                   # Storage for generated plots and visualizations
```

## Command-line Interface

The framework provides a command-line interface through `run_backtest.py`. Here are the available commands:

### Fetch Historical Data

```bash
python run_backtest.py fetch --symbols BTCUSDC ETHUSDC --intervals 1h 4h --days 60
```

This command fetches 60 days of historical market and social data for BTC and ETH in both 1-hour and 4-hour intervals.

Options:
- `--symbols`: One or more trading pairs (required)
- `--intervals`: One or more timeframe intervals (default: 1h)
- `--days`: Number of days to fetch (default: 30)
- `--no-social`: Skip fetching social data

### Run Backtests

```bash
python run_backtest.py backtest --symbols BTCUSDC --intervals 1h 4h --days 30 --balance 10000
```

This command runs backtests for BTC with 1-hour and 4-hour intervals over the past 30 days, starting with a $10,000 balance.

Options:
- `--symbols`: One or more trading pairs to backtest (required)
- `--intervals`: One or more timeframe intervals (default: 1h)
- `--days`: Number of days to backtest (default: 30)
- `--balance`: Initial balance for backtest (default: 10000.0)
- `--start-date`: Start date in YYYY-MM-DD format (overrides --days)
- `--end-date`: End date in YYYY-MM-DD format (defaults to today)

### List Available Data

```bash
python run_backtest.py list --symbols BTCUSDC --intervals 1h
```

This command lists available historical data, optionally filtered by symbols and intervals.

Options:
- `--symbols`: Filter by one or more symbols
- `--intervals`: Filter by one or more intervals

### Analyze Results

```bash
python run_backtest.py analyze --metric sharpe_ratio
```

This command analyzes all available backtest results, comparing them by Sharpe ratio.

Options:
- `--results`: Specific result file(s) to analyze
- `--symbols`: Filter results by symbol(s)
- `--intervals`: Filter results by interval(s)
- `--metric`: Metric to compare (default: return_pct)

## Advanced Usage

### Implementing Custom Strategies

To implement and test a custom strategy, you would need to:

1. Create a new strategy class in `backtesting/strategies/`
2. Implement the required interface methods
3. Register your strategy with the `StrategyTester` class
4. Run backtests using your custom strategy

### Integration with AI Models

The backtesting framework integrates with the existing AI Trader:

1. Historical market data is processed using the same technical indicators
2. Historical social metrics are included when available
3. The AI trader's decision logic is applied as in live trading
4. Results are stored for analysis and optimization

## Visualization Examples

The backtesting framework generates visualizations for:

1. Equity curves showing account growth over time
2. Drawdown analysis
3. Trade analysis showing win/loss ratios, trade durations, etc.
4. Performance comparisons across different parameters and time periods

## Configuration

The framework uses the same configuration file as the main trading system (`config.json`), with the following key configuration options:

- Connection settings for data sources
- Technical indicator parameters
- Trading strategy parameters
- Risk management settings
- Performance metrics thresholds