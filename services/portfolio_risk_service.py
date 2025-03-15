import os
import json
import redis
import socket
import asyncio
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import logging as logger
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Any, Optional, Tuple
from binance.client import Client

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure rotating file handler
rotating_handler = RotatingFileHandler(
    'logs/portfolio_risk.log',
    maxBytes=10*1024*1024,  # 10MB per file
    backupCount=5,  # Keep 5 backup files
    encoding='utf-8'
)

# Configure logging with rotation
logger.basicConfig(
    level=logger.DEBUG,
    format='%(asctime)s - %(levelname)s - [PortfolioRisk] %(message)s',
    handlers=[
        rotating_handler,
        logger.StreamHandler()
    ]
)

class PortfolioRiskService:
    """
    Service for managing portfolio-wide risk, implementing:
    - RISK-01: Portfolio-wide risk management
    - RISK-08: Portfolio-wide Value at Risk (VaR) calculations
    - RISK-09: Adaptive stop-losses based on market volatility
    """
    
    def __init__(self):
        """Initialize the portfolio risk management service"""
        logger.debug("Initializing Portfolio Risk Service...")
        
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        logger.debug(f"Loaded configuration")
        
        # Initialize Binance client for historical data
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        if not api_key or not api_secret:
            raise ValueError("Binance API credentials not set")
            
        self.client = Client(api_key, api_secret)
        logger.info("Initialized Binance client")
        
        # Redis configuration
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        logger.debug(f"Redis configuration - Host: {self.redis_host}, Port: {self.redis_port}")
        
        # Redis connection (will be initialized later)
        self.redis = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            decode_responses=True
        )
        self.pubsub = None
        
        # Service state
        self.running = True
        self.service_port = int(os.getenv('PORTFOLIO_RISK_PORT', 8008))
        
        # Risk management parameters
        self.risk_config = self.config.get('risk_management', {})
        if not self.risk_config:
            # Set default risk parameters if not found in config
            self.risk_config = {
                'max_portfolio_var': 0.05,  # Maximum Value at Risk (5%)
                'confidence_level': 0.95,  # 95% confidence for VaR
                'var_lookback_days': 30,  # Days to look back for VaR calculation
                'max_portfolio_allocation': 0.25,  # Maximum allocation to a single asset
                'correlation_threshold': 0.7,  # Correlation threshold for diversification
                'min_volatility_factor': 0.5,  # Minimum volatility factor for stop-loss
                'max_volatility_factor': 2.0,  # Maximum volatility factor for stop-loss
                'volatility_lookback_days': 14,  # Days to look back for volatility
                'max_drawdown_limit': 0.15,  # Maximum drawdown limit
                'position_sizing_method': 'equal_risk'  # Options: equal_risk, kelly, fixed
            }
        
        # Portfolio state
        self.portfolio = {}
        self.position_limits = {}
        self.historical_data = {}
        self.asset_correlations = {}
        self.var_estimates = {}
        self.volatility_estimates = {}
        self.adjusted_stop_losses = {}
        
        logger.debug("Portfolio Risk Service initialization complete")
        
    async def connect_redis(self, max_retries=5, retry_delay=5):
        """Connect to Redis with retries"""
        retries = 0
        
        while retries < max_retries and self.running:
            try:
                # Initialize async Redis connection
                self.redis = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    decode_responses=True
                )
                
                # Test connection
                self.redis.ping()
                logger.info(f"Successfully connected to Redis at {self.redis_host}:{self.redis_port}")
                return True
                
            except Exception as e:
                retries += 1
                logger.error(f"Failed to connect to Redis (attempt {retries}/{max_retries}): {str(e)}")
                
                if retries < max_retries:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("Max retries reached. Could not connect to Redis.")
                    return False
                    
    async def setup_pubsub(self):
        """Set up Redis pubsub for various risk-related channels"""
        try:
            self.pubsub = self.redis.pubsub()
            
            # Subscribe to relevant channels
            self.pubsub.subscribe('trading_signals', 'market_updates', 'strategy_update')
            logger.info("Successfully subscribed to Redis channels")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up pubsub: {str(e)}")
            return False
            
    async def fetch_holdings(self) -> Dict:
        """Fetch current portfolio holdings from Redis"""
        try:
            holdings_json = self.redis.get('holdings')
            if holdings_json:
                holdings = json.loads(holdings_json)
                logger.debug(f"Fetched holdings: {len(holdings['assets'])} assets, total value: ${holdings['total_value']:.2f}")
                return holdings
            else:
                logger.warning("No holdings data found in Redis")
                return {'assets': {}, 'total_value': 0, 'available_usdc': 0}
                
        except Exception as e:
            logger.error(f"Error fetching holdings: {str(e)}")
            return {'assets': {}, 'total_value': 0, 'available_usdc': 0}
            
    async def fetch_active_trades(self) -> Dict:
        """Fetch current active trades from Redis"""
        try:
            trades_json = self.redis.get('active_trades')
            if trades_json:
                trades = json.loads(trades_json)
                logger.debug(f"Fetched active trades: {len(trades)} positions")
                return trades
            else:
                logger.debug("No active trades found in Redis")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching active trades: {str(e)}")
            return {}
            
    def fetch_historical_prices(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Fetch historical price data for a symbol"""
        try:
            logger.debug(f"Fetching {days} days of historical data for {symbol}")
            
            # Get klines (candlestick data)
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_1DAY,
                limit=days
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close'] = pd.to_numeric(df['close'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            # Calculate daily returns
            df['returns'] = df['close'].pct_change()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical prices for {symbol}: {str(e)}")
            return pd.DataFrame()
            
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95, value: float = 1.0) -> float:
        """
        Calculate Value at Risk (VaR) using historical method
        
        Parameters:
        - returns: Series of historical returns
        - confidence_level: Confidence level for VaR (e.g., 0.95 for 95%)
        - value: Current value of the position
        
        Returns:
        - VaR amount (absolute dollar value that could be lost)
        """
        try:
            # Remove NaN values
            returns = returns.dropna()
            
            if len(returns) < 2:
                logger.warning("Not enough data points to calculate VaR")
                return 0.0
                
            # Calculate percentile
            var_percentile = np.percentile(returns, 100 * (1 - confidence_level))
            
            # Calculate VaR
            var_amount = abs(var_percentile * value)
            
            return var_amount
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return 0.0
            
    def calculate_conditional_var(self, returns: pd.Series, confidence_level: float = 0.95, value: float = 1.0) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
        This represents the expected loss given that the loss exceeds VaR
        
        Parameters:
        - returns: Series of historical returns
        - confidence_level: Confidence level for VaR (e.g., 0.95 for 95%)
        - value: Current value of the position
        
        Returns:
        - CVaR amount (absolute dollar value that could be lost in tail scenarios)
        """
        try:
            # Remove NaN values
            returns = returns.dropna()
            
            if len(returns) < 2:
                logger.warning("Not enough data points to calculate CVaR")
                return 0.0
                
            # Calculate percentile for VaR
            var_percentile = np.percentile(returns, 100 * (1 - confidence_level))
            
            # Calculate CVaR (expected return in the worst scenarios)
            cvar_returns = returns[returns <= var_percentile]
            cvar_percentile = cvar_returns.mean()
            
            # Calculate CVaR amount
            cvar_amount = abs(cvar_percentile * value)
            
            return cvar_amount
            
        except Exception as e:
            logger.error(f"Error calculating CVaR: {str(e)}")
            return 0.0
            
    def calculate_asset_correlation(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlation matrix between assets based on historical returns
        
        Parameters:
        - symbols: List of trading symbols
        
        Returns:
        - Dictionary of correlations between assets
        """
        try:
            # Create a dictionary to store return series for each symbol
            returns_dict = {}
            
            # Get historical data for each symbol
            for symbol in symbols:
                if symbol in self.historical_data:
                    df = self.historical_data[symbol]
                    returns_dict[symbol] = df['returns']
                    
            # Create a DataFrame with all return series
            returns_df = pd.DataFrame(returns_dict)
            
            # Calculate correlation matrix
            corr_matrix = returns_df.corr()
            
            # Convert to dictionary format
            correlations = {}
            for symbol1 in symbols:
                correlations[symbol1] = {}
                for symbol2 in symbols:
                    if symbol1 in corr_matrix.index and symbol2 in corr_matrix.columns:
                        correlations[symbol1][symbol2] = corr_matrix.loc[symbol1, symbol2]
                    else:
                        correlations[symbol1][symbol2] = 0.0
                        
            return correlations
            
        except Exception as e:
            logger.error(f"Error calculating asset correlations: {str(e)}")
            return {symbol: {s: 0.0 for s in symbols} for symbol in symbols}
            
    def calculate_portfolio_var(self, holdings: Dict, var_estimates: Dict) -> float:
        """
        Calculate portfolio-wide Value at Risk
        
        Parameters:
        - holdings: Dictionary of current holdings
        - var_estimates: VaR estimates for individual assets
        
        Returns:
        - Portfolio-wide VaR amount
        """
        try:
            # Extract list of assets that have both holdings and VaR estimates
            assets = [
                asset for asset in holdings['assets']
                if asset in var_estimates and asset != 'USDC'
            ]
            
            if not assets:
                logger.debug("No assets with VaR estimates in portfolio")
                return 0.0
                
            # Initialize values for calculation
            weights = []
            vars = []
            total_value = 0.0
            
            # Calculate total value of assets with VaR estimates
            for asset in assets:
                asset_value = holdings['assets'][asset]['value_usdc']
                total_value += asset_value
                
            # If total value is zero, return 0
            if total_value == 0:
                return 0.0
                
            # Calculate weights and collect VaR values
            for asset in assets:
                asset_value = holdings['assets'][asset]['value_usdc']
                weight = asset_value / total_value
                weights.append(weight)
                vars.append(var_estimates[asset])
                
            # Get correlation matrix for these assets
            corr_matrix = np.array([
                [self.asset_correlations.get(a1, {}).get(a2, 0.0) for a2 in assets]
                for a1 in assets
            ])
            
            # Ensure the correlation matrix is valid
            if not np.all(np.linalg.eigvals(corr_matrix) > 0):
                # If correlation matrix is not positive definite, use identity matrix
                logger.warning("Correlation matrix is not positive definite, using identity matrix")
                corr_matrix = np.eye(len(assets))
                
            # Calculate portfolio VaR using the diversification formula
            weights = np.array(weights)
            vars = np.array(vars)
            
            # Portfolio VaR calculation
            var_matrix = np.outer(vars, vars) * corr_matrix
            portfolio_var = np.sqrt(weights @ var_matrix @ weights)
            
            # Scale by total portfolio value
            portfolio_var_amount = portfolio_var * holdings['total_value']
            
            return portfolio_var_amount
            
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {str(e)}")
            return sum(var_estimates.get(asset, 0) for asset in holdings['assets'] if asset != 'USDC')
            
    def calculate_optimal_position_sizes(self, holdings: Dict, var_estimates: Dict) -> Dict[str, float]:
        """
        Calculate optimal position sizes to manage portfolio risk
        
        Parameters:
        - holdings: Dictionary of current holdings
        - var_estimates: VaR estimates for individual assets
        
        Returns:
        - Dictionary of recommended position sizes as fraction of portfolio
        """
        try:
            # Get risk management parameters
            position_sizing_method = self.risk_config.get('position_sizing_method', 'equal_risk')
            max_allocation = self.risk_config.get('max_portfolio_allocation', 0.25)
            
            # Extract list of assets (excluding USDC)
            assets = [asset for asset in holdings['assets'] if asset != 'USDC']
            
            if not assets:
                logger.debug("No non-USDC assets in portfolio")
                return {}
                
            # Calculate total portfolio value
            portfolio_value = holdings['total_value']
            
            # Initialize position sizes
            position_sizes = {}
            
            # Different position sizing methods
            if position_sizing_method == 'equal_risk':
                # Equal risk contribution method
                total_var = sum(var_estimates.get(asset, 0) for asset in assets)
                
                if total_var > 0:
                    for asset in assets:
                        var = var_estimates.get(asset, 0)
                        if var > 0:
                            # Inverse of VaR (higher VaR = lower allocation)
                            position_sizes[asset] = (1 / var) / sum(1 / var_estimates.get(a, float('inf')) for a in assets)
                        else:
                            position_sizes[asset] = 0
                else:
                    # Default to equal weighting if no VaR data
                    for asset in assets:
                        position_sizes[asset] = 1.0 / len(assets)
                        
            elif position_sizing_method == 'kelly':
                # Kelly Criterion (simplified version)
                for asset in assets:
                    if asset in self.historical_data:
                        returns = self.historical_data[asset]['returns'].dropna()
                        if len(returns) > 0:
                            win_rate = len(returns[returns > 0]) / len(returns)
                            avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
                            avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0
                            
                            if avg_loss > 0:
                                kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)
                                # Fractional Kelly (using half of full Kelly)
                                position_sizes[asset] = max(0, kelly * 0.5)
                            else:
                                position_sizes[asset] = 0
                        else:
                            position_sizes[asset] = 0
                    else:
                        position_sizes[asset] = 0
                        
            else:  # Default to fixed allocation
                # Equal allocation to all assets
                for asset in assets:
                    position_sizes[asset] = 1.0 / len(assets)
                    
            # Apply maximum allocation limit
            for asset in position_sizes:
                position_sizes[asset] = min(position_sizes[asset], max_allocation)
                
            # Normalize position sizes to sum to 1.0
            total_size = sum(position_sizes.values())
            if total_size > 0:
                for asset in position_sizes:
                    position_sizes[asset] /= total_size
                    
            return position_sizes
            
        except Exception as e:
            logger.error(f"Error calculating optimal position sizes: {str(e)}")
            return {asset: 1.0 / len(holdings['assets']) for asset in holdings['assets'] if asset != 'USDC'}
            
    def calculate_adaptive_stop_loss(self, symbol: str, entry_price: float) -> Tuple[float, Dict]:
        """
        Calculate adaptive stop-loss based on historical volatility
        
        Parameters:
        - symbol: Trading symbol
        - entry_price: Entry price of the position
        
        Returns:
        - Adaptive stop-loss price
        - Dictionary with calculation details
        """
        try:
            # Get volatility data
            if symbol not in self.historical_data:
                logger.warning(f"No historical data for {symbol}, using default stop-loss")
                stop_pct = self.config['trading_params']['stop_loss_pct']
                return entry_price * (1 - stop_pct / 100), {'method': 'default', 'volatility': 0, 'factor': 0}
                
            df = self.historical_data[symbol]
            
            # Calculate historical volatility (standard deviation of returns)
            volatility = df['returns'].std() * np.sqrt(252)  # Annualized volatility
            
            # Get volatility scaling parameters
            min_factor = self.risk_config.get('min_volatility_factor', 0.5)
            max_factor = self.risk_config.get('max_volatility_factor', 2.0)
            base_stop_pct = self.config['trading_params']['stop_loss_pct']
            
            # Calculate adaptive factor based on volatility
            # Higher volatility = wider stop-loss
            vol_percentile = min(max(0, volatility / 0.5), 1)  # Normalize volatility (0.5 = 50% annual volatility)
            factor = min_factor + (max_factor - min_factor) * vol_percentile
            
            # Calculate adaptive stop-loss percentage
            adaptive_stop_pct = base_stop_pct * factor
            
            # Calculate stop-loss price
            stop_loss_price = entry_price * (1 - adaptive_stop_pct / 100)
            
            # Save calculation details
            details = {
                'method': 'adaptive',
                'volatility': volatility,
                'volatility_percentile': vol_percentile,
                'factor': factor,
                'base_stop_pct': base_stop_pct,
                'adaptive_stop_pct': adaptive_stop_pct
            }
            
            logger.debug(f"Calculated adaptive stop-loss for {symbol}: {adaptive_stop_pct:.2f}% (factor: {factor:.2f})")
            
            return stop_loss_price, details
            
        except Exception as e:
            logger.error(f"Error calculating adaptive stop-loss for {symbol}: {str(e)}")
            stop_pct = self.config['trading_params']['stop_loss_pct']
            return entry_price * (1 - stop_pct / 100), {'method': 'default', 'error': str(e)}
            
    async def update_portfolio_risk_metrics(self):
        """
        Update portfolio risk metrics including VaR and position limits
        This is the main function implementing RISK-01 and RISK-08
        """
        try:
            # Fetch current holdings and active trades
            holdings = await self.fetch_holdings()
            active_trades = await self.fetch_active_trades()
            
            # Exit if no holdings
            if not holdings or not holdings['assets']:
                logger.debug("No holdings to analyze")
                return
                
            # Get list of assets in portfolio
            assets = list(holdings['assets'].keys())
            
            # Update historical data for all assets
            for asset in assets:
                if asset != 'USDC':
                    symbol = f"{asset}USDC"
                    self.historical_data[asset] = self.fetch_historical_prices(
                        symbol, 
                        days=self.risk_config.get('var_lookback_days', 30)
                    )
                    
            # Calculate asset correlations
            self.asset_correlations = self.calculate_asset_correlation(assets)
            
            # Calculate VaR for each asset
            for asset in assets:
                if asset != 'USDC' and asset in self.historical_data:
                    asset_value = holdings['assets'][asset]['value_usdc']
                    returns = self.historical_data[asset]['returns']
                    
                    # Calculate VaR
                    var = self.calculate_var(
                        returns,
                        confidence_level=self.risk_config.get('confidence_level', 0.95),
                        value=asset_value
                    )
                    
                    # Calculate CVaR (Conditional VaR / Expected Shortfall)
                    cvar = self.calculate_conditional_var(
                        returns,
                        confidence_level=self.risk_config.get('confidence_level', 0.95),
                        value=asset_value
                    )
                    
                    # Store results
                    self.var_estimates[asset] = {
                        'var': var,
                        'cvar': cvar,
                        'var_pct': var / asset_value if asset_value > 0 else 0,
                        'cvar_pct': cvar / asset_value if asset_value > 0 else 0,
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Calculate portfolio-wide VaR
            portfolio_var = self.calculate_portfolio_var(holdings, {
                asset: self.var_estimates[asset]['var']
                for asset in self.var_estimates
            })
            
            # Calculate optimal position sizes
            optimal_positions = self.calculate_optimal_position_sizes(holdings, {
                asset: self.var_estimates[asset]['var']
                for asset in self.var_estimates
            })
            
            # Store results in Redis
            portfolio_risk = {
                'portfolio_var': portfolio_var,
                'portfolio_var_pct': portfolio_var / holdings['total_value'] if holdings['total_value'] > 0 else 0,
                'asset_var': self.var_estimates,
                'correlations': self.asset_correlations,
                'optimal_positions': optimal_positions,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to Redis
            self.redis.set('portfolio_risk', json.dumps(portfolio_risk))
            
            # Log summary
            logger.info(f"Updated portfolio risk metrics:")
            logger.info(f"Portfolio VaR: ${portfolio_var:.2f} ({portfolio_var / holdings['total_value'] * 100:.2f}%)")
            logger.info(f"Assets analyzed: {len(self.var_estimates)}")
            
            # Check for portfolio risk limit violations
            max_var_pct = self.risk_config.get('max_portfolio_var', 0.05)
            if portfolio_var / holdings['total_value'] > max_var_pct:
                logger.warning(f"Portfolio VaR exceeds limit: {portfolio_var / holdings['total_value'] * 100:.2f}% > {max_var_pct * 100:.2f}%")
                
                # Publish risk alert
                risk_alert = {
                    'type': 'portfolio_var_exceeded',
                    'var_pct': portfolio_var / holdings['total_value'],
                    'limit_pct': max_var_pct,
                    'timestamp': datetime.now().isoformat()
                }
                self.redis.publish('risk_alerts', json.dumps(risk_alert))
                
        except Exception as e:
            logger.error(f"Error updating portfolio risk metrics: {str(e)}", exc_info=True)
            
    async def update_adaptive_stop_losses(self):
        """
        Update adaptive stop-losses for active trades
        This implements RISK-09: Adaptive stop-losses based on market volatility
        """
        try:
            # Fetch active trades
            active_trades = await self.fetch_active_trades()
            
            # Exit if no active trades
            if not active_trades:
                logger.debug("No active trades to update stop-losses for")
                return
                
            # Process each active trade
            for symbol, trade in active_trades.items():
                try:
                    # Extract asset name
                    asset = symbol.replace('USDC', '')
                    
                    # Update historical data if needed
                    if asset not in self.historical_data:
                        self.historical_data[asset] = self.fetch_historical_prices(
                            symbol,
                            days=self.risk_config.get('volatility_lookback_days', 14)
                        )
                        
                    # Calculate adaptive stop-loss
                    entry_price = trade['entry_price']
                    stop_loss_price, details = self.calculate_adaptive_stop_loss(asset, entry_price)
                    
                    # Store details
                    self.adjusted_stop_losses[symbol] = {
                        'current_stop_loss': trade['stop_loss_price'],
                        'adaptive_stop_loss': stop_loss_price,
                        'details': details,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Check if stop-loss should be updated
                    current_stop = trade['stop_loss_price']
                    if stop_loss_price > current_stop:
                        logger.info(f"Recommending stop-loss update for {symbol}: ${current_stop:.8f} → ${stop_loss_price:.8f}")
                        
                        # Publish stop-loss adjustment recommendation
                        adjustment = {
                            'symbol': symbol,
                            'current_stop_price': current_stop,
                            'recommended_stop_price': stop_loss_price,
                            'reason': 'adaptive_volatility',
                            'details': details,
                            'timestamp': datetime.now().isoformat()
                        }
                        self.redis.publish('stop_loss_adjustments', json.dumps(adjustment))
                        
                except Exception as e:
                    logger.error(f"Error updating stop-loss for {symbol}: {str(e)}")
                    
            # Save all adaptive stop-losses to Redis
            self.redis.set('adaptive_stop_losses', json.dumps(self.adjusted_stop_losses))
            
        except Exception as e:
            logger.error(f"Error updating adaptive stop-losses: {str(e)}", exc_info=True)
            
    async def analyze_portfolio_diversification(self):
        """
        Analyze portfolio diversification based on correlations
        Part of RISK-01: Portfolio-wide risk management
        """
        try:
            # Fetch current holdings
            holdings = await self.fetch_holdings()
            
            # Exit if no holdings
            if not holdings or not holdings['assets']:
                logger.debug("No holdings to analyze diversification")
                return
                
            # Get assets (excluding USDC)
            assets = [asset for asset in holdings['assets'] if asset != 'USDC']
            
            if len(assets) < 2:
                logger.debug("Less than 2 assets in portfolio, cannot analyze diversification")
                return
                
            # Check correlation matrix
            high_correlations = []
            correlation_threshold = self.risk_config.get('correlation_threshold', 0.7)
            
            for i, asset1 in enumerate(assets):
                for asset2 in assets[i+1:]:
                    if asset1 in self.asset_correlations and asset2 in self.asset_correlations[asset1]:
                        corr = self.asset_correlations[asset1][asset2]
                        if abs(corr) > correlation_threshold:
                            high_correlations.append({
                                'asset1': asset1,
                                'asset2': asset2,
                                'correlation': corr
                            })
                            
            # Calculate diversification score (average of 1-|correlation|)
            all_correlations = []
            for i, asset1 in enumerate(assets):
                for asset2 in assets[i+1:]:
                    if asset1 in self.asset_correlations and asset2 in self.asset_correlations[asset1]:
                        all_correlations.append(abs(self.asset_correlations[asset1][asset2]))
                        
            if all_correlations:
                diversification_score = sum(1 - corr for corr in all_correlations) / len(all_correlations)
            else:
                diversification_score = 1.0  # Assume perfect diversification if no data
                
            # Store diversification analysis in Redis
            diversification_analysis = {
                'diversification_score': diversification_score,
                'high_correlations': high_correlations,
                'correlation_threshold': correlation_threshold,
                'timestamp': datetime.now().isoformat()
            }
            
            self.redis.set('portfolio_diversification', json.dumps(diversification_analysis))
            
            # Log summary
            logger.info(f"Portfolio diversification score: {diversification_score:.2f}")
            logger.info(f"High correlations detected: {len(high_correlations)}")
            
            # Publish alert if diversification is poor
            if diversification_score < 0.5 and high_correlations:
                logger.warning(f"Poor portfolio diversification detected: score={diversification_score:.2f}")
                
                # Publish risk alert
                risk_alert = {
                    'type': 'poor_diversification',
                    'diversification_score': diversification_score,
                    'high_correlations': high_correlations,
                    'timestamp': datetime.now().isoformat()
                }
                self.redis.publish('risk_alerts', json.dumps(risk_alert))
                
        except Exception as e:
            logger.error(f"Error analyzing portfolio diversification: {str(e)}", exc_info=True)
            
    async def process_trading_signals(self):
        """Process and enrich trading signals with risk information"""
        try:
            # Process messages from pubsub if available
            if self.pubsub:
                message = self.pubsub.get_message(ignore_subscribe_messages=True)
                if message and message['type'] == 'message':
                    try:
                        channel = message['channel']
                        data = json.loads(message['data'])
                        
                        # Handle trading signals
                        if channel == 'trading_signals':
                            # Enrich with risk information
                            symbol = data['symbol']
                            asset = symbol.replace('USDC', '')
                            
                            # Add VaR information if available
                            if asset in self.var_estimates:
                                data['risk_info'] = {
                                    'var': self.var_estimates[asset]['var'],
                                    'var_pct': self.var_estimates[asset]['var_pct'],
                                    'cvar': self.var_estimates[asset]['cvar'],
                                    'cvar_pct': self.var_estimates[asset]['cvar_pct']
                                }
                                
                            # Add portfolio risk information
                            holdings = await self.fetch_holdings()
                            if holdings and holdings['total_value'] > 0:
                                portfolio_var = self.calculate_portfolio_var(holdings, {
                                    a: self.var_estimates[a]['var'] for a in self.var_estimates
                                })
                                data['risk_info']['portfolio_var'] = portfolio_var
                                data['risk_info']['portfolio_var_pct'] = portfolio_var / holdings['total_value']
                                
                            # Calculate optimal position size
                            optimal_positions = self.calculate_optimal_position_sizes(holdings, {
                                a: self.var_estimates[a]['var'] for a in self.var_estimates
                            })
                            if asset in optimal_positions:
                                data['risk_info']['optimal_position_pct'] = optimal_positions[asset]
                                
                            # Calculate adaptive stop-loss if it's a BUY signal
                            if data['decision'] == 'BUY':
                                current_price = data.get('market_data', {}).get('current_price', 0)
                                if current_price > 0 and asset in self.historical_data:
                                    stop_price, details = self.calculate_adaptive_stop_loss(asset, current_price)
                                    data['risk_info']['adaptive_stop_loss'] = stop_price
                                    data['risk_info']['adaptive_stop_pct'] = details['adaptive_stop_pct']
                                    
                            # Republish enriched signal
                            self.redis.publish('risk_enriched_signals', json.dumps(data))
                            logger.debug(f"Enriched trading signal for {symbol} with risk information")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse message: {e}")
                    except Exception as e:
                        logger.error(f"Error processing message: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error in process_trading_signals: {str(e)}")
            
    async def run_health_check_server(self):
        """Run a simple TCP server for health checks"""
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('0.0.0.0', self.service_port))
            server.listen(1)
            server.setblocking(False)
            
            logger.info(f"Health check server listening on port {self.service_port}")
            
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in health check server: {str(e)}")
        finally:
            server.close()
            
    async def run(self):
        """Main service loop"""
        try:
            logger.info("Starting Portfolio Risk Service...")
            
            # Connect to Redis
            await self.connect_redis()
            
            # Setup pubsub
            await self.setup_pubsub()
            
            # Start health check server
            health_check_task = asyncio.create_task(self.run_health_check_server())
            
            # Initial update of portfolio risk metrics
            await self.update_portfolio_risk_metrics()
            
            # Main loop
            while self.running:
                try:
                    # Update portfolio risk metrics every 5 minutes
                    await self.update_portfolio_risk_metrics()
                    
                    # Update adaptive stop-losses every minute
                    await self.update_adaptive_stop_losses()
                    
                    # Check portfolio diversification every 15 minutes
                    await self.analyze_portfolio_diversification()
                    
                    # Process trading signals
                    await self.process_trading_signals()
                    
                    # Wait before next cycle
                    await asyncio.sleep(60)  # Run main risk check every minute
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    await asyncio.sleep(30)  # Shorter delay on error
                    
        except Exception as e:
            logger.error(f"Error starting service: {str(e)}")
            
        finally:
            # Cleanup
            self.running = False
            
            # Cancel health check task
            if 'health_check_task' in locals():
                health_check_task.cancel()
                try:
                    await health_check_task
                except asyncio.CancelledError:
                    pass
                    
            logger.info("Portfolio Risk Service stopped")
            
    def stop(self):
        """Stop the service"""
        logger.info("Stopping Portfolio Risk Service...")
        self.running = False
        if self.redis:
            self.redis.close()
            
# Run the service if executed directly
if __name__ == "__main__":
    service = PortfolioRiskService()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        service.stop()
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        service.stop()