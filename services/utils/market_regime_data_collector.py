import os
import json
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from redis.asyncio import Redis
import logging as logger

class MarketRegimeDataCollector:
    """
    Collects and preprocesses market data for regime detection and feature importance analysis.
    """
    
    def __init__(self, redis_client: Redis):
        """
        Initialize the market regime data collector.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client
        
        # Set default parameters
        self.lookback_days = 30  # Default lookback period
        self.default_symbols = ['BTCUSDC', 'ETHUSDC']  # Default symbols to collect
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']  # Supported timeframes
        self.default_timeframe = '1h'  # Default timeframe
        
        # Feature sets
        self.features = {
            'price': ['open', 'high', 'low', 'close', 'volume'],
            'returns': ['return_1', 'return_5', 'return_10', 'return_20'],
            'volatility': ['volatility_10', 'volatility_20', 'volatility_50'],
            'momentum': ['rsi_14', 'cci_20', 'stoch_k_14', 'stoch_d_14', 'williams_r_14'],
            'trend': ['sma_20', 'sma_50', 'sma_200', 'ema_20', 'trend_strength'],
            'oscillators': ['macd', 'macd_signal', 'macd_hist', 'adx_14'],
            'volatility_indicators': ['atr_14', 'bbands_width_20']
        }
        
        logger.debug("MarketRegimeDataCollector initialized")
    
    async def get_market_data(self, symbol: str, 
                            timeframe: str = '1h',
                            lookback_days: int = 30) -> pd.DataFrame:
        """
        Get historical market data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g. 'BTCUSDC')
            timeframe: Timeframe for data ('1m', '5m', '15m', '1h', '4h', '1d')
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Check if timeframe is supported
            if timeframe not in self.timeframes:
                logger.warning(f"Unsupported timeframe: {timeframe}, using default: {self.default_timeframe}")
                timeframe = self.default_timeframe
                
            # Get historical data from Redis
            data_key = f"historical_data:{symbol}:{timeframe}"
            raw_data = await self.redis.get(data_key)
            
            # If no data in Redis or stale data, return None
            if not raw_data:
                logger.warning(f"No historical data found for {symbol} at {timeframe} timeframe")
                return None
                
            # Parse data
            data = json.loads(raw_data)
            
            if not data or not isinstance(data, list):
                logger.warning(f"Invalid data format for {symbol} at {timeframe} timeframe")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Expected columns
            expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in expected_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns in historical data: {missing_columns}")
                return None
                
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert numeric columns
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
            
            # Filter by lookback period
            if lookback_days > 0:
                cutoff_date = datetime.now() - timedelta(days=lookback_days)
                df = df[df.index >= cutoff_date]
            
            logger.debug(f"Retrieved {len(df)} rows of {timeframe} data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical market data for {symbol}: {str(e)}")
            return None
    
    async def get_trading_signals(self, 
                                lookback_days: int = 30) -> pd.DataFrame:
        """
        Get historical trading signals with features.
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with trading signals and features
        """
        try:
            # Get raw data from Redis
            signals_data = await self.redis.get('historical_trading_signals')
            
            if not signals_data:
                logger.warning("No historical trading signals found in Redis")
                return None
                
            # Parse JSON data
            signals = json.loads(signals_data)
            
            if not signals or not isinstance(signals, dict):
                logger.warning("Invalid signals data format")
                return None
                
            # Convert to list of dictionaries
            signal_list = []
            for signal_id, signal in signals.items():
                # Extract signal data
                processed_signal = {
                    'signal_id': signal_id,
                    'timestamp': signal.get('timestamp'),
                    'symbol': signal.get('symbol'),
                    'decision': signal.get('decision'),
                    'confidence': signal.get('confidence'),
                }
                
                # Add market_data features if available
                market_data = signal.get('market_data', {})
                for key, value in market_data.items():
                    # Skip non-numeric and non-relevant fields
                    if key not in ['timestamp', 'symbol', 'market_context', 'recent_news']:
                        try:
                            processed_signal[key] = float(value)
                        except (ValueError, TypeError):
                            pass
                
                signal_list.append(processed_signal)
            
            # Convert to DataFrame
            df = pd.DataFrame(signal_list)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by lookback period
            if lookback_days > 0:
                cutoff_date = datetime.now() - timedelta(days=lookback_days)
                df = df[df['timestamp'] >= cutoff_date]
            
            logger.debug(f"Retrieved {len(df)} historical trading signals")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical trading signals: {str(e)}")
            return None
    
    async def get_trading_outcomes(self, lookback_days: int = 30) -> pd.DataFrame:
        """
        Get historical trading outcomes.
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with trading outcomes
        """
        try:
            # Get raw data from Redis
            outcomes_data = await self.redis.get('historical_trading_outcomes')
            
            if not outcomes_data:
                logger.warning("No historical trading outcomes found in Redis")
                return None
                
            # Parse JSON data
            outcomes = json.loads(outcomes_data)
            
            if not outcomes or not isinstance(outcomes, dict):
                logger.warning("Invalid outcomes data format")
                return None
                
            # Convert to list of dictionaries
            outcome_list = []
            for signal_id, outcome in outcomes.items():
                # Extract outcome data
                processed_outcome = {
                    'signal_id': signal_id,
                    'timestamp': outcome.get('timestamp'),
                    'symbol': outcome.get('symbol'),
                    'outcome': outcome.get('outcome'),
                    'profit_pct': outcome.get('profit_pct', 0),
                    'holding_time': outcome.get('holding_time', 0),
                    'exit_price': outcome.get('exit_price', 0),
                    'entry_price': outcome.get('entry_price', 0),
                }
                
                outcome_list.append(processed_outcome)
            
            # Convert to DataFrame
            df = pd.DataFrame(outcome_list)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by lookback period
            if lookback_days > 0:
                cutoff_date = datetime.now() - timedelta(days=lookback_days)
                df = df[df['timestamp'] >= cutoff_date]
            
            logger.debug(f"Retrieved {len(df)} historical trading outcomes")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical trading outcomes: {str(e)}")
            return None
    
    async def get_historical_regime_data(self, lookback_days: int = 30) -> pd.DataFrame:
        """
        Get historical market regime data.
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with market regime data
        """
        try:
            # Get raw data from Redis
            regime_data = await self.redis.get('market_regime_history')
            
            if not regime_data:
                logger.warning("No historical market regime data found in Redis")
                return None
                
            # Parse JSON data
            regimes = json.loads(regime_data)
            
            if not regimes or not isinstance(regimes, list):
                logger.warning("Invalid market regime data format")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(regimes)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by lookback period
            if lookback_days > 0:
                cutoff_date = datetime.now() - timedelta(days=lookback_days)
                df = df[df['timestamp'] >= cutoff_date]
            
            logger.debug(f"Retrieved {len(df)} historical market regime records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical market regime data: {str(e)}")
            return None
    
    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for a price DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        try:
            if df is None or df.empty:
                logger.warning("Cannot calculate technical features for empty DataFrame")
                return None
                
            # Create a copy to avoid modifying original
            result = df.copy()
            
            # Returns
            result['return_1'] = result['close'].pct_change(1)
            result['return_5'] = result['close'].pct_change(5)
            result['return_10'] = result['close'].pct_change(10)
            result['return_20'] = result['close'].pct_change(20)
            
            # Volatility (standard deviation of returns)
            result['volatility_10'] = result['return_1'].rolling(10).std()
            result['volatility_20'] = result['return_1'].rolling(20).std()
            result['volatility_50'] = result['return_1'].rolling(50).std()
            
            # Moving Averages
            result['sma_20'] = result['close'].rolling(20).mean()
            result['sma_50'] = result['close'].rolling(50).mean()
            result['sma_200'] = result['close'].rolling(200).mean()
            result['ema_20'] = result['close'].ewm(span=20, adjust=False).mean()
            
            # RSI
            delta = result['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
            result['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Stochastic Oscillator
            low_14 = result['low'].rolling(14).min()
            high_14 = result['high'].rolling(14).max()
            result['stoch_k_14'] = 100 * ((result['close'] - low_14) / (high_14 - low_14).replace(0, np.finfo(float).eps))
            result['stoch_d_14'] = result['stoch_k_14'].rolling(3).mean()
            
            # Williams %R
            result['williams_r_14'] = -100 * ((high_14 - result['close']) / (high_14 - low_14).replace(0, np.finfo(float).eps))
            
            # MACD
            ema_12 = result['close'].ewm(span=12, adjust=False).mean()
            ema_26 = result['close'].ewm(span=26, adjust=False).mean()
            result['macd'] = ema_12 - ema_26
            result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
            result['macd_hist'] = result['macd'] - result['macd_signal']
            
            # Bollinger Bands
            sma = result['close'].rolling(window=20).mean()
            std = result['close'].rolling(window=20).std()
            result['bbands_upper'] = sma + (std * 2)
            result['bbands_lower'] = sma - (std * 2)
            result['bbands_width_20'] = (result['bbands_upper'] - result['bbands_lower']) / sma
            
            # Average True Range (ATR)
            tr1 = result['high'] - result['low']
            tr2 = abs(result['high'] - result['close'].shift())
            tr3 = abs(result['low'] - result['close'].shift())
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            result['atr_14'] = tr.rolling(14).mean()
            
            # Commodity Channel Index (CCI)
            typical_price = (result['high'] + result['low'] + result['close']) / 3
            sma_tp = typical_price.rolling(window=20).mean()
            mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
            result['cci_20'] = (typical_price - sma_tp) / (0.015 * mad.replace(0, np.finfo(float).eps))
            
            # Average Directional Index (ADX)
            plus_dm = result['high'].diff()
            minus_dm = result['low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            minus_dm = abs(minus_dm)
            
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            
            atr = tr.rolling(14).mean()
            plus_di = 100 * (plus_dm.rolling(14).mean() / atr.replace(0, np.finfo(float).eps))
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr.replace(0, np.finfo(float).eps))
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.finfo(float).eps)
            result['adx_14'] = dx.rolling(14).mean()
            
            # Trend Strength (slope of linear regression)
            result['trend_strength'] = result['close'].rolling(window=20).apply(
                lambda x: np.abs(np.polyfit(np.arange(len(x)), x, 1)[0]), 
                raw=True
            )
            
            # Drop NaN values
            result.dropna(inplace=True)
            
            logger.debug(f"Calculated technical features for {len(result)} rows")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating technical features: {str(e)}")
            return df
    
    async def prepare_regime_detection_dataset(self, 
                                            symbol: str = 'BTCUSDC',
                                            timeframe: str = '1h',
                                            lookback_days: int = 30,
                                            include_signals: bool = True,
                                            include_outcomes: bool = True) -> Dict:
        """
        Prepare a comprehensive dataset for regime detection.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for price data
            lookback_days: Number of days to look back
            include_signals: Whether to include trading signals
            include_outcomes: Whether to include trading outcomes
            
        Returns:
            Dictionary with price_data, signals, outcomes, and regimes
        """
        try:
            result = {}
            
            # Get price data
            df_prices = await self.get_market_data(symbol, timeframe, lookback_days)
            
            if df_prices is not None:
                # Calculate technical features
                df_prices = self.calculate_technical_features(df_prices)
                result['price_data'] = df_prices
            else:
                logger.warning(f"No price data available for {symbol} at {timeframe} timeframe")
            
            # Get trading signals if requested
            if include_signals:
                df_signals = await self.get_trading_signals(lookback_days)
                
                if df_signals is not None:
                    # Filter signals for this symbol
                    df_signals = df_signals[df_signals['symbol'] == symbol]
                    result['signals'] = df_signals
                else:
                    logger.warning(f"No trading signals available for {symbol}")
            
            # Get trading outcomes if requested
            if include_outcomes:
                df_outcomes = await self.get_trading_outcomes(lookback_days)
                
                if df_outcomes is not None:
                    # Filter outcomes for this symbol
                    df_outcomes = df_outcomes[df_outcomes['symbol'] == symbol]
                    result['outcomes'] = df_outcomes
                else:
                    logger.warning(f"No trading outcomes available for {symbol}")
            
            # Get regime history
            df_regimes = await self.get_historical_regime_data(lookback_days)
            if df_regimes is not None:
                result['regimes'] = df_regimes
            else:
                logger.warning("No market regime history available")
            
            return result
            
        except Exception as e:
            logger.error(f"Error preparing regime detection dataset: {str(e)}")
            return {}
    
    async def store_market_regime(self, regime_data: Dict) -> bool:
        """
        Store market regime detection result in Redis.
        
        Args:
            regime_data: Dictionary with regime detection results
            
        Returns:
            True if successful
        """
        try:
            if not regime_data:
                logger.warning("No regime data to store")
                return False
                
            # Add timestamp if not present
            if 'timestamp' not in regime_data:
                regime_data['timestamp'] = datetime.now().isoformat()
                
            # Store in Redis
            await self.redis.set(
                f"market_regime:{regime_data.get('timestamp', datetime.now().isoformat())}",
                json.dumps(regime_data)
            )
            
            # Add to regime history list
            current_history = await self.redis.get('market_regime_history')
            
            if current_history:
                history = json.loads(current_history)
                if not isinstance(history, list):
                    history = []
            else:
                history = []
                
            # Add new regime to history
            history.append(regime_data)
            
            # Keep only the last 1000 regime data points
            if len(history) > 1000:
                history = history[-1000:]
                
            # Store updated history
            await self.redis.set('market_regime_history', json.dumps(history))
            
            logger.info(f"Stored market regime data for {regime_data.get('timestamp')}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing market regime data: {str(e)}")
            return False