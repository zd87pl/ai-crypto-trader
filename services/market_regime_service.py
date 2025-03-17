import os
import json
import asyncio
import logging as logger
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler
from redis.asyncio import Redis
from redis.exceptions import ConnectionError
import pickle
import matplotlib.pyplot as plt
import io
import base64

# Import our custom utility classes
from services.utils.market_regime_detector import MarketRegimeDetector
from services.utils.market_regime_data_collector import MarketRegimeDataCollector

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure rotating file handler
rotating_handler = RotatingFileHandler(
    'logs/market_regime.log',
    maxBytes=10*1024*1024,  # 10MB per file
    backupCount=5,  # Keep 5 backup files
    encoding='utf-8'
)

# Configure logging
logger.basicConfig(
    level=logger.INFO,
    format='%(asctime)s - %(levelname)s - [MarketRegime] %(message)s',
    handlers=[
        rotating_handler,
        logger.StreamHandler()
    ]
)

class MarketRegimeService:
    """
    Service for detecting market regimes and selecting optimal strategies for each regime.
    
    Market regimes include:
    - Bull market (strong uptrend)
    - Bear market (strong downtrend)
    - Sideways/Ranging market (low volatility, no clear trend)
    - Volatile market (high volatility, possibly with rapid trend changes)
    
    This service uses various machine learning approaches for market regime detection:
    1. Clustering-based: KMeans, GMM
    2. Hidden Markov Models
    3. Supervised classification (when labels are available)
    """
    
    def __init__(self):
        logger.debug("Initializing Market Regime Service...")
        
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        logger.debug(f"Loaded configuration")
        
        # Initialize data structures
        self.running = True
        self.current_regime = None
        self.regime_history = []
        self.strategy_performance_history = {}
        self.regime_strategy_mapping = {}
        
        # Redis configuration
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        logger.debug(f"Redis configuration - Host: {self.redis_host}, Port: {self.redis_port}")
        
        # Redis will be initialized in connect_redis
        self.redis = None
        
        # Load configuration parameters
        self.regime_change_threshold = float(os.getenv('REGIME_CHANGE_THRESHOLD', '0.7'))
        self.check_interval = int(os.getenv('REGIME_CHECK_INTERVAL', '3600'))  # Default: check every hour
        
        # Market regime detection parameters
        self.lookback_periods = {
            'short': int(os.getenv('REGIME_SHORT_LOOKBACK', '24')),     # 24 hours
            'medium': int(os.getenv('REGIME_MEDIUM_LOOKBACK', '72')),   # 3 days
            'long': int(os.getenv('REGIME_LONG_LOOKBACK', '168'))       # 7 days
        }
        
        # Traditional regime detection thresholds (still used as fallback)
        self.regime_thresholds = {
            'bull': {
                'trend_strength': float(os.getenv('BULL_TREND_STRENGTH', '0.6')),
                'price_change': float(os.getenv('BULL_PRICE_CHANGE', '5.0')),
                'volatility': float(os.getenv('BULL_VOLATILITY', '0.5'))
            },
            'bear': {
                'trend_strength': float(os.getenv('BEAR_TREND_STRENGTH', '0.6')),
                'price_change': float(os.getenv('BEAR_PRICE_CHANGE', '-5.0')),
                'volatility': float(os.getenv('BEAR_VOLATILITY', '0.5'))
            },
            'volatile': {
                'volatility': float(os.getenv('VOLATILE_THRESHOLD', '1.5'))
            },
            'ranging': {
                'trend_strength': float(os.getenv('RANGING_TREND_STRENGTH', '0.3')),
                'price_range': float(os.getenv('RANGING_PRICE_RANGE', '3.0'))
            }
        }
        
        # Service port
        self.service_port = int(os.getenv('MARKET_REGIME_PORT', 8006))
        logger.debug(f"Service port configured as: {self.service_port}")
        
        # Default market data and settings
        self.market_data_cache = []
        self.primary_symbol = os.getenv('PRIMARY_TRADING_SYMBOL', 'BTCUSDC')
        self.primary_timeframe = os.getenv('PRIMARY_TIMEFRAME', '1h')
        
        # Detection method configuration
        self.detection_method = os.getenv('REGIME_DETECTION_METHOD', 'ml')  # 'traditional', 'ml', or 'hybrid'
        self.ml_method = os.getenv('ML_REGIME_METHOD', 'kmeans')  # 'kmeans', 'gmm', 'hmm', 'supervised'
        
        # Initialize ML components (will be set up after Redis connection)
        self.regime_detector = None
        self.data_collector = None
        self.trained_models = {}
        
        # Model configuration
        self.model_config = {
            'method': self.ml_method,
            'window_size': int(os.getenv('ML_WINDOW_SIZE', '30')),
            'n_regimes': 4,  # Default: bull, bear, ranging, volatile
            'features': [
                'return', 'volatility', 'trend_strength', 'rsi', 
                'macd', 'bb_width', 'adx', 'cci'
            ]
        }
        
        # Model training settings
        self.model_training_interval = int(os.getenv('MODEL_TRAINING_INTERVAL', '86400'))  # Default: once per day
        self.last_training_time = datetime.min
        self.min_training_samples = int(os.getenv('MIN_TRAINING_SAMPLES', '100'))
        
        # Model path for saving/loading
        self.model_dir = os.getenv('MODEL_DIR', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger.info("Market Regime Service initialized with the following parameters:")
        logger.info(f"- Detection Method: {self.detection_method} (ML Method: {self.ml_method})")
        logger.info(f"- Regime Change Threshold: {self.regime_change_threshold}")
        logger.info(f"- Check Interval: {self.check_interval} seconds")
        logger.info(f"- Lookback Periods: {self.lookback_periods}")
        logger.info(f"- Primary Trading Symbol: {self.primary_symbol}")
    
    async def connect_redis(self, max_retries=10, retry_delay=5):
        """Establish Redis connection with retries."""
        retries = 0
        while retries < max_retries and self.running:
            try:
                if self.redis is None:
                    logger.debug(f"Attempting Redis connection (attempt {retries + 1}/{max_retries})")
                    self.redis = Redis(
                        host=self.redis_host,
                        port=self.redis_port,
                        decode_responses=True,
                        socket_connect_timeout=5.0,
                        socket_keepalive=True,
                        health_check_interval=15
                    )
                await self.redis.ping()
                logger.info(f"Successfully connected to Redis at {self.redis_host}:{self.redis_port}")
                return True
            except (ConnectionError, OSError) as e:
                retries += 1
                logger.error(f"Failed to connect to Redis (attempt {retries}/{max_retries}): {str(e)}")
                if self.redis:
                    await self.redis.close()
                    self.redis = None
                if retries < max_retries:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("Max retries reached. Could not connect to Redis.")
                    return False
            except Exception as e:
                logger.error(f"Unexpected error connecting to Redis: {str(e)}")
                if self.redis:
                    await self.redis.close()
                    self.redis = None
                await asyncio.sleep(retry_delay)
                retries += 1
    
    async def initialize_ml_components(self):
        """Initialize ML components for market regime detection."""
        try:
            # Initialize data collector
            self.data_collector = MarketRegimeDataCollector(self.redis)
            
            # Initialize regime detector
            self.regime_detector = MarketRegimeDetector(self.model_config)
            
            # Try to load existing model
            model_path = os.path.join(self.model_dir, f"market_regime_{self.ml_method}.pkl")
            if os.path.exists(model_path):
                if self.regime_detector.load_model(model_path):
                    logger.info(f"Loaded existing {self.ml_method} model from {model_path}")
                else:
                    logger.warning(f"Failed to load model from {model_path}, will train a new model")
            else:
                logger.info(f"No existing model found at {model_path}, will train a new model")
            
            # Train models if needed
            await self.train_regime_models()
            
            logger.info("ML components initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing ML components: {str(e)}")
            return False
            
    async def train_regime_models(self):
        """Train ML models for market regime detection."""
        try:
            # Check if training is needed
            current_time = datetime.now()
            time_since_training = (current_time - self.last_training_time).total_seconds()
            
            # Skip if trained recently and model exists
            if (time_since_training < self.model_training_interval and 
                self.regime_detector and self.regime_detector.is_trained):
                logger.debug(f"Skipping model training, last trained {time_since_training:.1f}s ago")
                return True
                
            # Collect training data
            logger.info(f"Collecting training data for {self.primary_symbol}...")
            dataset = await self.data_collector.prepare_regime_detection_dataset(
                symbol=self.primary_symbol,
                timeframe=self.primary_timeframe,
                lookback_days=self.lookback_periods['long'] // 24  # Convert hours to days
            )
            
            if not dataset or 'price_data' not in dataset:
                logger.warning("No price data available for training")
                return False
                
            price_data = dataset['price_data']
            
            if len(price_data) < self.min_training_samples:
                logger.warning(f"Insufficient data for model training: {len(price_data)} samples")
                return False
                
            # Train the model
            logger.info(f"Training {self.ml_method} model on {len(price_data)} samples...")
            success = self.regime_detector.train(price_data)
            
            if success:
                self.last_training_time = current_time
                
                # Save the model
                model_path = os.path.join(self.model_dir, f"market_regime_{self.ml_method}.pkl")
                if self.regime_detector.save_model(model_path):
                    logger.info(f"Model saved to {model_path}")
                    
                logger.info(f"Successfully trained {self.ml_method} model")
                return True
            else:
                logger.error(f"Failed to train {self.ml_method} model")
                return False
                
        except Exception as e:
            logger.error(f"Error training regime models: {str(e)}")
            return False
            
    async def detect_regime_with_ml(self):
        """Detect market regime using machine learning models."""
        try:
            if not self.regime_detector or not self.regime_detector.is_trained:
                logger.warning("ML regime detector not trained, cannot detect regime with ML")
                return "unknown"
                
            # Get recent market data
            dataset = await self.data_collector.prepare_regime_detection_dataset(
                symbol=self.primary_symbol,
                timeframe=self.primary_timeframe,
                lookback_days=7  # Use recent data for detection
            )
            
            if not dataset or 'price_data' not in dataset:
                logger.warning("No recent price data available for regime detection")
                return "unknown"
                
            price_data = dataset['price_data']
            
            if len(price_data) < self.model_config['window_size']:
                logger.warning(f"Insufficient recent data for regime detection: {len(price_data)} samples")
                return "unknown"
                
            # Detect regime
            regime_result = self.regime_detector.detect_regime(price_data)
            
            # Store regime detection result in Redis
            await self.data_collector.store_market_regime(regime_result)
            
            # Extract regime name and confidence
            regime = regime_result.get('regime', 'unknown')
            confidence = regime_result.get('confidence', 0)
            
            logger.info(f"ML-based regime detection: {regime} (confidence: {confidence:.4f})")
            
            # Create and save visualization of the regime detection
            self._create_regime_visualization(price_data, regime_result)
            
            return regime
            
        except Exception as e:
            logger.error(f"Error in ML regime detection: {str(e)}")
            return "unknown"
            
    def _create_regime_visualization(self, price_data, regime_result):
        """Create and save visualization of the regime detection."""
        try:
            if price_data is None or len(price_data) == 0:
                return
                
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot price chart
            price_data['close'].plot(ax=ax1, color='blue', label='Price')
            ax1.set_title(f"Market Regime Detection: {regime_result.get('regime', 'unknown')}")
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True)
            
            # Plot regime probabilities
            probs = regime_result.get('probs', {})
            if probs:
                regimes = list(probs.keys())
                values = list(probs.values())
                
                # Create probability bars
                ax2.bar(regimes, values, color=['green', 'red', 'blue', 'orange'])
                ax2.set_ylim(0, 1)
                ax2.set_ylabel('Probability')
                ax2.set_title('Regime Probabilities')
                ax2.grid(True)
            
            plt.tight_layout()
            
            # Save visualization to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Encode as base64 and store in Redis
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            data_uri = f"data:image/png;base64,{image_base64}"
            
            # Store visualization in Redis
            asyncio.create_task(self.redis.set('market_regime_visualization', data_uri))
            
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error creating regime visualization: {str(e)}")
            
    async def detect_current_regime_ml(self):
        """
        Detect the current market regime using ML methods with fallback to traditional methods.
        
        Returns:
            Detected market regime: 'bull', 'bear', 'ranging', 'volatile', or 'unknown'
        """
        try:
            if self.detection_method == 'ml':
                # Use only ML-based detection
                return await self.detect_regime_with_ml()
                
            elif self.detection_method == 'traditional':
                # Use only traditional rule-based detection
                return await self.detect_current_regime()
                
            elif self.detection_method == 'hybrid':
                # Try ML first, fall back to traditional if needed
                ml_regime = await self.detect_regime_with_ml()
                
                if ml_regime != 'unknown':
                    return ml_regime
                else:
                    logger.info("Falling back to traditional regime detection")
                    return await self.detect_current_regime()
            
            else:
                logger.error(f"Unsupported detection method: {self.detection_method}")
                return "unknown"
                
        except Exception as e:
            logger.error(f"Error in market regime detection: {str(e)}")
            return "unknown"
    
    async def get_historical_market_data(self, lookback_hours: int = 168) -> List[Dict]:
        """
        Retrieve historical market data from Redis.
        
        Args:
            lookback_hours: Number of hours to look back for historical data
            
        Returns:
            List of market data points
        """
        try:
            # Get data from Redis
            historical_data = await self.redis.get('historical_market_data')
            if historical_data:
                data = json.loads(historical_data)
                
                # Filter for the requested lookback period
                cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
                cutoff_time_str = cutoff_time.isoformat()
                
                filtered_data = [
                    point for point in data 
                    if 'timestamp' in point and point['timestamp'] > cutoff_time_str
                ]
                
                return filtered_data
            
            logger.warning("No historical market data found in Redis")
            return []
            
        except Exception as e:
            logger.error(f"Error fetching historical market data: {str(e)}")
            return []
    
    async def get_available_strategies(self) -> List[Dict]:
        """
        Get list of available trading strategies with their metadata.
        
        Returns:
            List of strategy dictionaries with metadata
        """
        try:
            # Get strategies from Redis
            strategies = []
            strategy_keys = await self.redis.keys('worker_deployment_*')
            
            for key in strategy_keys:
                strategy_data = await self.redis.get(key)
                if strategy_data:
                    strategy = json.loads(strategy_data)
                    strategies.append(strategy)
            
            return strategies
        
        except Exception as e:
            logger.error(f"Error fetching available strategies: {str(e)}")
            return []
    
    async def get_strategy_performance(self, strategy_id: str, 
                                     time_range: str = 'all') -> Dict:
        """
        Get performance metrics for a specific strategy.
        
        Args:
            strategy_id: The ID of the strategy to evaluate
            time_range: Time range for performance data ('all', 'recent', '24h', '7d')
            
        Returns:
            Performance metrics dictionary
        """
        try:
            # Get performance data from Redis
            performance_key = f'strategy_performance_{strategy_id}'
            performance_data = await self.redis.get(performance_key)
            
            if not performance_data:
                logger.warning(f"No performance data found for strategy {strategy_id}")
                return {}
            
            performance = json.loads(performance_data)
            
            # Filter by time range if needed
            if time_range != 'all':
                # Implement time filtering logic
                pass
                
            return performance
        
        except Exception as e:
            logger.error(f"Error fetching strategy performance: {str(e)}")
            return {}
    
    def detect_regime_from_data(self, market_data: List[Dict]) -> str:
        """
        Detect the current market regime based on market data.
        
        Args:
            market_data: List of market data points
            
        Returns:
            Detected market regime: 'bull', 'bear', 'ranging', or 'volatile'
        """
        try:
            if not market_data:
                logger.warning("No market data provided for regime detection")
                return "unknown"
                
            # Convert data to DataFrame for easier analysis
            df = pd.DataFrame(market_data)
            
            # Check if required columns exist
            required_columns = ['timestamp', 'current_price', 'trend', 
                               'trend_strength', 'price_change_24h']
            
            for col in required_columns:
                if col not in df.columns:
                    logger.warning(f"Required column {col} missing from market data")
                    return "unknown"
            
            # Convert timestamp to datetime if it's a string
            if isinstance(df['timestamp'].iloc[0], str):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Calculate volatility (standard deviation of returns)
            if len(df) > 1:
                # Calculate returns
                df['return'] = df['current_price'].pct_change()
                
                # Volatility is the standard deviation of returns
                volatility = df['return'].std() * np.sqrt(24)  # Annualized to 24-hour scale
                
                # Calculate overall price change
                start_price = df['current_price'].iloc[0]
                end_price = df['current_price'].iloc[-1]
                overall_price_change = ((end_price - start_price) / start_price) * 100
                
                # Calculate price range (max - min) / avg
                price_range = (df['current_price'].max() - df['current_price'].min()) / df['current_price'].mean() * 100
                
                # Determine trend strength (average of trend_strength values)
                trend_strength = df['trend_strength'].mean()
                
                # Count trend types
                trend_counts = df['trend'].value_counts(normalize=True)
                uptrend_pct = trend_counts.get('uptrend', 0)
                downtrend_pct = trend_counts.get('downtrend', 0)
                sideways_pct = trend_counts.get('sideways', 0)
                
                # Detect regime based on calculated metrics
                thresholds = self.regime_thresholds
                
                # Volatile market check (high priority)
                if volatility > thresholds['volatile']['volatility']:
                    logger.info(f"Detected VOLATILE regime (volatility: {volatility:.4f})")
                    return "volatile"
                
                # Bull market check
                if (overall_price_change > thresholds['bull']['price_change'] and 
                    uptrend_pct > thresholds['bull']['trend_strength']):
                    logger.info(f"Detected BULL regime (price change: {overall_price_change:.2f}%, uptrend: {uptrend_pct:.2f})")
                    return "bull"
                
                # Bear market check
                if (overall_price_change < thresholds['bear']['price_change'] and 
                    downtrend_pct > thresholds['bear']['trend_strength']):
                    logger.info(f"Detected BEAR regime (price change: {overall_price_change:.2f}%, downtrend: {downtrend_pct:.2f})")
                    return "bear"
                
                # Ranging market check
                if (abs(overall_price_change) < thresholds['ranging']['price_range'] and 
                    trend_strength < thresholds['ranging']['trend_strength']):
                    logger.info(f"Detected RANGING regime (price change: {overall_price_change:.2f}%, trend strength: {trend_strength:.2f})")
                    return "ranging"
                
                # Default to the dominant trend type if no specific regime is detected
                if uptrend_pct > downtrend_pct and uptrend_pct > sideways_pct:
                    logger.info(f"Defaulting to BULL regime (uptrend: {uptrend_pct:.2f})")
                    return "bull"
                elif downtrend_pct > uptrend_pct and downtrend_pct > sideways_pct:
                    logger.info(f"Defaulting to BEAR regime (downtrend: {downtrend_pct:.2f})")
                    return "bear"
                else:
                    logger.info(f"Defaulting to RANGING regime (sideways: {sideways_pct:.2f})")
                    return "ranging"
            
            return "unknown"
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return "unknown"
    
    async def detect_current_regime(self) -> str:
        """
        Detect the current market regime based on recent market data.
        
        Returns:
            Detected market regime: 'bull', 'bear', 'ranging', or 'volatile'
        """
        try:
            # Get short and medium-term market data
            short_term_data = await self.get_historical_market_data(
                lookback_hours=self.lookback_periods['short']
            )
            
            medium_term_data = await self.get_historical_market_data(
                lookback_hours=self.lookback_periods['medium']
            )
            
            # Detect regimes for different time periods
            short_term_regime = self.detect_regime_from_data(short_term_data)
            medium_term_regime = self.detect_regime_from_data(medium_term_data)
            
            # Prioritize short-term regime when detecting volatile markets
            if short_term_regime == "volatile":
                return "volatile"
            
            # Use medium-term regime as the baseline
            return medium_term_regime
            
        except Exception as e:
            logger.error(f"Error detecting current regime: {str(e)}")
            return "unknown"
    
    async def select_strategy_for_regime(self, regime: str) -> Dict:
        """
        Select the best strategy for the given market regime.
        
        Args:
            regime: The detected market regime
            
        Returns:
            Selected strategy details or None if none found
        """
        try:
            # Get all available strategies
            available_strategies = await self.get_available_strategies()
            
            if not available_strategies:
                logger.warning("No available strategies found")
                return None
            
            # If we already have a mapping for this regime, use it
            if regime in self.regime_strategy_mapping:
                strategy_id = self.regime_strategy_mapping[regime]
                
                # Check if the strategy still exists
                for strategy in available_strategies:
                    if strategy['worker_id'] == strategy_id:
                        logger.info(f"Using pre-mapped strategy {strategy_id} for {regime} regime")
                        return strategy
            
            # No existing mapping or the mapped strategy doesn't exist anymore
            # We need to find the best strategy for this regime
            
            # Step 1: Get performance data for all strategies in this regime
            performance_data = []
            
            for strategy in available_strategies:
                strategy_id = strategy['worker_id']
                
                # Get strategy parameters
                strategy_params = strategy.get('parameters', {})
                
                # Get performance data
                perf_data = await self.get_strategy_performance(strategy_id)
                
                if perf_data:
                    # Add strategy details to performance data
                    perf_data['strategy_id'] = strategy_id
                    perf_data['params'] = strategy_params
                    performance_data.append(perf_data)
            
            if not performance_data:
                logger.warning("No performance data available for strategy selection")
                return available_strategies[0]  # Return the first available as default
            
            # Step 2: Score strategies for the current regime
            scored_strategies = []
            
            for perf in performance_data:
                score = self._calculate_regime_score(perf, regime)
                scored_strategies.append({
                    'strategy_id': perf['strategy_id'],
                    'score': score,
                    'performance': perf
                })
            
            # Step 3: Select the highest-scoring strategy
            if scored_strategies:
                # Sort by score in descending order
                sorted_strategies = sorted(
                    scored_strategies, 
                    key=lambda x: x['score'], 
                    reverse=True
                )
                
                best_strategy_id = sorted_strategies[0]['strategy_id']
                best_score = sorted_strategies[0]['score']
                
                # Find the full strategy details from available_strategies
                selected_strategy = next(
                    (s for s in available_strategies if s['worker_id'] == best_strategy_id),
                    None
                )
                
                if selected_strategy:
                    # Update regime strategy mapping
                    self.regime_strategy_mapping[regime] = best_strategy_id
                    
                    logger.info(f"Selected strategy {best_strategy_id} for {regime} regime with score {best_score:.4f}")
                    return selected_strategy
            
            # If no strategy is found, return the first available
            logger.warning(f"No suitable strategy found for {regime} regime, using default")
            return available_strategies[0]
            
        except Exception as e:
            logger.error(f"Error selecting strategy for regime {regime}: {str(e)}")
            return None
    
    def _calculate_regime_score(self, perf_data: Dict, regime: str) -> float:
        """
        Calculate a strategy's score for a specific market regime.
        
        Args:
            perf_data: Performance data for the strategy
            regime: The market regime to score for
            
        Returns:
            Score between 0 and 1, higher is better
        """
        try:
            # Extract performance metrics
            sharpe_ratio = perf_data.get('sharpe_ratio', 0)
            win_rate = perf_data.get('win_rate', 0)
            max_drawdown = perf_data.get('max_drawdown', 100)
            profit_factor = perf_data.get('profit_factor', 0)
            
            # Normalize drawdown (lower is better, so invert)
            norm_drawdown = 1 - (max_drawdown / 100)
            
            # Base score is an average of key metrics
            base_score = (
                sharpe_ratio * 0.3 +
                win_rate * 0.2 +
                norm_drawdown * 0.2 +
                profit_factor * 0.3
            ) / 1.0  # Total weight
            
            # Get regime-specific performance if available
            regime_perf = perf_data.get('regime_performance', {}).get(regime, {})
            
            if regime_perf:
                # Extract regime-specific metrics
                regime_sharpe = regime_perf.get('sharpe_ratio', sharpe_ratio)
                regime_win_rate = regime_perf.get('win_rate', win_rate)
                regime_drawdown = regime_perf.get('max_drawdown', max_drawdown)
                regime_profit_factor = regime_perf.get('profit_factor', profit_factor)
                
                # Normalize regime drawdown
                regime_norm_drawdown = 1 - (regime_drawdown / 100)
                
                # Calculate regime-specific score
                regime_score = (
                    regime_sharpe * 0.3 +
                    regime_win_rate * 0.2 +
                    regime_norm_drawdown * 0.2 +
                    regime_profit_factor * 0.3
                ) / 1.0  # Total weight
                
                # Combined score with higher weight for regime-specific performance
                final_score = (base_score * 0.3) + (regime_score * 0.7)
            else:
                # Only have base score
                final_score = base_score
            
            # Adjust score based on strategy parameters' suitability for the regime
            strategy_params = perf_data.get('params', {})
            
            # Specific adjustments for each regime
            if regime == 'bull':
                # In bull markets, favor trend-following strategies with higher take_profit
                if strategy_params.get('ema_long', 0) < 30:  # Shorter long-term EMA for faster trend following
                    final_score *= 1.1
                if strategy_params.get('take_profit', 0) > 5:  # Higher take profit to ride trends
                    final_score *= 1.05
                    
            elif regime == 'bear':
                # In bear markets, favor strategies with tighter stop losses
                if strategy_params.get('stop_loss', 100) < 3:  # Tighter stop loss
                    final_score *= 1.1
                if strategy_params.get('rsi_oversold', 0) > 30:  # Higher oversold threshold
                    final_score *= 1.05
                    
            elif regime == 'ranging':
                # In ranging markets, favor mean reversion strategies
                if strategy_params.get('bollinger_std', 0) > 2.0:  # Wider Bollinger Bands
                    final_score *= 1.1
                if strategy_params.get('rsi_period', 0) < 14:  # More responsive RSI
                    final_score *= 1.05
                    
            elif regime == 'volatile':
                # In volatile markets, favor strategies with dynamic position sizing and risk management
                if strategy_params.get('atr_multiplier', 0) > 2.0:  # Higher ATR multiplier for dynamic stops
                    final_score *= 1.1
                if strategy_params.get('atr_period', 0) < 10:  # More responsive ATR
                    final_score *= 1.05
            
            # Ensure score is between 0 and 1
            return max(0, min(1, final_score))
            
        except Exception as e:
            logger.error(f"Error calculating regime score: {str(e)}")
            return 0.0
    
    async def update_strategy_performance_by_regime(self):
        """
        Update performance history for all strategies across different market regimes.
        Uses historical data to evaluate how strategies perform in different regimes.
        """
        try:
            # Get long-term historical data
            historical_data = await self.get_historical_market_data(
                lookback_hours=self.lookback_periods['long']
            )
            
            if not historical_data:
                logger.warning("No historical data available for regime performance analysis")
                return
                
            # Convert data to DataFrame for analysis
            df = pd.DataFrame(historical_data)
            
            # Check if required columns exist
            required_columns = ['timestamp', 'current_price', 'trend', 
                               'trend_strength', 'price_change_24h']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Required columns missing for regime analysis: {missing_columns}")
                return
                
            # Convert timestamp to datetime if it's a string
            if isinstance(df['timestamp'].iloc[0], str):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Segment data into windows (e.g., 24-hour periods)
            window_hours = 24
            window_size = pd.Timedelta(hours=window_hours)
            
            # Create windows of market data
            windows = []
            start_time = df['timestamp'].min()
            end_time = df['timestamp'].max()
            
            current_start = start_time
            while current_start < end_time:
                current_end = current_start + window_size
                window_data = df[(df['timestamp'] >= current_start) & (df['timestamp'] < current_end)]
                
                if len(window_data) > 0:
                    window_dict = {
                        'start_time': current_start,
                        'end_time': current_end,
                        'data': window_data.to_dict('records')
                    }
                    windows.append(window_dict)
                
                current_start = current_end
            
            # Classify each window into a market regime
            for window in windows:
                window['regime'] = self.detect_regime_from_data(window['data'])
            
            # Get all available strategies
            available_strategies = await self.get_available_strategies()
            
            if not available_strategies:
                logger.warning("No available strategies found for regime performance analysis")
                return
            
            # Initialize performance tracking structure
            regime_performance = {
                strategy['worker_id']: {
                    'bull': [],
                    'bear': [],
                    'ranging': [],
                    'volatile': []
                }
                for strategy in available_strategies
            }
            
            # TODO: In a real implementation, we would simulate each strategy's
            # performance during each time window and categorize the results by regime.
            # For now, we'll use a simpler approach based on available data.
            
            # For each strategy, get historical trades and match them to regime windows
            for strategy in available_strategies:
                strategy_id = strategy['worker_id']
                
                # Get strategy trades from Redis
                trades_data = await self.redis.get(f'strategy_trades_{strategy_id}')
                if not trades_data:
                    logger.warning(f"No trade data found for strategy {strategy_id}")
                    continue
                
                trades = json.loads(trades_data)
                
                # Match trades to windows and calculate regime-specific performance
                for regime in ['bull', 'bear', 'ranging', 'volatile']:
                    # Get windows for this regime
                    regime_windows = [w for w in windows if w['regime'] == regime]
                    
                    if not regime_windows:
                        continue
                    
                    # Match trades to this regime
                    regime_trades = []
                    
                    for trade in trades:
                        trade_time = pd.to_datetime(trade.get('timestamp'))
                        
                        # Find matching window
                        for window in regime_windows:
                            if window['start_time'] <= trade_time < window['end_time']:
                                regime_trades.append(trade)
                                break
                    
                    # Calculate performance for this regime if we have trades
                    if regime_trades:
                        wins = sum(1 for t in regime_trades if t.get('profit', 0) > 0)
                        losses = sum(1 for t in regime_trades if t.get('profit', 0) <= 0)
                        
                        total_profit = sum(t.get('profit', 0) for t in regime_trades)
                        total_loss = abs(sum(t.get('profit', 0) for t in regime_trades if t.get('profit', 0) < 0))
                        
                        regime_win_rate = wins / len(regime_trades) if len(regime_trades) > 0 else 0
                        regime_profit_factor = total_profit / total_loss if total_loss > 0 else 1
                        
                        # Add to performance tracking
                        regime_performance[strategy_id][regime].append({
                            'win_rate': regime_win_rate,
                            'profit_factor': regime_profit_factor,
                            'trade_count': len(regime_trades)
                        })
            
            # Calculate average performance for each strategy in each regime
            for strategy_id, regimes in regime_performance.items():
                for regime, performances in regimes.items():
                    if performances:
                        # Calculate weighted average based on trade count
                        total_trades = sum(p['trade_count'] for p in performances)
                        
                        if total_trades > 0:
                            avg_win_rate = sum(p['win_rate'] * p['trade_count'] for p in performances) / total_trades
                            avg_profit_factor = sum(p['profit_factor'] * p['trade_count'] for p in performances) / total_trades
                            
                            # Store average performance
                            self.strategy_performance_history.setdefault(strategy_id, {})
                            self.strategy_performance_history[strategy_id][regime] = {
                                'win_rate': avg_win_rate,
                                'profit_factor': avg_profit_factor,
                                'trade_count': total_trades
                            }
                            
                            logger.info(f"Strategy {strategy_id} in {regime} regime: win_rate={avg_win_rate:.2f}, profit_factor={avg_profit_factor:.2f}, trades={total_trades}")
            
            # Save the performance history to Redis for persistence
            await self.redis.set(
                'strategy_regime_performance',
                json.dumps(self.strategy_performance_history)
            )
            
            logger.info("Updated strategy performance by market regime")
            
        except Exception as e:
            logger.error(f"Error updating strategy performance by regime: {str(e)}")
    
    async def switch_to_regime_strategy(self, regime: str) -> bool:
        """
        Switch the active trading strategy to the best one for the detected regime.
        
        Args:
            regime: The detected market regime
            
        Returns:
            True if the switch was successful, False otherwise
        """
        try:
            # Select the best strategy for this regime
            selected_strategy = await self.select_strategy_for_regime(regime)
            
            if not selected_strategy:
                logger.warning(f"No strategy found for {regime} regime, cannot switch")
                return False
            
            strategy_id = selected_strategy['worker_id']
            
            # Check if a strategy switch is needed
            current_strategy_id = await self.redis.get('active_strategy_id')
            
            if current_strategy_id == strategy_id:
                logger.info(f"Strategy {strategy_id} already active for {regime} regime, no switch needed")
                return True
            
            # Prepare to switch to the new strategy
            logger.info(f"Switching to strategy {strategy_id} for {regime} regime")
            
            # Send signal to trade executor to switch strategies
            switch_signal = {
                'action': 'switch_strategy',
                'strategy_id': strategy_id,
                'market_regime': regime,
                'timestamp': datetime.now().isoformat()
            }
            
            # Publish switch signal to Redis
            await self.redis.publish(
                'strategy_switch',
                json.dumps(switch_signal)
            )
            
            # Store the active strategy ID
            await self.redis.set('active_strategy_id', strategy_id)
            
            # Record the strategy switch in history
            switch_record = {
                'timestamp': datetime.now().isoformat(),
                'previous_strategy': current_strategy_id,
                'new_strategy': strategy_id,
                'market_regime': regime,
                'reason': f"Market regime changed to {regime}"
            }
            
            await self.redis.lpush(
                'strategy_switch_history',
                json.dumps(switch_record)
            )
            
            logger.info(f"Successfully switched to strategy {strategy_id} for {regime} regime")
            return True
            
        except Exception as e:
            logger.error(f"Error switching to regime strategy: {str(e)}")
            return False
    
    async def detect_and_switch_if_needed(self) -> Tuple[str, bool]:
        """
        Detect current market regime and switch strategy if needed.
        
        Returns:
            Tuple of (detected_regime, switch_success_bool)
        """
        try:
            # Detect current market regime
            current_regime = await self.detect_current_regime()
            
            if current_regime == "unknown":
                logger.warning("Could not determine current market regime")
                return (current_regime, False)
            
            # Store the current regime
            previous_regime = self.current_regime
            self.current_regime = current_regime
            
            # Add to regime history
            self.regime_history.append({
                'timestamp': datetime.now().isoformat(),
                'regime': current_regime
            })
            
            # Keep only the last 100 entries in history
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            # Store regime history in Redis
            await self.redis.set(
                'market_regime_history',
                json.dumps(self.regime_history)
            )
            
            # Check if we need to switch strategies
            if previous_regime != current_regime:
                logger.info(f"Market regime changed from {previous_regime} to {current_regime}")
                
                # Switch to the best strategy for this regime
                switch_success = await self.switch_to_regime_strategy(current_regime)
                return (current_regime, switch_success)
            
            # No change in regime
            logger.info(f"Current market regime is still {current_regime}, no strategy switch needed")
            return (current_regime, True)
            
        except Exception as e:
            logger.error(f"Error in detect_and_switch_if_needed: {str(e)}")
            return ("unknown", False)
    
    async def health_check_server(self):
        """Run a simple TCP server for health checks."""
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('0.0.0.0', self.service_port))
            server.listen(1)
            server.setblocking(False)
            
            logger.info(f"Health check server listening on port {self.service_port}")
            
            while self.running:
                try:
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Health check server loop error: {str(e)}")
                    break
                    
        except Exception as e:
            logger.error(f"Failed to start health check server: {str(e)}")
            raise  # Re-raise the exception to trigger service restart
        finally:
            try:
                server.close()
            except Exception:
                pass
    
    async def maintain_redis(self):
        """Maintain Redis connection."""
        logger.debug("Starting Redis connection maintenance...")
        while self.running:
            try:
                if self.redis:
                    await self.redis.ping()
                else:
                    await self.connect_redis()
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Redis connection error: {str(e)}")
                self.redis = None
                await asyncio.sleep(5)
    
    async def run(self):
        """
        Main service loop.
        """
        try:
            logger.info("Starting Market Regime Service...")
            
            # First establish Redis connection
            if not await self.connect_redis(max_retries=15, retry_delay=2):
                raise Exception("Failed to establish initial Redis connection")
            
            # Initialize ML components
            await self.initialize_ml_components()
            
            # Initial load of performance history from Redis
            perf_history = await self.redis.get('strategy_regime_performance')
            if perf_history:
                self.strategy_performance_history = json.loads(perf_history)
                logger.info(f"Loaded performance history for {len(self.strategy_performance_history)} strategies")
            
            # Initial regime detection and strategy selection using ML or hybrid approach
            if self.detection_method in ['ml', 'hybrid']:
                current_regime = await self.detect_current_regime_ml()
            else:
                current_regime = await self.detect_current_regime()
                
            self.current_regime = current_regime
            logger.info(f"Initial market regime: {current_regime}")
            
            # Initialize strategy for the detected regime
            switch_result = await self.switch_to_regime_strategy(current_regime)
            if not switch_result:
                logger.warning(f"Failed to initialize strategy for {current_regime} regime")
            
            # Create tasks for all service components
            tasks = [
                asyncio.create_task(self.maintain_redis()),
                asyncio.create_task(self.health_check_server()),
                asyncio.create_task(self.regime_detection_loop())
            ]
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Critical error in Market Regime Service: {str(e)}")
        finally:
            await self.stop()
            
    async def regime_detection_loop(self):
        """Main loop for regime detection and strategy switching."""
        logger.info("Starting market regime detection loop...")
        
        while self.running:
            try:
                # Check for regime changes and switch strategies if needed
                if self.detection_method in ['ml', 'hybrid']:
                    current_regime = await self.detect_current_regime_ml()
                else:
                    current_regime = await self.detect_current_regime()
                
                previous_regime = self.current_regime
                self.current_regime = current_regime
                
                # Handle regime change
                if previous_regime != current_regime:
                    logger.info(f"Market regime changed from {previous_regime} to {current_regime}")
                    
                    # Update regime history
                    self.regime_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'regime': current_regime
                    })
                    
                    # Keep only the last 100 entries in history
                    if len(self.regime_history) > 100:
                        self.regime_history = self.regime_history[-100:]
                    
                    # Store regime history in Redis
                    await self.redis.set(
                        'market_regime_history',
                        json.dumps(self.regime_history)
                    )
                    
                    # Switch to the best strategy for this regime
                    switch_result = await self.switch_to_regime_strategy(current_regime)
                    if not switch_result:
                        logger.warning(f"Failed to switch strategy for {current_regime} regime")
                else:
                    logger.info(f"Current market regime is still {current_regime}, no strategy switch needed")
                
                # Periodically update strategy performance by regime (less frequently)
                if datetime.now().hour % 12 == 0 and datetime.now().minute < 5:
                    await self.update_strategy_performance_by_regime()
                
                # Periodically retrain ML models if needed
                time_since_training = (datetime.now() - self.last_training_time).total_seconds()
                if time_since_training > self.model_training_interval:
                    logger.info("Scheduling periodic model retraining...")
                    asyncio.create_task(self.train_regime_models())
                
                # Sleep until next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in regime detection loop: {str(e)}")
                await asyncio.sleep(60)  # Sleep for a minute on error
    
    async def stop(self):
        """
        Stop the market regime service.
        """
        logger.info("Stopping Market Regime Service...")
        self.running = False
        # Save current regime detector model if trained
        if self.regime_detector and self.regime_detector.is_trained:
            model_path = os.path.join(self.model_dir, f"market_regime_{self.ml_method}.pkl")
            self.regime_detector.save_model(model_path)
            logger.info(f"Saved market regime model to {model_path}")
        
        # Close Redis connection
        if self.redis:
            await self.redis.close()
            logger.info("Closed Redis connection")

if __name__ == "__main__":
    service = MarketRegimeService()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
        asyncio.run(service.stop())
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        asyncio.run(service.stop())