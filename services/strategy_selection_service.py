import os
import json
import redis
import asyncio
import logging as logger
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

# Import our custom modules
from services.market_regime_service import MarketRegimeService
from services.model_integration import FeatureImportanceIntegrator

# Load environment variables
load_dotenv()

# Configure logging
logger.basicConfig(
    level=logger.INFO,
    format='%(asctime)s - %(levelname)s - [StrategySelection] %(message)s',
    handlers=[
        logger.FileHandler('logs/strategy_selection.log'),
        logger.StreamHandler()
    ]
)

class StrategySelectionService:
    """
    Service for automated strategy selection based on multiple factors:
    - Market regime (bull, bear, ranging, volatile)
    - Asset-specific performance history
    - Portfolio risk profile
    - Social sentiment metrics
    - Market volatility
    - Trading frequency requirements
    - Time of day / market activity patterns
    """
    
    def __init__(self):
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Initialize Redis connection
        self.redis = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=True
        )
        
        # Service state
        self.running = True
        self.check_interval = int(os.getenv('STRATEGY_CHECK_INTERVAL', '300'))  # Default: check every 5 minutes
        
        # Initialize market regime service
        self.market_regime_service = MarketRegimeService()
        
        # Initialize feature importance integrator
        self.feature_importance = FeatureImportanceIntegrator()
        
        # Strategy mapping and scoring
        self.strategy_scores = {}
        self.active_strategy_id = None
        self.strategy_history = []
        self.asset_strategy_mapping = {}  # Maps assets to their best strategies
        
        # Selection weighting factors (how much each component influences selection)
        self.selection_weights = {
            'market_regime': float(os.getenv('WEIGHT_MARKET_REGIME', '0.30')),
            'historical_performance': float(os.getenv('WEIGHT_HISTORICAL_PERF', '0.25')),
            'risk_profile': float(os.getenv('WEIGHT_RISK_PROFILE', '0.15')),
            'social_sentiment': float(os.getenv('WEIGHT_SOCIAL_SENTIMENT', '0.15')),
            'market_volatility': float(os.getenv('WEIGHT_MARKET_VOLATILITY', '0.10')),
            'feature_importance': float(os.getenv('WEIGHT_FEATURE_IMPORTANCE', '0.05'))
        }
        
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(self.selection_weights.values())
        for key in self.selection_weights:
            self.selection_weights[key] /= total_weight
        
        # Strategy switching thresholds
        self.min_improvement_threshold = float(os.getenv('MIN_STRATEGY_IMPROVEMENT', '0.15'))
        self.min_confidence_threshold = float(os.getenv('MIN_STRATEGY_CONFIDENCE', '0.7'))
        
        # Time-based factors
        self.time_windows = {
            'high_volatility': {'start': '14:00', 'end': '22:00'},  # UTC times for market opens
            'low_activity': {'start': '00:00', 'end': '08:00'}
        }
        
        # Risk profiles configurations
        self.risk_profiles = {
            'conservative': {
                'max_drawdown': 0.05,
                'min_sharpe': 1.5,
                'volatility_preference': 'low'
            },
            'moderate': {
                'max_drawdown': 0.10,
                'min_sharpe': 1.2,
                'volatility_preference': 'medium'
            },
            'aggressive': {
                'max_drawdown': 0.15,
                'min_sharpe': 1.0,
                'volatility_preference': 'high'
            }
        }
        
        # Current risk profile (can be changed at runtime)
        self.current_risk_profile = os.getenv('RISK_PROFILE', 'moderate')
        
        # Social sentiment thresholds
        self.sentiment_thresholds = {
            'very_bearish': -0.6,
            'bearish': -0.2,
            'neutral': 0.2,
            'bullish': 0.6,
            'very_bullish': 0.8
        }
        
        # Strategy metrics cache
        self.strategy_metrics_cache = {}
        self.cache_expiry = 3600  # 1 hour in seconds
        
        logger.info("Strategy Selection Service initialized with the following weights:")
        for factor, weight in self.selection_weights.items():
            logger.info(f"- {factor}: {weight:.2f}")
        logger.info(f"Current risk profile: {self.current_risk_profile}")
    
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
    
    async def get_strategy_metrics(self, strategy_id: str) -> Dict:
        """
        Get comprehensive metrics for a specific strategy.
        
        Args:
            strategy_id: The ID of the strategy to evaluate
            
        Returns:
            Metrics dictionary with performance, risk, and other metrics
        """
        try:
            # Check cache first
            current_time = datetime.now().timestamp()
            if strategy_id in self.strategy_metrics_cache:
                cache_entry = self.strategy_metrics_cache[strategy_id]
                if current_time - cache_entry['timestamp'] < self.cache_expiry:
                    return cache_entry['data']
            
            # Fetch base performance metrics
            performance_key = f'strategy_performance_{strategy_id}'
            performance_data = await self.redis.get(performance_key)
            
            if not performance_data:
                logger.warning(f"No performance data found for strategy {strategy_id}")
                return {}
            
            performance = json.loads(performance_data)
            
            # Fetch regime-specific performance if available
            regime_key = f'strategy_regime_performance_{strategy_id}'
            regime_data = await self.redis.get(regime_key)
            
            if regime_data:
                regime_performance = json.loads(regime_data)
                performance['regime_performance'] = regime_performance
            
            # Get social metrics influence if available
            social_key = f'strategy_social_performance_{strategy_id}'
            social_data = await self.redis.get(social_key)
            
            if social_data:
                social_performance = json.loads(social_data)
                performance['social_performance'] = social_performance
            
            # Fetch trade history for additional analytics
            trades_key = f'strategy_trades_{strategy_id}'
            trades_data = await self.redis.get(trades_key)
            
            if trades_data:
                trades = json.loads(trades_data)
                
                # Calculate additional metrics from trades
                if trades:
                    # Calculate win streaks, loss streaks, time-based performance
                    win_streak, max_win_streak, loss_streak, max_loss_streak = 0, 0, 0, 0
                    
                    for trade in trades:
                        if trade.get('profit', 0) > 0:
                            win_streak += 1
                            loss_streak = 0
                            max_win_streak = max(max_win_streak, win_streak)
                        else:
                            loss_streak += 1
                            win_streak = 0
                            max_loss_streak = max(max_loss_streak, loss_streak)
                    
                    performance['max_win_streak'] = max_win_streak
                    performance['max_loss_streak'] = max_loss_streak
                    
                    # Add time-based analysis if timestamps available
                    if 'timestamp' in trades[0]:
                        try:
                            # Convert trades to DataFrame for time analysis
                            trades_df = pd.DataFrame(trades)
                            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                            
                            # Add hour of day
                            trades_df['hour'] = trades_df['timestamp'].dt.hour
                            
                            # Calculate hourly performance
                            hourly_performance = {}
                            for hour in range(24):
                                hour_trades = trades_df[trades_df['hour'] == hour]
                                if len(hour_trades) > 0:
                                    wins = sum(hour_trades['profit'] > 0)
                                    win_rate = wins / len(hour_trades)
                                    hourly_performance[str(hour)] = {
                                        'win_rate': win_rate,
                                        'trade_count': len(hour_trades)
                                    }
                            
                            performance['hourly_performance'] = hourly_performance
                        except Exception as e:
                            logger.error(f"Error analyzing time-based performance: {str(e)}")
            
            # Cache the metrics
            self.strategy_metrics_cache[strategy_id] = {
                'timestamp': current_time,
                'data': performance
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"Error fetching strategy metrics: {str(e)}")
            return {}
    
    async def get_market_conditions(self) -> Dict:
        """Retrieve current market conditions from Redis"""
        try:
            market_data = await self.redis.get('market_conditions')
            if market_data:
                return json.loads(market_data)
            return {}
        except Exception as e:
            logger.error(f"Error getting market conditions: {str(e)}")
            return {}
    
    async def get_social_sentiment(self) -> Dict:
        """Retrieve current social sentiment metrics from Redis"""
        try:
            sentiment_data = await self.redis.get('social_metrics')
            if sentiment_data:
                return json.loads(sentiment_data)
            return {}
        except Exception as e:
            logger.error(f"Error getting social sentiment: {str(e)}")
            return {}
    
    async def get_portfolio_risk(self) -> Dict:
        """Retrieve current portfolio risk metrics from Redis"""
        try:
            risk_data = await self.redis.get('portfolio_risk')
            if risk_data:
                return json.loads(risk_data)
            return {}
        except Exception as e:
            logger.error(f"Error getting portfolio risk: {str(e)}")
            return {}
    
    async def calculate_risk_scores(self, strategies: List[Dict]) -> Dict[str, float]:
        """
        Calculate risk scores for strategies based on current risk profile.
        
        Args:
            strategies: List of strategy dictionaries
            
        Returns:
            Dictionary mapping strategy IDs to risk scores
        """
        try:
            risk_scores = {}
            risk_profile = self.risk_profiles[self.current_risk_profile]
            
            for strategy in strategies:
                strategy_id = strategy['worker_id']
                metrics = await self.get_strategy_metrics(strategy_id)
                
                if not metrics:
                    risk_scores[strategy_id] = 0.0
                    continue
                
                # Extract risk metrics
                max_drawdown = metrics.get('max_drawdown', 1.0)
                sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
                sortino_ratio = metrics.get('sortino_ratio', 0.0)
                recovery_factor = metrics.get('recovery_factor', 0.0)
                avg_vol = metrics.get('avg_volatility', 0.5)
                
                # Calculate base risk score
                if max_drawdown > 0:
                    dd_score = 1.0 - (max_drawdown / risk_profile['max_drawdown'])
                else:
                    dd_score = 1.0
                
                # Apply appropriate volatility preference based on risk profile
                volatility_score = 0.5  # Default middle value
                vol_pref = risk_profile['volatility_preference']
                
                if vol_pref == 'low':
                    # Lower is better for conservative
                    volatility_score = 1.0 - min(avg_vol, 1.0)
                elif vol_pref == 'medium':
                    # Middle is better for moderate
                    volatility_score = 1.0 - abs(avg_vol - 0.5)
                elif vol_pref == 'high':
                    # Higher is better for aggressive
                    volatility_score = min(avg_vol, 1.0)
                
                # Calculate combined risk score
                sharpe_score = min(sharpe_ratio / risk_profile['min_sharpe'], 1.0)
                
                # Recovery factor (ability to recover from drawdowns) - higher is better
                recovery_score = min(recovery_factor, 2.0) / 2.0
                
                # Combine scores with weights
                combined_score = (
                    dd_score * 0.3 +
                    sharpe_score * 0.3 +
                    recovery_score * 0.2 +
                    volatility_score * 0.2
                )
                
                # Ensure score is between 0 and 1
                risk_scores[strategy_id] = max(0.0, min(1.0, combined_score))
            
            return risk_scores
            
        except Exception as e:
            logger.error(f"Error calculating risk scores: {str(e)}")
            return {strategy['worker_id']: 0.0 for strategy in strategies}
    
    async def calculate_historical_performance_scores(self, strategies: List[Dict]) -> Dict[str, float]:
        """
        Calculate scores based on historical performance metrics.
        
        Args:
            strategies: List of strategy dictionaries
            
        Returns:
            Dictionary mapping strategy IDs to performance scores
        """
        try:
            performance_scores = {}
            
            for strategy in strategies:
                strategy_id = strategy['worker_id']
                metrics = await self.get_strategy_metrics(strategy_id)
                
                if not metrics:
                    performance_scores[strategy_id] = 0.0
                    continue
                
                # Extract key performance metrics
                win_rate = metrics.get('win_rate', 0.0)
                profit_factor = metrics.get('profit_factor', 0.0)
                sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
                sortino_ratio = metrics.get('sortino_ratio', 0.0)
                avg_trade = metrics.get('average_trade', 0.0)
                
                # Calculate score from metrics
                win_rate_score = min(win_rate, 1.0)
                pf_score = min(profit_factor / 3.0, 1.0)  # Scale profit factor 0-3
                sharpe_score = min(sharpe_ratio / 3.0, 1.0)  # Scale sharpe ratio 0-3
                
                # For average trade, we want it to be positive but not overly weight it
                avg_trade_score = 0.5
                if avg_trade > 0:
                    avg_trade_score = min(0.5 + (avg_trade / 0.02), 1.0)  # Assume 2% is excellent
                elif avg_trade < 0:
                    avg_trade_score = max(0.0, 0.5 + (avg_trade / 0.02))
                
                # Combine scores
                performance_scores[strategy_id] = (
                    win_rate_score * 0.25 +
                    pf_score * 0.25 +
                    sharpe_score * 0.3 +
                    avg_trade_score * 0.2
                )
            
            return performance_scores
            
        except Exception as e:
            logger.error(f"Error calculating performance scores: {str(e)}")
            return {strategy['worker_id']: 0.0 for strategy in strategies}
    
    async def calculate_social_sentiment_scores(self, strategies: List[Dict]) -> Dict[str, float]:
        """
        Calculate strategy scores based on social sentiment metrics.
        
        Args:
            strategies: List of strategy dictionaries
            
        Returns:
            Dictionary mapping strategy IDs to social sentiment scores
        """
        try:
            social_scores = {}
            social_sentiment = await self.get_social_sentiment()
            
            # If no social sentiment data available, return neutral scores
            if not social_sentiment:
                return {strategy['worker_id']: 0.5 for strategy in strategies}
            
            # Get overall sentiment metric
            overall_sentiment = social_sentiment.get('overall_sentiment', 0.0)
            sentiment_volume = social_sentiment.get('volume', 0.0)
            
            # Categorize current sentiment
            sentiment_category = 'neutral'
            for category, threshold in sorted(self.sentiment_thresholds.items(), key=lambda x: x[1]):
                if overall_sentiment <= threshold:
                    sentiment_category = category
                    break
            
            logger.debug(f"Current social sentiment: {sentiment_category} ({overall_sentiment:.2f})")
            
            for strategy in strategies:
                strategy_id = strategy['worker_id']
                metrics = await self.get_strategy_metrics(strategy_id)
                
                # Default score if no metrics available
                if not metrics or 'social_performance' not in metrics:
                    # Neutral score if no social data
                    social_scores[strategy_id] = 0.5
                    continue
                
                # Get strategy's historical performance with different sentiment levels
                social_performance = metrics.get('social_performance', {})
                
                # If strategy has performance data for current sentiment category, use it
                if sentiment_category in social_performance:
                    category_perf = social_performance[sentiment_category]
                    win_rate = category_perf.get('win_rate', 0.5)
                    profit_factor = category_perf.get('profit_factor', 1.0)
                    
                    # Calculate social score based on how well strategy performs in this sentiment
                    social_scores[strategy_id] = (win_rate * 0.5) + (min(profit_factor / 2.0, 1.0) * 0.5)
                else:
                    # Default to neutral score if no specific data for this sentiment
                    social_scores[strategy_id] = 0.5
            
            return social_scores
            
        except Exception as e:
            logger.error(f"Error calculating social sentiment scores: {str(e)}")
            return {strategy['worker_id']: 0.5 for strategy in strategies}
    
    async def calculate_volatility_scores(self, strategies: List[Dict]) -> Dict[str, float]:
        """
        Calculate strategy scores based on market volatility.
        
        Args:
            strategies: List of strategy dictionaries
            
        Returns:
            Dictionary mapping strategy IDs to volatility-based scores
        """
        try:
            volatility_scores = {}
            market_conditions = await self.get_market_conditions()
            
            # Get current volatility metric
            current_volatility = market_conditions.get('volatility', 0.5)
            
            # Categorize volatility
            volatility_category = 'medium'
            if current_volatility > 0.8:
                volatility_category = 'very_high'
            elif current_volatility > 0.6:
                volatility_category = 'high'
            elif current_volatility < 0.2:
                volatility_category = 'very_low'
            elif current_volatility < 0.4:
                volatility_category = 'low'
            
            logger.debug(f"Current market volatility: {volatility_category} ({current_volatility:.2f})")
            
            for strategy in strategies:
                strategy_id = strategy['worker_id']
                metrics = await self.get_strategy_metrics(strategy_id)
                
                if not metrics:
                    # Default score if no metrics
                    volatility_scores[strategy_id] = 0.5
                    continue
                
                # Check if strategy has volatility-specific performance metrics
                volatility_performance = metrics.get('volatility_performance', {})
                
                if volatility_category in volatility_performance:
                    # Use category-specific performance
                    category_perf = volatility_performance[volatility_category]
                    win_rate = category_perf.get('win_rate', 0.5)
                    profit_factor = category_perf.get('profit_factor', 1.0)
                    
                    volatility_scores[strategy_id] = (win_rate * 0.6) + (min(profit_factor / 2.0, 1.0) * 0.4)
                else:
                    # No specific data, infer from strategy parameters
                    # For this we need to analyze the strategy parameters
                    try:
                        params = strategy.get('parameters', {})
                        
                        # Get risk management parameters that indicate volatility handling
                        stop_loss = params.get('stop_loss', 2.0)
                        take_profit = params.get('take_profit', 4.0)
                        atr_multiplier = params.get('atr_multiplier', 1.5)
                        
                        # Score strategy's suitability for current volatility
                        if volatility_category in ['high', 'very_high']:
                            # High volatility - need adaptive stops, wider ATR, balanced risk/reward
                            if atr_multiplier > 1.5 and stop_loss < 3.0:
                                volatility_scores[strategy_id] = 0.8  # Good for high volatility
                            elif take_profit / stop_loss > 2.5:
                                volatility_scores[strategy_id] = 0.3  # Too aggressive for high volatility
                            else:
                                volatility_scores[strategy_id] = 0.5  # Neutral
                        elif volatility_category in ['low', 'very_low']:
                            # Low volatility - tighter stops, narrower ATR, more aggressive R:R
                            if take_profit / stop_loss > 2.0 and atr_multiplier < 1.5:
                                volatility_scores[strategy_id] = 0.8  # Good for low volatility
                            elif atr_multiplier > 2.0:
                                volatility_scores[strategy_id] = 0.3  # Too conservative for low volatility
                            else:
                                volatility_scores[strategy_id] = 0.5  # Neutral
                        else:
                            # Medium volatility - balanced approach
                            volatility_scores[strategy_id] = 0.6  # Slightly favor balanced strategies
                    except Exception as e:
                        logger.error(f"Error analyzing strategy parameters for volatility: {str(e)}")
                        volatility_scores[strategy_id] = 0.5  # Default to neutral score
            
            return volatility_scores
            
        except Exception as e:
            logger.error(f"Error calculating volatility scores: {str(e)}")
            return {strategy['worker_id']: 0.5 for strategy in strategies}
    
    async def calculate_feature_importance_scores(self, strategies: List[Dict]) -> Dict[str, float]:
        """
        Calculate strategy scores based on feature importance analysis.
        
        Args:
            strategies: List of strategy dictionaries
            
        Returns:
            Dictionary mapping strategy IDs to scores (0-1)
        """
        try:
            # Update feature importance data
            self.feature_importance.update_feature_importance_data()
            
            # Get feature importance data
            feature_importance_data = self.feature_importance.feature_importance_data
            if not feature_importance_data:
                logger.warning("No feature importance data available, using default scores")
                return {strategy['worker_id']: 0.5 for strategy in strategies}
            
            feature_importance_scores = {}
            
            # Get category weights and top features
            category_weights = feature_importance_data.get('top_categories', {})
            top_features = feature_importance_data.get('top_features_permutation', {})
            
            for strategy in strategies:
                strategy_id = strategy['worker_id']
                
                try:
                    # Get strategy category/type
                    strategy_category = strategy.get('category', '').lower()
                    strategy_type = strategy.get('type', '').lower()
                    parameters = strategy.get('parameters', {})
                    
                    # Default score
                    score = 0.5
                    
                    # Adjust based on strategy category and feature importance
                    # Social-based strategies
                    if 'social' in strategy_category or 'sentiment' in strategy_type:
                        if 'social' in category_weights and category_weights['social'] > 0.2:
                            # Social features are important
                            score += 0.3
                        elif 'social_sentiment' in top_features and top_features['social_sentiment'] > 0.1:
                            # Social sentiment specifically is important
                            score += 0.2
                    
                    # Momentum-based strategies
                    if 'momentum' in strategy_category or 'momentum' in strategy_type:
                        if 'momentum' in category_weights and category_weights['momentum'] > 0.2:
                            # Momentum features are important
                            score += 0.25
                        elif any(f in top_features and top_features[f] > 0.1 for f in ['rsi', 'macd', 'stoch_k']):
                            # Specific momentum indicators are important
                            score += 0.2
                    
                    # Price action strategies
                    if 'price_action' in strategy_category or 'price' in strategy_type:
                        if 'price_action' in category_weights and category_weights['price_action'] > 0.2:
                            # Price action features are important
                            score += 0.25
                        elif any(f in top_features and top_features[f] > 0.1 for f in 
                                ['price_change_5m', 'price_change_15m']):
                            # Specific price change indicators are important
                            score += 0.2
                    
                    # Volatility-based strategies
                    if 'volatility' in strategy_category or 'volatility' in strategy_type:
                        if 'volatility' in category_weights and category_weights['volatility'] > 0.2:
                            # Volatility features are important
                            score += 0.2
                        elif any(f in top_features and top_features[f] > 0.1 for f in 
                                ['atr', 'bb_width', 'bb_position']):
                            # Specific volatility indicators are important
                            score += 0.15
                    
                    # Trend-based strategies
                    if 'trend' in strategy_category or 'trend' in strategy_type:
                        if 'trend' in category_weights and category_weights['trend'] > 0.2:
                            # Trend features are important
                            score += 0.2
                        elif any(f in top_features and top_features[f] > 0.1 for f in 
                                ['trend_strength', 'ema_12', 'ema_26']):
                            # Specific trend indicators are important
                            score += 0.15
                    
                    # Check if strategy uses high-importance features
                    if 'indicators' in parameters:
                        used_indicators = parameters['indicators']
                        important_indicators = list(top_features.keys())[:5]  # Top 5 important features
                        
                        # Count how many important indicators the strategy uses
                        matching_indicators = [ind for ind in used_indicators if ind in important_indicators]
                        if len(matching_indicators) >= 3:
                            score += 0.2
                        elif len(matching_indicators) >= 1:
                            score += 0.1
                    
                    # Ensure score is between 0 and 1
                    feature_importance_scores[strategy_id] = max(0.0, min(1.0, score))
                    
                except Exception as e:
                    logger.error(f"Error analyzing strategy for feature importance: {str(e)}")
                    feature_importance_scores[strategy_id] = 0.5  # Default to neutral score
            
            return feature_importance_scores
            
        except Exception as e:
            logger.error(f"Error calculating feature importance scores: {str(e)}")
            return {strategy['worker_id']: 0.5 for strategy in strategies}
    
    async def apply_time_based_adjustments(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Apply time-based adjustments to strategy scores.
        
        Args:
            scores: Dictionary of strategy scores
            
        Returns:
            Adjusted scores dictionary
        """
        try:
            adjusted_scores = scores.copy()
            current_time = datetime.now().time()
            current_hour = current_time.hour
            
            # Check time windows
            in_high_volatility = False
            in_low_activity = False
            
            # Parse time windows
            for window_name, window in self.time_windows.items():
                start_hour = int(window['start'].split(':')[0])
                end_hour = int(window['end'].split(':')[0])
                
                if start_hour <= current_hour < end_hour:
                    if window_name == 'high_volatility':
                        in_high_volatility = True
                    elif window_name == 'low_activity':
                        in_low_activity = True
            
            # Apply adjustments based on time windows
            for strategy_id in adjusted_scores:
                try:
                    # Get strategy metrics to see how it performs at this time
                    metrics = await self.get_strategy_metrics(strategy_id)
                    
                    if 'hourly_performance' in metrics:
                        hourly_perf = metrics['hourly_performance'].get(str(current_hour), {})
                        
                        if hourly_perf:
                            # Adjust score based on historical performance during this hour
                            win_rate = hourly_perf.get('win_rate', 0.5)
                            trade_count = hourly_perf.get('trade_count', 0)
                            
                            # Only apply significant adjustment if we have enough data
                            if trade_count >= 10:
                                hour_factor = (win_rate - 0.5) * 2  # Scale to -1.0 to 1.0
                                adjusted_scores[strategy_id] += hour_factor * 0.1  # 10% adjustment
                    
                    # Apply volatility window adjustments
                    if in_high_volatility:
                        # For high volatility windows, favor strategies with good volatility handling
                        volatility_handling = 0.0
                        params = await self.redis.get(f'worker_parameters_{strategy_id}')
                        
                        if params:
                            params_dict = json.loads(params)
                            atr_multiplier = params_dict.get('atr_multiplier', 1.0)
                            volatility_handling = min(atr_multiplier / 2.0, 1.0)
                            
                        adjusted_scores[strategy_id] += volatility_handling * 0.05  # 5% adjustment
                    
                    # Apply low activity window adjustments
                    if in_low_activity:
                        # For low activity, prefer strategies with lower trade frequency
                        avg_trades_per_hour = metrics.get('avg_trades_per_hour', 10.0)
                        frequency_score = max(0, 1.0 - (avg_trades_per_hour / 20.0))
                        
                        adjusted_scores[strategy_id] += frequency_score * 0.05  # 5% adjustment
                
                except Exception as e:
                    logger.error(f"Error applying time adjustments to strategy {strategy_id}: {str(e)}")
            
            # Ensure scores remain in valid range
            for strategy_id in adjusted_scores:
                adjusted_scores[strategy_id] = max(0.0, min(1.0, adjusted_scores[strategy_id]))
            
            return adjusted_scores
            
        except Exception as e:
            logger.error(f"Error applying time-based adjustments: {str(e)}")
            return scores
    
    async def select_optimal_strategy(self) -> Dict:
        """
        Select the optimal trading strategy based on all relevant factors.
        
        Returns:
            Selected strategy details or None if none available
        """
        try:
            # Get all available strategies
            strategies = await self.get_available_strategies()
            
            if not strategies:
                logger.warning("No available strategies found")
                return None
            
            # Get current market regime
            current_regime = await self.market_regime_service.detect_current_regime()
            logger.info(f"Current market regime: {current_regime}")
            
            # Get regime-specific scores using market regime service
            regime_scores = {}
            for strategy in strategies:
                strategy_id = strategy['worker_id']
                
                # Calculate regime score using the market regime service's method
                # This leverages the existing regime-specific logic we already built
                if hasattr(self.market_regime_service, '_calculate_regime_score'):
                    metrics = await self.get_strategy_metrics(strategy_id)
                    regime_score = self.market_regime_service._calculate_regime_score(
                        metrics, current_regime)
                    regime_scores[strategy_id] = regime_score
                else:
                    # Fallback if method not available
                    regime_scores[strategy_id] = 0.5
            
            # Calculate factor-specific scores
            risk_scores = await self.calculate_risk_scores(strategies)
            performance_scores = await self.calculate_historical_performance_scores(strategies)
            social_scores = await self.calculate_social_sentiment_scores(strategies)
            volatility_scores = await self.calculate_volatility_scores(strategies)
            feature_importance_scores = await self.calculate_feature_importance_scores(strategies)
            
            # Combine all scores using selection weights
            combined_scores = {}
            
            for strategy in strategies:
                strategy_id = strategy['worker_id']
                
                # Combine scores with weights
                combined_score = (
                    regime_scores.get(strategy_id, 0.0) * self.selection_weights['market_regime'] +
                    performance_scores.get(strategy_id, 0.0) * self.selection_weights['historical_performance'] +
                    risk_scores.get(strategy_id, 0.0) * self.selection_weights['risk_profile'] +
                    social_scores.get(strategy_id, 0.0) * self.selection_weights['social_sentiment'] +
                    volatility_scores.get(strategy_id, 0.0) * self.selection_weights['market_volatility'] +
                    feature_importance_scores.get(strategy_id, 0.0) * self.selection_weights['feature_importance']
                )
                
                combined_scores[strategy_id] = combined_score
            
            # Apply time-based adjustments
            final_scores = await self.apply_time_based_adjustments(combined_scores)
            
            # Log all scores for transparency
            logger.debug("Strategy scores:")
            for strategy_id, score in final_scores.items():
                logger.debug(f"  {strategy_id}: {score:.4f}")
            
            # Select the highest-scoring strategy
            if not final_scores:
                logger.warning("No strategy scores available")
                return strategies[0]  # Return first available as fallback
            
            # Find strategy with highest score
            best_strategy_id = max(final_scores.items(), key=lambda x: x[1])[0]
            best_score = final_scores[best_strategy_id]
            
            # Find matching strategy object
            selected_strategy = next(
                (s for s in strategies if s['worker_id'] == best_strategy_id),
                None
            )
            
            if selected_strategy:
                # Store score with strategy
                selected_strategy['selection_score'] = best_score
                selected_strategy['selection_confidence'] = best_score  # Use score as confidence
                selected_strategy['market_regime'] = current_regime
                selected_strategy['selection_time'] = datetime.now().isoformat()
                selected_strategy['factor_scores'] = {
                    'market_regime': regime_scores.get(best_strategy_id, 0.0),
                    'historical_performance': performance_scores.get(best_strategy_id, 0.0),
                    'risk_profile': risk_scores.get(best_strategy_id, 0.0),
                    'social_sentiment': social_scores.get(best_strategy_id, 0.0),
                    'market_volatility': volatility_scores.get(best_strategy_id, 0.0),
                    'feature_importance': feature_importance_scores.get(best_strategy_id, 0.0)
                }
                
                logger.info(f"Selected strategy {best_strategy_id} with score {best_score:.4f}")
                return selected_strategy
            
            # Fallback to first strategy if best not found
            logger.warning(f"Could not find selected strategy with ID {best_strategy_id}")
            return strategies[0]
            
        except Exception as e:
            logger.error(f"Error selecting optimal strategy: {str(e)}")
            # Return the first available strategy as fallback
            if strategies:
                return strategies[0]
            return None
    
    async def should_switch_strategy(self, current_strategy_id: str, new_strategy: Dict) -> bool:
        """
        Determine if we should switch from current strategy to new strategy.
        
        Args:
            current_strategy_id: ID of currently active strategy
            new_strategy: Dictionary with new strategy details
            
        Returns:
            True if strategy should be switched, False otherwise
        """
        try:
            if not current_strategy_id or current_strategy_id != new_strategy['worker_id']:
                # Different strategy selected, check if improvement is significant
                new_score = new_strategy.get('selection_score', 0.0)
                new_confidence = new_strategy.get('selection_confidence', 0.0)
                
                # Get current strategy score
                current_score = 0.0
                if current_strategy_id:
                    current_strategy = await self.redis.get(f'strategy_selection_{current_strategy_id}')
                    if current_strategy:
                        current_strategy_data = json.loads(current_strategy)
                        current_score = current_strategy_data.get('selection_score', 0.0)
                
                # Calculate improvement
                improvement = new_score - current_score
                
                # Only switch if improvement is significant and confidence is high enough
                should_switch = (
                    improvement > self.min_improvement_threshold and 
                    new_confidence > self.min_confidence_threshold
                )
                
                if should_switch:
                    logger.info(f"Strategy switch recommended: {current_strategy_id} -> {new_strategy['worker_id']}")
                    logger.info(f"Improvement: {improvement:.4f}, Confidence: {new_confidence:.4f}")
                else:
                    if improvement <= self.min_improvement_threshold:
                        logger.debug(f"Improvement not significant enough: {improvement:.4f} <= {self.min_improvement_threshold}")
                    if new_confidence <= self.min_confidence_threshold:
                        logger.debug(f"Confidence not high enough: {new_confidence:.4f} <= {self.min_confidence_threshold}")
                
                return should_switch
            else:
                # Already using this strategy
                return False
            
        except Exception as e:
            logger.error(f"Error in should_switch_strategy: {str(e)}")
            return False
    
    async def switch_strategy(self, new_strategy: Dict) -> bool:
        """
        Switch to a new trading strategy.
        
        Args:
            new_strategy: Dictionary with new strategy details
            
        Returns:
            True if switch was successful, False otherwise
        """
        try:
            strategy_id = new_strategy['worker_id']
            logger.info(f"Switching to strategy {strategy_id}")
            
            # Get strategy parameters
            params_key = f'worker_parameters_{strategy_id}'
            params_data = await self.redis.get(params_key)
            
            if not params_data:
                logger.error(f"No parameters found for strategy {strategy_id}")
                return False
            
            params = json.loads(params_data)
            
            # Publish strategy parameters to Redis for other services
            await self.redis.set('strategy_params', json.dumps(params))
            
            # Send signal to strategy evolution service to reload parameters
            await self.redis.publish('strategy_update', 'reload')
            
            # Update active strategy
            self.active_strategy_id = strategy_id
            await self.redis.set('active_strategy_id', strategy_id)
            
            # Store strategy selection data
            await self.redis.set(f'strategy_selection_{strategy_id}', json.dumps(new_strategy))
            
            # Record the strategy switch in history
            switch_record = {
                'timestamp': datetime.now().isoformat(),
                'previous_strategy_id': self.active_strategy_id,
                'new_strategy_id': strategy_id,
                'selection_score': new_strategy.get('selection_score', 0.0),
                'factor_scores': new_strategy.get('factor_scores', {}),
                'market_regime': new_strategy.get('market_regime', 'unknown'),
                'reason': "Automated strategy selection"
            }
            
            # Store switch history
            self.strategy_history.append(switch_record)
            await self.redis.lpush('strategy_switch_history', json.dumps(switch_record))
            
            # Notify about strategy switch
            await self.redis.publish('strategy_switch', json.dumps({
                'timestamp': datetime.now().isoformat(),
                'strategy_id': strategy_id,
                'selection_score': new_strategy.get('selection_score', 0.0),
                'market_regime': new_strategy.get('market_regime', 'unknown'),
                'reason': "Automated strategy selection"
            }))
            
            logger.info(f"Successfully switched to strategy {strategy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching strategy: {str(e)}")
            return False
    
    async def update_risk_profile(self, profile: str) -> bool:
        """
        Update the current risk profile.
        
        Args:
            profile: New risk profile name ('conservative', 'moderate', 'aggressive')
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            if profile not in self.risk_profiles:
                logger.error(f"Invalid risk profile: {profile}")
                return False
            
            logger.info(f"Updating risk profile: {self.current_risk_profile} -> {profile}")
            self.current_risk_profile = profile
            
            # Store current risk profile in Redis
            await self.redis.set('current_risk_profile', profile)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating risk profile: {str(e)}")
            return False
    
    async def run(self):
        """Main service loop"""
        try:
            logger.info("Starting Strategy Selection Service...")
            
            # Load current active strategy
            active_strategy = await self.redis.get('active_strategy_id')
            if active_strategy:
                self.active_strategy_id = active_strategy
                logger.info(f"Current active strategy: {self.active_strategy_id}")
            
            # Load selection history
            history_data = await self.redis.lrange('strategy_switch_history', 0, 20)
            if history_data:
                for entry in history_data:
                    self.strategy_history.append(json.loads(entry))
                logger.info(f"Loaded {len(self.strategy_history)} history entries")
            
            # Main service loop
            while self.running:
                try:
                    # Select optimal strategy
                    optimal_strategy = await self.select_optimal_strategy()
                    
                    if optimal_strategy:
                        # Check if we should switch to the selected strategy
                        if await self.should_switch_strategy(self.active_strategy_id, optimal_strategy):
                            # Switch to the new strategy
                            success = await self.switch_strategy(optimal_strategy)
                            if success:
                                logger.info(f"Successfully switched to strategy {optimal_strategy['worker_id']}")
                            else:
                                logger.error(f"Failed to switch to strategy {optimal_strategy['worker_id']}")
                        else:
                            logger.debug(f"No strategy switch needed, keeping {self.active_strategy_id}")
                    
                    # Check for risk profile updates
                    risk_profile = await self.redis.get('current_risk_profile')
                    if risk_profile and risk_profile != self.current_risk_profile:
                        await self.update_risk_profile(risk_profile)
                    
                    # Ensure feature importance data is up-to-date
                    self.feature_importance.update_feature_importance_data()
                    
                    # Publish current selection metrics to Redis
                    if optimal_strategy:
                        # Get feature importance summary
                        feature_importance_summary = self.feature_importance.get_feature_importance_summary()
                        
                        await self.redis.set('strategy_selection_metrics', json.dumps({
                            'timestamp': datetime.now().isoformat(),
                            'active_strategy_id': self.active_strategy_id,
                            'optimal_strategy_id': optimal_strategy['worker_id'],
                            'optimal_score': optimal_strategy.get('selection_score', 0.0),
                            'optimal_confidence': optimal_strategy.get('selection_confidence', 0.0),
                            'market_regime': optimal_strategy.get('market_regime', 'unknown'),
                            'factor_scores': optimal_strategy.get('factor_scores', {}),
                            'feature_importance': feature_importance_summary
                        }))
                    
                    # Sleep until next check
                    await asyncio.sleep(self.check_interval)
                    
                except Exception as e:
                    logger.error(f"Error in main service loop: {str(e)}")
                    await asyncio.sleep(60)  # Sleep for a minute on error
            
        except Exception as e:
            logger.error(f"Critical error in Strategy Selection Service: {str(e)}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the strategy selection service"""
        logger.info("Stopping Strategy Selection Service...")
        self.running = False
        self.redis.close()

if __name__ == "__main__":
    service = StrategySelectionService()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
        asyncio.run(service.stop())
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        asyncio.run(service.stop())