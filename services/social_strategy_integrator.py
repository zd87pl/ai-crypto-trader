import os
import json
import redis
import asyncio
import logging as logger
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger.basicConfig(
    level=logger.INFO,
    format='%(asctime)s - %(levelname)s - [SocialStrategyIntegrator] %(message)s',
    handlers=[
        logger.FileHandler('logs/social_strategy.log'),
        logger.StreamHandler()
    ]
)

class SocialStrategyIntegrator:
    """
    Service for integrating social sentiment data into trading strategy evolution.
    This class provides mechanisms to:
    
    1. Analyze historical correlation between social metrics and price movements
    2. Generate social sensitivity factors for different assets and market conditions
    3. Provide optimized strategy parameters based on social sentiment data
    4. Adapt strategies to changing social sentiment patterns
    5. Backtest strategies with social sentiment as a key factor
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
        self.check_interval = int(os.getenv('SOCIAL_STRATEGY_CHECK_INTERVAL', '300'))  # Default: check every 5 minutes
        
        # Social sentiment configuration
        self.sentiment_thresholds = {
            'very_negative': -0.7,
            'negative': -0.3,
            'neutral': 0.3,
            'positive': 0.7,
            'very_positive': 0.9
        }
        
        # Social volume and engagement thresholds
        self.volume_thresholds = {
            'low': 1000,
            'medium': 5000,
            'high': 20000,
            'very_high': 50000
        }
        
        self.engagement_thresholds = {
            'low': 500,
            'medium': 2000,
            'high': 10000,
            'very_high': 30000
        }
        
        # Strategy parameters affected by social metrics
        self.social_impacted_parameters = {
            'entry_threshold': {
                'positive_sentiment_effect': 0.9,  # Lower entry threshold when sentiment is positive
                'negative_sentiment_effect': 1.1   # Raise entry threshold when sentiment is negative
            },
            'position_size': {
                'positive_sentiment_effect': 1.2,  # Increase position size when sentiment is positive
                'negative_sentiment_effect': 0.8   # Decrease position size when sentiment is negative
            },
            'take_profit': {
                'positive_sentiment_effect': 1.1,  # Higher take profit when sentiment is positive
                'negative_sentiment_effect': 0.9   # Lower take profit when sentiment is negative
            },
            'stop_loss': {
                'positive_sentiment_effect': 0.9,  # Tighter stop loss when sentiment is positive
                'negative_sentiment_effect': 1.1   # Wider stop loss when sentiment is negative
            },
            'holding_period': {
                'positive_sentiment_effect': 1.2,  # Longer holding period when sentiment is positive
                'negative_sentiment_effect': 0.8   # Shorter holding period when sentiment is negative
            }
        }
        
        # Asset-specific social sensitivity - will be learned over time
        self.asset_social_sensitivity = {}
        
        # Correlation tracking - maps assets to their social-price correlation
        self.social_price_correlation = {}
        
        # Social sentiment strategies
        self.social_strategies = {
            'trend_following': {
                'description': 'Follows the social sentiment trend',
                'parameters': {
                    'sentiment_threshold': 0.5,
                    'volume_threshold': 5000,
                    'engagement_threshold': 2000,
                    'sentiment_lookback': 24,  # hours
                    'entry_weight': 0.6,       # How much sentiment affects entry
                    'exit_weight': 0.4         # How much sentiment affects exit
                }
            },
            'contrarian': {
                'description': 'Takes positions contrary to extreme social sentiment',
                'parameters': {
                    'sentiment_threshold': 0.8,  # Only act on extreme sentiment
                    'volume_threshold': 10000,
                    'engagement_threshold': 5000,
                    'sentiment_lookback': 12,    # hours
                    'entry_weight': 0.7,
                    'exit_weight': 0.5
                }
            },
            'news_reactive': {
                'description': 'Reacts quickly to news sentiment changes',
                'parameters': {
                    'sentiment_threshold': 0.3,  # Lower threshold to be more reactive
                    'volume_threshold': 3000,
                    'engagement_threshold': 1500,
                    'sentiment_lookback': 6,     # hours
                    'entry_weight': 0.8,
                    'exit_weight': 0.7
                }
            },
            'volume_driven': {
                'description': 'Focuses on social volume rather than sentiment',
                'parameters': {
                    'sentiment_threshold': 0.2,   # Low sentiment threshold
                    'volume_threshold': 15000,    # High volume threshold
                    'engagement_threshold': 7500,
                    'sentiment_lookback': 48,     # hours
                    'entry_weight': 0.5,
                    'exit_weight': 0.3
                }
            },
            'engagement_focused': {
                'description': 'Focuses on social engagement metrics',
                'parameters': {
                    'sentiment_threshold': 0.4,
                    'volume_threshold': 3000,
                    'engagement_threshold': 10000,  # High engagement threshold
                    'sentiment_lookback': 36,       # hours
                    'entry_weight': 0.6,
                    'exit_weight': 0.5
                }
            }
        }
        
        # Historical data cache
        self.historical_data_cache = {}
        self.cache_expiry = 60 * 60  # 1 hour in seconds
        
        logger.info("Social Strategy Integrator initialized")
        
    async def get_social_metrics(self, symbol: str) -> Dict:
        """
        Get the latest social metrics for a symbol from Redis.
        
        Args:
            symbol: The crypto asset symbol
            
        Returns:
            Dictionary of social metrics or empty dict if not found
        """
        try:
            social_data = await self.redis.hget('social_metrics', symbol)
            if social_data:
                return json.loads(social_data)
            return {}
        except Exception as e:
            logger.error(f"Error getting social metrics for {symbol}: {str(e)}")
            return {}
    
    async def get_historical_social_data(self, symbol: str, lookback_hours: int = 24) -> List[Dict]:
        """
        Get historical social data for a symbol.
        
        Args:
            symbol: The crypto asset symbol
            lookback_hours: How many hours to look back
            
        Returns:
            List of historical social metrics
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{lookback_hours}"
            current_time = datetime.now().timestamp()
            
            if cache_key in self.historical_data_cache:
                cache_entry = self.historical_data_cache[cache_key]
                if current_time - cache_entry['timestamp'] < self.cache_expiry:
                    return cache_entry['data']
            
            # Get historical data from Redis
            history_key = f'social_history:{symbol}'
            history_data = await self.redis.lrange(history_key, 0, lookback_hours - 1)
            
            if not history_data:
                logger.warning(f"No historical social data found for {symbol}")
                return []
            
            # Parse JSON data
            social_history = [json.loads(item) for item in history_data]
            
            # Sort by timestamp
            social_history.sort(key=lambda x: x.get('timestamp', ''))
            
            # Update cache
            self.historical_data_cache[cache_key] = {
                'timestamp': current_time,
                'data': social_history
            }
            
            return social_history
            
        except Exception as e:
            logger.error(f"Error getting historical social data for {symbol}: {str(e)}")
            return []
    
    async def calculate_social_price_correlation(self, symbol: str, lookback_days: int = 30) -> float:
        """
        Calculate correlation between social sentiment and price movements.
        
        Args:
            symbol: The crypto asset symbol
            lookback_days: How many days to analyze
            
        Returns:
            Correlation coefficient [-1 to 1]
        """
        try:
            # Get historical price data
            price_history_key = f'price_history:{symbol}'
            price_data = await self.redis.lrange(price_history_key, 0, lookback_days * 24 - 1)
            
            if not price_data:
                logger.warning(f"No price history found for {symbol}")
                return 0
            
            # Get historical social data
            social_history = await self.get_historical_social_data(symbol, lookback_days * 24)
            
            if not social_history:
                logger.warning(f"No social history found for {symbol}")
                return 0
            
            # Convert to DataFrame for analysis
            price_df = pd.DataFrame([json.loads(item) for item in price_data])
            social_df = pd.DataFrame(social_history)
            
            # Ensure we have timestamps
            if 'timestamp' not in price_df.columns or 'timestamp' not in social_df.columns:
                logger.warning(f"Missing timestamp data for correlation analysis")
                return 0
            
            # Convert timestamps to datetime
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
            social_df['timestamp'] = pd.to_datetime(social_df['timestamp'])
            
            # Sort by timestamp
            price_df = price_df.sort_values('timestamp')
            social_df = social_df.sort_values('timestamp')
            
            # Resample to hourly data to align the datasets
            if 'close' in price_df.columns and 'metrics' in social_df.columns:
                price_df.set_index('timestamp', inplace=True)
                social_df.set_index('timestamp', inplace=True)
                
                # Extract sentiment from nested structure
                social_df['sentiment'] = social_df['metrics'].apply(
                    lambda x: x.get('social_sentiment', 0) if isinstance(x, dict) else 0
                )
                
                # Resample both to hourly data
                price_hourly = price_df['close'].resample('1H').last()
                sentiment_hourly = social_df['sentiment'].resample('1H').mean()
                
                # Align the datasets
                aligned_data = pd.concat([price_hourly, sentiment_hourly], axis=1).dropna()
                
                if len(aligned_data) > 5:  # Need enough data points for meaningful correlation
                    # Calculate correlation
                    correlation = aligned_data['close'].pct_change().corr(aligned_data['sentiment'])
                    
                    # Store the correlation
                    self.social_price_correlation[symbol] = correlation
                    
                    logger.info(f"Social-price correlation for {symbol}: {correlation:.4f}")
                    return correlation
            
            logger.warning(f"Insufficient data for correlation analysis")
            return 0
            
        except Exception as e:
            logger.error(f"Error calculating social-price correlation: {str(e)}")
            return 0
    
    async def get_social_strategy_parameters(self, strategy_type: str, 
                                          symbol: str, base_params: Dict) -> Dict:
        """
        Get social strategy parameters adjusted for a specific asset.
        
        Args:
            strategy_type: Type of social strategy ('trend_following', 'contrarian', etc.)
            symbol: The crypto asset symbol
            base_params: Base trading parameters to adjust
            
        Returns:
            Adjusted parameters dictionary
        """
        try:
            # Get the social strategy configuration
            if strategy_type not in self.social_strategies:
                logger.warning(f"Unknown social strategy type: {strategy_type}")
                strategy_type = 'trend_following'  # Default
            
            strategy_config = self.social_strategies[strategy_type]
            
            # Get current social metrics
            social_metrics = await self.get_social_metrics(symbol)
            if not social_metrics or 'metrics' not in social_metrics:
                logger.warning(f"No social metrics available for {symbol}")
                return base_params
            
            metrics = social_metrics['metrics']
            
            # Extract the key metrics
            sentiment = metrics.get('social_sentiment', 0)
            volume = metrics.get('social_volume', 0)
            engagement = metrics.get('social_engagement', 0)
            
            # Get correlation coefficient (or calculate if not available)
            correlation = self.social_price_correlation.get(symbol)
            if correlation is None:
                correlation = await self.calculate_social_price_correlation(symbol)
            
            # Skip adjustment if correlation is weak
            if abs(correlation) < 0.2:
                logger.info(f"Weak social-price correlation for {symbol}: {correlation:.4f}, skipping adjustment")
                return base_params
            
            # Determine sentiment effect
            sentiment_effect = 'neutral'
            if sentiment > self.sentiment_thresholds['positive']:
                sentiment_effect = 'positive_sentiment_effect'
            elif sentiment < self.sentiment_thresholds['negative']:
                sentiment_effect = 'negative_sentiment_effect'
            
            # Adjust parameters based on sentiment effect and correlation strength
            adjusted_params = base_params.copy()
            correlation_strength = abs(correlation)
            
            # Only apply adjustments if we have decent correlation
            if correlation_strength >= 0.2:
                for param_name, effects in self.social_impacted_parameters.items():
                    if param_name in adjusted_params and sentiment_effect in effects:
                        # Apply the effect scaled by correlation strength
                        adjustment_factor = effects[sentiment_effect]
                        
                        # Scale adjustment based on correlation strength
                        scaled_adjustment = 1.0 + ((adjustment_factor - 1.0) * correlation_strength)
                        
                        # Apply adjustment
                        adjusted_params[param_name] *= scaled_adjustment
                
                logger.info(f"Applied social sentiment adjustments for {symbol} with correlation {correlation:.4f}")
            
            return adjusted_params
            
        except Exception as e:
            logger.error(f"Error getting social strategy parameters: {str(e)}")
            return base_params
    
    async def analyze_social_sentiment_impact(self, symbol: str, lookback_days: int = 30) -> Dict:
        """
        Analyze the historical impact of social sentiment on price movements.
        
        Args:
            symbol: The crypto asset symbol
            lookback_days: Number of days to analyze
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Get historical price data
            price_history_key = f'price_history:{symbol}'
            price_data = await self.redis.lrange(price_history_key, 0, lookback_days * 24 - 1)
            
            if not price_data:
                logger.warning(f"No price history found for {symbol}")
                return {'error': 'No price history found'}
            
            # Get historical social data
            social_history = await self.get_historical_social_data(symbol, lookback_days * 24)
            
            if not social_history:
                logger.warning(f"No social history found for {symbol}")
                return {'error': 'No social history found'}
            
            # Convert to DataFrame for analysis
            price_df = pd.DataFrame([json.loads(item) for item in price_data])
            social_df = pd.DataFrame(social_history)
            
            # Ensure we have timestamps
            if 'timestamp' not in price_df.columns or 'timestamp' not in social_df.columns:
                logger.warning(f"Missing timestamp data for impact analysis")
                return {'error': 'Missing timestamp data'}
            
            # Convert timestamps to datetime
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
            social_df['timestamp'] = pd.to_datetime(social_df['timestamp'])
            
            # Extract sentiment from nested structure
            social_df['sentiment'] = social_df.apply(
                lambda row: row['metrics'].get('social_sentiment', 0) 
                if isinstance(row.get('metrics'), dict) else 0,
                axis=1
            )
            
            # Calculate price changes (future returns)
            price_df['next_1h_return'] = price_df['close'].pct_change(1).shift(-1)
            price_df['next_4h_return'] = price_df['close'].pct_change(4).shift(-4)
            price_df['next_24h_return'] = price_df['close'].pct_change(24).shift(-24)
            
            # Merge the dataframes
            merged_df = pd.merge_asof(
                price_df.sort_values('timestamp'),
                social_df[['timestamp', 'sentiment']].sort_values('timestamp'),
                on='timestamp',
                direction='nearest'
            )
            
            # Drop rows with NaN values
            merged_df = merged_df.dropna(subset=['sentiment', 'next_1h_return', 'next_4h_return', 'next_24h_return'])
            
            # Calculate correlations
            correlation_1h = merged_df['sentiment'].corr(merged_df['next_1h_return'])
            correlation_4h = merged_df['sentiment'].corr(merged_df['next_4h_return'])
            correlation_24h = merged_df['sentiment'].corr(merged_df['next_24h_return'])
            
            # Calculate returns by sentiment category
            sentiment_categories = []
            for threshold_name, threshold_value in self.sentiment_thresholds.items():
                if threshold_name == 'very_negative':
                    sentiment_categories.append((threshold_name, merged_df['sentiment'] <= threshold_value))
                elif threshold_name == 'very_positive':
                    sentiment_categories.append((threshold_name, merged_df['sentiment'] > threshold_value))
                else:
                    # Find the next threshold value
                    thresholds = list(self.sentiment_thresholds.items())
                    for i, (name, value) in enumerate(thresholds):
                        if name == threshold_name and i < len(thresholds) - 1:
                            next_value = thresholds[i+1][1]
                            sentiment_categories.append(
                                (threshold_name, 
                                 (merged_df['sentiment'] > value) & (merged_df['sentiment'] <= next_value))
                            )
            
            returns_by_sentiment = {}
            for category_name, category_mask in sentiment_categories:
                if category_mask.sum() > 0:  # If we have data points in this category
                    returns_by_sentiment[category_name] = {
                        '1h': merged_df.loc[category_mask, 'next_1h_return'].mean() * 100,
                        '4h': merged_df.loc[category_mask, 'next_4h_return'].mean() * 100,
                        '24h': merged_df.loc[category_mask, 'next_24h_return'].mean() * 100,
                        'count': int(category_mask.sum())
                    }
            
            # Determine which timeframe has the strongest correlation
            timeframes = [
                ('1h', abs(correlation_1h)),
                ('4h', abs(correlation_4h)),
                ('24h', abs(correlation_24h))
            ]
            strongest_timeframe = max(timeframes, key=lambda x: x[1])
            
            # Calculate sentiment lead/lag
            lead_lag_correlations = []
            for lag in range(-24, 25):  # -24 hours to +24 hours
                if lag < 0:
                    # Sentiment lags price (price leads)
                    lagged_df = pd.merge_asof(
                        social_df[['timestamp', 'sentiment']].sort_values('timestamp'),
                        price_df[['timestamp', 'close']].sort_values('timestamp'),
                        on='timestamp',
                        direction='nearest'
                    )
                    lagged_df['return'] = lagged_df['close'].pct_change(abs(lag))
                    corr = lagged_df['sentiment'].corr(lagged_df['return'])
                else:
                    # Sentiment leads price
                    lagged_df = pd.merge_asof(
                        price_df[['timestamp', 'close']].sort_values('timestamp'),
                        social_df[['timestamp', 'sentiment']].sort_values('timestamp'),
                        on='timestamp',
                        direction='nearest'
                    )
                    lagged_df['future_return'] = lagged_df['close'].pct_change(lag).shift(-lag)
                    corr = lagged_df['sentiment'].corr(lagged_df['future_return'])
                
                lead_lag_correlations.append((lag, corr))
            
            # Find the lag with the strongest correlation
            max_lag = max(lead_lag_correlations, key=lambda x: abs(x[1]))
            
            # Determine if sentiment leads or lags price
            if max_lag[0] < 0:
                lead_lag_relationship = f"Price leads sentiment by {abs(max_lag[0])} hours"
            elif max_lag[0] > 0:
                lead_lag_relationship = f"Sentiment leads price by {max_lag[0]} hours"
            else:
                lead_lag_relationship = "No lead/lag relationship detected"
            
            # Prepare results
            analysis_results = {
                'symbol': symbol,
                'correlations': {
                    '1h': correlation_1h,
                    '4h': correlation_4h,
                    '24h': correlation_24h
                },
                'strongest_timeframe': {
                    'timeframe': strongest_timeframe[0],
                    'correlation': strongest_timeframe[1]
                },
                'returns_by_sentiment': returns_by_sentiment,
                'lead_lag_relationship': lead_lag_relationship,
                'optimal_lag': max_lag[0],
                'optimal_lag_correlation': max_lag[1],
                'data_points': len(merged_df),
                'analysis_period': f"{lookback_days} days"
            }
            
            # Store the analysis in Redis for future use
            await self.redis.set(
                f'social_impact_analysis:{symbol}',
                json.dumps(analysis_results)
            )
            
            logger.info(f"Completed social sentiment impact analysis for {symbol}")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing social sentiment impact: {str(e)}")
            return {'error': str(e)}
    
    async def generate_social_trading_strategy(self, symbol: str) -> Dict:
        """
        Generate a complete trading strategy optimized for social sentiment.
        
        Args:
            symbol: The crypto asset symbol
            
        Returns:
            Strategy configuration dictionary
        """
        try:
            # First analyze social sentiment impact
            impact_analysis = await self.analyze_social_sentiment_impact(symbol)
            
            if 'error' in impact_analysis:
                logger.warning(f"Cannot generate social strategy: {impact_analysis['error']}")
                return {'error': impact_analysis['error']}
            
            # Determine the best strategy type based on analysis
            best_strategy_type = 'trend_following'  # Default
            
            # If correlation is strong and positive
            correlation_24h = impact_analysis['correlations']['24h']
            if abs(correlation_24h) > 0.4:
                if correlation_24h > 0:
                    # Positive correlation - sentiment moves with price
                    best_strategy_type = 'trend_following'
                else:
                    # Negative correlation - sentiment moves against price
                    best_strategy_type = 'contrarian'
            
            # If sentiment leads price by several hours, use news_reactive
            if impact_analysis['optimal_lag'] > 3 and impact_analysis['optimal_lag_correlation'] > 0.3:
                best_strategy_type = 'news_reactive'
            
            # Get the base strategy configuration
            base_strategy = self.social_strategies[best_strategy_type].copy()
            strategy_params = base_strategy['parameters'].copy()
            
            # Optimize parameters based on analysis
            # Set sentiment threshold based on which sentiment category has best returns
            best_sentiment_category = None
            best_return = -float('inf')
            
            for category, returns in impact_analysis['returns_by_sentiment'].items():
                if returns['count'] >= 5:  # Need enough data points
                    if returns['24h'] > best_return:
                        best_return = returns['24h']
                        best_sentiment_category = category
            
            if best_sentiment_category:
                # Set sentiment threshold based on best performing category
                if best_sentiment_category in ['positive', 'very_positive']:
                    strategy_params['sentiment_threshold'] = self.sentiment_thresholds['positive']
                elif best_sentiment_category in ['negative', 'very_negative']:
                    strategy_params['sentiment_threshold'] = self.sentiment_thresholds['negative']
            
            # Set lookback period based on optimal lag
            optimal_lag = abs(impact_analysis['optimal_lag'])
            if optimal_lag > 0:
                strategy_params['sentiment_lookback'] = max(6, optimal_lag * 2)  # At least 6 hours
            
            # Adjust entry and exit weights based on correlation strength
            correlation_strength = abs(impact_analysis['strongest_timeframe']['correlation'])
            if correlation_strength > 0.3:
                strategy_params['entry_weight'] = min(0.8, 0.4 + correlation_strength)
                strategy_params['exit_weight'] = min(0.7, 0.3 + correlation_strength)
            else:
                # Weak correlation - reduce influence of social metrics
                strategy_params['entry_weight'] = 0.3
                strategy_params['exit_weight'] = 0.2
            
            # Generate the complete strategy
            strategy = {
                'symbol': symbol,
                'strategy_type': best_strategy_type,
                'description': base_strategy['description'],
                'parameters': strategy_params,
                'impact_analysis': {
                    'correlation': impact_analysis['strongest_timeframe']['correlation'],
                    'timeframe': impact_analysis['strongest_timeframe']['timeframe'],
                    'lead_lag': impact_analysis['lead_lag_relationship']
                },
                'generation_time': datetime.now().isoformat()
            }
            
            # Store in Redis
            await self.redis.set(
                f'social_strategy:{symbol}',
                json.dumps(strategy)
            )
            
            logger.info(f"Generated {best_strategy_type} social strategy for {symbol}")
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating social trading strategy: {str(e)}")
            return {'error': str(e)}
    
    async def run(self):
        """Main service loop"""
        try:
            logger.info("Starting Social Strategy Integrator service...")
            
            # Main service loop
            while self.running:
                try:
                    # Get all symbols with social data
                    social_symbols = await self.redis.hkeys('social_metrics')
                    
                    for symbol in social_symbols:
                        try:
                            # Calculate correlation if we haven't already
                            if symbol not in self.social_price_correlation:
                                correlation = await self.calculate_social_price_correlation(symbol)
                                logger.info(f"Calculated initial correlation for {symbol}: {correlation:.4f}")
                            
                            # Generate/update social strategy if sufficient data exists
                            strategy = await self.redis.get(f'social_strategy:{symbol}')
                            
                            if not strategy or self.should_update_strategy(strategy):
                                logger.info(f"Generating social strategy for {symbol}")
                                await self.generate_social_trading_strategy(symbol)
                                
                        except Exception as e:
                            logger.error(f"Error processing {symbol}: {str(e)}")
                    
                    # Sleep until next check
                    await asyncio.sleep(self.check_interval)
                    
                except Exception as e:
                    logger.error(f"Error in main service loop: {str(e)}")
                    await asyncio.sleep(60)  # Sleep for a minute on error
            
        except Exception as e:
            logger.error(f"Critical error in Social Strategy Integrator: {str(e)}")
        finally:
            await self.stop()
    
    def should_update_strategy(self, strategy_json: str) -> bool:
        """
        Determine if a strategy should be updated based on its age.
        
        Args:
            strategy_json: Strategy JSON string
            
        Returns:
            True if strategy should be updated, False otherwise
        """
        try:
            strategy = json.loads(strategy_json)
            generation_time = datetime.fromisoformat(strategy.get('generation_time', '2000-01-01'))
            age_hours = (datetime.now() - generation_time).total_seconds() / 3600
            
            # Update every 24 hours
            return age_hours >= 24
            
        except Exception:
            # If any error occurs, assume we should update
            return True
    
    async def stop(self):
        """Stop the service"""
        logger.info("Stopping Social Strategy Integrator...")
        self.running = False
        self.redis.close()

if __name__ == "__main__":
    service = SocialStrategyIntegrator()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
        asyncio.run(service.stop())
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        asyncio.run(service.stop())