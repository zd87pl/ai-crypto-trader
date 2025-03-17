import os
import json
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging as logger
from typing import Dict, List, Optional, Tuple, Union
from redis.asyncio import Redis
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

class SocialMetricsAnalyzer:
    """
    Enhanced social metrics analyzer for improving the accuracy of social data analysis
    
    This class provides:
    1. Advanced social metrics normalization
    2. Time-weighted sentiment calculation
    3. Anomaly detection for identifying outliers
    4. Multi-source sentiment aggregation
    5. Lead-lag relationship detection
    6. Accuracy assessment and monitoring
    7. Adaptive weights based on predictive power
    """
    
    def __init__(self, redis: Redis = None):
        """
        Initialize the social metrics analyzer
        
        Args:
            redis: Redis client instance (optional)
        """
        self.redis = redis
        
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Default weights for different sources
        self.source_weights = {
            'twitter': 0.35,
            'reddit': 0.30,
            'news': 0.25,
            'telegram': 0.10
        }
        
        # Decay constants for time weighting
        self.decay_constants = {
            'very_short': 1.0,  # 1 hour half-life
            'short': 0.5,       # 2 hour half-life
            'medium': 0.25,     # 4 hour half-life
            'long': 0.125,      # 8 hour half-life
            'very_long': 0.0625 # 16 hour half-life
        }
        
        # Historical metrics cache
        self.historical_metrics = {}
        
        # Accuracy metrics tracking
        self.accuracy_metrics = defaultdict(dict)
        
        # Adaptive weights based on historical predictive power
        self.adaptive_weights = {}
        
        # Outlier thresholds
        self.anomaly_threshold = 3.0  # Standard deviations
        self.anomaly_models = {}
        
        logger.info("Social Metrics Analyzer initialized")
    
    def normalize_metrics(self, metrics: Dict) -> Dict:
        """
        Apply advanced normalization to social metrics
        
        Args:
            metrics: Raw social metrics dictionary
            
        Returns:
            Normalized metrics dictionary
        """
        normalized = {}
        
        # Define expected ranges for each metric type
        ranges = {
            'volume': (0, 50000),       # Social volume typically 0-50K
            'engagement': (0, 25000),   # Engagement typically 0-25K
            'contributors': (0, 5000),  # Contributors typically 0-5K
            'sentiment': (-1, 1)        # Sentiment typically -1 to 1
        }
        
        for key, value in metrics.items():
            # Determine the metric type based on the key name
            metric_type = None
            for t in ranges.keys():
                if t in key.lower():
                    metric_type = t
                    break
            
            if metric_type:
                # Get the range for this metric type
                min_val, max_val = ranges[metric_type]
                
                # Normalize to 0-1 range
                if max_val > min_val:
                    normalized[key] = min(1.0, max(0.0, (value - min_val) / (max_val - min_val)))
                else:
                    normalized[key] = 0.0
            else:
                # Default normalization if type not recognized
                normalized[key] = value
        
        return normalized
    
    def calculate_time_weighted_sentiment(self, sentiment_history: List[Dict], 
                                         decay_mode: str = 'medium') -> float:
        """
        Calculate time-weighted sentiment score
        
        Args:
            sentiment_history: List of sentiment data points with timestamps
            decay_mode: Decay rate to use ('very_short', 'short', 'medium', 'long', 'very_long')
            
        Returns:
            Time-weighted sentiment score
        """
        if not sentiment_history:
            return 0.5  # Neutral default
        
        # Sort by timestamp (newest first)
        sorted_history = sorted(
            sentiment_history, 
            key=lambda x: datetime.fromisoformat(x.get('timestamp', '2000-01-01T00:00:00')),
            reverse=True
        )
        
        # Get decay constant
        decay_constant = self.decay_constants.get(decay_mode, self.decay_constants['medium'])
        
        # Calculate weighted sum
        now = datetime.now()
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for entry in sorted_history:
            # Get sentiment value
            sentiment = entry.get('sentiment', 0.5)
            
            # Get timestamp
            try:
                timestamp = datetime.fromisoformat(entry.get('timestamp', now.isoformat()))
            except ValueError:
                timestamp = now
            
            # Calculate time difference in hours
            time_diff = (now - timestamp).total_seconds() / 3600.0
            
            # Calculate weight using exponential decay
            weight = np.exp(-decay_constant * time_diff)
            
            # Add to weighted sum
            weighted_sum += sentiment * weight
            weight_sum += weight
        
        # Return weighted average
        if weight_sum > 0:
            return weighted_sum / weight_sum
        else:
            return 0.5  # Neutral default
    
    def detect_anomalies(self, symbol: str, metrics: Dict) -> Tuple[Dict, bool]:
        """
        Detect and filter anomalies in social metrics
        
        Args:
            symbol: Asset symbol
            metrics: Social metrics dictionary
            
        Returns:
            Tuple of (filtered_metrics, is_anomaly)
        """
        # Create a DataFrame for the current metrics
        df = pd.DataFrame([metrics])
        
        # Check if we already have an anomaly detection model for this symbol
        if symbol in self.anomaly_models:
            # Use existing model
            model = self.anomaly_models[symbol]
            
            # Predict anomaly
            try:
                # Get only numeric features
                numeric_data = df.select_dtypes(include=['number'])
                if not numeric_data.empty:
                    # Apply model
                    predictions = model.predict(numeric_data)
                    is_anomaly = predictions[0] == -1  # -1 means anomaly
                    
                    if is_anomaly:
                        logger.warning(f"Anomaly detected in {symbol} social metrics")
                        
                        # Filter the metrics by replacing anomalous values with None
                        filtered_metrics = metrics.copy()
                        scores = model.decision_function(numeric_data)
                        
                        # Identify the most anomalous features
                        for i, col in enumerate(numeric_data.columns):
                            if abs(scores[0]) > self.anomaly_threshold and col in filtered_metrics:
                                logger.debug(f"Anomalous value detected for {col}: {filtered_metrics[col]}")
                                filtered_metrics[col] = None
                        
                        return filtered_metrics, True
            except Exception as e:
                logger.error(f"Error detecting anomalies: {str(e)}")
        
        # Return original metrics if no anomaly detected or error
        return metrics, False
    
    async def train_anomaly_model(self, symbol: str, history_length: int = 100) -> bool:
        """
        Train an anomaly detection model for a symbol
        
        Args:
            symbol: Asset symbol
            history_length: Number of historical data points to use
            
        Returns:
            True if model was trained successfully, False otherwise
        """
        try:
            # Get historical data
            if self.redis:
                history_key = f'social_history:{symbol}'
                history_data = await self.redis.lrange(history_key, 0, history_length - 1)
                
                if not history_data or len(history_data) < 10:  # Need at least 10 data points
                    logger.warning(f"Not enough historical data for {symbol} to train anomaly model")
                    return False
                
                # Parse data
                history = [json.loads(item) for item in history_data]
                
                # Extract metrics
                metrics_list = []
                for item in history:
                    if 'metrics' in item and isinstance(item['metrics'], dict):
                        metrics_list.append(item['metrics'])
                
                if len(metrics_list) < 10:
                    logger.warning(f"Not enough valid metrics for {symbol} to train anomaly model")
                    return False
                
                # Create DataFrame
                df = pd.DataFrame(metrics_list)
                
                # Keep only numeric columns
                numeric_df = df.select_dtypes(include=['number'])
                
                if numeric_df.empty:
                    logger.warning(f"No numeric metrics found for {symbol}")
                    return False
                
                # Train isolation forest model
                model = IsolationForest(
                    n_estimators=100,
                    max_samples='auto',
                    contamination=0.05,  # Assume 5% of data are anomalies
                    random_state=42
                )
                
                model.fit(numeric_df)
                
                # Store model
                self.anomaly_models[symbol] = model
                
                logger.info(f"Trained anomaly detection model for {symbol} with {len(numeric_df)} data points")
                return True
                
            else:
                logger.warning("Redis client not available, cannot train anomaly model")
                return False
                
        except Exception as e:
            logger.error(f"Error training anomaly model: {str(e)}")
            return False
    
    def aggregate_sentiment(self, sentiments: Dict[str, float]) -> float:
        """
        Aggregate sentiment from multiple sources with weighted average
        
        Args:
            sentiments: Dictionary mapping source names to sentiment values
            
        Returns:
            Aggregated sentiment value (0-1)
        """
        if not sentiments:
            return 0.5  # Neutral default
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for source, sentiment in sentiments.items():
            # Get weight for this source
            weight = self.source_weights.get(source, 0.25)  # Default weight 0.25
            
            # Add to weighted sum
            weighted_sum += sentiment * weight
            weight_sum += weight
        
        # Return weighted average
        if weight_sum > 0:
            return weighted_sum / weight_sum
        else:
            return 0.5  # Neutral default
    
    async def detect_lead_lag_relationship(self, symbol: str, lookback_days: int = 30) -> Dict:
        """
        Detect lead-lag relationship between social metrics and price
        
        Args:
            symbol: Asset symbol
            lookback_days: Number of days to analyze
            
        Returns:
            Dictionary with lead-lag analysis results
        """
        try:
            if self.redis:
                # Get price history
                price_history_key = f'price_history:{symbol}'
                price_data = await self.redis.lrange(price_history_key, 0, lookback_days * 24 - 1)
                
                # Get social history
                social_history_key = f'social_history:{symbol}'
                social_data = await self.redis.lrange(social_history_key, 0, lookback_days * 24 - 1)
                
                if not price_data or not social_data:
                    return {'error': 'Insufficient historical data'}
                
                # Parse data
                price_history = [json.loads(item) for item in price_data]
                social_history = [json.loads(item) for item in social_data]
                
                # Convert to DataFrames
                price_df = pd.DataFrame(price_history)
                social_df = pd.DataFrame(social_history)
                
                # Ensure we have timestamps
                if 'timestamp' not in price_df.columns or 'timestamp' not in social_df.columns:
                    return {'error': 'Missing timestamp data'}
                
                # Convert timestamps to datetime
                price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
                social_df['timestamp'] = pd.to_datetime(social_df['timestamp'])
                
                # Sort by timestamp
                price_df = price_df.sort_values('timestamp')
                social_df = social_df.sort_values('timestamp')
                
                # Extract sentiment and price data
                if 'metrics' in social_df.columns and 'close' in price_df.columns:
                    # Extract sentiment
                    social_df['sentiment'] = social_df['metrics'].apply(
                        lambda x: x.get('social_sentiment', 0.5) if isinstance(x, dict) else 0.5
                    )
                    
                    # Create price returns
                    price_df['returns'] = price_df['close'].pct_change()
                    
                    # Test different lags from -24 to +24 hours
                    results = []
                    
                    for lag in range(-24, 25):
                        # For negative lags, price leads sentiment
                        # For positive lags, sentiment leads price
                        
                        if lag < 0:
                            # Price leading sentiment
                            lagged_returns = price_df['returns'].shift(abs(lag))
                            merged = pd.merge_asof(
                                social_df[['timestamp', 'sentiment']],
                                price_df[['timestamp', 'returns']].assign(returns=lagged_returns),
                                on='timestamp'
                            )
                        else:
                            # Sentiment leading price
                            merged = pd.merge_asof(
                                price_df[['timestamp', 'returns']],
                                social_df[['timestamp', 'sentiment']],
                                on='timestamp'
                            )
                            # Shift returns to align with lagged sentiment
                            merged['returns'] = merged['returns'].shift(-lag)
                        
                        # Drop NaN values
                        merged = merged.dropna()
                        
                        # Calculate correlation if we have enough data
                        if len(merged) >= 5:
                            pearson, p_value = pearsonr(merged['sentiment'], merged['returns'])
                            spearman, sp_value = spearmanr(merged['sentiment'], merged['returns'])
                            
                            results.append({
                                'lag': lag,
                                'pearson': pearson,
                                'pearson_p': p_value,
                                'spearman': spearman,
                                'spearman_p': sp_value,
                                'data_points': len(merged)
                            })
                    
                    # Find the lag with the strongest correlation (by absolute value)
                    if results:
                        # Sort by absolute Pearson correlation
                        sorted_results = sorted(results, key=lambda x: abs(x['pearson']), reverse=True)
                        strongest_pearson = sorted_results[0]
                        
                        # Sort by absolute Spearman correlation
                        sorted_results = sorted(results, key=lambda x: abs(x['spearman']), reverse=True)
                        strongest_spearman = sorted_results[0]
                        
                        # Create the result dictionary
                        lead_lag_result = {
                            'symbol': symbol,
                            'strongest_pearson_lag': strongest_pearson['lag'],
                            'strongest_pearson_corr': strongest_pearson['pearson'],
                            'strongest_pearson_p': strongest_pearson['pearson_p'],
                            'strongest_spearman_lag': strongest_spearman['lag'],
                            'strongest_spearman_corr': strongest_spearman['spearman'],
                            'strongest_spearman_p': strongest_spearman['spearman_p'],
                            'data_points': strongest_pearson['data_points'],
                            'analysis_period': f"{lookback_days} days",
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Store the result in Redis
                        await self.redis.set(
                            f'social_lead_lag:{symbol}',
                            json.dumps(lead_lag_result)
                        )
                        
                        return lead_lag_result
                    
                return {'error': 'Could not calculate correlation'}
            else:
                return {'error': 'Redis client not available'}
        
        except Exception as e:
            logger.error(f"Error detecting lead-lag relationship: {str(e)}")
            return {'error': str(e)}
    
    async def evaluate_sentiment_accuracy(self, symbol: str, lookback_days: int = 30) -> Dict:
        """
        Evaluate the accuracy of sentiment analysis for price prediction
        
        Args:
            symbol: Asset symbol
            lookback_days: Number of days to analyze
            
        Returns:
            Dictionary with accuracy metrics
        """
        try:
            if self.redis:
                # Get lead-lag relationship
                lead_lag_key = f'social_lead_lag:{symbol}'
                lead_lag_data = await self.redis.get(lead_lag_key)
                
                if not lead_lag_data:
                    # Calculate lead-lag relationship if not available
                    lead_lag_result = await self.detect_lead_lag_relationship(symbol, lookback_days)
                    
                    if 'error' in lead_lag_result:
                        return {'error': lead_lag_result['error']}
                else:
                    lead_lag_result = json.loads(lead_lag_data)
                
                # Get optimal lag (use Pearson by default)
                optimal_lag = lead_lag_result.get('strongest_pearson_lag', 0)
                
                # Get price history
                price_history_key = f'price_history:{symbol}'
                price_data = await self.redis.lrange(price_history_key, 0, lookback_days * 24 - 1)
                
                # Get social history
                social_history_key = f'social_history:{symbol}'
                social_data = await self.redis.lrange(social_history_key, 0, lookback_days * 24 - 1)
                
                if not price_data or not social_data:
                    return {'error': 'Insufficient historical data'}
                
                # Parse data
                price_history = [json.loads(item) for item in price_data]
                social_history = [json.loads(item) for item in social_data]
                
                # Convert to DataFrames
                price_df = pd.DataFrame(price_history)
                social_df = pd.DataFrame(social_history)
                
                # Ensure we have timestamps
                if 'timestamp' not in price_df.columns or 'timestamp' not in social_df.columns:
                    return {'error': 'Missing timestamp data'}
                
                # Convert timestamps to datetime
                price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
                social_df['timestamp'] = pd.to_datetime(social_df['timestamp'])
                
                # Sort by timestamp
                price_df = price_df.sort_values('timestamp')
                social_df = social_df.sort_values('timestamp')
                
                # Extract sentiment and price data
                if 'metrics' in social_df.columns and 'close' in price_df.columns:
                    # Extract sentiment
                    social_df['sentiment'] = social_df['metrics'].apply(
                        lambda x: x.get('social_sentiment', 0.5) if isinstance(x, dict) else 0.5
                    )
                    
                    # Calculate future returns based on optimal lag
                    if optimal_lag > 0:
                        # Sentiment leads price - calculate future returns
                        price_df['future_return'] = price_df['close'].pct_change(optimal_lag).shift(-optimal_lag)
                        
                        # Merge data
                        merged = pd.merge_asof(
                            social_df[['timestamp', 'sentiment']],
                            price_df[['timestamp', 'future_return']],
                            on='timestamp'
                        )
                    else:
                        # Price leads sentiment or no lag - use concurrent returns
                        price_df['return'] = price_df['close'].pct_change()
                        
                        # Merge data
                        merged = pd.merge_asof(
                            social_df[['timestamp', 'sentiment']],
                            price_df[['timestamp', 'return']],
                            on='timestamp'
                        )
                        merged['future_return'] = merged['return']
                    
                    # Drop NaN values
                    merged = merged.dropna()
                    
                    # Only proceed if we have enough data
                    if len(merged) < 10:
                        return {'error': 'Insufficient data for accuracy evaluation'}
                    
                    # Calculate accuracy metrics
                    
                    # 1. Correlation metrics
                    correlation, p_value = pearsonr(merged['sentiment'], merged['future_return'])
                    spearman, sp_value = spearmanr(merged['sentiment'], merged['future_return'])
                    
                    # 2. Direction accuracy
                    # Convert sentiment to direction (>0.5 is up, <0.5 is down)
                    merged['sentiment_direction'] = (merged['sentiment'] > 0.5).astype(int) * 2 - 1
                    merged['actual_direction'] = np.sign(merged['future_return'])
                    
                    # Calculate direction accuracy
                    correct_directions = (merged['sentiment_direction'] == merged['actual_direction']).sum()
                    direction_accuracy = correct_directions / len(merged)
                    
                    # 3. Linear regression performance
                    X = merged[['sentiment']]
                    y = merged['future_return']
                    
                    # Scale the data
                    scaler_X = StandardScaler()
                    scaler_y = StandardScaler()
                    
                    X_scaled = scaler_X.fit_transform(X)
                    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
                    
                    # Fit linear regression
                    model = LinearRegression()
                    model.fit(X_scaled, y_scaled)
                    
                    # Make predictions
                    y_pred_scaled = model.predict(X_scaled)
                    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                    
                    # Calculate regression metrics
                    r2 = r2_score(y, y_pred)
                    mae = mean_absolute_error(y, y_pred)
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    
                    # 4. Calculate information coefficient (IC)
                    # IC is the correlation between predicted and actual returns
                    ic, ic_p_value = pearsonr(y_pred, y)
                    
                    # Create the result dictionary
                    accuracy_result = {
                        'symbol': symbol,
                        'correlation': correlation,
                        'correlation_p_value': p_value,
                        'spearman_correlation': spearman,
                        'spearman_p_value': sp_value,
                        'direction_accuracy': direction_accuracy,
                        'r2': r2,
                        'mae': mae,
                        'rmse': rmse,
                        'information_coefficient': ic,
                        'ic_p_value': ic_p_value,
                        'optimal_lag': optimal_lag,
                        'data_points': len(merged),
                        'analysis_period': f"{lookback_days} days",
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Store the accuracy metrics
                    self.accuracy_metrics[symbol] = accuracy_result
                    
                    # Store the result in Redis
                    await self.redis.set(
                        f'social_accuracy:{symbol}',
                        json.dumps(accuracy_result)
                    )
                    
                    return accuracy_result
                
                return {'error': 'Missing required metrics or price data'}
            else:
                return {'error': 'Redis client not available'}
                
        except Exception as e:
            logger.error(f"Error evaluating sentiment accuracy: {str(e)}")
            return {'error': str(e)}
    
    async def update_adaptive_weights(self, symbol: str) -> Dict:
        """
        Update adaptive weights based on historical predictive power
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Dictionary with updated weights
        """
        try:
            # Get accuracy metrics
            if symbol in self.accuracy_metrics:
                accuracy = self.accuracy_metrics[symbol]
            else:
                # Try to get from Redis
                if self.redis:
                    accuracy_key = f'social_accuracy:{symbol}'
                    accuracy_data = await self.redis.get(accuracy_key)
                    
                    if accuracy_data:
                        accuracy = json.loads(accuracy_data)
                    else:
                        # Calculate accuracy if not available
                        accuracy = await self.evaluate_sentiment_accuracy(symbol)
                        
                        if 'error' in accuracy:
                            return {'error': accuracy['error']}
                else:
                    return {'error': 'Redis client not available'}
            
            # Calculate weights based on accuracy metrics
            weights = {}
            
            # Get base direction accuracy
            direction_accuracy = accuracy.get('direction_accuracy', 0.5)
            
            # Calculate information value
            # Information value is higher when direction accuracy is significantly different from 0.5
            # We use a sigmoid-like function to map direction accuracy to information value
            iv = 2 * abs(direction_accuracy - 0.5)
            
            # Adjust weights based on information value
            if iv > 0.2:  # Only adjust if we have meaningful predictive power
                # For predictive sources, increase weights for more accurate sources
                # Start with default weights
                weights = self.source_weights.copy()
                
                # Adjust based on optimal lag
                optimal_lag = accuracy.get('optimal_lag', 0)
                
                if optimal_lag > 6:
                    # Sentiment leads price by significant amount - increase news weight
                    weights['news'] = min(0.5, weights['news'] * 1.5)
                    
                    # Reduce weights for other sources
                    total_reduction = weights['news'] - self.source_weights['news']
                    for source in ['twitter', 'reddit', 'telegram']:
                        reduction = total_reduction * self.source_weights[source] / sum(
                            self.source_weights[s] for s in ['twitter', 'reddit', 'telegram']
                        )
                        weights[source] = max(0.1, weights[source] - reduction)
                        
                elif optimal_lag > 0:
                    # Sentiment leads price by small amount - increase Twitter weight
                    weights['twitter'] = min(0.5, weights['twitter'] * 1.3)
                    
                    # Reduce weights for other sources
                    total_reduction = weights['twitter'] - self.source_weights['twitter']
                    for source in ['news', 'reddit', 'telegram']:
                        reduction = total_reduction * self.source_weights[source] / sum(
                            self.source_weights[s] for s in ['news', 'reddit', 'telegram']
                        )
                        weights[source] = max(0.1, weights[source] - reduction)
                        
                elif optimal_lag < -6:
                    # Price leads sentiment by significant amount - reduce all weights
                    # This means social sentiment is a lagging indicator
                    for source in weights:
                        weights[source] *= 0.8
                
                # Normalize weights to sum to 1
                weight_sum = sum(weights.values())
                if weight_sum > 0:
                    for source in weights:
                        weights[source] /= weight_sum
            else:
                # Poor predictive power - use default weights
                weights = self.source_weights.copy()
            
            # Store the updated weights
            self.adaptive_weights[symbol] = weights
            
            # Create result dictionary
            result = {
                'symbol': symbol,
                'weights': weights,
                'information_value': iv,
                'direction_accuracy': direction_accuracy,
                'optimal_lag': accuracy.get('optimal_lag', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            # Store in Redis
            if self.redis:
                await self.redis.set(
                    f'social_adaptive_weights:{symbol}',
                    json.dumps(result)
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error updating adaptive weights: {str(e)}")
            return {'error': str(e)}
    
    async def get_enhanced_sentiment(self, symbol: str, metrics: Dict) -> Dict:
        """
        Get enhanced sentiment for a symbol
        
        Args:
            symbol: Asset symbol
            metrics: Raw social metrics dictionary
            
        Returns:
            Enhanced sentiment dictionary
        """
        try:
            # 1. Normalize the metrics
            normalized_metrics = self.normalize_metrics(metrics)
            
            # 2. Detect and filter anomalies
            filtered_metrics, is_anomaly = self.detect_anomalies(symbol, normalized_metrics)
            
            # 3. Get adaptive weights
            if symbol in self.adaptive_weights:
                weights = self.adaptive_weights[symbol].copy()
            else:
                # Try to get from Redis
                if self.redis:
                    weights_key = f'social_adaptive_weights:{symbol}'
                    weights_data = await self.redis.get(weights_key)
                    
                    if weights_data:
                        weights_result = json.loads(weights_data)
                        weights = weights_result.get('weights', self.source_weights)
                    else:
                        # Use default weights
                        weights = self.source_weights.copy()
                else:
                    weights = self.source_weights.copy()
            
            # 4. Extract sentiments by source
            sentiments = {}
            
            # Map metric names to sources
            source_mappings = {
                'twitter': ['twitter_sentiment'],
                'reddit': ['reddit_sentiment'],
                'news': ['news_sentiment'],
                'telegram': ['telegram_sentiment']
            }
            
            # Extract sentiments from metrics
            for source, metric_names in source_mappings.items():
                for metric_name in metric_names:
                    if metric_name in filtered_metrics:
                        sentiments[source] = filtered_metrics[metric_name]
                        break
            
            # 5. Get time-weighted sentiment if history is available
            if self.redis:
                # Get sentiment history
                history_key = f'social_history:{symbol}'
                history_data = await self.redis.lrange(history_key, 0, 24)  # Last 24 hours
                
                if history_data:
                    # Parse data
                    history = [json.loads(item) for item in history_data]
                    
                    # Calculate time-weighted sentiment
                    time_weighted = self.calculate_time_weighted_sentiment(history, 'medium')
                    
                    # Add to sentiments
                    sentiments['time_weighted'] = time_weighted
                    weights['time_weighted'] = 0.2  # Add weight for time-weighted sentiment
                    
                    # Normalize weights
                    weight_sum = sum(weights.values())
                    for source in weights:
                        weights[source] /= weight_sum
            
            # 6. Aggregate sentiments
            aggregated_sentiment = self.aggregate_sentiment(sentiments)
            
            # 7. Get optimal lag information
            optimal_lag = 0
            if self.redis:
                lead_lag_key = f'social_lead_lag:{symbol}'
                lead_lag_data = await self.redis.get(lead_lag_key)
                
                if lead_lag_data:
                    lead_lag_result = json.loads(lead_lag_data)
                    optimal_lag = lead_lag_result.get('strongest_pearson_lag', 0)
            
            # 8. Create result dictionary
            result = {
                'symbol': symbol,
                'raw_sentiment': metrics.get('social_sentiment', 0.5),
                'enhanced_sentiment': aggregated_sentiment,
                'source_sentiments': sentiments,
                'weights': weights,
                'is_anomaly': is_anomaly,
                'optimal_lag': optimal_lag,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting enhanced sentiment: {str(e)}")
            return {
                'symbol': symbol,
                'raw_sentiment': metrics.get('social_sentiment', 0.5),
                'enhanced_sentiment': metrics.get('social_sentiment', 0.5),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }