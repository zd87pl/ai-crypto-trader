import pandas as pd
import numpy as np
import logging as logger
from datetime import datetime, timedelta
from typing import Dict, Optional, Union
from pathlib import Path

from .data_manager import HistoricalDataManager

class SocialDataProvider:
    """Provider for social data during backtesting"""
    
    def __init__(self, data_manager: HistoricalDataManager = None):
        """Initialize with optional data manager"""
        self.data_manager = data_manager or HistoricalDataManager()
        self.social_data = {}  # Cache for loaded social data
        self.default_metrics = {
            'social_volume': 0,
            'social_engagement': 0,
            'social_contributors': 0,
            'social_sentiment': 0.5,  # Neutral sentiment
            'twitter_volume': 0,
            'reddit_volume': 0,
            'news_volume': 0
        }
    
    def load_social_data(self, symbol: str, start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
        """Load social data for the given symbol and date range"""
        # Generate cache key
        cache_key = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d') if end_date else 'now'}"
        
        # Check if already in cache
        if cache_key in self.social_data:
            return self.social_data[cache_key]
        
        # Load social data using data manager
        data = self.data_manager.load_social_data(symbol, start_date, end_date)
        
        # Store in cache
        self.social_data[cache_key] = data
        
        return data
    
    def get_social_metrics_at(self, symbol: str, timestamp: datetime) -> Dict:
        """Get social metrics for a specific point in time"""
        # Check if we have data loaded for this symbol
        if symbol not in self.social_data:
            # Try to load data for the past 90 days
            start_date = timestamp - timedelta(days=90)
            end_date = timestamp + timedelta(days=1)  # Include the target day
            self.load_social_data(symbol, start_date, end_date)
        
        # If we have data for this symbol, find the closest data point
        for cache_key, data in self.social_data.items():
            if symbol in cache_key and not data.empty:
                # Get the closest date not exceeding the timestamp
                try:
                    # Find dates that are less than or equal to the timestamp
                    valid_dates = data.index[data.index <= timestamp]
                    
                    if len(valid_dates) > 0:
                        # Get the most recent date
                        closest_date = valid_dates[-1]
                        row = data.loc[closest_date]
                        
                        # Extract relevant social metrics
                        metrics = {}
                        for metric in self.default_metrics.keys():
                            if metric in row:
                                metrics[metric] = row[metric]
                            else:
                                metrics[metric] = self.default_metrics[metric]
                        
                        return metrics
                except Exception as e:
                    logger.error(f"Error getting social metrics for {symbol} at {timestamp}: {str(e)}")
        
        # If no data found, return default metrics
        logger.warning(f"No social data found for {symbol} at {timestamp}, using defaults")
        return self.default_metrics.copy()
    
    def get_news_sentiment(self, symbol: str, timestamp: datetime, lookback_days: int = 7) -> Dict:
        """Get news sentiment from social data"""
        # Define default response
        default_response = {
            'sentiment': 0.5,  # Neutral sentiment
            'recent_news': []
        }
        
        try:
            # Calculate start date for lookback
            start_date = timestamp - timedelta(days=lookback_days)
            
            # Load social data if needed
            data = self.load_social_data(symbol, start_date, timestamp)
            
            if data.empty:
                return default_response
            
            # Extract news sentiment if available
            if 'news_sentiment' in data.columns:
                # Get the most recent sentiment value
                recent_indices = data.index[data.index <= timestamp]
                if len(recent_indices) > 0:
                    latest_idx = recent_indices[-1]
                    sentiment = data.loc[latest_idx, 'news_sentiment']
                    return {
                        'sentiment': sentiment,
                        'recent_news': []  # We don't have the actual news content in historical data
                    }
            
            # If no specific news sentiment column, use general social sentiment
            if 'social_sentiment' in data.columns:
                recent_indices = data.index[data.index <= timestamp]
                if len(recent_indices) > 0:
                    latest_idx = recent_indices[-1]
                    sentiment = data.loc[latest_idx, 'social_sentiment']
                    return {
                        'sentiment': sentiment,
                        'recent_news': []
                    }
            
            return default_response
            
        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol}: {str(e)}")
            return default_response
    
    def get_social_indicators(self, symbol: str, timestamp: datetime, lookback_days: int = 30) -> Dict:
        """Get derived social indicators like momentum, trend, etc."""
        try:
            # Calculate start date for lookback
            start_date = timestamp - timedelta(days=lookback_days)
            
            # Load social data if needed
            data = self.load_social_data(symbol, start_date, timestamp)
            
            if data.empty:
                return {
                    'social_momentum': 0,
                    'social_trend': 'neutral',
                    'social_intensity': 0,
                    'social_engagement_rate': 0
                }
            
            # Filter data to timestamp
            data = data[data.index <= timestamp]
            
            if len(data) < 2:
                return {
                    'social_momentum': 0,
                    'social_trend': 'neutral',
                    'social_intensity': 0,
                    'social_engagement_rate': 0
                }
            
            # Calculate social momentum (rate of change in volume)
            if 'social_volume' in data.columns:
                recent_volume = data['social_volume'].iloc[-1]
                prev_volume = data['social_volume'].iloc[-2]
                social_momentum = ((recent_volume - prev_volume) / max(prev_volume, 1)) * 100
            else:
                social_momentum = 0
            
            # Determine social trend
            if social_momentum > 20:
                social_trend = 'bullish'
            elif social_momentum < -20:
                social_trend = 'bearish'
            else:
                social_trend = 'neutral'
            
            # Calculate social intensity (volatility of social metrics)
            if 'social_volume' in data.columns and len(data) > 5:
                social_intensity = data['social_volume'].pct_change().std() * 100
            else:
                social_intensity = 0
            
            # Calculate engagement rate
            if 'social_engagement' in data.columns and 'social_volume' in data.columns:
                social_engagement_rate = data['social_engagement'].iloc[-1] / max(data['social_volume'].iloc[-1], 1)
            else:
                social_engagement_rate = 0
            
            return {
                'social_momentum': social_momentum,
                'social_trend': social_trend,
                'social_intensity': social_intensity,
                'social_engagement_rate': social_engagement_rate
            }
            
        except Exception as e:
            logger.error(f"Error calculating social indicators for {symbol}: {str(e)}")
            return {
                'social_momentum': 0,
                'social_trend': 'neutral',
                'social_intensity': 0,
                'social_engagement_rate': 0
            }
            
    def generate_market_update_with_social(self, market_data: Dict, timestamp: datetime) -> Dict:
        """Enrich market data with social metrics for backtesting"""
        symbol = market_data['symbol']
        
        # Get social metrics
        social_metrics = self.get_social_metrics_at(symbol, timestamp)
        
        # Get news sentiment
        news_sentiment = self.get_news_sentiment(symbol, timestamp)
        
        # Get social indicators
        social_indicators = self.get_social_indicators(symbol, timestamp)
        
        # Merge all the data
        enriched_data = market_data.copy()
        enriched_data.update({
            'social_volume': social_metrics.get('social_volume', 0),
            'social_engagement': social_metrics.get('social_engagement', 0),
            'social_contributors': social_metrics.get('social_contributors', 0),
            'social_sentiment': social_metrics.get('social_sentiment', 0.5),
            'twitter_volume': social_metrics.get('twitter_volume', 0),
            'reddit_volume': social_metrics.get('reddit_volume', 0),
            'news_volume': social_metrics.get('news_volume', 0),
            'news_sentiment': news_sentiment.get('sentiment', 0.5),
            'recent_news': news_sentiment.get('recent_news', []),
            'social_momentum': social_indicators.get('social_momentum', 0),
            'social_trend': social_indicators.get('social_trend', 'neutral'),
            'social_intensity': social_indicators.get('social_intensity', 0),
            'social_engagement_rate': social_indicators.get('social_engagement_rate', 0)
        })
        
        return enriched_data