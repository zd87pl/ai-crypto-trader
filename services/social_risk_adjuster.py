import os
import json
import socket
import asyncio
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging as logger
from logging.handlers import RotatingFileHandler
from redis.asyncio import Redis
from redis.exceptions import ConnectionError
from typing import Dict, List, Optional

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure rotating file handler
rotating_handler = RotatingFileHandler(
    'logs/social_risk_adjuster.log',
    maxBytes=10*1024*1024,  # 10MB per file
    backupCount=5,  # Keep 5 backup files
    encoding='utf-8'
)

# Configure logging with rotation
logger.basicConfig(
    level=logger.DEBUG,
    format='%(asctime)s - %(levelname)s - [SocialRiskAdjuster] %(message)s',
    handlers=[
        rotating_handler,
        logger.StreamHandler()
    ]
)

class SocialRiskAdjuster:
    """
    Service that adjusts trading risk parameters based on social sentiment data
    Implementation of RISK-07: Implement social sentiment-based risk adjustments
    """
    
    def __init__(self):
        logger.debug("Initializing Social Risk Adjuster Service...")
        
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Redis configuration
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis = None
        
        # Service state
        self.running = True
        self.social_data_cache = {}
        self.last_update = {}
        self.adjustment_history = {}
        self.sentiment_threshold_bullish = 0.65  # Threshold for positive sentiment
        self.sentiment_threshold_bearish = 0.35  # Threshold for negative sentiment
        self.max_adjustment_percent = 0.5  # Maximum adjustment percentage (+/-)
        
        # Track symbols with active risk adjustments
        self.active_adjustments = {}
        
        # Get service port from environment variable
        self.service_port = int(os.getenv('SOCIAL_RISK_PORT', 8005))
        logger.debug(f"Service port configured as: {self.service_port}")
        
        # Initialize risk adjustment parameters
        self._initialize_risk_parameters()
        
        logger.debug("Social Risk Adjuster Service initialization complete")
    
    def _initialize_risk_parameters(self):
        """Initialize risk adjustment parameters"""
        # Default risk adjustment parameters
        self.risk_params = {
            # Position sizing adjustments
            "position_size_impact": 0.3,  # Adjustment impact on position size (0-1)
            
            # Stop loss adjustments
            "stop_loss_impact": 0.2,  # Adjustment impact on stop loss percentage
            
            # Take profit adjustments
            "take_profit_impact": 0.4,  # Adjustment impact on take profit percentage
            
            # Correlation thresholds
            "correlation_impact": 0.25,  # Adjustment impact on correlation threshold
            
            # Sentiment importance by source
            "sentiment_weights": {
                "twitter_sentiment": 0.35,
                "reddit_sentiment": 0.30,
                "news_sentiment": 0.25,
                "overall_sentiment": 0.10
            },
            
            # Risk factor importance
            "risk_factor_weights": {
                "sentiment_score": 0.45,
                "social_volume": 0.20,
                "social_engagement": 0.25,
                "social_contributors": 0.10
            },
            
            # Volatility adjustment based on social engagement
            "volatility_modifier": 0.15,
            
            # News impact factor
            "news_impact_factor": 0.35,
            
            # Time decay for sentiment (half-life in hours)
            "sentiment_half_life": 6,
            
            # Minimum data quality requirement (0-1)
            "min_data_quality": 0.5
        }
        
        # Override with config if provided
        if "social_risk_adjustment" in self.config:
            self.risk_params.update(self.config["social_risk_adjustment"])
            logger.info("Loaded risk adjustment parameters from config")
        else:
            logger.info("Using default risk adjustment parameters")
    
    async def connect_redis(self, max_retries=5, retry_delay=5):
        """Establish Redis connection with retries"""
        retries = 0
        while retries < max_retries:
            try:
                if self.redis is None:
                    self.redis = Redis(
                        host=self.redis_host,
                        port=self.redis_port,
                        decode_responses=True
                    )
                await self.redis.ping()
                logger.info("Successfully connected to Redis")
                return True
            except (ConnectionError, Exception) as e:
                retries += 1
                logger.error(f"Failed to connect to Redis (attempt {retries}/{max_retries}): {str(e)}")
                if retries < max_retries:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("Max retries reached. Could not connect to Redis.")
                    return False
    
    def calculate_sentiment_score(self, social_data):
        """Calculate a consolidated sentiment score from social data"""
        try:
            # Extract metrics from social data
            metrics = social_data.get('metrics', {})
            
            # Normalize scores to 0-1 range
            sentiment = metrics.get('social_sentiment', 0.5)
            # If sentiment is -100 to 100 scale, normalize to 0-1
            if sentiment < 0 or sentiment > 1:
                sentiment = (sentiment + 100) / 200 if sentiment >= -100 else 0.0
            
            # Calculate weighted score using configured weights
            weights = self.risk_params["risk_factor_weights"]
            
            weighted_score = (
                weights["sentiment_score"] * sentiment +
                weights["social_volume"] * min(1.0, metrics.get('social_volume', 0) / 10000) +
                weights["social_engagement"] * min(1.0, metrics.get('social_engagement', 0) / 5000) +
                weights["social_contributors"] * min(1.0, metrics.get('social_contributors', 0) / 1000)
            )
            
            # Apply news impact if available
            news_list = social_data.get('recent_news', [])
            if news_list:
                news_sentiment = 0
                news_count = 0
                
                for news in news_list:
                    if 'sentiment' in news:
                        news_sent = news.get('sentiment', 0)
                        # Normalize to 0-1 if needed
                        if news_sent < 0 or news_sent > 1:
                            news_sent = (news_sent + 100) / 200 if news_sent >= -100 else 0.0
                        
                        news_sentiment += news_sent
                        news_count += 1
                
                if news_count > 0:
                    avg_news_sentiment = news_sentiment / news_count
                    weighted_score = (
                        weighted_score * (1 - self.risk_params["news_impact_factor"]) +
                        avg_news_sentiment * self.risk_params["news_impact_factor"]
                    )
            
            # Ensure final score is between 0 and 1
            final_score = max(0.0, min(1.0, weighted_score))
            logger.debug(f"Calculated sentiment score: {final_score:.4f} from sentiment {sentiment:.4f}")
            
            return final_score
        
        except Exception as e:
            logger.error(f"Error calculating sentiment score: {str(e)}")
            return 0.5  # Neutral fallback
    
    def apply_time_decay(self, score, timestamp):
        """Apply time decay to sentiment score based on age"""
        try:
            # Parse timestamp and calculate age in hours
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            now = datetime.now()
            age_hours = (now - timestamp).total_seconds() / 3600
            
            # Calculate decay factor based on half-life
            half_life = self.risk_params["sentiment_half_life"]
            decay_factor = 0.5 ** (age_hours / half_life)
            
            # Apply decay: move score toward neutral (0.5) based on age
            decayed_score = 0.5 + (score - 0.5) * decay_factor
            
            logger.debug(f"Applied time decay: original={score:.4f}, age={age_hours:.2f}h, decayed={decayed_score:.4f}")
            return decayed_score
            
        except Exception as e:
            logger.error(f"Error applying time decay: {str(e)}")
            return score  # Return original score if error
    
    def calculate_risk_adjustments(self, symbol, sentiment_score):
        """Calculate risk parameter adjustments based on sentiment score"""
        try:
            # Center the score around neutral (0.5)
            centered_score = sentiment_score - 0.5
            
            # Max adjustment percentage 
            max_adj = self.max_adjustment_percent
            
            # Calculate adjustment factor (-max_adj to +max_adj)
            adjustment_factor = centered_score * 2 * max_adj
            
            # Only adjust if sentiment is strong enough
            if sentiment_score > self.sentiment_threshold_bullish:
                sentiment_type = "BULLISH"
                # Bullish sentiment: increase position size, widen stop-loss, increase take-profit
                position_size_adj = adjustment_factor * self.risk_params["position_size_impact"]
                stop_loss_adj = -adjustment_factor * self.risk_params["stop_loss_impact"]
                take_profit_adj = adjustment_factor * self.risk_params["take_profit_impact"]
                correlation_adj = adjustment_factor * self.risk_params["correlation_impact"]
                
            elif sentiment_score < self.sentiment_threshold_bearish:
                sentiment_type = "BEARISH"
                # Bearish sentiment: decrease position size, tighten stop-loss, decrease take-profit
                position_size_adj = adjustment_factor * self.risk_params["position_size_impact"]
                stop_loss_adj = -adjustment_factor * self.risk_params["stop_loss_impact"]
                take_profit_adj = adjustment_factor * self.risk_params["take_profit_impact"]
                correlation_adj = adjustment_factor * self.risk_params["correlation_impact"]
                
            else:
                sentiment_type = "NEUTRAL"
                # Neutral sentiment: no adjustments
                position_size_adj = 0
                stop_loss_adj = 0
                take_profit_adj = 0
                correlation_adj = 0
            
            # Create adjustment object
            adjustments = {
                "symbol": symbol,
                "sentiment_score": sentiment_score,
                "sentiment_type": sentiment_type,
                "position_size_adj": position_size_adj,
                "stop_loss_adj": stop_loss_adj,
                "take_profit_adj": take_profit_adj,
                "correlation_threshold_adj": correlation_adj,
                "timestamp": datetime.now().isoformat(),
                "confidence": abs(centered_score) * 2  # 0-1 scale based on distance from neutral
            }
            
            logger.info(f"Calculated risk adjustments for {symbol}: {sentiment_type} (score: {sentiment_score:.4f})")
            logger.debug(f"Adjustments: {json.dumps(adjustments)}")
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Error calculating risk adjustments: {str(e)}")
            return {
                "symbol": symbol,
                "sentiment_score": 0.5,
                "sentiment_type": "ERROR",
                "position_size_adj": 0,
                "stop_loss_adj": 0,
                "take_profit_adj": 0,
                "correlation_threshold_adj": 0,
                "timestamp": datetime.now().isoformat(),
                "confidence": 0,
                "error": str(e)
            }
    
    def calculate_volatility_adjustment(self, social_data):
        """Calculate volatility adjustment based on social engagement"""
        try:
            # Extract engagement metrics
            metrics = social_data.get('metrics', {})
            engagement = metrics.get('social_engagement', 0)
            volume = metrics.get('social_volume', 0)
            
            # Normalize metrics to 0-1 scale
            norm_engagement = min(1.0, engagement / 10000)
            norm_volume = min(1.0, volume / 20000)
            
            # Calculate volatility score (high engagement and volume suggests higher volatility)
            volatility_score = (norm_engagement * 0.7) + (norm_volume * 0.3)
            
            # Apply modifier
            volatility_adj = volatility_score * self.risk_params["volatility_modifier"]
            
            return volatility_adj
            
        except Exception as e:
            logger.error(f"Error calculating volatility adjustment: {str(e)}")
            return 0.0
    
    def assess_data_quality(self, social_data):
        """Assess the quality of social data for reliability"""
        try:
            quality_score = 1.0
            metrics = social_data.get('metrics', {})
            
            # Check for missing metrics
            required_metrics = ['social_volume', 'social_engagement', 'social_contributors', 'social_sentiment']
            missing_metrics = [m for m in required_metrics if m not in metrics or metrics.get(m) is None]
            
            if missing_metrics:
                # Reduce quality score for each missing metric
                quality_score -= len(missing_metrics) * 0.2
            
            # Check age of data
            if 'timestamp' in social_data:
                try:
                    timestamp = datetime.fromisoformat(social_data['timestamp'])
                    age_hours = (datetime.now() - timestamp).total_seconds() / 3600
                    
                    # Reduce quality for older data
                    if age_hours > 24:
                        quality_score -= min(0.5, (age_hours - 24) / 48)
                except:
                    quality_score -= 0.2  # Invalid timestamp
            else:
                quality_score -= 0.2  # Missing timestamp
            
            # Check for insufficient data
            if metrics.get('social_volume', 0) < 100 or metrics.get('social_engagement', 0) < 100:
                quality_score -= 0.3
            
            # Ensure quality is between 0 and 1
            quality_score = max(0.0, min(1.0, quality_score))
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {str(e)}")
            return 0.5  # Mediocre quality as fallback
    
    async def get_active_trades(self):
        """Get active trades from Redis"""
        try:
            if not self.redis or not await self.redis.ping():
                if not await self.connect_redis():
                    return {}
            
            active_trades_json = await self.redis.get('active_trades')
            if active_trades_json:
                return json.loads(active_trades_json)
            return {}
            
        except Exception as e:
            logger.error(f"Error getting active trades: {str(e)}")
            return {}
    
    async def get_social_data(self, symbol):
        """Get social data for a symbol from Redis"""
        try:
            if not self.redis or not await self.redis.ping():
                if not await self.connect_redis():
                    return None
            
            # Get social metrics
            social_data_json = await self.redis.hget('social_metrics', symbol)
            if not social_data_json:
                logger.debug(f"No social data found for {symbol}")
                return None
            
            social_data = json.loads(social_data_json)
            
            # Assess data quality
            quality_score = self.assess_data_quality(social_data)
            if quality_score < self.risk_params["min_data_quality"]:
                logger.warning(f"Low quality social data for {symbol} (score: {quality_score:.2f})")
                return None
            
            # Cache the data
            self.social_data_cache[symbol] = social_data
            self.last_update[symbol] = datetime.now()
            
            return social_data
            
        except Exception as e:
            logger.error(f"Error getting social data for {symbol}: {str(e)}")
            return None
    
    async def process_symbol(self, symbol):
        """Process a symbol to generate risk adjustments"""
        try:
            # Get social data
            social_data = await self.get_social_data(symbol)
            if not social_data:
                logger.debug(f"No valid social data available for {symbol}")
                # Remove any existing adjustments
                if symbol in self.active_adjustments:
                    await self.publish_risk_adjustment(symbol, None)
                    self.active_adjustments.pop(symbol, None)
                return
            
            # Calculate sentiment score
            raw_score = self.calculate_sentiment_score(social_data)
            
            # Apply time decay based on data age
            timestamp = datetime.fromisoformat(social_data.get('timestamp', datetime.now().isoformat()))
            decayed_score = self.apply_time_decay(raw_score, timestamp)
            
            # Calculate risk adjustments
            adjustments = self.calculate_risk_adjustments(symbol, decayed_score)
            
            # Add volatility adjustment
            volatility_adj = self.calculate_volatility_adjustment(social_data)
            adjustments["volatility_adj"] = volatility_adj
            
            # Store in adjustment history
            self.adjustment_history[symbol] = adjustments
            
            # Update active adjustments
            self.active_adjustments[symbol] = adjustments
            
            # Publish adjustments to Redis
            await self.publish_risk_adjustment(symbol, adjustments)
            
            logger.info(f"Processed {symbol}: sentiment={decayed_score:.4f}, type={adjustments['sentiment_type']}")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
    
    async def publish_risk_adjustment(self, symbol, adjustment):
        """Publish risk adjustment to Redis"""
        try:
            if not self.redis or not await self.redis.ping():
                if not await self.connect_redis():
                    return False
            
            if adjustment is None:
                # Remove adjustment
                await self.redis.hdel('social_risk_adjustments', symbol)
                logger.info(f"Removed risk adjustment for {symbol}")
                return True
            
            # Store adjustment in hash
            await self.redis.hset('social_risk_adjustments', symbol, json.dumps(adjustment))
            
            # Publish event notification
            await self.redis.publish(
                'risk_adjustment_updates',
                json.dumps({
                    'symbol': symbol,
                    'adjustment': adjustment,
                    'timestamp': datetime.now().isoformat()
                })
            )
            
            logger.debug(f"Published risk adjustment for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error publishing risk adjustment: {str(e)}")
            return False
    
    async def update_risk_adjustments(self):
        """Update risk adjustments for monitored symbols"""
        while self.running:
            try:
                if not self.redis or not await self.redis.ping():
                    if not await self.connect_redis():
                        await asyncio.sleep(5)
                        continue
                
                # Get active trades
                active_trades = await self.get_active_trades()
                
                # Process each active symbol
                for symbol in active_trades.keys():
                    await self.process_symbol(symbol)
                
                # Get other symbols to monitor from market updates
                try:
                    # Get recent market updates
                    market_updates = []
                    async for msg in self.redis.xrevrange('market_data', count=50):
                        try:
                            data = json.loads(msg[1]['data'])
                            market_updates.append(data)
                        except:
                            pass
                    
                    # Extract symbols
                    symbols = set()
                    for update in market_updates:
                        if 'symbol' in update:
                            symbols.add(update['symbol'])
                    
                    # Process additional symbols not in active trades
                    for symbol in symbols:
                        if symbol not in active_trades:
                            await self.process_symbol(symbol)
                            
                except Exception as e:
                    logger.error(f"Error processing market symbols: {str(e)}")
                
                # Publish overall risk adjustment report
                await self.publish_risk_report()
                
                # Wait for next update cycle
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in update_risk_adjustments: {str(e)}")
                await asyncio.sleep(30)
    
    async def publish_risk_report(self):
        """Publish overall risk adjustment report to Redis"""
        try:
            if not self.redis or not await self.redis.ping():
                if not await self.connect_redis():
                    return False
            
            # Create report
            report = {
                "active_adjustments": len(self.active_adjustments),
                "adjustments": self.active_adjustments,
                "parameters": self.risk_params,
                "timestamp": datetime.now().isoformat()
            }
            
            # Publish to Redis
            await self.redis.set('social_risk_report', json.dumps(report))
            logger.debug(f"Published social risk report with {len(self.active_adjustments)} active adjustments")
            
            return True
            
        except Exception as e:
            logger.error(f"Error publishing risk report: {str(e)}")
            return False
    
    async def handle_config_updates(self):
        """Listen for configuration updates"""
        while self.running:
            try:
                if not self.redis or not await self.redis.ping():
                    if not await self.connect_redis():
                        await asyncio.sleep(5)
                        continue
                
                # Subscribe to configuration updates
                pubsub = self.redis.pubsub()
                await pubsub.subscribe('config_updates')
                
                while self.running:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if message:
                        try:
                            config_update = json.loads(message['data'])
                            
                            # Check if the update is for social risk adjustment
                            if 'social_risk_adjustment' in config_update:
                                self.risk_params.update(config_update['social_risk_adjustment'])
                                logger.info(f"Updated risk parameters from config update: {self.risk_params}")
                                
                                # Re-process all symbols with new parameters
                                for symbol in list(self.active_adjustments.keys()):
                                    await self.process_symbol(symbol)
                                    
                        except Exception as e:
                            logger.error(f"Error processing config update: {str(e)}")
                    
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in handle_config_updates: {str(e)}")
                await asyncio.sleep(5)
    
    async def maintain_redis(self):
        """Maintain Redis connection"""
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
    
    async def health_check_server(self):
        """Run a simple TCP server for health checks"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server.bind(('0.0.0.0', self.service_port))
            server.listen(1)
            server.setblocking(False)
            
            logger.info(f"Health check server listening on port {self.service_port}")
            
            while self.running:
                try:
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Health check server error: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to start health check server: {str(e)}")
        finally:
            server.close()
    
    async def run(self):
        """Run the social risk adjuster service"""
        try:
            logger.info("Starting Social Risk Adjuster Service...")
            
            # First establish Redis connection
            if not await self.connect_redis():
                raise Exception("Failed to establish initial Redis connection")
            
            # Create tasks
            tasks = [
                asyncio.create_task(self.update_risk_adjustments()),
                asyncio.create_task(self.handle_config_updates()),
                asyncio.create_task(self.maintain_redis()),
                asyncio.create_task(self.health_check_server())
            ]
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Error in Social Risk Adjuster Service: {str(e)}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the social risk adjuster service"""
        logger.info("Stopping Social Risk Adjuster Service...")
        self.running = False
        if self.redis:
            await self.redis.close()

if __name__ == "__main__":
    service = SocialRiskAdjuster()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        asyncio.run(service.stop())
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        asyncio.run(service.stop())