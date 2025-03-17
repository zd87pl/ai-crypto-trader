import os
import json
import socket
import asyncio
import logging as logger
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import aiohttp
import numpy as np
import pandas as pd
from redis.asyncio import Redis
from redis.exceptions import ConnectionError
from typing import Dict, List, Optional, Tuple, Any

from services.utils.social_metrics_analyzer import SocialMetricsAnalyzer

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure rotating file handler
rotating_handler = RotatingFileHandler(
    'logs/enhanced_social_monitor.log',
    maxBytes=10*1024*1024,  # 10MB per file
    backupCount=5,  # Keep 5 backup files
    encoding='utf-8'
)

# Configure logging with rotation
logger.basicConfig(
    level=logger.DEBUG,
    format='%(asctime)s - %(levelname)s - [EnhancedSocialMonitor] %(message)s',
    handlers=[
        rotating_handler,
        logger.StreamHandler()
    ]
)

class EnhancedSocialMonitorService:
    """
    Enhanced service for monitoring and analyzing social metrics for crypto assets.
    
    Improvements over the original SocialMonitorService:
    1. More accurate sentiment analysis with multi-source aggregation
    2. Anomaly detection to filter out unusual data
    3. Time-weighted sentiment calculation
    4. Lead-lag relationship detection between social metrics and price
    5. Historical accuracy assessment
    6. Adaptive source weighting based on predictive power
    7. Comprehensive performance reporting
    """
    
    def __init__(self):
        logger.debug("Initializing Enhanced Social Monitor Service...")
        
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # LunarCrush configuration
        self.api_key = os.getenv('LUNARCRUSH_API_KEY', self.config['lunarcrush']['api_key'])
        if not self.api_key:
            raise ValueError("LUNARCRUSH_API_KEY environment variable or config value not set")
        
        self.base_url = self.config['lunarcrush']['base_url']
        self.endpoints = self.config['lunarcrush']['endpoints']
        self.update_interval = self.config['lunarcrush']['update_interval']
        self.required_metrics = self.config['lunarcrush']['metrics']['required']
        self.sentiment_weights = self.config['lunarcrush']['sentiment_weights']
        self.cache_duration = self.config['lunarcrush']['cache_duration']
        self.max_news_age = self.config['lunarcrush']['max_news_age']
        self.min_engagement = self.config['lunarcrush']['min_engagement']
        
        # Redis configuration
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis = None
        
        # Service state
        self.running = True
        self.monitored_symbols = set()
        self.cache = {}
        self.last_update = {}
        
        # Enhanced settings
        self.anomaly_detection_enabled = True
        self.adaptive_weights_enabled = True
        self.accuracy_assessment_interval = 43200  # 12 hours in seconds
        self.lead_lag_analysis_interval = 86400  # 24 hours in seconds
        self.accuracy_assessment_symbols = set()  # Symbols to assess for accuracy
        
        # Performance metrics
        self.performance_metrics = {
            'api_requests': 0,
            'api_errors': 0,
            'anomalies_detected': 0,
            'cache_hits': 0,
            'processing_times': []
        }
        
        # Get service port from environment variable
        self.service_port = int(os.getenv('ENHANCED_SOCIAL_MONITOR_PORT', 8013))
        logger.debug(f"Service port configured as: {self.service_port}")
        
        # Initialize the social metrics analyzer (will connect it to Redis later)
        self.analyzer = SocialMetricsAnalyzer()
        
        logger.debug("Enhanced Social Monitor Service initialization complete")
    
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
                    # Also connect the analyzer to Redis
                    self.analyzer.redis = self.redis
                    
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
    
    async def fetch_social_metrics(self, symbol: str) -> Optional[Dict]:
        """Fetch social metrics from LunarCrush API with enhanced processing"""
        try:
            # Record processing start time
            start_time = datetime.now()
            
            # Check cache first
            if symbol in self.cache:
                cache_time = self.last_update.get(symbol, datetime.min)
                if (datetime.now() - cache_time).total_seconds() < self.cache_duration:
                    self.performance_metrics['cache_hits'] += 1
                    return self.cache[symbol]
            
            # Prepare API request
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Accept': 'application/json'
            }
            
            params = {
                'symbol': symbol,
                'interval': '1h',  # Get hourly data
                'limit': 1  # Get most recent data point
            }
            
            url = f"{self.base_url}{self.endpoints['assets']}"
            
            # Track API requests
            self.performance_metrics['api_requests'] += 1
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'data' in data and data['data']:
                            asset_data = data['data'][0]
                            
                            # Extract required metrics
                            metrics = {
                                'social_volume': asset_data.get('social_volume', 0),
                                'social_engagement': asset_data.get('social_engagement', 0),
                                'social_contributors': asset_data.get('social_contributors', 0),
                                'social_sentiment': asset_data.get('social_sentiment', 0),
                                'twitter_volume': asset_data.get('twitter_volume', 0),
                                'twitter_sentiment': asset_data.get('twitter_sentiment', 0.5),
                                'reddit_volume': asset_data.get('reddit_volume', 0),
                                'reddit_sentiment': asset_data.get('reddit_sentiment', 0.5),
                                'news_volume': asset_data.get('news_volume', 0),
                                'news_sentiment': asset_data.get('news_sentiment', 0.5)
                            }
                            
                            # Enhanced processing: Detect anomalies if enabled
                            if self.anomaly_detection_enabled:
                                # First, check if we need to train an anomaly model
                                if symbol not in self.analyzer.anomaly_models:
                                    await self.analyzer.train_anomaly_model(symbol)
                                
                                # Detect anomalies
                                filtered_metrics, is_anomaly = self.analyzer.detect_anomalies(symbol, metrics)
                                
                                if is_anomaly:
                                    self.performance_metrics['anomalies_detected'] += 1
                                    logger.info(f"Anomaly detected in social metrics for {symbol}")
                                    metrics = filtered_metrics
                            
                            # Get enhanced sentiment
                            enhanced_sentiment = await self.analyzer.get_enhanced_sentiment(symbol, metrics)
                            
                            # Fetch recent news
                            news_url = f"{self.base_url}{self.endpoints['feeds']}"
                            news_params = {
                                'symbol': symbol,
                                'limit': 5,
                                'sources': 'news'
                            }
                            
                            # Track API requests
                            self.performance_metrics['api_requests'] += 1
                            
                            recent_news = []
                            try:
                                async with session.get(news_url, headers=headers, params=news_params) as news_response:
                                    if news_response.status == 200:
                                        news_data = await news_response.json()
                                        
                                        if 'data' in news_data:
                                            for news_item in news_data['data']:
                                                # Check if news is within max age
                                                news_time = datetime.fromtimestamp(news_item['time'])
                                                if (datetime.now() - news_time).total_seconds() <= self.max_news_age:
                                                    recent_news.append({
                                                        'title': news_item['title'],
                                                        'sentiment': news_item.get('sentiment', 0),
                                                        'engagement': news_item.get('engagement', 0),
                                                        'source': news_item.get('source', 'unknown'),
                                                        'timestamp': news_time.isoformat()
                                                    })
                            except Exception as e:
                                logger.error(f"Error fetching news for {symbol}: {str(e)}")
                            
                            # Prepare social data with enhanced metrics
                            social_data = {
                                'metrics': metrics,
                                'enhanced_sentiment': enhanced_sentiment,
                                'recent_news': recent_news,
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            # Update cache
                            self.cache[symbol] = social_data
                            self.last_update[symbol] = datetime.now()
                            
                            # Record processing time
                            processing_time = (datetime.now() - start_time).total_seconds()
                            self.performance_metrics['processing_times'].append(processing_time)
                            
                            # Log processing time for monitoring
                            logger.debug(f"Processed metrics for {symbol} in {processing_time:.3f}s")
                            
                            return social_data
                    
                    # Handle API errors
                    self.performance_metrics['api_errors'] += 1
                    logger.error(f"Failed to fetch social metrics for {symbol}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching social metrics for {symbol}: {str(e)}")
            self.performance_metrics['api_errors'] += 1
            return None
    
    async def process_market_updates(self):
        """Process market updates and maintain monitored symbols"""
        while self.running:
            try:
                if not self.redis or not await self.redis.ping():
                    if not await self.connect_redis():
                        await asyncio.sleep(5)
                        continue
                
                # Subscribe to market updates to track active symbols
                pubsub = self.redis.pubsub()
                await pubsub.subscribe('market_updates')
                
                while self.running:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if message:
                        try:
                            market_update = json.loads(message['data'])
                            symbol = market_update['symbol']
                            self.monitored_symbols.add(symbol)
                            
                            # Also add to accuracy assessment set periodically
                            if len(self.accuracy_assessment_symbols) < 10:  # Limit to 10 symbols for efficiency
                                self.accuracy_assessment_symbols.add(symbol)
                        except Exception as e:
                            logger.error(f"Error processing market update: {str(e)}")
                    
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in process_market_updates: {str(e)}")
                await asyncio.sleep(5)
    
    async def update_social_data(self):
        """Update social data for monitored symbols with enhanced metrics"""
        while self.running:
            try:
                if not self.redis or not await self.redis.ping():
                    if not await self.connect_redis():
                        await asyncio.sleep(5)
                        continue
                
                for symbol in self.monitored_symbols:
                    try:
                        # Check if update is needed
                        last_update = self.last_update.get(symbol, datetime.min)
                        if (datetime.now() - last_update).total_seconds() >= self.update_interval:
                            social_data = await self.fetch_social_metrics(symbol)
                            
                            if social_data:
                                # Publish enhanced social update
                                await self.redis.publish(
                                    'enhanced_social_updates',
                                    json.dumps({
                                        'symbol': symbol,
                                        'data': social_data
                                    })
                                )
                                
                                # Also publish to regular social_updates for backward compatibility
                                await self.redis.publish(
                                    'social_updates',
                                    json.dumps({
                                        'symbol': symbol,
                                        'data': social_data
                                    })
                                )
                                
                                # Store latest data
                                await self.redis.hset(
                                    'social_metrics',  # Original key for backward compatibility
                                    symbol,
                                    json.dumps(social_data)
                                )
                                
                                await self.redis.hset(
                                    'enhanced_social_metrics',  # New key for enhanced metrics
                                    symbol,
                                    json.dumps(social_data)
                                )
                                
                                # Store in historical list
                                history_key = f'social_history:{symbol}'
                                await self.redis.lpush(history_key, json.dumps(social_data))
                                
                                # Trim history list to last 168 hours (7 days)
                                await self.redis.ltrim(history_key, 0, 167)
                                
                                logger.info(f"Updated enhanced social metrics for {symbol}")
                    
                    except Exception as e:
                        logger.error(f"Error updating social data for {symbol}: {str(e)}")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in update_social_data: {str(e)}")
                await asyncio.sleep(5)
    
    async def analyze_accuracy(self):
        """Periodically analyze accuracy of social metrics"""
        last_accuracy_analysis = datetime.min
        last_lead_lag_analysis = datetime.min
        
        while self.running:
            try:
                if not self.redis or not await self.redis.ping():
                    if not await self.connect_redis():
                        await asyncio.sleep(5)
                        continue
                
                now = datetime.now()
                
                # Analyze lead-lag relationship periodically
                if (now - last_lead_lag_analysis).total_seconds() >= self.lead_lag_analysis_interval:
                    logger.info("Running lead-lag relationship analysis")
                    
                    for symbol in self.accuracy_assessment_symbols:
                        try:
                            # Detect lead-lag relationship
                            lead_lag_result = await self.analyzer.detect_lead_lag_relationship(symbol)
                            
                            if 'error' not in lead_lag_result:
                                logger.info(f"Lead-lag analysis for {symbol}: "
                                           f"Optimal lag={lead_lag_result.get('strongest_pearson_lag', 0)}h, "
                                           f"Correlation={lead_lag_result.get('strongest_pearson_corr', 0):.4f}")
                        except Exception as e:
                            logger.error(f"Error analyzing lead-lag for {symbol}: {str(e)}")
                    
                    last_lead_lag_analysis = now
                
                # Assess accuracy periodically
                if (now - last_accuracy_analysis).total_seconds() >= self.accuracy_assessment_interval:
                    logger.info("Running social metrics accuracy assessment")
                    
                    # Prepare accuracy report
                    accuracy_report = {
                        'symbols': {},
                        'timestamp': now.isoformat(),
                        'average_direction_accuracy': 0.0,
                        'total_symbols': 0
                    }
                    
                    for symbol in self.accuracy_assessment_symbols:
                        try:
                            # Evaluate sentiment accuracy
                            accuracy_result = await self.analyzer.evaluate_sentiment_accuracy(symbol)
                            
                            if 'error' not in accuracy_result:
                                # Record in report
                                accuracy_report['symbols'][symbol] = {
                                    'direction_accuracy': accuracy_result.get('direction_accuracy', 0.0),
                                    'correlation': accuracy_result.get('correlation', 0.0),
                                    'optimal_lag': accuracy_result.get('optimal_lag', 0),
                                    'r2': accuracy_result.get('r2', 0.0)
                                }
                                
                                accuracy_report['total_symbols'] += 1
                                accuracy_report['average_direction_accuracy'] += accuracy_result.get('direction_accuracy', 0.0)
                                
                                # Update adaptive weights if enabled
                                if self.adaptive_weights_enabled:
                                    await self.analyzer.update_adaptive_weights(symbol)
                        except Exception as e:
                            logger.error(f"Error assessing accuracy for {symbol}: {str(e)}")
                    
                    # Calculate average metrics
                    if accuracy_report['total_symbols'] > 0:
                        accuracy_report['average_direction_accuracy'] /= accuracy_report['total_symbols']
                    
                    # Store report in Redis
                    await self.redis.set(
                        'social_accuracy_report',
                        json.dumps(accuracy_report)
                    )
                    
                    logger.info(f"Completed accuracy assessment for {accuracy_report['total_symbols']} symbols. "
                               f"Average direction accuracy: {accuracy_report['average_direction_accuracy']:.4f}")
                    
                    last_accuracy_analysis = now
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in analyze_accuracy: {str(e)}")
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
                self.analyzer.redis = None
                await asyncio.sleep(5)
    
    async def report_performance(self):
        """Report service performance metrics"""
        while self.running:
            try:
                if not self.redis or not await self.redis.ping():
                    if not await self.connect_redis():
                        await asyncio.sleep(5)
                        continue
                
                # Prepare performance report
                report = {
                    'api_requests': self.performance_metrics['api_requests'],
                    'api_errors': self.performance_metrics['api_errors'],
                    'cache_hits': self.performance_metrics['cache_hits'],
                    'anomalies_detected': self.performance_metrics['anomalies_detected'],
                    'monitored_symbols': len(self.monitored_symbols),
                    'accuracy_assessment_symbols': len(self.accuracy_assessment_symbols)
                }
                
                # Calculate average processing time
                if self.performance_metrics['processing_times']:
                    report['avg_processing_time'] = np.mean(self.performance_metrics['processing_times'])
                    # Keep only the latest 100 processing times
                    if len(self.performance_metrics['processing_times']) > 100:
                        self.performance_metrics['processing_times'] = self.performance_metrics['processing_times'][-100:]
                else:
                    report['avg_processing_time'] = 0.0
                
                # Get memory usage
                report['memory_usage_mb'] = 0.0  # Placeholder
                
                # Add timestamp
                report['timestamp'] = datetime.now().isoformat()
                
                # Store in Redis
                await self.redis.set('enhanced_social_monitor_performance', json.dumps(report))
                
                # Log performance summary
                logger.info(f"Performance: {report['api_requests']} API requests, "
                           f"{report['api_errors']} errors, "
                           f"{report['anomalies_detected']} anomalies, "
                           f"{report['avg_processing_time']:.3f}s avg. processing time")
                
                await asyncio.sleep(300)  # Report every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in report_performance: {str(e)}")
                await asyncio.sleep(60)
    
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
        """Run the enhanced social monitor service"""
        try:
            logger.info("Starting Enhanced Social Monitor Service...")
            
            # First establish Redis connection
            if not await self.connect_redis():
                raise Exception("Failed to establish initial Redis connection")
            
            # Create tasks
            tasks = [
                asyncio.create_task(self.process_market_updates()),
                asyncio.create_task(self.update_social_data()),
                asyncio.create_task(self.analyze_accuracy()),
                asyncio.create_task(self.maintain_redis()),
                asyncio.create_task(self.report_performance()),
                asyncio.create_task(self.health_check_server())
            ]
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Error in Enhanced Social Monitor Service: {str(e)}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the enhanced social monitor service"""
        logger.info("Stopping Enhanced Social Monitor Service...")
        self.running = False
        if self.redis:
            await self.redis.close()

if __name__ == "__main__":
    service = EnhancedSocialMonitorService()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        asyncio.run(service.stop())
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        asyncio.run(service.stop())