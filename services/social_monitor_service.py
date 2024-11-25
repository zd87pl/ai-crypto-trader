import os
import json
import socket
import asyncio
import aiohttp
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
    'logs/social_monitor.log',
    maxBytes=10*1024*1024,  # 10MB per file
    backupCount=5,  # Keep 5 backup files
    encoding='utf-8'
)

# Configure logging with rotation
logger.basicConfig(
    level=logger.DEBUG,
    format='%(asctime)s - %(levelname)s - [SocialMonitor] %(message)s',
    handlers=[
        rotating_handler,
        logger.StreamHandler()
    ]
)

class SocialMonitorService:
    def __init__(self):
        logger.debug("Initializing Social Monitor Service...")
        
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
        
        # Get service port from environment variable
        self.service_port = int(os.getenv('SOCIAL_MONITOR_PORT', 8004))
        logger.debug(f"Service port configured as: {self.service_port}")
        logger.debug("Social Monitor Service initialization complete")

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

    async def fetch_social_metrics(self, symbol: str) -> Optional[Dict]:
        """Fetch social metrics from LunarCrush API"""
        try:
            # Check cache first
            if symbol in self.cache:
                cache_time = self.last_update.get(symbol, datetime.min)
                if (datetime.now() - cache_time).total_seconds() < self.cache_duration:
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
                                'reddit_volume': asset_data.get('reddit_volume', 0),
                                'news_volume': asset_data.get('news_volume', 0)
                            }
                            
                            # Calculate weighted sentiment
                            weighted_sentiment = sum(
                                metrics[metric] * weight
                                for metric, weight in self.sentiment_weights.items()
                                if metric in metrics
                            )
                            
                            # Fetch recent news
                            news_url = f"{self.base_url}{self.endpoints['feeds']}"
                            news_params = {
                                'symbol': symbol,
                                'limit': 5,
                                'sources': 'news'
                            }
                            
                            async with session.get(news_url, headers=headers, params=news_params) as news_response:
                                if news_response.status == 200:
                                    news_data = await news_response.json()
                                    recent_news = []
                                    
                                    if 'data' in news_data:
                                        for news_item in news_data['data']:
                                            # Check if news is within max age
                                            news_time = datetime.fromtimestamp(news_item['time'])
                                            if (datetime.now() - news_time).total_seconds() <= self.max_news_age:
                                                recent_news.append({
                                                    'title': news_item['title'],
                                                    'sentiment': news_item.get('sentiment', 0),
                                                    'engagement': news_item.get('engagement', 0)
                                                })
                            
                            # Prepare social data
                            social_data = {
                                'metrics': metrics,
                                'weighted_sentiment': weighted_sentiment,
                                'recent_news': recent_news,
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            # Update cache
                            self.cache[symbol] = social_data
                            self.last_update[symbol] = datetime.now()
                            
                            return social_data
                    
                    logger.error(f"Failed to fetch social metrics for {symbol}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching social metrics for {symbol}: {str(e)}")
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
                        except Exception as e:
                            logger.error(f"Error processing market update: {str(e)}")
                    
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in process_market_updates: {str(e)}")
                await asyncio.sleep(5)

    async def update_social_data(self):
        """Update social data for monitored symbols"""
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
                                # Publish social update
                                await self.redis.publish(
                                    'social_updates',
                                    json.dumps({
                                        'symbol': symbol,
                                        'data': social_data
                                    })
                                )
                                
                                # Store latest data
                                await self.redis.hset(
                                    'social_metrics',
                                    symbol,
                                    json.dumps(social_data)
                                )
                                
                                logger.info(f"Updated social metrics for {symbol}")
                                logger.debug(f"Social data: {json.dumps(social_data, indent=2)}")
                    
                    except Exception as e:
                        logger.error(f"Error updating social data for {symbol}: {str(e)}")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in update_social_data: {str(e)}")
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
        """Run the social monitor service"""
        try:
            logger.info("Starting Social Monitor Service...")
            
            # First establish Redis connection
            if not await self.connect_redis():
                raise Exception("Failed to establish initial Redis connection")
            
            # Create tasks
            tasks = [
                asyncio.create_task(self.process_market_updates()),
                asyncio.create_task(self.update_social_data()),
                asyncio.create_task(self.maintain_redis()),
                asyncio.create_task(self.health_check_server())
            ]
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Error in Social Monitor Service: {str(e)}")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the social monitor service"""
        logger.info("Stopping Social Monitor Service...")
        self.running = False
        if self.redis:
            await self.redis.close()

if __name__ == "__main__":
    service = SocialMonitorService()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        asyncio.run(service.stop())
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        asyncio.run(service.stop())
