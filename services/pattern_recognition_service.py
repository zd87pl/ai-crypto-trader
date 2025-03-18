import os
import json
import socket
import asyncio
from datetime import datetime
import logging as logger
from logging.handlers import RotatingFileHandler
from redis.asyncio import Redis
from redis.exceptions import ConnectionError
import pandas as pd
import numpy as np
from services.utils.pattern_recognition import ChartPatternRecognitionService

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure rotating file handler
rotating_handler = RotatingFileHandler(
    'logs/pattern_recognition.log',
    maxBytes=10*1024*1024,  # 10MB per file
    backupCount=5,  # Keep 5 backup files
    encoding='utf-8'
)

# Configure logging with rotation
logger.basicConfig(
    level=logger.DEBUG,
    format='%(asctime)s - %(levelname)s - [PatternRecognition] %(message)s',
    handlers=[
        rotating_handler,
        logger.StreamHandler()
    ]
)

class PatternRecognitionService:
    """
    Service that detects chart patterns using deep learning models
    and provides pattern-based trading signals.
    """
    
    def __init__(self):
        """Initialize the Pattern Recognition Service"""
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
            
        # Redis will be initialized in connect_redis
        self.redis = None
        self.market_data = {}
        self.historical_data = {}
        self.pattern_data = {}
        self.running = True
        
        # Get service port from environment variable
        self.service_port = int(os.getenv('SERVICE_PORT', 8004))
        
        # Initialize pattern recognition service
        self.pattern_service = ChartPatternRecognitionService(self.config)
        
        # Flag to indicate if we have a trained model
        self.model_available = True
        
        # Set update interval for pattern detection
        pattern_config = self.config.get('pattern_recognition', {})
        self.update_interval = pattern_config.get('analysis_interval', 300)  # 5 minutes by default
        self.last_update_time = {}
        self.confidence_threshold = pattern_config.get('confidence_threshold', 0.6)
        
    async def connect_redis(self, max_retries=10, retry_delay=5):
        """Establish Redis connection with retries"""
        retries = 0
        while retries < max_retries and self.running:
            try:
                if self.redis is None:
                    logger.debug(f"Attempting Redis connection (attempt {retries + 1}/{max_retries})")
                    self.redis = Redis(
                        host=os.getenv('REDIS_HOST', 'redis'),
                        port=int(os.getenv('REDIS_PORT', 6379)),
                        decode_responses=True,
                        socket_connect_timeout=5.0,
                        socket_keepalive=True,
                        health_check_interval=15
                    )
                await self.redis.ping()
                logger.info(f"Successfully connected to Redis")
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
    
    async def setup_pubsub(self):
        """Set up Redis pubsub connection"""
        try:
            # Verify Redis connection
            if not self.redis or not await self.redis.ping():
                logger.error("Redis connection not available for pubsub setup")
                return False
                
            # Create new pubsub instance
            logger.debug("Creating new pubsub instance")
            self.pubsub = self.redis.pubsub()
            
            # Subscribe to market data channel
            logger.debug("Subscribing to market_updates channel")
            await self.pubsub.subscribe('market_updates')
            
            # Get first message to confirm subscription
            message = await self.pubsub.get_message(timeout=1.0)
            if message and message['type'] == 'subscribe':
                logger.info("Successfully subscribed to market_updates channel")
                return True
            else:
                logger.error("Failed to subscribe to market_updates channel")
                return False
                
        except Exception as e:
            logger.error(f"Error in pubsub setup: {str(e)}", exc_info=True)
            if hasattr(self, 'pubsub') and self.pubsub:
                await self.pubsub.close()
                self.pubsub = None
            return False
    
    async def analyze_chart_patterns(self, market_update):
        """
        Analyze chart patterns for a specific symbol
        
        Args:
            market_update: Market data update from market_monitor
        """
        try:
            symbol = market_update['symbol']
            current_time = datetime.now()
            
            # Check if we need to analyze patterns for this symbol
            if symbol in self.last_update_time:
                time_since_last = (current_time - self.last_update_time[symbol]).seconds
                if time_since_last < self.update_interval:
                    logger.debug(f"Skipping pattern analysis for {symbol}, last analysis was {time_since_last}s ago")
                    return
            
            # Get historical data for pattern detection
            if symbol not in self.historical_data or 'data_1m' not in self.historical_data[symbol]:
                # We need to obtain historical data
                logger.debug(f"No historical data available for {symbol}, fetching from Redis")
                
                # Try to get historical data from Redis
                data_key = f"historical_data:{symbol}"
                historical_data_json = await self.redis.get(data_key)
                
                if not historical_data_json:
                    logger.warning(f"No historical data found in Redis for {symbol}")
                    return
                
                try:
                    self.historical_data[symbol] = json.loads(historical_data_json)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON data for historical data of {symbol}")
                    return
            
            # Use 5-minute data for pattern recognition if available
            if 'data_5m' in self.historical_data[symbol]:
                df = pd.DataFrame(self.historical_data[symbol]['data_5m'])
            elif 'data_1m' in self.historical_data[symbol]:
                df = pd.DataFrame(self.historical_data[symbol]['data_1m'])
            else:
                logger.warning(f"No suitable timeframe data found for {symbol}")
                return
            
            # Perform pattern detection
            logger.info(f"Analyzing chart patterns for {symbol}")
            pattern_results = self.pattern_service.analyze_chart_patterns(df, symbol)
            
            if "error" in pattern_results:
                logger.error(f"Error detecting patterns for {symbol}: {pattern_results['error']}")
                return
            
            if pattern_results.get("enabled", True) == False:
                logger.info(f"Pattern recognition is disabled for {symbol}")
                return
            
            # Store pattern results
            self.pattern_data[symbol] = pattern_results
            self.last_update_time[symbol] = current_time
            
            # Log detected pattern
            pattern = pattern_results.get("primary_pattern", "unknown")
            confidence = pattern_results.get("confidence", 0.0)
            logger.info(f"Detected pattern for {symbol}: {pattern} (confidence: {confidence:.2f})")
            
            # Get trading signals from pattern
            trading_signals = self.pattern_service.get_pattern_trading_signals(pattern_results)
            
            # Only publish signals with sufficient strength
            if trading_signals['signal'] != 'neutral' and trading_signals['strength'] >= 0.3:
                # Add symbol and timestamp to signals
                trading_signals['symbol'] = symbol
                trading_signals['timestamp'] = current_time.isoformat()
                trading_signals['source'] = 'pattern_recognition'
                
                # Publish pattern signals to Redis
                await self.redis.publish('pattern_signals', json.dumps(trading_signals))
                logger.info(f"Published pattern signal for {symbol}: {trading_signals['signal']} (strength: {trading_signals['strength']})")
                
                # Store pattern data in Redis for other services
                pattern_key = f"pattern:{symbol}"
                await self.redis.set(pattern_key, json.dumps(pattern_results))
                
        except Exception as e:
            logger.error(f"Error in pattern analysis for {market_update['symbol']}: {str(e)}", exc_info=True)
    
    async def process_updates(self):
        """Process market updates from Redis"""
        logger.debug("Starting market updates processing...")
        pubsub_retry_count = 0
        max_pubsub_retries = 10
        
        while self.running:
            try:
                # Ensure Redis connection
                if not self.redis or not await self.redis.ping():
                    logger.debug("Redis connection not available, attempting to connect...")
                    if not await self.connect_redis():
                        await asyncio.sleep(5)
                        continue
                
                # Setup pubsub if needed
                if not hasattr(self, 'pubsub') or self.pubsub is None:
                    logger.debug(f"Setting up pubsub (attempt {pubsub_retry_count + 1}/{max_pubsub_retries})")
                    if await self.setup_pubsub():
                        logger.info("Pubsub setup successful")
                        pubsub_retry_count = 0  # Reset counter on success
                    else:
                        pubsub_retry_count += 1
                        if pubsub_retry_count >= max_pubsub_retries:
                            logger.error("Max pubsub retry attempts reached")
                            raise Exception("Failed to set up pubsub after maximum retries")
                        await asyncio.sleep(5)
                        continue
                
                # Process messages
                try:
                    logger.debug("Waiting for message from Redis...")
                    message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    
                    if message:
                        logger.debug(f"Received message type: {message['type']}")
                        if message['type'] == 'message' and message['channel'] == 'market_updates':
                            try:
                                logger.debug(f"Raw message data: {message['data']}")
                                data = json.loads(message['data'])
                                
                                # Update market data
                                symbol = data['symbol']
                                self.market_data[symbol] = data
                                
                                # Update historical data from the market update if available
                                if 'historical_data' in data:
                                    self.historical_data[symbol] = data['historical_data']
                                
                                # Analyze chart patterns
                                await self.analyze_chart_patterns(data)
                                
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse market update: {e}")
                            except KeyError as e:
                                logger.error(f"Missing required field in market update: {e}")
                            except Exception as e:
                                logger.error(f"Error processing market update: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error getting message from pubsub: {str(e)}")
                    self.pubsub = None  # Force pubsub reconnection
                    continue
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in process_updates: {str(e)}", exc_info=True)
                if hasattr(self, 'pubsub') and self.pubsub:
                    await self.pubsub.close()
                    self.pubsub = None
                await asyncio.sleep(5)
    
    async def generate_combined_analysis(self):
        """
        Generate and publish combined pattern analysis report
        This runs periodically to provide a summary of all detected patterns
        """
        while self.running:
            try:
                # Wait for next scheduled run
                await asyncio.sleep(300)  # Run every 5 minutes
                
                if not self.pattern_data:
                    logger.debug("No pattern data available for combined analysis")
                    continue
                
                # Get signals for all patterns
                pattern_signals = self.pattern_service.get_all_pattern_signals()
                
                if not pattern_signals:
                    logger.debug("No significant pattern signals found")
                    continue
                
                # Create combined report
                now = datetime.now()
                report = {
                    'timestamp': now.isoformat(),
                    'signals': pattern_signals,
                    'summary': {
                        'bullish_patterns': sum(1 for s in pattern_signals.values() if s['bias'] == 'bullish'),
                        'bearish_patterns': sum(1 for s in pattern_signals.values() if s['bias'] == 'bearish'),
                        'neutral_patterns': sum(1 for s in pattern_signals.values() if s['bias'] == 'neutral'),
                        'strongest_signal': max(
                            pattern_signals.items(), 
                            key=lambda x: x[1]['strength'], 
                            default=(None, {'strength': 0})
                        )
                    }
                }
                
                # If we have Redis connection, publish the report
                if self.redis and await self.redis.ping():
                    await self.redis.set('pattern_analysis_report', json.dumps(report))
                    logger.info(f"Published combined pattern analysis report with {len(pattern_signals)} signals")
                
            except Exception as e:
                logger.error(f"Error generating combined pattern analysis: {str(e)}", exc_info=True)
                await asyncio.sleep(60)  # Wait a bit and retry
    
    async def maintain_redis(self):
        """Maintain Redis connection"""
        logger.debug("Starting Redis connection maintenance...")
        while self.running:
            try:
                if self.redis:
                    await self.redis.ping()
                else:
                    await self.connect_redis()
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Redis connection error: {str(e)}")
                self.redis = None
                await asyncio.sleep(5)
    
    async def health_check_server(self):
        """Run a simple TCP server for health checks"""
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
    
    async def run(self):
        """Run the pattern recognition service"""
        try:
            logger.info("Starting Pattern Recognition Service...")
            
            # First establish Redis connection with increased retries
            if not await self.connect_redis(max_retries=15, retry_delay=2):
                raise Exception("Failed to establish initial Redis connection")
            
            # Create tasks for updates processing, Redis maintenance, and health check
            tasks = [
                asyncio.create_task(self.process_updates()),
                asyncio.create_task(self.maintain_redis()),
                asyncio.create_task(self.health_check_server()),
                asyncio.create_task(self.generate_combined_analysis())
            ]
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Error in Pattern Recognition Service: {str(e)}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the pattern recognition service"""
        logger.info("Stopping Pattern Recognition Service...")
        self.running = False
        if hasattr(self, 'pubsub') and self.pubsub:
            await self.pubsub.close()
        if self.redis:
            await self.redis.close()

if __name__ == "__main__":
    service = PatternRecognitionService()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        asyncio.run(service.stop())
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        asyncio.run(service.stop())