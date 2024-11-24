import os
import json
import socket
import asyncio
from datetime import datetime
import logging as logger
from openai import AsyncOpenAI
from redis.asyncio import Redis
from redis.exceptions import ConnectionError
from ai_trader import AITrader  # Simple import since we run from root directory

# Configure logging with more debug information
logger.basicConfig(
    level=logger.DEBUG,  # Ensure DEBUG level logging
    format='%(asctime)s - %(levelname)s - [AIAnalyzer] %(message)s',
    handlers=[
        logger.FileHandler('logs/ai_analyzer.log'),
        logger.StreamHandler()
    ]
)

class AIAnalyzerService:
    def __init__(self):
        logger.debug("Initializing AI Analyzer Service...")
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        logger.debug(f"Loaded configuration: {json.dumps(self.config, indent=2)}")

        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = AsyncOpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI client with model: {self.config['openai']['model']}")

        # Initialize AITrader
        logger.debug("Initializing AITrader...")
        self.ai_trader = AITrader(self.config)  # Instantiate AITrader
        logger.debug("AITrader initialized successfully")

        # Redis configuration
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        logger.debug(f"Redis configuration - Host: {self.redis_host}, Port: {self.redis_port}")

        # Redis will be initialized in connect_redis
        self.redis = None
        self.pubsub = None
        self.running = True
        self.market_data = {}
        self.last_analysis_time = {}
        
        # Get service port from environment variable
        self.service_port = int(os.getenv('SERVICE_PORT', 8003))  # Default to 8003 (AI_ANALYZER_PORT)
        logger.debug(f"Service port configured as: {self.service_port}")
        logger.debug("AI Analyzer Service initialization complete")

    async def connect_redis(self, max_retries=10, retry_delay=5):
        """Establish Redis connection with retries"""
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

    async def setup_pubsub(self):
        """Set up Redis pubsub connection"""
        try:
            # Verify Redis connection
            if not self.redis or not await self.redis.ping():
                logger.error("Redis connection not available for pubsub setup")
                return False

            # Close existing pubsub if any
            if self.pubsub:
                logger.debug("Closing existing pubsub connection")
                await self.pubsub.close()
                self.pubsub = None

            # Create new pubsub instance
            logger.debug("Creating new pubsub instance")
            self.pubsub = self.redis.pubsub()
            
            # Subscribe to channel
            logger.debug("Subscribing to market_updates channel")
            await self.pubsub.subscribe('market_updates')
            
            # Get first message to confirm subscription
            logger.debug("Waiting for subscription confirmation message")
            message = await self.pubsub.get_message(timeout=1.0)
            
            if message and message['type'] == 'subscribe':
                logger.info("Successfully subscribed to market_updates channel")
                return True
            else:
                logger.error(f"Unexpected subscription response: {message}")
                return False

        except Exception as e:
            logger.error(f"Error in pubsub setup: {str(e)}", exc_info=True)
            if self.pubsub:
                await self.pubsub.close()
                self.pubsub = None
            return False

    async def analyze_market_data(self, market_update):
        """Analyze market data using AITrader"""
        try:
            symbol = market_update['symbol']
            current_time = datetime.now()

            # Check if we need to analyze this symbol
            if symbol in self.last_analysis_time:
                time_since_last = (current_time - self.last_analysis_time[symbol]).seconds
                if time_since_last < self.config['trading_params']['ai_analysis_interval']:
                    logger.debug(f"Skipping analysis for {symbol}, last analysis was {time_since_last}s ago")
                    return

            logger.info(f"Starting analysis for {symbol}")
            logger.debug(f"Market update data: {json.dumps(market_update, indent=2)}")

            # Use AITrader to analyze the trading opportunity
            logger.debug("Calling AITrader.analyze_trade_opportunity...")
            analysis = await self.ai_trader.analyze_trade_opportunity(market_update)
            logger.debug(f"Received analysis from AITrader: {json.dumps(analysis, indent=2)}")

            # Add metadata
            analysis['timestamp'] = current_time.isoformat()
            analysis['symbol'] = symbol
            analysis['market_data'] = {
                'price': market_update['current_price'],
                'volume': market_update['avg_volume'],
                'price_change_5m': market_update['price_change_5m'],
                'price_change_15m': market_update['price_change_15m'],
                'technical_indicators': {
                    'rsi': market_update['rsi'],
                    'stoch_k': market_update['stoch_k'],
                    'macd': market_update['macd'],
                    'williams_r': market_update['williams_r'],
                    'bb_position': market_update['bb_position']
                },
                'trend': {
                    'direction': market_update['trend'],
                    'strength': market_update['trend_strength']
                }
            }

            # Log the analysis
            logger.info(f"AI Analysis for {symbol}:")
            logger.info(f"Decision: {analysis['decision']}")
            logger.info(f"Confidence: {analysis['confidence']}")
            logger.info(f"Reasoning: {analysis['reasoning']}")

            # Publish analysis to Redis
            if self.redis and await self.redis.ping():
                logger.info(f"Publishing trading signal for {symbol}")
                await self.redis.publish('trading_signals', json.dumps(analysis))
                logger.info(f"Published trading signal for {symbol}: {json.dumps(analysis)}")
            else:
                logger.error("Redis connection lost during analysis publishing")
                await self.connect_redis()

            # Update last analysis time
            self.last_analysis_time[symbol] = current_time

        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}", exc_info=True)

    async def process_market_updates(self):
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
                if not self.pubsub:
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
                        if message['type'] == 'message':
                            try:
                                logger.debug(f"Raw message data: {message['data']}")
                                market_update = json.loads(message['data'])
                                logger.info(f"Processing market update for {market_update['symbol']}")
                                await self.analyze_market_data(market_update)
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse market update: {e}")
                                logger.error(f"Invalid JSON data: {message['data']}")
                            except KeyError as e:
                                logger.error(f"Missing required field in market update: {e}")
                                logger.error(f"Market update data: {market_update}")
                            except Exception as e:
                                logger.error(f"Error processing market update: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error getting message from pubsub: {str(e)}")
                    self.pubsub = None  # Force pubsub reconnection
                    continue

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in process_market_updates: {str(e)}", exc_info=True)
                if self.pubsub:
                    await self.pubsub.close()
                    self.pubsub = None
                await asyncio.sleep(5)

    async def maintain_redis(self):
        """Maintain Redis connection"""
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
        """Run the AI analyzer service"""
        try:
            logger.info("Starting AI Analyzer Service...")
            
            # First establish Redis connection with increased retries
            if not await self.connect_redis(max_retries=15, retry_delay=2):
                raise Exception("Failed to establish initial Redis connection")
            
            # Create tasks for market updates processing, Redis maintenance, and health check
            tasks = [
                asyncio.create_task(self.process_market_updates()),
                asyncio.create_task(self.maintain_redis()),
                asyncio.create_task(self.health_check_server())
            ]
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Error in AI Analyzer Service: {str(e)}")
        finally:
            await self.stop()

    def start(self):
        """Start the service"""
        try:
            # Create and run event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.run())
        except KeyboardInterrupt:
            asyncio.run(self.stop())
        except Exception as e:
            logger.error(f"Critical error: {str(e)}")
            asyncio.run(self.stop())

    async def stop(self):
        """Stop the AI analyzer service"""
        logger.info("Stopping AI Analyzer Service...")
        self.running = False
        if self.pubsub:
            await self.pubsub.close()
        if self.redis:
            await self.redis.close()

if __name__ == "__main__":
    service = AIAnalyzerService()
    service.start()
