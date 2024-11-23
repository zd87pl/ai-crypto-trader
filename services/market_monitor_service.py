import os
import json
import redis
import asyncio
from datetime import datetime
from binance.client import Client
from binance.streams import ThreadedWebsocketManager
import logging as logger

# Configure logging
logger.basicConfig(
    level=logger.INFO,
    format='%(asctime)s - %(levelname)s - [MarketMonitor] %(message)s',
    handlers=[
        logger.FileHandler('logs/market_monitor.log'),
        logger.StreamHandler()
    ]
)

class MarketMonitorService:
    def __init__(self):
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)

        # Initialize Binance client
        self.client = Client(
            os.getenv('BINANCE_API_KEY'),
            os.getenv('BINANCE_API_SECRET')
        )

        # Initialize Redis connection
        self.redis = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=True
        )

        # Initialize WebSocket manager
        self.twm = ThreadedWebsocketManager(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_API_SECRET')
        )

        self.market_data = {}
        self.running = True

    def process_market_data(self, msg):
        """Process incoming market data and publish to Redis"""
        try:
            if isinstance(msg, dict) and msg.get('e') == '24hrMiniTicker':
                data = msg
            elif isinstance(msg, dict) and isinstance(msg.get('data'), dict):
                data = msg['data']
            else:
                return

            if data.get('e') == '24hrMiniTicker':
                symbol = data['s']
                if symbol.endswith('USDC'):
                    # Process market data
                    price = float(data['c'])
                    volume = float(data['v']) * price
                    price_change = ((price - float(data['o'])) / float(data['o'])) * 100

                    market_update = {
                        'symbol': symbol,
                        'price': price,
                        'volume': volume,
                        'price_change': price_change,
                        'timestamp': datetime.now().isoformat()
                    }

                    # Store in local cache
                    self.market_data[symbol] = market_update

                    # Publish to Redis
                    self.redis.publish(
                        'market_updates',
                        json.dumps(market_update)
                    )

                    # Store latest price in Redis
                    self.redis.hset(
                        'current_prices',
                        symbol,
                        json.dumps(market_update)
                    )

                    logger.info(f"Market update - {symbol}: ${price:.8f} (24h volume: ${volume:.2f}, change: {price_change:.2f}%)")

                    # Check for trading opportunities
                    if volume >= self.config['trading_params']['min_volume_usdc']:
                        self.redis.publish(
                            'trading_opportunities',
                            json.dumps(market_update)
                        )
                        logger.info(f"Found opportunity: {symbol} at ${price:.8f}")

        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")
            logger.error(f"Message content: {msg}")

    def start(self):
        """Start the market monitor service"""
        try:
            logger.info("Starting Market Monitor Service...")

            # Start WebSocket manager
            self.twm.start()

            # Subscribe to market data
            self.twm.start_miniticker_socket(
                callback=self.process_market_data
            )

            logger.info("Successfully subscribed to market data streams")

            # Keep service running
            while self.running:
                # Ping Redis to maintain connection
                self.redis.ping()
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error in Market Monitor Service: {str(e)}")
        finally:
            self.stop()

    def stop(self):
        """Stop the market monitor service"""
        logger.info("Stopping Market Monitor Service...")
        self.running = False
        self.twm.stop()
        self.redis.close()

if __name__ == "__main__":
    service = MarketMonitorService()
    try:
        service.start()
    except KeyboardInterrupt:
        service.stop()
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        service.stop()
