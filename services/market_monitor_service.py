import os
import json
import socket
import asyncio
import websockets
from datetime import datetime
from binance.client import Client
import logging as logger
from redis.asyncio import Redis
from redis.exceptions import ConnectionError
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands

# Configure logging
logger.basicConfig(
    level=logger.DEBUG,  # Changed to DEBUG level
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

        # Redis will be initialized in connect_redis
        self.redis = None
        self.market_data = {}
        self.historical_data = {}  # Store historical data for technical analysis
        self.running = True
        self.ws_url = "wss://stream.binance.com:9443/ws/!miniTicker@arr"
        self.health_check_port = int(os.getenv('MARKET_MONITOR_PORT', 8001))

    async def connect_redis(self, max_retries=5, retry_delay=5):
        """Establish Redis connection with retries"""
        retries = 0
        while retries < max_retries:
            try:
                if self.redis is None:
                    self.redis = Redis(
                        host=os.getenv('REDIS_HOST', 'redis'),
                        port=int(os.getenv('REDIS_PORT', 6379)),
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

    def get_historical_data(self, symbol):
        """Get historical klines/candlestick data"""
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval='5m',  # 5-minute candles
                limit=100  # Last 100 candles
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Convert numeric columns to float
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return None

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        try:
            # RSI
            rsi = RSIIndicator(close=df['close'])
            df['rsi'] = rsi.rsi()

            # Stochastic
            stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
            df['stoch_k'] = stoch.stoch()

            # MACD
            macd = MACD(close=df['close'])
            df['macd'] = macd.macd()

            # Williams %R
            williams = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'])
            df['williams_r'] = williams.williams_r()

            # Bollinger Bands
            bb = BollingerBands(close=df['close'])
            df['bb_high'] = bb.bollinger_hband()
            df['bb_low'] = bb.bollinger_lband()
            df['bb_mid'] = bb.bollinger_mavg()
            
            # Calculate BB position
            bb_range = df['bb_high'] - df['bb_low']
            df['bb_position'] = (df['close'] - df['bb_low']) / bb_range.replace(0, np.nan)

            # Moving Averages for Trend
            sma20 = SMAIndicator(close=df['close'], window=20)
            sma50 = SMAIndicator(close=df['close'], window=50)
            df['sma_20'] = sma20.sma_indicator()
            df['sma_50'] = sma50.sma_indicator()

            # Determine trend
            last_close = float(df['close'].iloc[-1])
            sma20_last = float(df['sma_20'].iloc[-1])
            sma50_last = float(df['sma_50'].iloc[-1])
            
            # Calculate trend strength
            strength = ((last_close - sma20_last) / sma20_last * 100 +
                       (last_close - sma50_last) / sma50_last * 100) / 2
            
            if last_close > sma20_last and sma20_last > sma50_last:
                trend = 'uptrend'
            elif last_close < sma20_last and sma20_last < sma50_last:
                trend = 'downtrend'
            else:
                trend = 'sideways'

            return {
                'rsi': float(df['rsi'].iloc[-1]),
                'stoch_k': float(df['stoch_k'].iloc[-1]),
                'macd': float(df['macd'].iloc[-1]),
                'williams_r': float(df['williams_r'].iloc[-1]),
                'bb_position': float(df['bb_position'].iloc[-1]),
                'trend': trend,
                'trend_strength': abs(strength)
            }
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return None

    async def process_market_data(self, msg):
        """Process incoming market data and publish to Redis"""
        try:
            if not self.redis or not await self.redis.ping():
                if not await self.connect_redis():
                    return

            data = json.loads(msg)
            if not isinstance(data, list):
                return

            for ticker in data:
                if not isinstance(ticker, dict) or 's' not in ticker:
                    continue

                symbol = ticker['s']
                if symbol.endswith('USDC'):
                    # Process market data
                    price = float(ticker['c'])
                    volume = float(ticker['v']) * price
                    price_change = ((price - float(ticker['o'])) / float(ticker['o'])) * 100

                    # Get historical data and calculate indicators
                    df = self.get_historical_data(symbol)
                    if df is not None:
                        indicators = self.calculate_technical_indicators(df)
                        if indicators:
                            # Create complete market update
                            market_update = {
                                'symbol': symbol,
                                'current_price': price,
                                'avg_volume': volume,
                                'price_change': price_change,
                                'timestamp': datetime.now().isoformat(),
                                'rsi': indicators['rsi'],
                                'stoch_k': indicators['stoch_k'],
                                'macd': indicators['macd'],
                                'williams_r': indicators['williams_r'],
                                'bb_position': indicators['bb_position'],
                                'trend': indicators['trend'],
                                'trend_strength': indicators['trend_strength'],
                                'price_change_5m': price_change,  # Using current price change as 5m for now
                                'price_change_15m': price_change * 3  # Approximating 15m change
                            }

                            # Store in local cache
                            self.market_data[symbol] = market_update

                            # Log market update before publishing
                            logger.debug(f"Publishing market update to Redis: {json.dumps(market_update)}")

                            # Publish to Redis
                            await self.redis.publish(
                                'market_updates',
                                json.dumps(market_update)
                            )

                            # Store latest data in Redis
                            await self.redis.hset(
                                'current_prices',
                                symbol,
                                json.dumps(market_update)
                            )

                            logger.info(f"Market update - {symbol}: ${price:.8f} (24h volume: ${volume:.2f}, RSI: {indicators['rsi']:.2f})")

                            # Check for trading opportunities
                            if abs(price_change) >= self.config['trading_params'].get('min_price_change_pct', 1.0) and \
                               volume >= self.config['trading_params']['min_volume_usdc']:
                                await self.redis.publish(
                                    'trading_opportunities',
                                    json.dumps(market_update)
                                )
                                logger.info(f"Found opportunity: {symbol} at ${price:.8f} (change: {price_change:.2f}%)")

        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")
            logger.error(f"Message content: {msg}")

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

    async def websocket_handler(self):
        """Handle WebSocket connection"""
        while self.running:
            try:
                async with websockets.connect(self.ws_url) as websocket:
                    logger.info("WebSocket connection established")
                    while self.running:
                        try:
                            message = await websocket.recv()
                            await self.process_market_data(message)
                        except websockets.exceptions.ConnectionClosed:
                            logger.error("WebSocket connection closed")
                            break
                        except Exception as e:
                            logger.error(f"Error in WebSocket handler: {str(e)}")
                            break
            except Exception as e:
                logger.error(f"WebSocket connection error: {str(e)}")
                await asyncio.sleep(5)  # Wait before reconnecting

    async def health_check_server(self):
        """Run a simple TCP server for health checks"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server.bind(('0.0.0.0', self.health_check_port))
            server.listen(1)
            server.setblocking(False)
            
            logger.info(f"Health check server listening on port {self.health_check_port}")
            
            while self.running:
                try:
                    await asyncio.sleep(1)
                    # The socket exists and is listening, which is enough for the Docker healthcheck
                except Exception as e:
                    logger.error(f"Health check server error: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to start health check server: {str(e)}")
        finally:
            server.close()

    async def run(self):
        """Run the market monitor service"""
        try:
            logger.info("Starting Market Monitor Service...")
            
            # First establish Redis connection
            if not await self.connect_redis():
                raise Exception("Failed to establish initial Redis connection")
            
            # Create tasks for WebSocket, Redis maintenance, and health check server
            tasks = [
                asyncio.create_task(self.websocket_handler()),
                asyncio.create_task(self.maintain_redis()),
                asyncio.create_task(self.health_check_server())
            ]
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Error in Market Monitor Service: {str(e)}")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the market monitor service"""
        logger.info("Stopping Market Monitor Service...")
        self.running = False
        if self.redis:
            await self.redis.close()

if __name__ == "__main__":
    service = MarketMonitorService()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        asyncio.run(service.stop())
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        asyncio.run(service.stop())
