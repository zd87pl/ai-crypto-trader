import os
import json
import socket
import asyncio
import websockets
from datetime import datetime
from binance.client import Client
import logging as logger
from logging.handlers import RotatingFileHandler
from redis.asyncio import Redis
from redis.exceptions import ConnectionError
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands
from services.utils.indicator_combinations import calculate_indicator_combinations
from services.utils.volume_profile_analyzer import VolumeProfileAnalyzer

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure rotating file handler
rotating_handler = RotatingFileHandler(
    'logs/market_monitor.log',
    maxBytes=10*1024*1024,  # 10MB per file
    backupCount=5,  # Keep 5 backup files
    encoding='utf-8'
)

# Configure logging with rotation
logger.basicConfig(
    level=logger.DEBUG,
    format='%(asctime)s - %(levelname)s - [MarketMonitor] %(message)s',
    handlers=[
        rotating_handler,
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
        self.volume_profiles = {}  # Store volume profile analysis by symbol
        self.running = True
        self.ws_url = "wss://stream.binance.com:9443/ws/!miniTicker@arr"
        self.health_check_port = int(os.getenv('MARKET_MONITOR_PORT', 8001))
        
        # Initialize volume profile analyzer
        self.volume_profile_analyzer = VolumeProfileAnalyzer(
            num_bins=self.config.get('volume_profile', {}).get('num_bins', 20),
            value_area_pct=self.config.get('volume_profile', {}).get('value_area_pct', 0.7)
        )
        
        # Rate limiting settings
        self.update_interval = 5  # Reduced from 15 to 5 seconds between updates
        self.last_update_time = {}  # Track last update time for each symbol
        self.batch_size = 5  # Increased from 3 to 5 symbols per batch
        self.batch_interval = 2  # Reduced from 5 to 2 seconds between batches
        self.pending_updates = asyncio.Queue()  # Queue for pending market updates
        
        # Volume analysis settings
        self.volume_profile_enabled = self.config.get('volume_profile', {}).get('enabled', True)
        self.volume_profile_update_interval = self.config.get('volume_profile', {}).get('update_interval', 300)  # 5 minutes
        self.volume_delta_enabled = self.config.get('volume_profile', {}).get('delta_enabled', True)
        self.volume_anomaly_detection = self.config.get('volume_profile', {}).get('anomaly_detection', True)
        self.last_volume_profile_update = {}

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
            # Check if we have cached data that's still valid
            if symbol in self.historical_data:
                last_update = self.historical_data[symbol]['timestamp']
                if (datetime.now() - last_update).total_seconds() < 60:  # Cache for 1 minute
                    return self.historical_data[symbol]

            # Get 1m, 3m, 5m, and 15m candles
            klines_1m = self.client.get_klines(
                symbol=symbol,
                interval='1m',
                limit=100
            )
            
            klines_3m = self.client.get_klines(
                symbol=symbol,
                interval='3m',
                limit=100
            )
            
            klines_5m = self.client.get_klines(
                symbol=symbol,
                interval='5m',
                limit=100
            )
            
            klines_15m = self.client.get_klines(
                symbol=symbol,
                interval='15m',
                limit=100
            )
            
            # Process candles for all timeframes
            df_1m = pd.DataFrame(klines_1m, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df_3m = pd.DataFrame(klines_3m, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df_5m = pd.DataFrame(klines_5m, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df_15m = pd.DataFrame(klines_15m, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert timestamp to datetime for all dataframes
            for df in [df_1m, df_3m, df_5m, df_15m]:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Cache the data
            self.historical_data[symbol] = {
                'data_1m': df_1m,
                'data_3m': df_3m,
                'data_5m': df_5m,
                'data_15m': df_15m,
                'timestamp': datetime.now()
            }
            
            return self.historical_data[symbol]
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return None

    def calculate_technical_indicators(self, data):
        """Calculate technical indicators using multiple timeframes"""
        try:
            df_1m = data['data_1m']
            df_3m = data['data_3m']
            df_5m = data['data_5m']
            df_15m = data['data_15m']
            
            # Calculate indicators for 1m timeframe
            rsi_1m = RSIIndicator(close=df_1m['close'])
            stoch_1m = StochasticOscillator(high=df_1m['high'], low=df_1m['low'], close=df_1m['close'])
            macd_1m = MACD(close=df_1m['close'])
            williams_1m = WilliamsRIndicator(high=df_1m['high'], low=df_1m['low'], close=df_1m['close'])
            bb_1m = BollingerBands(close=df_1m['close'])
            
            # Calculate indicators for 3m timeframe
            rsi_3m = RSIIndicator(close=df_3m['close'])
            macd_3m = MACD(close=df_3m['close'])
            
            # Calculate indicators for 5m timeframe
            rsi_5m = RSIIndicator(close=df_5m['close'])
            macd_5m = MACD(close=df_5m['close'])
            
            # Calculate price changes
            last_price = float(df_1m['close'].iloc[-1])
            price_change_1m = ((last_price - float(df_1m['open'].iloc[-1])) / float(df_1m['open'].iloc[-1])) * 100
            price_change_3m = ((last_price - float(df_3m['open'].iloc[-1])) / float(df_3m['open'].iloc[-1])) * 100
            price_change_5m = ((last_price - float(df_5m['open'].iloc[-1])) / float(df_5m['open'].iloc[-1])) * 100
            price_change_15m = ((last_price - float(df_15m['open'].iloc[-1])) / float(df_15m['open'].iloc[-1])) * 100
            
            # Calculate BB position
            bb_range = bb_1m.bollinger_hband() - bb_1m.bollinger_lband()
            bb_position = (df_1m['close'] - bb_1m.bollinger_lband()) / bb_range.replace(0, np.nan)
            
            # Determine trend using multiple timeframes
            sma20_1m = SMAIndicator(close=df_1m['close'], window=20)
            sma50_1m = SMAIndicator(close=df_1m['close'], window=50)
            sma20_5m = SMAIndicator(close=df_5m['close'], window=20)
            
            df_1m['sma_20'] = sma20_1m.sma_indicator()
            df_1m['sma_50'] = sma50_1m.sma_indicator()
            df_5m['sma_20'] = sma20_5m.sma_indicator()
            
            # Calculate trend strength using multiple timeframes
            last_close = float(df_1m['close'].iloc[-1])
            sma20_last_1m = float(df_1m['sma_20'].iloc[-1])
            sma50_last_1m = float(df_1m['sma_50'].iloc[-1])
            sma20_last_5m = float(df_5m['sma_20'].iloc[-1])
            
            strength_1m = ((last_close - sma20_last_1m) / sma20_last_1m * 100)
            strength_5m = ((last_close - sma20_last_5m) / sma20_last_5m * 100)
            
            # Combined trend strength
            trend_strength = (strength_1m * 0.6 + strength_5m * 0.4)  # Weight recent data more heavily
            
            # Determine overall trend
            if last_close > sma20_last_1m and sma20_last_1m > sma50_last_1m:
                trend = 'uptrend'
            elif last_close < sma20_last_1m and sma20_last_1m < sma50_last_1m:
                trend = 'downtrend'
            else:
                trend = 'sideways'

            return {
                'rsi': float(rsi_1m.rsi().iloc[-1]),
                'rsi_3m': float(rsi_3m.rsi().iloc[-1]),
                'rsi_5m': float(rsi_5m.rsi().iloc[-1]),
                'stoch_k': float(stoch_1m.stoch().iloc[-1]),
                'macd': float(macd_1m.macd().iloc[-1]),
                'macd_3m': float(macd_3m.macd().iloc[-1]),
                'macd_5m': float(macd_5m.macd().iloc[-1]),
                'williams_r': float(williams_1m.williams_r().iloc[-1]),
                'bb_position': float(bb_position.iloc[-1]),
                'trend': trend,
                'trend_strength': abs(trend_strength),
                'price_change_1m': price_change_1m,
                'price_change_3m': price_change_3m,
                'price_change_5m': price_change_5m,
                'price_change_15m': price_change_15m
            }
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return None
            
    def analyze_volume_profile(self, symbol, data):
        """
        Analyze volume profile for a given symbol
        
        Args:
            symbol: The trading symbol
            data: Historical price and volume data
            
        Returns:
            Dictionary with volume profile analysis or None if error
        """
        try:
            if not self.volume_profile_enabled:
                logger.debug(f"Volume profile analysis is disabled, skipping for {symbol}")
                return None
            
            # Check if we should update the volume profile
            current_time = datetime.now()
            last_update = self.last_volume_profile_update.get(symbol, datetime.min)
            
            if (current_time - last_update).total_seconds() < self.volume_profile_update_interval:
                logger.debug(f"Using cached volume profile for {symbol}, last update was {(current_time - last_update).seconds}s ago")
                return self.volume_profiles.get(symbol)
                
            # Use 5m timeframe for volume profile analysis
            df_5m = data['data_5m'] if 'data_5m' in data else data['data_1m']
            
            # Make sure we have necessary columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df_5m.columns for col in required_columns):
                logger.warning(f"Missing required columns for volume profile analysis: {symbol}")
                return None
            
            # Convert to numeric if not already
            for col in required_columns:
                if not pd.api.types.is_numeric_dtype(df_5m[col]):
                    df_5m[col] = pd.to_numeric(df_5m[col], errors='coerce')
            
            # Analyze volume profile
            logger.info(f"Analyzing volume profile for {symbol}")
            profile_result = self.volume_profile_analyzer.analyze_volume_profile(df_5m)
            
            # Check for error
            if 'error' in profile_result:
                logger.error(f"Error in volume profile analysis for {symbol}: {profile_result['error']}")
                return None
            
            # Add additional volume analysis if enabled
            if self.volume_delta_enabled:
                volume_delta = self.volume_profile_analyzer.analyze_volume_delta(df_5m)
                if 'error' not in volume_delta:
                    profile_result['volume_delta'] = volume_delta
            
            # Add volume anomaly detection if enabled
            if self.volume_anomaly_detection:
                anomalies = self.volume_profile_analyzer.detect_volume_anomalies(df_5m)
                if 'error' not in anomalies:
                    profile_result['volume_anomalies'] = anomalies
            
            # Store in cache
            self.volume_profiles[symbol] = profile_result
            self.last_volume_profile_update[symbol] = current_time
            
            logger.info(f"Completed volume profile analysis for {symbol}")
            
            return profile_result
            
        except Exception as e:
            logger.error(f"Error analyzing volume profile for {symbol}: {str(e)}")
            return None

    async def process_market_data(self, msg):
        """Process incoming market data and queue updates"""
        try:
            data = json.loads(msg)
            if not isinstance(data, list):
                return

            current_time = datetime.now()
            
            for ticker in data:
                if not isinstance(ticker, dict) or 's' not in ticker:
                    continue

                symbol = ticker['s']
                if not symbol.endswith('USDC'):
                    continue

                # Check if we should update this symbol based on the interval
                last_update = self.last_update_time.get(symbol, datetime.min)
                if (current_time - last_update).total_seconds() < self.update_interval:
                    continue

                # Queue the ticker data for processing
                await self.pending_updates.put((symbol, ticker))

        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")
            logger.error(f"Message content: {msg}")

    async def process_pending_updates(self):
        """Process pending market updates in batches"""
        while self.running:
            try:
                if not self.redis or not await self.redis.ping():
                    if not await self.connect_redis():
                        await asyncio.sleep(5)
                        continue

                # Process a batch of updates
                batch = []
                try:
                    for _ in range(self.batch_size):
                        if self.pending_updates.empty():
                            break
                        batch.append(await self.pending_updates.get())
                except asyncio.QueueEmpty:
                    pass

                if not batch:
                    await asyncio.sleep(1)
                    continue

                current_time = datetime.now()

                for symbol, ticker in batch:
                    try:
                        # Process market data
                        price = float(ticker['c'])
                        volume = float(ticker['v']) * price

                        # Get historical data and calculate indicators
                        data = self.get_historical_data(symbol)
                        if data is not None:
                            indicators = self.calculate_technical_indicators(data)
                            if indicators:
                                # Create complete market update with all fields at the top level
                                market_update = {
                                    'symbol': symbol,
                                    'current_price': price,
                                    'avg_volume': volume,
                                    'timestamp': current_time.isoformat(),
                                    'rsi': indicators['rsi'],
                                    'rsi_3m': indicators['rsi_3m'],
                                    'rsi_5m': indicators['rsi_5m'],
                                    'stoch_k': indicators['stoch_k'],
                                    'macd': indicators['macd'],
                                    'macd_3m': indicators['macd_3m'],
                                    'macd_5m': indicators['macd_5m'],
                                    'williams_r': indicators['williams_r'],
                                    'bb_position': indicators['bb_position'],
                                    'trend': indicators['trend'],
                                    'trend_strength': indicators['trend_strength'],
                                    'price_change_1m': indicators['price_change_1m'],
                                    'price_change_3m': indicators['price_change_3m'],
                                    'price_change_5m': indicators['price_change_5m'],
                                    'price_change_15m': indicators['price_change_15m']
                                }
                                
                                # Calculate indicator combinations
                                try:
                                    combined_indicators = calculate_indicator_combinations(market_update)
                                    if 'error' not in combined_indicators:
                                        # Add combined indicators to market update
                                        market_update['combined_indicators'] = combined_indicators
                                        logger.debug(f"Added {len(combined_indicators)} combined indicators")
                                    else:
                                        logger.warning(f"Error calculating combined indicators: {combined_indicators['error']}")
                                except Exception as e:
                                    logger.error(f"Failed to calculate combined indicators: {str(e)}")
                                    # Continue without combined indicators
                                
                                # Add volume profile analysis
                                try:
                                    volume_profile = self.analyze_volume_profile(symbol, data)
                                    if volume_profile:
                                        # Add core volume profile metrics to market update
                                        volume_profile_summary = {
                                            'poc': volume_profile['poc'],
                                            'value_area_high': volume_profile['value_area_high'],
                                            'value_area_low': volume_profile['value_area_low'],
                                            'signal': volume_profile['signals']['overall'],
                                            'summary': volume_profile['summary']
                                        }
                                        
                                        # Add volume delta insights if available
                                        if 'volume_delta' in volume_profile and 'signals' in volume_profile['volume_delta']:
                                            volume_profile_summary['volume_pressure'] = volume_profile['volume_delta']['signals']['overall_pressure']
                                            
                                            # Add divergence signal if present
                                            if volume_profile['volume_delta']['signals']['divergence'] != 'no_divergence':
                                                volume_profile_summary['volume_divergence'] = volume_profile['volume_delta']['signals']['divergence']
                                        
                                        # Add volume anomaly info if available
                                        if 'volume_anomalies' in volume_profile and volume_profile['volume_anomalies'].get('anomalies_detected', False):
                                            volume_profile_summary['volume_anomalies'] = True
                                            volume_profile_summary['recent_anomaly_percentage'] = volume_profile['volume_anomalies'].get('recent_anomaly_percentage', 0)
                                        
                                        # Add to market update
                                        market_update['volume_profile'] = volume_profile_summary
                                        
                                        # Keep the full analysis separate for detailed queries
                                        self.volume_profiles[symbol] = volume_profile
                                        
                                        logger.debug(f"Added volume profile analysis for {symbol}")
                                    else:
                                        logger.debug(f"No volume profile available for {symbol}")
                                except Exception as e:
                                    logger.error(f"Error adding volume profile to market update: {str(e)}")
                                    # Continue without volume profile

                                # Store in local cache
                                self.market_data[symbol] = market_update
                                self.last_update_time[symbol] = current_time

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
                                if abs(indicators['price_change_1m']) >= self.config['trading_params'].get('min_price_change_pct', 0.5) and \
                                   volume >= self.config['trading_params']['min_volume_usdc']:
                                    await self.redis.publish(
                                        'trading_opportunities',
                                        json.dumps(market_update)
                                    )
                                    logger.info(f"Found opportunity: {symbol} at ${price:.8f} (1m change: {indicators['price_change_1m']:.2f}%)")

                    except Exception as e:
                        logger.error(f"Error processing update for {symbol}: {str(e)}")

                # Wait between batches
                await asyncio.sleep(self.batch_interval)

            except Exception as e:
                logger.error(f"Error in process_pending_updates: {str(e)}")
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
            
            # Create tasks for WebSocket, Redis maintenance, health check server, and update processing
            tasks = [
                asyncio.create_task(self.websocket_handler()),
                asyncio.create_task(self.maintain_redis()),
                asyncio.create_task(self.health_check_server()),
                asyncio.create_task(self.process_pending_updates())
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
