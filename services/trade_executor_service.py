import os
import json
import socket
import asyncio
from datetime import datetime
from binance.client import Client
from binance.enums import *
import logging as logger
from redis.asyncio import Redis
from redis.exceptions import ConnectionError

# Configure logging with more debug information
logger.basicConfig(
    level=logger.DEBUG,  # Changed to DEBUG for more detailed logging
    format='%(asctime)s - %(levelname)s - [TradeExecutor] %(message)s',
    handlers=[
        logger.FileHandler('logs/trade_executor.log'),
        logger.StreamHandler()
    ]
)

class TradeExecutorService:
    def __init__(self):
        logger.debug("Initializing Trade Executor Service...")
        
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        logger.debug(f"Loaded configuration: {json.dumps(self.config, indent=2)}")

        # Initialize Binance client
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        if not api_key or not api_secret:
            raise ValueError("Binance API credentials not set")
            
        self.client = Client(api_key, api_secret)
        logger.info("Initialized Binance client")

        # Redis configuration
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        logger.debug(f"Redis configuration - Host: {self.redis_host}, Port: {self.redis_port}")

        # Redis will be initialized in connect_redis
        self.redis = None
        self.pubsub = None
        self.running = True
        self.active_trades = {}
        self.symbol_info = {}
        self.available_usdc = 0.0
        
        # Get service port from environment variable
        self.service_port = int(os.getenv('SERVICE_PORT', 8002))  # Default to 8002 (TRADE_EXECUTOR_PORT)
        logger.debug(f"Service port configured as: {self.service_port}")
        
        # Load initial trading rules
        self.load_trading_rules()
        logger.debug("Trade Executor Service initialization complete")

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

    def load_trading_rules(self):
        """Load trading rules for all USDC pairs"""
        try:
            logger.info("Loading trading rules from Binance...")
            exchange_info = self.client.get_exchange_info()
            
            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'].endswith('USDC') and symbol_info['status'] == 'TRADING':
                    filters = {f['filterType']: f for f in symbol_info['filters']}
                    
                    self.symbol_info[symbol_info['symbol']] = {
                        'baseAsset': symbol_info['baseAsset'],
                        'quoteAsset': symbol_info['quoteAsset'],
                        'status': symbol_info['status'],
                        'min_price': float(filters['PRICE_FILTER']['minPrice']),
                        'max_price': float(filters['PRICE_FILTER']['maxPrice']),
                        'tick_size': float(filters['PRICE_FILTER']['tickSize']),
                        'min_qty': float(filters['LOT_SIZE']['minQty']),
                        'max_qty': float(filters['LOT_SIZE']['maxQty']),
                        'step_size': float(filters['LOT_SIZE']['stepSize']),
                        'min_notional': float(filters['MIN_NOTIONAL']['minNotional']) if 'MIN_NOTIONAL' in filters else 0.0
                    }
            
            logger.info(f"Loaded trading rules for {len(self.symbol_info)} USDC pairs")
            
        except Exception as e:
            logger.error(f"Error loading trading rules: {str(e)}")
            raise

    def update_usdc_balance(self):
        """Update available USDC balance"""
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == 'USDC':
                    self.available_usdc = float(balance['free'])
                    logger.info(f"USDC balance updated: ${self.available_usdc:.2f}")
                    break
        except Exception as e:
            logger.error(f"Error updating USDC balance: {str(e)}")

    def round_step_size(self, quantity: float, step_size: float) -> float:
        """Round quantity to valid step size"""
        precision = len(str(step_size).split('.')[-1])
        return round(quantity - (quantity % step_size), precision)

    async def execute_trade(self, signal: dict):
        """Execute a trade based on the trading signal"""
        try:
            symbol = signal['symbol']
            decision = signal['decision']
            confidence = signal['confidence']
            
            logger.debug(f"Processing trade signal for {symbol}: {decision} (confidence: {confidence})")

            # Skip if confidence is too low
            if confidence < self.config['trading_params']['ai_confidence_threshold']:
                logger.debug(f"Skipping trade due to low confidence: {confidence}")
                return

            # Handle SELL signals
            if decision == 'SELL' and symbol in self.active_trades:
                position = self.active_trades[symbol]
                await self.close_position(symbol, position['quantity'], "AI SELL Signal")
                return

            # Handle BUY signals
            if decision == 'BUY':
                # Get current price
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
                logger.debug(f"Current price for {symbol}: ${current_price:.8f}")

                # Calculate position size
                position_size = min(
                    self.available_usdc * self.config['trading_params']['position_size'],
                    self.available_usdc * 0.95  # Max 95% of available balance
                )

                # Skip if position size is too small
                if position_size < self.config['trading_params']['min_trade_amount']:
                    logger.info(f"Position size too small: ${position_size:.2f}")
                    return

                # Calculate quantity
                quantity = position_size / current_price
                
                # Round quantity to valid step size
                rules = self.symbol_info.get(symbol)
                if not rules:
                    logger.error(f"No trading rules found for {symbol}")
                    return
                
                quantity = self.round_step_size(quantity, rules['step_size'])
                logger.debug(f"Calculated quantity for {symbol}: {quantity}")
                
                # Check minimum notional value
                if quantity * current_price < rules['min_notional']:
                    logger.error(f"Order does not meet minimum notional value")
                    return

                # Place market buy order
                logger.debug(f"Placing market buy order for {symbol}")
                order = self.client.create_order(
                    symbol=symbol,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )

                # Get fill price
                fill_price = float(order['fills'][0]['price'])
                filled_quantity = float(order['executedQty'])
                logger.info(f"Market buy order filled at ${fill_price:.8f}")

                # Calculate stop loss and take profit prices
                stop_loss_price = fill_price * (1 - self.config['trading_params']['stop_loss_pct'] / 100)
                take_profit_price = fill_price * (1 + self.config['trading_params']['take_profit_pct'] / 100)

                # Place stop loss order
                logger.debug(f"Placing stop loss order at ${stop_loss_price:.8f}")
                stop_loss_order = self.client.create_order(
                    symbol=symbol,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_STOP_LOSS_LIMIT,
                    timeInForce=TIME_IN_FORCE_GTC,
                    quantity=filled_quantity,
                    stopPrice=stop_loss_price,
                    price=stop_loss_price * 0.99
                )

                # Place take profit order
                logger.debug(f"Placing take profit order at ${take_profit_price:.8f}")
                take_profit_order = self.client.create_order(
                    symbol=symbol,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_LIMIT,
                    timeInForce=TIME_IN_FORCE_GTC,
                    quantity=filled_quantity,
                    price=take_profit_price
                )

                # Record trade
                self.active_trades[symbol] = {
                    'entry_price': fill_price,
                    'quantity': filled_quantity,
                    'stop_loss_price': stop_loss_price,
                    'take_profit_price': take_profit_price,
                    'stop_loss_order': stop_loss_order['orderId'],
                    'take_profit_order': take_profit_order['orderId'],
                    'entry_time': datetime.now().isoformat(),
                    'status': 'ACTIVE'
                }

                logger.info(f"Opened position for {symbol}:")
                logger.info(f"Entry Price: ${fill_price:.8f}")
                logger.info(f"Quantity: {filled_quantity}")
                logger.info(f"Stop Loss: ${stop_loss_price:.8f}")
                logger.info(f"Take Profit: ${take_profit_price:.8f}")

                # Update balance
                self.update_usdc_balance()

        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}", exc_info=True)

    async def close_position(self, symbol: str, quantity: float, reason: str):
        """Close a specific trading position"""
        try:
            logger.debug(f"Closing position for {symbol} ({reason})")
            
            # Place market sell order
            order = self.client.create_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            
            fill_price = float(order['fills'][0]['price'])
            logger.info(f"Closed position for {symbol} at ${fill_price:.8f} ({reason})")
            
            # Remove from active trades
            self.active_trades.pop(symbol, None)
            
            # Update balance
            self.update_usdc_balance()
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {str(e)}", exc_info=True)

    async def monitor_active_trades(self):
        """Monitor and manage active trades"""
        try:
            for symbol, trade in list(self.active_trades.items()):
                try:
                    # Get current price
                    ticker = self.client.get_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    
                    # Calculate current P&L
                    entry_price = trade['entry_price']
                    quantity = trade['quantity']
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    
                    logger.debug(f"Monitoring {symbol} - Current: ${current_price:.8f}, P&L: {pnl_pct:.2f}%")
                    
                    # Check stop loss
                    if current_price <= trade['stop_loss_price']:
                        logger.info(f"Stop loss triggered for {symbol} at ${current_price:.8f} ({pnl_pct:.2f}%)")
                        await self.close_position(symbol, quantity, "Stop Loss")
                    
                    # Check take profit
                    elif current_price >= trade['take_profit_price']:
                        logger.info(f"Take profit triggered for {symbol} at ${current_price:.8f} ({pnl_pct:.2f}%)")
                        await self.close_position(symbol, quantity, "Take Profit")
                    
                except Exception as e:
                    logger.error(f"Error monitoring trade for {symbol}: {str(e)}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Error in trade monitoring: {str(e)}", exc_info=True)

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
            logger.debug("Subscribing to trading_signals channel")
            await self.pubsub.subscribe('trading_signals')
            
            # Get first message to confirm subscription
            logger.debug("Waiting for subscription confirmation message")
            message = await self.pubsub.get_message(timeout=1.0)
            
            if message and message['type'] == 'subscribe':
                logger.info("Successfully subscribed to trading_signals channel")
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

    async def process_trading_signals(self):
        """Process trading signals from Redis"""
        logger.debug("Starting trading signals processing...")
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
                    logger.debug("Waiting for trading signals...")
                    message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    
                    if message:
                        logger.debug(f"Received message type: {message['type']}")
                        if message['type'] == 'message':
                            try:
                                logger.debug(f"Raw message data: {message['data']}")
                                signal = json.loads(message['data'])
                                await self.execute_trade(signal)
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse trading signal: {e}")
                                logger.error(f"Invalid JSON data: {message['data']}")
                            except Exception as e:
                                logger.error(f"Error processing trading signal: {str(e)}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error getting message from pubsub: {str(e)}")
                    self.pubsub = None  # Force pubsub reconnection
                    continue

                # Monitor active trades
                await self.monitor_active_trades()
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in process_trading_signals: {str(e)}", exc_info=True)
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
        """Run the trade executor service"""
        try:
            logger.info("Starting Trade Executor Service...")
            
            # First establish Redis connection with increased retries
            if not await self.connect_redis(max_retries=15, retry_delay=2):
                raise Exception("Failed to establish initial Redis connection")
            
            self.update_usdc_balance()
            
            # Create tasks for trading signals processing, Redis maintenance, and health check
            tasks = [
                asyncio.create_task(self.process_trading_signals()),
                asyncio.create_task(self.maintain_redis()),
                asyncio.create_task(self.health_check_server())
            ]
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Error in Trade Executor Service: {str(e)}")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the trade executor service"""
        logger.info("Stopping Trade Executor Service...")
        self.running = False
        if self.pubsub:
            await self.pubsub.close()
        if self.redis:
            await self.redis.close()

if __name__ == "__main__":
    service = TradeExecutorService()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        asyncio.run(service.stop())
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        asyncio.run(service.stop())
