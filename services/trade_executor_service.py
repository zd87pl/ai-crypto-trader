import os
import json
import socket
import asyncio
from datetime import datetime
from binance.client import Client
from binance.enums import *
import logging as logger
from logging.handlers import RotatingFileHandler
from redis.asyncio import Redis
from redis.exceptions import ConnectionError

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure rotating file handler
rotating_handler = RotatingFileHandler(
    'logs/trade_executor.log',
    maxBytes=10*1024*1024,  # 10MB per file
    backupCount=5,  # Keep 5 backup files
    encoding='utf-8'
)

# Configure logging with rotation
logger.basicConfig(
    level=logger.DEBUG,
    format='%(asctime)s - %(levelname)s - [TradeExecutor] %(message)s',
    handlers=[
        rotating_handler,
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
        self.holdings = {}  # Track all asset holdings
        self.total_portfolio_value = 0.0  # Track total portfolio value in USDC
        
        # Get service port from environment variable
        self.service_port = int(os.getenv('SERVICE_PORT', 8002))
        logger.debug(f"Service port configured as: {self.service_port}")
        
        # Load initial trading rules
        self.load_trading_rules()
        logger.debug("Trade Executor Service initialization complete")

    async def initialize(self):
        """Initialize async components"""
        try:
            # Update holdings
            await self.update_holdings()
            
            # Initial cleanup to convert all assets to USDC
            await self.cleanup_positions()
            
            logger.info("Async initialization complete")
        except Exception as e:
            logger.error(f"Error in async initialization: {str(e)}")
            raise

    async def cleanup_positions(self):
        """Sell all non-USDC assets to start fresh"""
        try:
            logger.info("Starting cleanup - Converting all assets to USDC...")
            account = self.client.get_account()
            
            for balance in account['balances']:
                asset = balance['asset']
                free_amount = float(balance['free'])
                
                if asset != 'USDC' and free_amount > 0:
                    symbol = f"{asset}USDC"
                    try:
                        # Check if trading pair exists
                        ticker = self.client.get_symbol_ticker(symbol=symbol)
                        if ticker:
                            # Get symbol info for precision
                            symbol_info = self.symbol_info.get(symbol)
                            if not symbol_info:
                                logger.error(f"No trading rules found for {symbol}")
                                continue
                            
                            # Check minimum quantity before logging sell attempt
                            if free_amount < symbol_info['min_qty']:
                                logger.debug(f"Skipping {asset}: amount too small to sell ({free_amount} < {symbol_info['min_qty']})")
                                continue
                            
                            logger.info(f"Selling {free_amount} {asset} to USDC")
                            
                            # Round quantity to valid step size
                            quantity = self.round_step_size(free_amount, symbol_info['step_size'])
                            
                            # Check if rounded quantity is still valid
                            if quantity < symbol_info['min_qty']:
                                logger.debug(f"Skipping {asset}: rounded quantity too small ({quantity} < {symbol_info['min_qty']})")
                                continue
                            
                            # Place market sell order
                            order = self.client.create_order(
                                symbol=symbol,
                                side=SIDE_SELL,
                                type=ORDER_TYPE_MARKET,
                                quantity=quantity
                            )
                            
                            fill_price = float(order['fills'][0]['price'])
                            logger.info(f"Sold {quantity} {asset} at ${fill_price:.8f}")
                            
                    except Exception as e:
                        logger.error(f"Error selling {asset}: {str(e)}")
            
            # Wait a moment for orders to settle
            await asyncio.sleep(5)
            
            # Update holdings
            await self.update_holdings()
            logger.info("Cleanup complete - All assets converted to USDC")
            
        except Exception as e:
            logger.error(f"Error in cleanup_positions: {str(e)}")

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
            
            # Subscribe to channels
            logger.debug("Subscribing to trading_signals and strategy_update channels")
            await self.pubsub.subscribe('trading_signals', 'strategy_update')
            
            # Get first messages to confirm subscriptions
            logger.debug("Waiting for subscription confirmation messages")
            subscribed_count = 0
            while subscribed_count < 2:  # Wait for both subscriptions
                message = await self.pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'subscribe':
                    subscribed_count += 1
            
            if subscribed_count == 2:
                logger.info("Successfully subscribed to all channels")
                return True
            else:
                logger.error("Failed to subscribe to all channels")
                return False

        except Exception as e:
            logger.error(f"Error in pubsub setup: {str(e)}", exc_info=True)
            if self.pubsub:
                await self.pubsub.close()
                self.pubsub = None
            return False

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

    async def update_holdings(self):
        """Update holdings and portfolio value"""
        try:
            account = self.client.get_account()
            
            # Reset holdings
            self.holdings = {}
            total_value = 0.0
            
            # Update all balances
            for balance in account['balances']:
                asset = balance['asset']
                free_amount = float(balance['free'])
                locked_amount = float(balance['locked'])
                total_amount = free_amount + locked_amount
                
                if total_amount > 0:
                    if asset == 'USDC':
                        self.available_usdc = free_amount
                        total_value += total_amount
                        self.holdings[asset] = {
                            'amount': total_amount,
                            'free': free_amount,
                            'locked': locked_amount,
                            'value_usdc': total_amount
                        }
                    else:
                        # Get current price in USDC if available
                        try:
                            ticker = self.client.get_symbol_ticker(symbol=f"{asset}USDC")
                            price = float(ticker['price'])
                            value_usdc = total_amount * price
                            total_value += value_usdc
                            
                            self.holdings[asset] = {
                                'amount': total_amount,
                                'free': free_amount,
                                'locked': locked_amount,
                                'value_usdc': value_usdc,
                                'current_price': price
                            }
                        except Exception as e:
                            logger.debug(f"Could not get USDC price for {asset}: {str(e)}")
            
            self.total_portfolio_value = total_value
            logger.info(f"Updated holdings - Total Portfolio Value: ${self.total_portfolio_value:.2f}")
            logger.info(f"Available USDC: ${self.available_usdc:.2f}")
            
            # Store holdings in Redis for dashboard
            if self.redis and await self.redis.ping():
                await self.redis.set('holdings', json.dumps({
                    'assets': self.holdings,
                    'total_value': self.total_portfolio_value,
                    'available_usdc': self.available_usdc,
                    'timestamp': datetime.now().isoformat()
                }))
            
        except Exception as e:
            logger.error(f"Error updating holdings: {str(e)}")

    async def check_trading_conditions(self, symbol: str, decision: str) -> bool:
        """Check if trading conditions are met"""
        try:
            # Update holdings first
            await self.update_holdings()
            
            # Check if we have enough USDC for new trades
            min_usdc_required = self.config['trading_params']['min_trade_amount']
            if self.available_usdc < min_usdc_required:
                logger.info(f"Insufficient USDC balance (${self.available_usdc:.2f}) for trading. Minimum required: ${min_usdc_required:.2f}")
                return False
            
            # Check if we're already at max positions
            if len(self.active_trades) >= self.config['trading_params']['max_positions']:
                logger.info("Maximum number of positions reached")
                return False
            
            # For sell decisions, check if we have the asset
            if decision == 'SELL':
                asset = symbol.replace('USDC', '')
                if asset not in self.holdings or self.holdings[asset]['free'] <= 0:
                    logger.info(f"No free {asset} balance available for selling")
                    return False
            
            # Check if total portfolio value is above minimum required
            min_portfolio_value = self.config['trading_params'].get('min_portfolio_value', 100)
            if self.total_portfolio_value < min_portfolio_value:
                logger.info(f"Total portfolio value (${self.total_portfolio_value:.2f}) below minimum required (${min_portfolio_value:.2f})")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking trading conditions: {str(e)}")
            return False

    def round_step_size(self, quantity: float, step_size: float) -> float:
        """Round quantity to valid step size"""
        precision = len(str(step_size).split('.')[-1])
        return round(quantity - (quantity % step_size), precision)

    def round_price(self, price: float, tick_size: float) -> float:
        """Round price to valid tick size"""
        precision = len(str(tick_size).split('.')[-1])
        return round(price - (price % tick_size), precision)

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

            # Check trading conditions
            if not await self.check_trading_conditions(symbol, decision):
                logger.info("Trading conditions not met, skipping trade")
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

                # Calculate and round stop loss and take profit prices
                stop_loss_price = self.round_price(
                    fill_price * (1 - self.config['trading_params']['stop_loss_pct'] / 100),
                    rules['tick_size']
                )
                take_profit_price = self.round_price(
                    fill_price * (1 + self.config['trading_params']['take_profit_pct'] / 100),
                    rules['tick_size']
                )

                # Place stop loss order
                logger.debug(f"Placing stop loss order at ${stop_loss_price:.8f}")
                stop_loss_order = self.client.create_order(
                    symbol=symbol,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_STOP_LOSS_LIMIT,
                    timeInForce=TIME_IN_FORCE_GTC,
                    quantity=filled_quantity,
                    stopPrice=stop_loss_price,
                    price=self.round_price(stop_loss_price * 0.99, rules['tick_size'])
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

                # Update holdings after trade
                await self.update_holdings()

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
            
            # Update holdings after trade
            await self.update_holdings()
            
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
                    logger.debug("Waiting for messages...")
                    message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    
                    if message:
                        logger.debug(f"Received message type: {message['type']}")
                        if message['type'] == 'message':
                            try:
                                logger.debug(f"Raw message data: {message['data']}")
                                signal = json.loads(message['data'])
                                logger.info(f"Processing trading signal for {signal['symbol']}")
                                await self.execute_trade(signal)
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse trading signal: {e}")
                                logger.error(f"Invalid JSON data: {message['data']}")
                            except KeyError as e:
                                logger.error(f"Missing required field in signal: {e}")
                                logger.error(f"Signal data: {signal}")
                            except Exception as e:
                                logger.error(f"Error processing trading signal: {e}", exc_info=True)
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
            
            # Initialize async components
            await self.initialize()
            
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
            # Cleanup positions before stopping
            logger.info("Cleaning up positions before stopping...")
            await self.cleanup_positions()
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
