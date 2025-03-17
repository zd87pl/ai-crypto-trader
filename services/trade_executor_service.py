import os
import json
import socket
import asyncio
import time
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
from binance.enums import *
import logging as logger
from logging.handlers import RotatingFileHandler
from redis.asyncio import Redis
from redis.exceptions import ConnectionError

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Create a separate log file for trailing stops
trailing_stops_handler = RotatingFileHandler(
    'logs/trailing_stops.log',
    maxBytes=10*1024*1024,  # 10MB per file
    backupCount=5,  # Keep 5 backup files
    encoding='utf-8'
)
trailing_stops_handler.setFormatter(
    logger.Formatter('%(asctime)s - %(levelname)s - [TrailingStops] %(message)s')
)

# Configure trailing stops logger
trailing_stops_logger = logger.getLogger('trailing_stops')
trailing_stops_logger.setLevel(logger.DEBUG)
trailing_stops_logger.addHandler(trailing_stops_handler)
trailing_stops_logger.addHandler(logger.StreamHandler())

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

class TrailingStopManager:
    """Manages trailing stop-losses for active trades"""
    
    def __init__(self, config, client, symbol_info):
        self.config = config
        self.client = client
        self.symbol_info = symbol_info
        self.logger = trailing_stops_logger
        self.trailing_stops = {}  # Symbol -> trailing stop data
        self.settings = config["risk_management"]["trailing_stop_settings"]
        self.enabled = config["risk_management"]["trailing_stop_loss_enabled"]
        self.logger.info(f"Trailing Stop Manager initialized (enabled: {self.enabled})")
    
    def calculate_atr(self, symbol, periods=14):
        """Calculate Average True Range (ATR) for volatility-based trailing stops"""
        try:
            # Get historical klines
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_5MINUTE,
                limit=periods + 1
            )
            
            # Calculate true ranges
            true_ranges = []
            prev_close = float(klines[0][4])  # First candle's close
            
            for i in range(1, len(klines)):
                high = float(klines[i][2])
                low = float(klines[i][3])
                prev_close = float(klines[i-1][4])
                
                tr1 = high - low  # Current high - current low
                tr2 = abs(high - prev_close)  # Current high - previous close
                tr3 = abs(low - prev_close)  # Current low - previous close
                
                true_range = max(tr1, tr2, tr3)
                true_ranges.append(true_range)
            
            # Calculate ATR as average of true ranges
            if true_ranges:
                atr = sum(true_ranges) / len(true_ranges)
                return atr
            return 0
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR for {symbol}: {str(e)}")
            return 0
    
    def register_trailing_stop(self, symbol, entry_price, initial_stop_price, quantity, order_id):
        """Register a new trailing stop for a position"""
        if not self.enabled:
            self.logger.debug(f"Trailing stops disabled, skipping registration for {symbol}")
            return False
            
        # Calculate volatility metrics if needed
        atr = 0
        if self.settings["strategy"] in ["atr_based", "volatility_based"]:
            atr = self.calculate_atr(
                symbol, 
                self.settings["strategies"]["atr_based"]["min_periods"]
            )
        
        # Initialize trailing stop data
        self.trailing_stops[symbol] = {
            "symbol": symbol,
            "entry_price": entry_price,
            "current_price": entry_price,  # Initial price when position opened
            "highest_price": entry_price,  # Track highest price seen
            "initial_stop_price": initial_stop_price,
            "current_stop_price": initial_stop_price,
            "quantity": quantity,
            "order_id": order_id,
            "activation_price": entry_price * (1 + self.settings["activation_threshold_pct"] / 100),
            "activated": False,
            "last_update_time": time.time(),
            "strategy": self.settings["strategy"],
            "atr": atr,
            "trail_percent": self.settings["trail_percent"],
            "adjustments": [],
            "status": "PENDING_ACTIVATION"
        }
        
        self.logger.info(f"Registered trailing stop for {symbol}: entry=${entry_price:.8f}, "
                        f"initial stop=${initial_stop_price:.8f}, activation=${self.trailing_stops[symbol]['activation_price']:.8f}")
        return True
    
    def update_price(self, symbol, current_price):
        """Update the current price for a symbol with trailing stop"""
        if symbol not in self.trailing_stops or not self.enabled:
            return False
        
        stop_data = self.trailing_stops[symbol]
        prev_highest = stop_data["highest_price"]
        stop_data["current_price"] = current_price
        
        # Update highest price if current price is higher
        if current_price > stop_data["highest_price"]:
            stop_data["highest_price"] = current_price
            self.logger.debug(f"Updated highest price for {symbol}: ${current_price:.8f}")
        
        # Check if trailing stop should be activated
        if not stop_data["activated"] and current_price >= stop_data["activation_price"]:
            stop_data["activated"] = True
            stop_data["status"] = "ACTIVE"
            self.logger.info(f"Trailing stop activated for {symbol} at ${current_price:.8f}")
        
        # If activated and price has moved up, adjust trailing stop
        if stop_data["activated"] and current_price > prev_highest:
            self._adjust_trailing_stop(symbol)
            
        return True
    
    def _adjust_trailing_stop(self, symbol):
        """Adjust the trailing stop based on the selected strategy"""
        if symbol not in self.trailing_stops:
            return False
            
        stop_data = self.trailing_stops[symbol]
        
        # Check if enough time has passed since last adjustment
        current_time = time.time()
        time_since_last_update = current_time - stop_data["last_update_time"]
        if time_since_last_update < self.settings["max_adjustment_frequency_seconds"]:
            self.logger.debug(f"Skipping adjustment for {symbol} - too soon since last update")
            return False
            
        # Get current price and highest price
        current_price = stop_data["current_price"]
        highest_price = stop_data["highest_price"]
        current_stop = stop_data["current_stop_price"]
        
        # Calculate new stop price based on strategy
        new_stop_price = current_stop
        
        if stop_data["strategy"] == "percent_based":
            trail_pct = self.settings["strategies"]["percent_based"]["trail_percent"]
            min_dist = self.settings["strategies"]["percent_based"]["min_trail_distance"]
            
            # Calculate new stop as a percentage below highest price
            new_stop_price = highest_price * (1 - trail_pct / 100)
            
            # Ensure minimum distance from highest price
            min_stop_price = highest_price * (1 - min_dist / 100)
            if new_stop_price > min_stop_price:
                new_stop_price = min_stop_price
                
        elif stop_data["strategy"] == "atr_based":
            atr_multiplier = self.settings["strategies"]["atr_based"]["atr_multiplier"]
            
            # Update ATR for current volatility
            atr = self.calculate_atr(
                symbol, 
                self.settings["strategies"]["atr_based"]["min_periods"]
            )
            stop_data["atr"] = atr
            
            # Calculate new stop as ATR multiplier below highest price
            new_stop_price = highest_price - (atr * atr_multiplier)
            
        elif stop_data["strategy"] == "volatility_based":
            # Calculate volatility as standard deviation of recent prices
            try:
                lookback = self.settings["strategies"]["volatility_based"]["lookback_periods"]
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=Client.KLINE_INTERVAL_1MINUTE,
                    limit=lookback
                )
                
                prices = [float(kline[4]) for kline in klines]  # closing prices
                volatility = np.std(prices)
                multiplier = self.settings["strategies"]["volatility_based"]["volatility_multiplier"]
                
                # Higher volatility = wider trailing stop
                new_stop_price = highest_price - (volatility * multiplier)
                
            except Exception as e:
                self.logger.error(f"Error calculating volatility-based stop for {symbol}: {str(e)}")
                # Fall back to percent-based if volatility calculation fails
                new_stop_price = highest_price * (1 - self.settings["trail_percent"] / 100)
                
        elif stop_data["strategy"] == "fixed_amount":
            trail_amount = self.settings["strategies"]["fixed_amount"]["trail_amount_usdc"]
            
            # For higher priced assets, ensure minimum percentage trail
            min_trail_pct = self.settings["strategies"]["fixed_amount"]["min_adjustment_pct"]
            min_trail_amount = highest_price * (min_trail_pct / 100)
            
            # Use the larger of fixed amount or minimum percentage
            trail_amount = max(trail_amount, min_trail_amount)
            new_stop_price = highest_price - trail_amount
        
        # Only move stop price up, never down
        if new_stop_price <= current_stop:
            self.logger.debug(f"No adjustment needed for {symbol} - new stop (${new_stop_price:.8f}) <= current stop (${current_stop:.8f})")
            return False
            
        # Round to valid tick size
        symbol_info = self.symbol_info.get(symbol, {})
        if symbol_info and 'tick_size' in symbol_info:
            new_stop_price = self._round_price(new_stop_price, symbol_info['tick_size'])
        
        # Record the adjustment
        adjustment = {
            "timestamp": datetime.now().isoformat(),
            "previous_stop": current_stop,
            "new_stop": new_stop_price,
            "highest_price": highest_price,
            "price_movement": ((highest_price - stop_data["entry_price"]) / stop_data["entry_price"]) * 100
        }
        
        stop_data["adjustments"].append(adjustment)
        stop_data["current_stop_price"] = new_stop_price
        stop_data["last_update_time"] = current_time
        
        self.logger.info(f"Adjusted trailing stop for {symbol}: ${current_stop:.8f} → ${new_stop_price:.8f} "
                        f"(highest: ${highest_price:.8f}, movement: {adjustment['price_movement']:.2f}%)")
        
        return True
    
    def _round_price(self, price, tick_size):
        """Round price to valid tick size"""
        precision = len(str(tick_size).split('.')[-1])
        return round(price - (price % tick_size), precision)
        
    def get_current_stop(self, symbol):
        """Get current trailing stop price for a symbol"""
        if symbol not in self.trailing_stops:
            return None
        return self.trailing_stops[symbol]["current_stop_price"]
        
    def is_activated(self, symbol):
        """Check if trailing stop is activated for a symbol"""
        if symbol not in self.trailing_stops:
            return False
        return self.trailing_stops[symbol]["activated"]
        
    def should_execute_stop(self, symbol, current_price):
        """Check if trailing stop should be executed"""
        if symbol not in self.trailing_stops or not self.enabled:
            return False
            
        stop_data = self.trailing_stops[symbol]
        
        # If not activated, use initial stop price
        if not stop_data["activated"]:
            return current_price <= stop_data["initial_stop_price"]
            
        # If activated, use current trailing stop price
        return current_price <= stop_data["current_stop_price"]
        
    def remove_trailing_stop(self, symbol):
        """Remove trailing stop for a symbol"""
        if symbol in self.trailing_stops:
            self.logger.info(f"Removing trailing stop for {symbol}")
            stop_data = self.trailing_stops.pop(symbol)
            
            # Log summary of trailing stop performance
            if stop_data["activated"]:
                entry_price = stop_data["entry_price"]
                highest_price = stop_data["highest_price"]
                final_stop = stop_data["current_stop_price"]
                
                max_profit_pct = ((highest_price - entry_price) / entry_price) * 100
                protected_profit_pct = ((final_stop - entry_price) / entry_price) * 100
                
                self.logger.info(f"Trailing stop summary for {symbol}:")
                self.logger.info(f"  Entry price: ${entry_price:.8f}")
                self.logger.info(f"  Highest price: ${highest_price:.8f} (+{max_profit_pct:.2f}%)")
                self.logger.info(f"  Final stop price: ${final_stop:.8f} ({protected_profit_pct:.2f}%)")
                self.logger.info(f"  Protected profit: {protected_profit_pct:.2f}% of {max_profit_pct:.2f}%")
                self.logger.info(f"  Strategy: {stop_data['strategy']}")
                self.logger.info(f"  Adjustments: {len(stop_data['adjustments'])}")
            
            return True
        return False
        
    def update_stop_order(self, symbol, new_stop_price):
        """Update the stop-loss order for a symbol"""
        if symbol not in self.trailing_stops:
            return False
            
        stop_data = self.trailing_stops[symbol]
        
        try:
            # Cancel existing stop-loss order
            self.client.cancel_order(
                symbol=symbol,
                orderId=stop_data["order_id"]
            )
            
            # Get symbol info for tick size
            symbol_info = self.symbol_info.get(symbol, {})
            if not symbol_info:
                raise Exception(f"No trading rules found for {symbol}")
                
            # Create new stop-loss order
            new_stop_order = self.client.create_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_STOP_LOSS_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=stop_data["quantity"],
                stopPrice=new_stop_price,
                price=self._round_price(new_stop_price * 0.99, symbol_info['tick_size'])
            )
            
            # Update trailing stop data
            stop_data["order_id"] = new_stop_order["orderId"]
            
            self.logger.info(f"Updated stop-loss order for {symbol}: ${stop_data['current_stop_price']:.8f} → ${new_stop_price:.8f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating stop-loss order for {symbol}: {str(e)}")
            return False
            
    def get_status_report(self):
        """Get a status report of all trailing stops"""
        report = {
            "active_trailing_stops": len(self.trailing_stops),
            "stops": {}
        }
        
        for symbol, stop_data in self.trailing_stops.items():
            entry_price = stop_data["entry_price"]
            current_price = stop_data["current_price"]
            current_profit_pct = ((current_price - entry_price) / entry_price) * 100
            protected_profit_pct = ((stop_data["current_stop_price"] - entry_price) / entry_price) * 100
            
            report["stops"][symbol] = {
                "status": stop_data["status"],
                "entry_price": entry_price,
                "current_price": current_price,
                "current_stop": stop_data["current_stop_price"],
                "highest_price": stop_data["highest_price"],
                "current_profit_pct": current_profit_pct,
                "protected_profit_pct": protected_profit_pct,
                "strategy": stop_data["strategy"],
                "activated": stop_data["activated"],
                "adjustments": len(stop_data["adjustments"])
            }
            
        return report

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
        
        # Initialize trailing stop manager
        self.trailing_stop_manager = None  # Will be initialized after trading rules are loaded
        self._initialize_trailing_stop_manager()
        
        logger.debug("Trade Executor Service initialization complete")
        
    def _initialize_trailing_stop_manager(self):
        """Initialize the trailing stop manager"""
        try:
            # Check if trailing stops are enabled in config
            trailing_stops_enabled = self.config.get('risk_management', {}).get('trailing_stop_loss_enabled', False)
            
            # Create trailing stop manager
            self.trailing_stop_manager = TrailingStopManager(
                self.config,
                self.client,
                self.symbol_info
            )
            
            if trailing_stops_enabled:
                logger.info("Trailing stop manager initialized and enabled")
            else:
                logger.info("Trailing stop manager initialized but disabled")
                
        except Exception as e:
            logger.error(f"Error initializing trailing stop manager: {str(e)}")
            # Still create the manager but it will be disabled
            self.trailing_stop_manager = TrailingStopManager(
                self.config,
                self.client,
                self.symbol_info
            )

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
                
            # Check portfolio risk metrics if available
            if decision == 'BUY':
                try:
                    # Get portfolio risk metrics from Redis
                    portfolio_risk_json = await self.redis.get('portfolio_risk')
                    if portfolio_risk_json:
                        portfolio_risk = json.loads(portfolio_risk_json)
                        
                        # Check if portfolio VaR exceeds limit
                        portfolio_var_pct = portfolio_risk.get('portfolio_var_pct', 0)
                        max_var_limit = self.config.get('risk_management', {}).get('max_portfolio_var', 0.05)
                        
                        if portfolio_var_pct > max_var_limit:
                            logger.info(f"Portfolio VaR ({portfolio_var_pct:.2%}) exceeds limit ({max_var_limit:.2%}), blocking new trades")
                            return False
                            
                        # Check asset diversification if adding a new asset
                        asset = symbol.replace('USDC', '')
                        if asset not in self.holdings and 'correlations' in portfolio_risk:
                            # Check high correlations with existing assets
                            high_correlation = False
                            correlation_threshold = self.config.get('risk_management', {}).get('correlation_threshold', 0.7)
                            
                            for existing_asset in [a for a in self.holdings if a != 'USDC']:
                                corr = portfolio_risk['correlations'].get(asset, {}).get(existing_asset, 0)
                                if abs(corr) > correlation_threshold:
                                    logger.info(f"High correlation ({corr:.2f}) between {asset} and {existing_asset}, limiting position size")
                                    high_correlation = True
                                    break
                                    
                except Exception as e:
                    logger.error(f"Error checking portfolio risk metrics: {str(e)}")
                    # Continue with the trade even if risk check fails
            
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

    async def get_social_risk_adjustment(self, symbol):
        """Get social risk adjustment for a symbol"""
        try:
            if not self.redis or not await self.redis.ping():
                if not await self.connect_redis():
                    return None
            
            # Get adjustment from Redis
            adjustment_json = await self.redis.hget('social_risk_adjustments', symbol)
            if adjustment_json:
                return json.loads(adjustment_json)
            return None
            
        except Exception as e:
            logger.error(f"Error getting social risk adjustment for {symbol}: {str(e)}")
            return None
    
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

                # Get social sentiment-based risk adjustment (RISK-07)
                social_risk_adjustment = await self.get_social_risk_adjustment(symbol)
                
                # Check for risk information in the signal
                risk_info = signal.get('risk_info', {})
                
                # Calculate position size using risk-based approach if available
                default_position_pct = self.config['trading_params']['position_size']
                optimal_position_pct = risk_info.get('optimal_position_pct', default_position_pct)
                
                # Apply social sentiment-based position size adjustment if available
                if social_risk_adjustment:
                    logger.info(f"Applying social sentiment-based risk adjustment for {symbol}")
                    sentiment_type = social_risk_adjustment.get('sentiment_type', 'NEUTRAL')
                    position_adj = social_risk_adjustment.get('position_size_adj', 0)
                    
                    # Apply adjustment
                    adjusted_position_pct = optimal_position_pct * (1 + position_adj)
                    
                    # Log adjustment details
                    logger.info(f"Social sentiment: {sentiment_type} - Position size adjustment: {position_adj:+.2%}")
                    logger.info(f"Original position size: {optimal_position_pct:.2%} → Adjusted: {adjusted_position_pct:.2%}")
                    
                    # Use adjusted position size
                    optimal_position_pct = adjusted_position_pct
                
                # Apply position size
                position_size = min(
                    self.available_usdc * optimal_position_pct,
                    self.available_usdc * 0.95  # Max 95% of available balance
                )

                # Log position sizing information
                logger.debug(f"Position sizing for {symbol}: {optimal_position_pct:.2%} of available capital")
                if optimal_position_pct != default_position_pct:
                    logger.info(f"Using risk-optimized position size: {optimal_position_pct:.2%} (default: {default_position_pct:.2%})")

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

                # Default stop loss percentage from config
                stop_loss_pct = self.config['trading_params']['stop_loss_pct']
                
                # Use adaptive stop-loss if available, otherwise use default
                if 'adaptive_stop_loss' in risk_info:
                    adaptive_stop_price = risk_info['adaptive_stop_loss']
                    adaptive_stop_pct = risk_info.get('adaptive_stop_pct', stop_loss_pct)
                    
                    logger.info(f"Using adaptive stop-loss: {adaptive_stop_pct:.2f}% (default: {stop_loss_pct:.2f}%)")
                    
                    stop_loss_price = self.round_price(
                        adaptive_stop_price,
                        rules['tick_size']
                    )
                    
                    # Update stop_loss_pct for social adjustment
                    stop_loss_pct = adaptive_stop_pct
                else:
                    # Calculate default stop-loss price
                    stop_loss_price = fill_price * (1 - stop_loss_pct / 100)
                
                # Default take profit percentage from config
                take_profit_pct = self.config['trading_params']['take_profit_pct']
                
                # Apply social sentiment-based stop loss and take profit adjustments
                if social_risk_adjustment:
                    # Get adjustments
                    stop_loss_adj = social_risk_adjustment.get('stop_loss_adj', 0)
                    take_profit_adj = social_risk_adjustment.get('take_profit_adj', 0)
                    
                    # Apply stop loss adjustment
                    adjusted_stop_loss_pct = stop_loss_pct * (1 + stop_loss_adj)
                    
                    # Apply take profit adjustment
                    adjusted_take_profit_pct = take_profit_pct * (1 + take_profit_adj)
                    
                    # Log adjustments
                    logger.info(f"Stop loss: {stop_loss_pct:.2f}% → {adjusted_stop_loss_pct:.2f}% (adj: {stop_loss_adj:+.2%})")
                    logger.info(f"Take profit: {take_profit_pct:.2f}% → {adjusted_take_profit_pct:.2f}% (adj: {take_profit_adj:+.2%})")
                    
                    # Use adjusted percentages
                    stop_loss_pct = adjusted_stop_loss_pct
                    take_profit_pct = adjusted_take_profit_pct
                    
                    # Calculate adjusted stop loss price only if not using adaptive stop loss
                    if 'adaptive_stop_loss' not in risk_info:
                        stop_loss_price = fill_price * (1 - stop_loss_pct / 100)
                
                # Round stop loss price
                stop_loss_price = self.round_price(stop_loss_price, rules['tick_size'])
                
                # Calculate and round take profit price
                take_profit_price = self.round_price(
                    fill_price * (1 + take_profit_pct / 100),
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

                # Record trade with social sentiment information
                self.active_trades[symbol] = {
                    'entry_price': fill_price,
                    'quantity': filled_quantity,
                    'stop_loss_price': stop_loss_price,
                    'take_profit_price': take_profit_price,
                    'stop_loss_order': stop_loss_order['orderId'],
                    'take_profit_order': take_profit_order['orderId'],
                    'entry_time': datetime.now().isoformat(),
                    'status': 'ACTIVE',
                    'trailing_stop_active': False,
                    'highest_price': fill_price,
                    'social_sentiment': social_risk_adjustment.get('sentiment_type', 'NEUTRAL') if social_risk_adjustment else 'NEUTRAL',
                    'social_score': social_risk_adjustment.get('sentiment_score', 0.5) if social_risk_adjustment else 0.5
                }
                
                # Register trailing stop if enabled
                if self.trailing_stop_manager and self.config["risk_management"]["trailing_stop_loss_enabled"]:
                    try:
                        registered = self.trailing_stop_manager.register_trailing_stop(
                            symbol=symbol,
                            entry_price=fill_price,
                            initial_stop_price=stop_loss_price,
                            quantity=filled_quantity,
                            order_id=stop_loss_order['orderId']
                        )
                        
                        if registered:
                            logger.info(f"Registered trailing stop for {symbol}")
                            self.active_trades[symbol]['trailing_stop_active'] = True
                        
                    except Exception as e:
                        logger.error(f"Error setting up trailing stop for {symbol}: {str(e)}")
                        # Trade will still be valid with regular stop loss

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
            
            # Check if this position has a trailing stop
            if (self.trailing_stop_manager and 
                symbol in self.active_trades and 
                self.active_trades[symbol].get('trailing_stop_active', False)):
                
                # Remove from trailing stop manager first
                self.trailing_stop_manager.remove_trailing_stop(symbol)
                logger.debug(f"Removed trailing stop for {symbol}")
            
            # Place market sell order
            order = self.client.create_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            
            fill_price = float(order['fills'][0]['price'])
            
            # Calculate P&L if we have entry price
            if symbol in self.active_trades and 'entry_price' in self.active_trades[symbol]:
                entry_price = self.active_trades[symbol]['entry_price']
                pnl_pct = ((fill_price - entry_price) / entry_price) * 100
                pnl_usdc = (fill_price - entry_price) * quantity
                
                # Log performance with trailing stop information if available
                if self.active_trades[symbol].get('trailing_stop_active', False):
                    highest_price = self.active_trades[symbol].get('highest_price', entry_price)
                    max_potential_pnl = ((highest_price - entry_price) / entry_price) * 100
                    
                    logger.info(f"Closed position for {symbol} at ${fill_price:.8f} (P&L: {pnl_pct:.2f}%, ${pnl_usdc:.2f}) - {reason}")
                    logger.info(f"Max potential profit was {max_potential_pnl:.2f}% from highest price ${highest_price:.8f}")
                    
                    # Log trailing stop effectiveness
                    if pnl_pct > 0 and max_potential_pnl > 0:
                        capture_pct = (pnl_pct / max_potential_pnl) * 100
                        logger.info(f"Trailing stop captured {capture_pct:.2f}% of maximum potential profit")
                else:
                    logger.info(f"Closed position for {symbol} at ${fill_price:.8f} (P&L: {pnl_pct:.2f}%, ${pnl_usdc:.2f}) - {reason}")
            else:
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
            # Check for stop-loss adjustment recommendations
            try:
                adjustments_json = await self.redis.get('adaptive_stop_losses')
                if adjustments_json:
                    stop_loss_adjustments = json.loads(adjustments_json)
                else:
                    stop_loss_adjustments = {}
            except Exception as e:
                logger.error(f"Error fetching stop-loss adjustments: {str(e)}")
                stop_loss_adjustments = {}
            
            # Get trailing stop manager status report
            trailing_stops_report = None
            if self.trailing_stop_manager:
                trailing_stops_report = self.trailing_stop_manager.get_status_report()
                
                # Publish trailing stops status to Redis for dashboard
                if self.redis and await self.redis.ping():
                    try:
                        await self.redis.set('trailing_stops', json.dumps({
                            "report": trailing_stops_report,
                            "timestamp": datetime.now().isoformat()
                        }))
                    except Exception as e:
                        logger.error(f"Error publishing trailing stops report: {str(e)}")
            
            # Monitor each active trade
            for symbol, trade in list(self.active_trades.items()):
                try:
                    # Get current price
                    ticker = self.client.get_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    
                    # Update highest price tracking
                    if current_price > trade.get('highest_price', 0):
                        trade['highest_price'] = current_price
                        self.active_trades[symbol] = trade
                    
                    # Calculate current P&L
                    entry_price = trade['entry_price']
                    quantity = trade['quantity']
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    
                    logger.debug(f"Monitoring {symbol} - Current: ${current_price:.8f}, P&L: {pnl_pct:.2f}%")
                    
                    # Handle trailing stop if enabled for this position
                    if (self.trailing_stop_manager and 
                        self.config["risk_management"]["trailing_stop_loss_enabled"] and
                        trade.get('trailing_stop_active', False)):
                        
                        # Update price in trailing stop manager
                        self.trailing_stop_manager.update_price(symbol, current_price)
                        
                        # Check if trailing stop should be executed
                        if self.trailing_stop_manager.should_execute_stop(symbol, current_price):
                            logger.info(f"Trailing stop triggered for {symbol} at ${current_price:.8f}")
                            await self.close_position(symbol, quantity, "Trailing Stop")
                            continue  # Skip to next position after closing
                        
                        # Check if trailing stop is activated and needs to update the order
                        if (self.trailing_stop_manager.is_activated(symbol) and 
                            self.trailing_stop_manager.get_current_stop(symbol) > trade['stop_loss_price']):
                            
                            new_stop_price = self.trailing_stop_manager.get_current_stop(symbol)
                            
                            try:
                                # Update the stop-loss order
                                await self._update_stop_loss_order(symbol, new_stop_price)
                                
                                # Update active trades record
                                trade['stop_loss_price'] = new_stop_price
                                self.active_trades[symbol] = trade
                                
                            except Exception as e:
                                logger.error(f"Error updating trailing stop order for {symbol}: {str(e)}")
                        
                    # Check regular adaptive stop-loss adjustments if not using trailing stops
                    elif symbol in stop_loss_adjustments and current_price > trade['stop_loss_price']:
                        adjustment = stop_loss_adjustments[symbol]
                        adaptive_stop = adjustment.get('adaptive_stop_loss', 0)
                        
                        # Only adjust stop-loss upward if price has moved in our favor
                        if adaptive_stop > trade['stop_loss_price'] and pnl_pct > 0:
                            # Update the stop-loss order
                            if await self._update_stop_loss_order(symbol, adaptive_stop):
                                # Update active trades record
                                trade['stop_loss_price'] = adaptive_stop
                                self.active_trades[symbol] = trade
                    
                    # Check stop loss - only if not using trailing stops or trailing stop not activated
                    if not trade.get('trailing_stop_active', False) or not self.trailing_stop_manager.is_activated(symbol):
                        if current_price <= trade['stop_loss_price']:
                            logger.info(f"Stop loss triggered for {symbol} at ${current_price:.8f} ({pnl_pct:.2f}%)")
                            await self.close_position(symbol, quantity, "Stop Loss")
                    
                    # Check take profit
                    elif current_price >= trade['take_profit_price']:
                        logger.info(f"Take profit triggered for {symbol} at ${current_price:.8f} ({pnl_pct:.2f}%)")
                        await self.close_position(symbol, quantity, "Take Profit")
                    
                except Exception as e:
                    logger.error(f"Error monitoring trade for {symbol}: {str(e)}", exc_info=True)
            
            # Publish active trades to Redis for other services
            try:
                await self.redis.set('active_trades', json.dumps(self.active_trades))
            except Exception as e:
                logger.error(f"Error publishing active trades to Redis: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error in trade monitoring: {str(e)}", exc_info=True)
            
    async def _update_stop_loss_order(self, symbol, new_stop_price):
        """Update a stop loss order with a new price"""
        if symbol not in self.active_trades:
            return False
            
        trade = self.active_trades[symbol]
        
        try:
            # Cancel existing stop-loss order
            self.client.cancel_order(
                symbol=symbol,
                orderId=trade['stop_loss_order']
            )
            
            # Get trading rules
            rules = self.symbol_info.get(symbol)
            if not rules:
                raise Exception(f"No trading rules found for {symbol}")
            
            # Round to valid tick size
            new_stop_price = self.round_price(new_stop_price, rules['tick_size'])
            logger.info(f"Adjusting stop-loss for {symbol}: ${trade['stop_loss_price']:.8f} → ${new_stop_price:.8f}")
            
            # Place new stop-loss order
            new_stop_order = self.client.create_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_STOP_LOSS_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=trade['quantity'],
                stopPrice=new_stop_price,
                price=self.round_price(new_stop_price * 0.99, rules['tick_size'])
            )
            
            # Update trade record with new order ID
            trade['stop_loss_order'] = new_stop_order['orderId']
            trade['stop_loss_price'] = new_stop_price
            self.active_trades[symbol] = trade
            
            # Update order ID in trailing stop manager if needed
            if (self.trailing_stop_manager and 
                self.config["risk_management"]["trailing_stop_loss_enabled"] and
                trade.get('trailing_stop_active', False)):
                
                if symbol in self.trailing_stop_manager.trailing_stops:
                    self.trailing_stop_manager.trailing_stops[symbol]['order_id'] = new_stop_order['orderId']
            
            logger.info(f"Stop-loss updated for {symbol} to ${new_stop_price:.8f}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating stop-loss order for {symbol}: {str(e)}")
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
