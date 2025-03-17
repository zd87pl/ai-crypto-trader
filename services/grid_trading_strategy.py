import os
import json
import logging
import asyncio
import time
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import redis
from binance.client import Client
from binance.exceptions import BinanceAPIException


class GridTradingStrategy:
    """
    Grid Trading Strategy implementation.
    
    The grid strategy places buy and sell orders at regular price intervals (a grid),
    allowing for automated trading as the price oscillates within the grid boundaries.
    When price moves up and hits a sell order, the order is executed and a new buy order
    is placed below. When price moves down and hits a buy order, a new sell order is
    placed above.
    
    Features:
    - Dynamic grid based on volatility
    - Multiple grid types (arithmetic, geometric, custom)
    - Risk management with auto-rebalancing
    - Market condition adaptation
    - Profit tracking and optimization
    """
    
    def __init__(self):
        """Initialize the grid trading strategy service"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Grid Trading Strategy")
        
        # Load configuration
        try:
            with open('config.json', 'r') as f:
                self.config = json.load(f)
                self.grid_config = self.config.get('grid_trading', {})
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            self.config = {}
            self.grid_config = {}
        
        # Set up Redis connection
        redis_host = os.environ.get('REDIS_HOST', 'localhost')
        redis_port = int(os.environ.get('REDIS_PORT', 6379))
        self.redis = redis.Redis(host=redis_host, port=redis_port, db=0)
        
        # Initialize Binance client
        self.api_key = os.environ.get('BINANCE_API_KEY', '')
        self.api_secret = os.environ.get('BINANCE_API_SECRET', '')
        
        try:
            self.client = Client(self.api_key, self.api_secret)
            self.logger.info("Binance client initialized")
        except Exception as e:
            self.logger.error(f"Error initializing Binance client: {str(e)}")
            self.client = None
        
        # Grid strategy parameters
        self.enabled = self.grid_config.get('enabled', False)
        self.symbols = self.grid_config.get('symbols', ['BTCUSDC', 'ETHUSDC'])
        self.service_port = self.grid_config.get('service_port', 8014)
        self.update_interval = self.grid_config.get('update_interval', 60)  # seconds
        
        # Define grid parameters (default values)
        self.grid_type = self.grid_config.get('grid_type', 'arithmetic')  # arithmetic, geometric, volatility_based
        self.num_grids = self.grid_config.get('num_grids', 10)
        self.grid_spread = self.grid_config.get('grid_spread', 0.5)  # percentage between grid levels
        
        # Grid boundaries
        self.auto_boundaries = self.grid_config.get('auto_boundaries', True)
        self.lower_boundary_pct = self.grid_config.get('lower_boundary_pct', 5.0)
        self.upper_boundary_pct = self.grid_config.get('upper_boundary_pct', 5.0)
        
        # Position sizing
        self.quantity_per_grid = self.grid_config.get('quantity_per_grid', 0.01)  # in BTC or ETH units
        self.max_active_grids = self.grid_config.get('max_active_grids', 5)  # maximum number of grid positions
        self.order_size_type = self.grid_config.get('order_size_type', 'fixed')  # fixed, decreasing, increasing
        
        # Risk management
        self.max_total_investment = self.grid_config.get('max_total_investment', 0.2)  # as portion of available capital
        self.stop_loss_enabled = self.grid_config.get('stop_loss_enabled', True)
        self.stop_loss_pct = self.grid_config.get('stop_loss_pct', 10.0)
        
        # State tracking
        self.active_grids = {}  # symbol -> grid configuration
        self.active_orders = {}  # symbol -> list of active orders
        self.grid_profits = {}  # symbol -> grid profit tracking
        self.initialized = False
        
        # Auto-adjustment parameters
        self.volatility_lookback = self.grid_config.get('volatility_lookback', 14)  # days
        self.auto_adjust_period = self.grid_config.get('auto_adjust_period', 86400)  # seconds (default: 1 day)
        self.last_adjustment = {}  # symbol -> last adjustment time
        
        # Market condition adaptation
        self.adapt_to_market_regime = self.grid_config.get('adapt_to_market_regime', True)
        
        # Rebalancing
        self.rebalance_frequency = self.grid_config.get('rebalance_frequency', 3600)  # seconds (default: 1 hour)
        self.last_rebalance = {}  # symbol -> last rebalance time
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_profit = 0
        
        # Simulation mode for testing (no real orders)
        self.simulation_mode = self.grid_config.get('simulation_mode', True)
        
        # Notification settings
        self.notify_on_trade = self.grid_config.get('notify_on_trade', True)
        
        self.logger.info(f"Grid trading strategy initialized for {len(self.symbols)} symbols")
    
    async def run(self):
        """Run the grid trading strategy service"""
        self.logger.info("Starting Grid Trading Strategy Service")
        
        # Check if trading is enabled
        if not self.enabled:
            self.logger.info("Grid trading is disabled in configuration")
            return
        
        # Check if Binance client is available
        if not self.client:
            self.logger.error("Cannot start service: Binance client not available")
            return
        
        # Publish service status to Redis
        self.redis.set(
            'grid_trading_status', 
            json.dumps({
                'status': 'starting',
                'timestamp': datetime.now().isoformat(),
                'simulation_mode': self.simulation_mode
            })
        )
        
        # Initialize grid strategies for each symbol
        await self._initialize_grids()
        
        # Update status to running
        self.redis.set(
            'grid_trading_status', 
            json.dumps({
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'symbols': list(self.active_grids.keys()),
                'simulation_mode': self.simulation_mode
            })
        )
        
        # Main service loop
        try:
            while True:
                start_time = time.time()
                
                # Process each symbol's grid
                for symbol in self.symbols:
                    if symbol in self.active_grids:
                        try:
                            await self._process_grid(symbol)
                        except Exception as e:
                            self.logger.error(f"Error processing grid for {symbol}: {str(e)}")
                    else:
                        # Try to initialize grid if not already done
                        try:
                            await self._initialize_grid(symbol)
                        except Exception as e:
                            self.logger.error(f"Error initializing grid for {symbol}: {str(e)}")
                
                # Check for rebalancing opportunity
                current_time = datetime.now()
                for symbol in self.active_grids:
                    last_rebalance = self.last_rebalance.get(symbol, self.start_time)
                    if (current_time - last_rebalance).total_seconds() > self.rebalance_frequency:
                        try:
                            await self._rebalance_grid(symbol)
                            self.last_rebalance[symbol] = current_time
                        except Exception as e:
                            self.logger.error(f"Error rebalancing grid for {symbol}: {str(e)}")
                
                # Check for adjustment opportunity
                for symbol in self.active_grids:
                    last_adjustment = self.last_adjustment.get(symbol, self.start_time)
                    if (current_time - last_adjustment).total_seconds() > self.auto_adjust_period:
                        try:
                            await self._adjust_grid_parameters(symbol)
                            self.last_adjustment[symbol] = current_time
                        except Exception as e:
                            self.logger.error(f"Error adjusting grid for {symbol}: {str(e)}")
                
                # Update status metrics
                await self._update_performance_metrics()
                
                # Sleep for the remainder of the interval
                elapsed = time.time() - start_time
                sleep_time = max(0.1, self.update_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            self.logger.info("Grid Trading Strategy Service shutting down")
            # Cancel all open orders if not in simulation mode
            if not self.simulation_mode:
                await self._cancel_all_orders()
            
            # Update service status in Redis
            self.redis.set(
                'grid_trading_status', 
                json.dumps({
                    'status': 'stopped',
                    'timestamp': datetime.now().isoformat()
                })
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in Grid Trading Strategy Service: {str(e)}")
            # Update service status in Redis with error
            self.redis.set(
                'grid_trading_status', 
                json.dumps({
                    'status': 'error',
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'simulation_mode': self.simulation_mode
                })
            )
    
    async def _initialize_grids(self):
        """Initialize grid trading strategies for all symbols"""
        for symbol in self.symbols:
            try:
                await self._initialize_grid(symbol)
            except Exception as e:
                self.logger.error(f"Error initializing grid for {symbol}: {str(e)}")
        
        self.initialized = True
    
    async def _initialize_grid(self, symbol):
        """Initialize a grid trading strategy for a single symbol"""
        self.logger.info(f"Initializing grid for {symbol}")
        
        try:
            # Get current price
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            
            # Get symbol information (for order precision)
            symbol_info = None
            exchange_info = self.client.get_exchange_info()
            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    symbol_info = s
                    break
            
            if not symbol_info:
                self.logger.error(f"Symbol information not found for {symbol}")
                return
            
            # Calculate grid boundaries
            if self.auto_boundaries:
                # Calculate boundaries based on volatility
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=Client.KLINE_INTERVAL_1DAY,
                    start_str=f"{self.volatility_lookback} days ago UTC"
                )
                closes = [float(kline[4]) for kline in klines]
                
                if not closes:
                    self.logger.error(f"No historical data available for {symbol}")
                    return
                
                # Calculate price volatility
                volatility = np.std(closes) / np.mean(closes) * 100
                
                # Adjust boundaries based on volatility
                lower_boundary = current_price * (1 - volatility / 100)
                upper_boundary = current_price * (1 + volatility / 100)
            else:
                # Use configured percentages
                lower_boundary = current_price * (1 - self.lower_boundary_pct / 100)
                upper_boundary = current_price * (1 + self.upper_boundary_pct / 100)
            
            # Generate grid levels
            grid_levels = self._generate_grid_levels(
                lower_boundary,
                upper_boundary,
                self.num_grids,
                grid_type=self.grid_type
            )
            
            # Calculate quantity for each grid level based on configuration
            quantity = self._calculate_grid_quantity(symbol, current_price)
            
            # Store grid configuration
            self.active_grids[symbol] = {
                'symbol': symbol,
                'current_price': current_price,
                'lower_boundary': lower_boundary,
                'upper_boundary': upper_boundary,
                'grid_levels': grid_levels,
                'quantity': quantity,
                'initialized_at': datetime.now().isoformat(),
                'last_price': current_price,
                'price_precision': self._get_price_precision(symbol_info),
                'quantity_precision': self._get_quantity_precision(symbol_info)
            }
            
            # Initialize orders if not in simulation mode
            if not self.simulation_mode:
                await self._place_initial_orders(symbol)
            
            # Initialize profit tracking
            self.grid_profits[symbol] = {
                'total_profit': 0.0,
                'total_trades': 0,
                'profitable_trades': 0,
                'last_trade_profit': 0.0,
                'start_time': datetime.now().isoformat()
            }
            
            # Set last adjustment time
            self.last_adjustment[symbol] = datetime.now()
            
            # Set last rebalance time
            self.last_rebalance[symbol] = datetime.now()
            
            # Update Redis with grid configuration
            self.redis.set(
                f'grid_config:{symbol}',
                json.dumps(self.active_grids[symbol])
            )
            
            self.logger.info(f"Grid initialized for {symbol} with {len(grid_levels)} levels between {lower_boundary:.6f} and {upper_boundary:.6f}")
            
        except Exception as e:
            self.logger.error(f"Error initializing grid for {symbol}: {str(e)}")
            raise
    
    def _generate_grid_levels(self, lower_boundary, upper_boundary, num_grids, grid_type='arithmetic'):
        """Generate grid price levels based on the chosen grid type"""
        grid_levels = []
        
        if grid_type == 'arithmetic':
            # Equal price difference between each level
            step = (upper_boundary - lower_boundary) / num_grids
            for i in range(num_grids + 1):
                grid_levels.append(lower_boundary + i * step)
        
        elif grid_type == 'geometric':
            # Equal percentage difference between each level
            ratio = (upper_boundary / lower_boundary) ** (1 / num_grids)
            for i in range(num_grids + 1):
                grid_levels.append(lower_boundary * (ratio ** i))
        
        elif grid_type == 'volatility_based':
            # More levels in areas of higher historical volatility
            # This is a simplified version - would need actual historic volatility data
            # for a more accurate implementation
            step = (upper_boundary - lower_boundary) / num_grids
            for i in range(num_grids + 1):
                grid_levels.append(lower_boundary + i * step)
            
            # Randomize slightly to simulate volatility-based distribution
            # (just for demonstration - a real implementation would use historical data)
            for i in range(1, len(grid_levels) - 1):
                grid_levels[i] += (np.random.random() - 0.5) * step * 0.2
            
            # Ensure levels are in ascending order
            grid_levels.sort()
        
        else:
            # Default to arithmetic grid
            step = (upper_boundary - lower_boundary) / num_grids
            for i in range(num_grids + 1):
                grid_levels.append(lower_boundary + i * step)
        
        return grid_levels
    
    def _calculate_grid_quantity(self, symbol, current_price):
        """Calculate quantity for each grid level based on configuration"""
        # This is a simplified calculation - in a real implementation, 
        # you'd need to consider account balance and risk parameters
        
        # Get symbol-specific configuration if available
        symbol_config = next((s for s in self.grid_config.get('symbol_settings', []) 
                             if s.get('symbol') == symbol), None)
        
        if symbol_config and 'quantity_per_grid' in symbol_config:
            return symbol_config.get('quantity_per_grid')
        
        # Default to global setting
        return self.quantity_per_grid
    
    def _get_price_precision(self, symbol_info):
        """Get price precision from symbol info"""
        for filter in symbol_info['filters']:
            if filter['filterType'] == 'PRICE_FILTER':
                tick_size = float(filter['tickSize'])
                return len(str(tick_size).rstrip('0').split('.')[1]) if '.' in str(tick_size) else 0
        return 6  # Default precision
    
    def _get_quantity_precision(self, symbol_info):
        """Get quantity precision from symbol info"""
        for filter in symbol_info['filters']:
            if filter['filterType'] == 'LOT_SIZE':
                step_size = float(filter['stepSize'])
                return len(str(step_size).rstrip('0').split('.')[1]) if '.' in str(step_size) else 0
        return 6  # Default precision
    
    async def _place_initial_orders(self, symbol):
        """Place initial grid orders based on current price"""
        grid_config = self.active_grids[symbol]
        current_price = grid_config['current_price']
        grid_levels = grid_config['grid_levels']
        quantity = grid_config['quantity']
        price_precision = grid_config['price_precision']
        quantity_precision = grid_config['quantity_precision']
        
        # Initialize active orders list
        self.active_orders[symbol] = []
        
        # Find grid levels above and below current price
        buy_levels = [level for level in grid_levels if level < current_price]
        sell_levels = [level for level in grid_levels if level > current_price]
        
        # Limit to max_active_grids/2 on each side
        max_orders_per_side = self.max_active_grids // 2
        buy_levels = sorted(buy_levels, reverse=True)[:max_orders_per_side]
        sell_levels = sorted(sell_levels)[:max_orders_per_side]
        
        self.logger.info(f"Placing initial orders for {symbol}: {len(buy_levels)} buy orders, {len(sell_levels)} sell orders")
        
        # Place buy orders
        for level in buy_levels:
            price = round(level, price_precision)
            qty = round(quantity, quantity_precision)
            
            try:
                # Place limit buy order
                order = self.client.create_order(
                    symbol=symbol,
                    side=Client.SIDE_BUY,
                    type=Client.ORDER_TYPE_LIMIT,
                    timeInForce=Client.TIME_IN_FORCE_GTC,
                    quantity=qty,
                    price=str(price)
                )
                
                self.active_orders[symbol].append({
                    'orderId': order['orderId'],
                    'side': 'BUY',
                    'price': price,
                    'quantity': qty,
                    'status': order['status'],
                    'grid_level': level
                })
                
                self.logger.info(f"Placed buy order for {symbol} at {price}")
                
            except BinanceAPIException as e:
                self.logger.error(f"Error placing buy order for {symbol} at {price}: {str(e)}")
        
        # Place sell orders
        for level in sell_levels:
            price = round(level, price_precision)
            qty = round(quantity, quantity_precision)
            
            try:
                # Place limit sell order
                order = self.client.create_order(
                    symbol=symbol,
                    side=Client.SIDE_SELL,
                    type=Client.ORDER_TYPE_LIMIT,
                    timeInForce=Client.TIME_IN_FORCE_GTC,
                    quantity=qty,
                    price=str(price)
                )
                
                self.active_orders[symbol].append({
                    'orderId': order['orderId'],
                    'side': 'SELL',
                    'price': price,
                    'quantity': qty,
                    'status': order['status'],
                    'grid_level': level
                })
                
                self.logger.info(f"Placed sell order for {symbol} at {price}")
                
            except BinanceAPIException as e:
                self.logger.error(f"Error placing sell order for {symbol} at {price}: {str(e)}")
        
        # Update Redis with active orders
        self.redis.set(
            f'grid_orders:{symbol}',
            json.dumps({
                'timestamp': datetime.now().isoformat(),
                'orders': self.active_orders[symbol]
            })
        )
    
    async def _process_grid(self, symbol):
        """Process grid trading for a symbol, checking for filled orders and placing new ones"""
        if self.simulation_mode:
            await self._process_grid_simulation(symbol)
        else:
            await self._process_grid_live(symbol)
    
    async def _process_grid_live(self, symbol):
        """Process grid trading for a symbol in live trading mode"""
        if symbol not in self.active_orders or not self.active_orders[symbol]:
            self.logger.warning(f"No active orders for {symbol}")
            return
        
        # Get current price
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        
        # Update last price in grid config
        if symbol in self.active_grids:
            self.active_grids[symbol]['last_price'] = current_price
        
        # Get grid configuration
        grid_config = self.active_grids[symbol]
        grid_levels = grid_config['grid_levels']
        quantity = grid_config['quantity']
        price_precision = grid_config['price_precision']
        quantity_precision = grid_config['quantity_precision']
        
        # Check for filled orders
        filled_orders = []
        active_orders = []
        
        for order_info in self.active_orders[symbol]:
            try:
                order = self.client.get_order(
                    symbol=symbol,
                    orderId=order_info['orderId']
                )
                
                if order['status'] == 'FILLED':
                    filled_orders.append(order_info)
                    self.logger.info(f"Order filled for {symbol}: {order_info['side']} at {order_info['price']}")
                else:
                    active_orders.append(order_info)
            
            except BinanceAPIException as e:
                self.logger.error(f"Error checking order status for {symbol}: {str(e)}")
        
        # Update active orders list
        self.active_orders[symbol] = active_orders
        
        # Process filled orders and place new ones
        for filled_order in filled_orders:
            # Update profit tracking
            filled_price = float(filled_order['price'])
            filled_qty = float(filled_order['quantity'])
            
            if filled_order['side'] == 'BUY':
                # Buy order filled, place a sell order above it
                buy_level_index = grid_levels.index(filled_order['grid_level'])
                
                if buy_level_index < len(grid_levels) - 1:
                    sell_level = grid_levels[buy_level_index + 1]
                    sell_price = round(sell_level, price_precision)
                    
                    try:
                        # Place sell order
                        order = self.client.create_order(
                            symbol=symbol,
                            side=Client.SIDE_SELL,
                            type=Client.ORDER_TYPE_LIMIT,
                            timeInForce=Client.TIME_IN_FORCE_GTC,
                            quantity=round(filled_qty, quantity_precision),
                            price=str(sell_price)
                        )
                        
                        self.active_orders[symbol].append({
                            'orderId': order['orderId'],
                            'side': 'SELL',
                            'price': sell_price,
                            'quantity': filled_qty,
                            'status': order['status'],
                            'grid_level': sell_level
                        })
                        
                        self.logger.info(f"Placed sell order for {symbol} at {sell_price} after buy fill")
                    
                    except BinanceAPIException as e:
                        self.logger.error(f"Error placing sell order for {symbol} at {sell_price}: {str(e)}")
            
            elif filled_order['side'] == 'SELL':
                # Sell order filled, place a buy order below it
                sell_level_index = grid_levels.index(filled_order['grid_level'])
                
                if sell_level_index > 0:
                    buy_level = grid_levels[sell_level_index - 1]
                    buy_price = round(buy_level, price_precision)
                    
                    try:
                        # Place buy order
                        order = self.client.create_order(
                            symbol=symbol,
                            side=Client.SIDE_BUY,
                            type=Client.ORDER_TYPE_LIMIT,
                            timeInForce=Client.TIME_IN_FORCE_GTC,
                            quantity=round(filled_qty, quantity_precision),
                            price=str(buy_price)
                        )
                        
                        self.active_orders[symbol].append({
                            'orderId': order['orderId'],
                            'side': 'BUY',
                            'price': buy_price,
                            'quantity': filled_qty,
                            'status': order['status'],
                            'grid_level': buy_level
                        })
                        
                        self.logger.info(f"Placed buy order for {symbol} at {buy_price} after sell fill")
                    
                    except BinanceAPIException as e:
                        self.logger.error(f"Error placing buy order for {symbol} at {buy_price}: {str(e)}")
                
                # Calculate and track profit
                profit = (filled_price - grid_levels[sell_level_index - 1]) * filled_qty
                
                if profit > 0:
                    self.grid_profits[symbol]['profitable_trades'] += 1
                
                self.grid_profits[symbol]['total_profit'] += profit
                self.grid_profits[symbol]['total_trades'] += 1
                self.grid_profits[symbol]['last_trade_profit'] = profit
                
                self.logger.info(f"Grid trade profit for {symbol}: {profit:.6f} USDC")
                
                # Update global stats
                self.total_trades += 1
                if profit > 0:
                    self.profitable_trades += 1
                self.total_profit += profit
                
                # Publish trade notification to Redis
                if self.notify_on_trade:
                    self.redis.publish(
                        'grid_trade_notifications',
                        json.dumps({
                            'timestamp': datetime.now().isoformat(),
                            'symbol': symbol,
                            'side': filled_order['side'],
                            'price': filled_price,
                            'quantity': filled_qty,
                            'profit': profit
                        })
                    )
        
        # Update Redis with active orders and profit
        self.redis.set(
            f'grid_orders:{symbol}',
            json.dumps({
                'timestamp': datetime.now().isoformat(),
                'orders': self.active_orders[symbol]
            })
        )
        
        self.redis.set(
            f'grid_profit:{symbol}',
            json.dumps(self.grid_profits[symbol])
        )
    
    async def _process_grid_simulation(self, symbol):
        """Process grid trading for a symbol in simulation mode"""
        # Get current price
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        
        # Initialize if not already done
        if symbol not in self.active_grids:
            await self._initialize_grid(symbol)
            return
        
        # Get grid configuration
        grid_config = self.active_grids[symbol]
        grid_levels = grid_config['grid_levels']
        last_price = grid_config['last_price']
        
        # Update last price in grid config
        self.active_grids[symbol]['last_price'] = current_price
        
        # Check if price crossed any grid levels
        crossed_levels = []
        
        if current_price > last_price:
            # Price moved up, check for crossed levels
            crossed_levels = [level for level in grid_levels 
                              if last_price <= level <= current_price]
            
            for level in crossed_levels:
                # Simulate a buy order being filled
                self.logger.info(f"[SIMULATION] Grid level {level:.6f} crossed upward for {symbol}")
                
                # Calculate and track profit
                quantity = self.active_grids[symbol]['quantity']
                profit = (level - grid_levels[grid_levels.index(level) - 1]) * quantity if grid_levels.index(level) > 0 else 0
                
                if profit > 0:
                    self.grid_profits[symbol]['profitable_trades'] += 1
                
                self.grid_profits[symbol]['total_profit'] += profit
                self.grid_profits[symbol]['total_trades'] += 1
                self.grid_profits[symbol]['last_trade_profit'] = profit
                
                self.logger.info(f"[SIMULATION] Grid trade profit for {symbol}: {profit:.6f} USDC")
                
                # Update global stats
                self.total_trades += 1
                if profit > 0:
                    self.profitable_trades += 1
                self.total_profit += profit
                
                # Publish trade notification to Redis
                if self.notify_on_trade:
                    self.redis.publish(
                        'grid_trade_notifications',
                        json.dumps({
                            'timestamp': datetime.now().isoformat(),
                            'symbol': symbol,
                            'side': 'SELL',
                            'price': level,
                            'quantity': quantity,
                            'profit': profit,
                            'simulation': True
                        })
                    )
        
        elif current_price < last_price:
            # Price moved down, check for crossed levels
            crossed_levels = [level for level in grid_levels 
                              if current_price <= level <= last_price]
            
            for level in crossed_levels:
                # Simulate a sell order being filled
                self.logger.info(f"[SIMULATION] Grid level {level:.6f} crossed downward for {symbol}")
                
                # No direct profit from buy orders being filled
                # But we track the trade
                self.grid_profits[symbol]['total_trades'] += 1
                
                # Update global stats
                self.total_trades += 1
                
                # Publish trade notification to Redis
                if self.notify_on_trade:
                    self.redis.publish(
                        'grid_trade_notifications',
                        json.dumps({
                            'timestamp': datetime.now().isoformat(),
                            'symbol': symbol,
                            'side': 'BUY',
                            'price': level,
                            'quantity': self.active_grids[symbol]['quantity'],
                            'profit': 0,
                            'simulation': True
                        })
                    )
        
        # Update Redis with profit
        self.redis.set(
            f'grid_profit:{symbol}',
            json.dumps(self.grid_profits[symbol])
        )
    
    async def _rebalance_grid(self, symbol):
        """Rebalance grid by adjusting boundaries based on current price"""
        if symbol not in self.active_grids:
            return
        
        self.logger.info(f"Rebalancing grid for {symbol}")
        
        # Get current price
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        
        # Get grid configuration
        grid_config = self.active_grids[symbol]
        lower_boundary = grid_config['lower_boundary']
        upper_boundary = grid_config['upper_boundary']
        
        # Check if current price is close to boundaries
        lower_threshold = lower_boundary * 1.05  # Within 5% of lower boundary
        upper_threshold = upper_boundary * 0.95  # Within 5% of upper boundary
        
        if current_price < lower_threshold or current_price > upper_threshold:
            self.logger.info(f"Current price {current_price:.6f} is close to grid boundaries, rebalancing grid for {symbol}")
            
            # Cancel all orders in live mode
            if not self.simulation_mode:
                await self._cancel_orders_for_symbol(symbol)
            
            # Recalculate grid boundaries
            new_lower = current_price * (1 - self.lower_boundary_pct / 100)
            new_upper = current_price * (1 + self.upper_boundary_pct / 100)
            
            # Generate new grid levels
            new_grid_levels = self._generate_grid_levels(
                new_lower,
                new_upper,
                self.num_grids,
                grid_type=self.grid_type
            )
            
            # Update grid configuration
            self.active_grids[symbol]['lower_boundary'] = new_lower
            self.active_grids[symbol]['upper_boundary'] = new_upper
            self.active_grids[symbol]['grid_levels'] = new_grid_levels
            self.active_grids[symbol]['current_price'] = current_price
            
            # Place new orders in live mode
            if not self.simulation_mode:
                await self._place_initial_orders(symbol)
            
            # Update Redis with new grid configuration
            self.redis.set(
                f'grid_config:{symbol}',
                json.dumps(self.active_grids[symbol])
            )
            
            self.logger.info(f"Grid rebalanced for {symbol} with new boundaries: {new_lower:.6f} - {new_upper:.6f}")
        else:
            self.logger.info(f"No rebalancing needed for {symbol}, price is within boundaries")
    
    async def _adjust_grid_parameters(self, symbol):
        """Adjust grid parameters based on market conditions and performance"""
        if symbol not in self.active_grids:
            return
        
        self.logger.info(f"Adjusting grid parameters for {symbol}")
        
        # Check if we should adapt to market regime
        if self.adapt_to_market_regime:
            # Get current market regime if available
            regime_json = self.redis.get('market_regime_history')
            
            if regime_json:
                try:
                    regime_history = json.loads(regime_json)
                    
                    if regime_history and len(regime_history) > 0:
                        # Get the most recent regime
                        current_regime = regime_history[-1]['regime']
                        
                        # Adjust grid parameters based on regime
                        if current_regime == 'ranging':
                            # For ranging markets, use more grids with tighter spacing
                            self.num_grids = self.grid_config.get('ranging_num_grids', 15)
                            self.lower_boundary_pct = self.grid_config.get('ranging_lower_boundary_pct', 3.0)
                            self.upper_boundary_pct = self.grid_config.get('ranging_upper_boundary_pct', 3.0)
                            self.logger.info(f"Adjusted to ranging market regime for {symbol}: {self.num_grids} grids, {self.lower_boundary_pct}% boundaries")
                        
                        elif current_regime == 'trending':
                            # For trending markets, use fewer grids with wider spacing
                            self.num_grids = self.grid_config.get('trending_num_grids', 8)
                            self.lower_boundary_pct = self.grid_config.get('trending_lower_boundary_pct', 8.0)
                            self.upper_boundary_pct = self.grid_config.get('trending_upper_boundary_pct', 8.0)
                            self.logger.info(f"Adjusted to trending market regime for {symbol}: {self.num_grids} grids, {self.lower_boundary_pct}% boundaries")
                        
                        elif current_regime == 'volatile':
                            # For volatile markets, use moderate grids with asymmetric spacing
                            self.num_grids = self.grid_config.get('volatile_num_grids', 12)
                            self.lower_boundary_pct = self.grid_config.get('volatile_lower_boundary_pct', 6.0)
                            self.upper_boundary_pct = self.grid_config.get('volatile_upper_boundary_pct', 6.0)
                            self.logger.info(f"Adjusted to volatile market regime for {symbol}: {self.num_grids} grids, {self.lower_boundary_pct}% boundaries")
                        
                except Exception as e:
                    self.logger.error(f"Error adjusting grid parameters based on market regime: {str(e)}")
        
        # Evaluate grid performance
        if symbol in self.grid_profits:
            profit_data = self.grid_profits[symbol]
            
            # If we have enough trades to evaluate
            if profit_data['total_trades'] > 10:
                win_rate = profit_data['profitable_trades'] / profit_data['total_trades'] if profit_data['total_trades'] > 0 else 0
                
                # If win rate is low, adjust grid parameters
                if win_rate < 0.4:
                    # Reduce number of grids and widen spacing
                    self.num_grids = max(5, self.num_grids - 2)
                    self.grid_spread = min(1.0, self.grid_spread * 1.2)
                    self.logger.info(f"Adjusted grid parameters due to low win rate for {symbol}: {self.num_grids} grids, {self.grid_spread}% spread")
                
                # If win rate is high, optimize grid parameters
                elif win_rate > 0.7:
                    # Increase number of grids and tighten spacing
                    self.num_grids = min(20, self.num_grids + 2)
                    self.grid_spread = max(0.2, self.grid_spread * 0.9)
                    self.logger.info(f"Adjusted grid parameters due to high win rate for {symbol}: {self.num_grids} grids, {self.grid_spread}% spread")
    
    async def _cancel_orders_for_symbol(self, symbol):
        """Cancel all active orders for a symbol"""
        if symbol not in self.active_orders:
            return
        
        self.logger.info(f"Cancelling all orders for {symbol}")
        
        try:
            # Cancel all open orders for the symbol
            result = self.client.cancel_open_orders(symbol=symbol)
            self.logger.info(f"Cancelled orders for {symbol}: {result}")
            
            # Clear active orders list
            self.active_orders[symbol] = []
            
            # Update Redis
            self.redis.set(
                f'grid_orders:{symbol}',
                json.dumps({
                    'timestamp': datetime.now().isoformat(),
                    'orders': []
                })
            )
        
        except BinanceAPIException as e:
            self.logger.error(f"Error cancelling orders for {symbol}: {str(e)}")
    
    async def _cancel_all_orders(self):
        """Cancel all active orders across all symbols"""
        self.logger.info("Cancelling all active orders")
        
        for symbol in self.active_orders:
            await self._cancel_orders_for_symbol(symbol)
    
    async def _update_performance_metrics(self):
        """Update and publish performance metrics"""
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'start_time': self.start_time.isoformat(),
            'running_time_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'win_rate': self.profitable_trades / self.total_trades if self.total_trades > 0 else 0,
            'total_profit': self.total_profit,
            'profit_per_day': self.total_profit / ((datetime.now() - self.start_time).total_seconds() / 86400) if (datetime.now() - self.start_time).total_seconds() > 0 else 0,
            'symbols': list(self.active_grids.keys()),
            'simulation_mode': self.simulation_mode,
            'symbol_profits': {symbol: self.grid_profits.get(symbol, {}).get('total_profit', 0) for symbol in self.active_grids}
        }
        
        # Publish to Redis
        self.redis.set('grid_performance', json.dumps(performance_data))
    
    async def stop(self):
        """Stop the grid trading strategy and clean up"""
        self.logger.info("Stopping Grid Trading Strategy")
        
        # Cancel all orders if not in simulation mode
        if not self.simulation_mode:
            await self._cancel_all_orders()
        
        # Update final performance
        await self._update_performance_metrics()
        
        # Update service status
        self.redis.set(
            'grid_trading_status', 
            json.dumps({
                'status': 'stopped',
                'timestamp': datetime.now().isoformat(),
                'simulation_mode': self.simulation_mode
            })
        )