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


class DCAStrategy:
    """
    Dollar Cost Averaging (DCA) Strategy implementation.
    
    DCA is a strategy where a fixed amount is invested at regular intervals, 
    regardless of the asset's price. This reduces the impact of volatility and
    eliminates the need to time the market.
    
    Features:
    - Regular automatic purchases on fixed schedule
    - Multiple scheduling options (daily, weekly, monthly)
    - Dip detection for opportunistic purchases
    - Market regime adaptation
    - Volatility and sentiment-based adjustments
    - Performance tracking and reporting
    """
    
    def __init__(self):
        """Initialize the DCA strategy service"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing DCA Strategy")
        
        # Load configuration
        try:
            with open('config.json', 'r') as f:
                self.config = json.load(f)
                self.dca_config = self.config.get('dca_strategy', {})
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            self.config = {}
            self.dca_config = {}
        
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
        
        # DCA strategy parameters
        self.enabled = self.dca_config.get('enabled', False)
        self.symbols = self.dca_config.get('symbols', ['BTCUSDC', 'ETHUSDC'])
        self.service_port = self.dca_config.get('service_port', 8015)
        self.update_interval = self.dca_config.get('update_interval', 60)  # seconds
        
        # DCA schedule parameters
        self.schedule_type = self.dca_config.get('schedule_type', 'fixed')  # fixed, weekly, monthly
        self.interval_hours = self.dca_config.get('interval_hours', 24)
        self.weekly_dca_day = self.dca_config.get('weekly_dca_day', 1)  # Monday = 1, Sunday = 7
        self.monthly_dca_day = self.dca_config.get('monthly_dca_day', 1)  # Day of month
        
        # Investment parameters
        self.total_allocation = self.dca_config.get('total_allocation', 0.3)  # Portion of available capital
        self.base_order_size_usdc = self.dca_config.get('base_order_size_usdc', 100)
        self.auto_adjust_order_size = self.dca_config.get('auto_adjust_order_size', True)
        self.min_order_size_usdc = self.dca_config.get('min_order_size_usdc', 20)
        self.max_order_size_usdc = self.dca_config.get('max_order_size_usdc', 500)
        
        # Adjustment parameters
        self.volatility_adjustment_enabled = self.dca_config.get('volatility_adjustment_enabled', True)
        self.sentiment_adjustment_enabled = self.dca_config.get('sentiment_adjustment_enabled', True)
        self.max_sentiment_impact = self.dca_config.get('max_sentiment_impact', 0.3)
        
        # Dip detection for opportunistic purchases
        self.price_dip_detection = self.dca_config.get('price_dip_detection', True)
        self.dip_threshold_pct = self.dca_config.get('dip_threshold_pct', 5.0)
        self.additional_dip_allocation = self.dca_config.get('additional_dip_allocation', 0.5)
        self.max_dip_frequency_days = self.dca_config.get('max_dip_frequency_days', 7)
        
        # Exit parameters (optional for DCA)
        self.take_profit_enabled = self.dca_config.get('take_profit_enabled', False)
        self.take_profit_pct = self.dca_config.get('take_profit_pct', 20.0)
        self.stop_loss_enabled = self.dca_config.get('stop_loss_enabled', False)
        self.stop_loss_pct = self.dca_config.get('stop_loss_pct', 30.0)
        
        # Adaptive scheduling
        self.adaptive_scheduling = self.dca_config.get('adaptive_scheduling', True)
        self.market_regime_scheduling = self.dca_config.get('scheduling_strategies', {}).get('market_regime_based', {})
        
        # Value averaging settings
        self.value_averaging = self.dca_config.get('scheduling_strategies', {}).get('value_averaging', {}).get('enabled', False)
        self.target_value_increase_pct = self.dca_config.get('scheduling_strategies', {}).get('value_averaging', {}).get('target_value_increase_pct', 5.0)
        
        # Weighted scheduling settings
        self.weighted_scheduling = self.dca_config.get('scheduling_strategies', {}).get('weighted_scheduling', {}).get('enabled', False)
        self.weighted_factors = self.dca_config.get('scheduling_strategies', {}).get('weighted_scheduling', {}).get('weight_factors', {})
        
        # Notification settings
        self.notify_on_purchase = self.dca_config.get('notify_on_purchase', True)
        
        # State tracking
        self.last_purchases = {}  # symbol -> last purchase time
        self.dca_positions = {}  # symbol -> position info (avg price, amount, etc.)
        self.next_scheduled_buys = {}  # symbol -> next scheduled buy time
        self.dip_purchases = {}  # symbol -> last dip purchase time
        self.initialized = False
        
        # Periodic rebalancing
        self.periodic_rebalancing = self.dca_config.get('periodic_rebalancing', {}).get('enabled', True)
        self.rebalance_interval_days = self.dca_config.get('periodic_rebalancing', {}).get('rebalance_interval_days', 30)
        self.last_rebalance = datetime.now() - timedelta(days=self.rebalance_interval_days)  # Ensure first check happens soon
        
        # Market condition rules
        self.market_condition_rules = self.dca_config.get('market_condition_rules', {})
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_purchases = 0
        self.total_invested = 0
        self.position_values = {}  # symbol -> current value
        
        # Simulation mode for testing (no real orders)
        self.simulation_mode = self.dca_config.get('simulation_mode', True)
        
        self.logger.info(f"DCA strategy initialized for {len(self.symbols)} symbols")
    
    async def run(self):
        """Run the DCA strategy service"""
        self.logger.info("Starting DCA Strategy Service")
        
        # Check if trading is enabled
        if not self.enabled:
            self.logger.info("DCA strategy is disabled in configuration")
            return
        
        # Check if Binance client is available
        if not self.client:
            self.logger.error("Cannot start service: Binance client not available")
            return
        
        # Publish service status to Redis
        self.redis.set(
            'dca_strategy_status', 
            json.dumps({
                'status': 'starting',
                'timestamp': datetime.now().isoformat(),
                'simulation_mode': self.simulation_mode
            })
        )
        
        # Initialize DCA strategy for each symbol
        await self._initialize_dca_strategy()
        
        # Update status to running
        self.redis.set(
            'dca_strategy_status', 
            json.dumps({
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'symbols': list(self.dca_positions.keys()),
                'simulation_mode': self.simulation_mode
            })
        )
        
        # Main service loop
        try:
            while True:
                start_time = time.time()
                
                # Process each symbol's DCA strategy
                for symbol in self.symbols:
                    try:
                        await self._process_dca(symbol)
                    except Exception as e:
                        self.logger.error(f"Error processing DCA for {symbol}: {str(e)}")
                
                # Check for rebalancing opportunity
                if self.periodic_rebalancing:
                    current_time = datetime.now()
                    days_since_rebalance = (current_time - self.last_rebalance).days
                    
                    if days_since_rebalance >= self.rebalance_interval_days:
                        try:
                            await self._rebalance_portfolio()
                            self.last_rebalance = current_time
                        except Exception as e:
                            self.logger.error(f"Error rebalancing portfolio: {str(e)}")
                
                # Check for dip buying opportunities
                if self.price_dip_detection:
                    try:
                        await self._check_for_dips()
                    except Exception as e:
                        self.logger.error(f"Error checking for dips: {str(e)}")
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Sleep for the remainder of the interval
                elapsed = time.time() - start_time
                sleep_time = max(0.1, self.update_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            self.logger.info("DCA Strategy Service shutting down")
            
            # Update service status in Redis
            self.redis.set(
                'dca_strategy_status', 
                json.dumps({
                    'status': 'stopped',
                    'timestamp': datetime.now().isoformat()
                })
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in DCA Strategy Service: {str(e)}")
            # Update service status in Redis with error
            self.redis.set(
                'dca_strategy_status', 
                json.dumps({
                    'status': 'error',
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'simulation_mode': self.simulation_mode
                })
            )
    
    async def _initialize_dca_strategy(self):
        """Initialize DCA strategies for all symbols"""
        self.logger.info("Initializing DCA strategy for all symbols")
        
        # Initialize positions dictionary
        for symbol in self.symbols:
            try:
                # Get current price
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
                
                # Initialize DCA tracking
                self.dca_positions[symbol] = {
                    'symbol': symbol,
                    'total_invested': 0.0,
                    'total_quantity': 0.0,
                    'average_price': 0.0,
                    'last_purchase_time': None,
                    'last_purchase_price': 0.0,
                    'current_price': current_price,
                    'current_value': 0.0,
                    'profit_loss': 0.0,
                    'profit_loss_pct': 0.0,
                    'purchases': []
                }
                
                # Get symbol-specific settings
                self._get_symbol_specific_settings(symbol)
                
                # Schedule first purchase
                self.next_scheduled_buys[symbol] = self._calculate_next_purchase_time(symbol)
                
            except Exception as e:
                self.logger.error(f"Error initializing DCA for {symbol}: {str(e)}")
        
        # Load position history from Redis if available
        for symbol in self.symbols:
            position_json = self.redis.get(f'dca_position:{symbol}')
            if position_json:
                try:
                    saved_position = json.loads(position_json)
                    
                    # Update current tracking with saved data
                    if symbol in self.dca_positions:
                        current_price = self.dca_positions[symbol]['current_price']
                        
                        self.dca_positions[symbol].update({
                            'total_invested': saved_position.get('total_invested', 0.0),
                            'total_quantity': saved_position.get('total_quantity', 0.0),
                            'average_price': saved_position.get('average_price', 0.0),
                            'purchases': saved_position.get('purchases', [])
                        })
                        
                        # Calculate current value and P/L
                        total_quantity = self.dca_positions[symbol]['total_quantity']
                        average_price = self.dca_positions[symbol]['average_price']
                        
                        current_value = total_quantity * current_price
                        profit_loss = current_value - self.dca_positions[symbol]['total_invested']
                        profit_loss_pct = (profit_loss / self.dca_positions[symbol]['total_invested']) * 100 if self.dca_positions[symbol]['total_invested'] > 0 else 0.0
                        
                        self.dca_positions[symbol].update({
                            'current_value': current_value,
                            'profit_loss': profit_loss,
                            'profit_loss_pct': profit_loss_pct
                        })
                        
                        # Parse last purchase time
                        if saved_position.get('last_purchase_time'):
                            self.dca_positions[symbol]['last_purchase_time'] = datetime.fromisoformat(saved_position['last_purchase_time'])
                            self.last_purchases[symbol] = datetime.fromisoformat(saved_position['last_purchase_time'])
                        
                except Exception as e:
                    self.logger.error(f"Error loading saved position data for {symbol}: {str(e)}")
        
        # Update Redis with current DCA positions
        self._update_redis_positions()
        
        self.initialized = True
        self.logger.info("DCA strategy initialization complete")
    
    def _get_symbol_specific_settings(self, symbol):
        """Get symbol-specific settings from configuration"""
        # Look for symbol-specific settings
        symbol_settings = next((s for s in self.dca_config.get('symbol_settings', []) 
                             if s.get('symbol') == symbol), None)
        
        if symbol_settings:
            # Store symbol-specific settings
            self.symbols_config = {
                symbol: {
                    'allocation': symbol_settings.get('allocation', 1.0 / len(self.symbols)),
                    'interval_hours': symbol_settings.get('interval_hours', self.interval_hours),
                    'base_order_size_usdc': symbol_settings.get('base_order_size_usdc', self.base_order_size_usdc)
                }
            }
        else:
            # Use global settings with equal allocation
            self.symbols_config = {
                symbol: {
                    'allocation': 1.0 / len(self.symbols),  # Equal allocation for all symbols
                    'interval_hours': self.interval_hours,
                    'base_order_size_usdc': self.base_order_size_usdc
                }
            }
    
    def _calculate_next_purchase_time(self, symbol):
        """Calculate the next scheduled purchase time based on the configuration"""
        current_time = datetime.now()
        
        # Get the schedule type and interval for this symbol
        symbol_config = self.symbols_config.get(symbol, {})
        interval_hours = symbol_config.get('interval_hours', self.interval_hours)
        
        # Use standard interval if no adaptive scheduling
        if not self.adaptive_scheduling:
            return current_time + timedelta(hours=interval_hours)
        
        # Apply market regime-based scheduling if enabled
        if self.adaptive_scheduling and self.market_regime_scheduling:
            try:
                # Get current market regime if available
                regime_json = self.redis.get('market_regime_history')
                
                if regime_json:
                    regime_history = json.loads(regime_json)
                    
                    if regime_history and len(regime_history) > 0:
                        # Get the most recent regime
                        current_regime = regime_history[-1]['regime']
                        
                        # Adjust interval based on regime
                        if current_regime == 'bull':
                            interval_hours = self.market_regime_scheduling.get('bull_interval_hours', interval_hours)
                        elif current_regime == 'bear':
                            interval_hours = self.market_regime_scheduling.get('bear_interval_hours', interval_hours)
                        elif current_regime == 'crab':
                            interval_hours = self.market_regime_scheduling.get('crab_interval_hours', interval_hours)
                        elif current_regime == 'volatile':
                            interval_hours = self.market_regime_scheduling.get('volatile_interval_hours', interval_hours)
            except Exception as e:
                self.logger.error(f"Error applying market regime scheduling: {str(e)}")
        
        # Apply weighted scheduling if enabled
        if self.adaptive_scheduling and self.weighted_scheduling:
            try:
                # Get weight factors
                day_weight = self.weighted_factors.get('day_of_week', 0.0)
                volatility_weight = self.weighted_factors.get('market_volatility', 0.0)
                sentiment_weight = self.weighted_factors.get('social_sentiment', 0.0)
                
                # Calculate day of week factor (weekend = longer intervals)
                day_of_week = current_time.weekday()  # 0 = Monday, 6 = Sunday
                day_factor = 1.0 + (0.2 * day_weight * (day_of_week >= 5))  # Increase by 20% on weekends
                
                # Calculate volatility factor if available
                volatility_factor = 1.0
                if volatility_weight > 0:
                    try:
                        # Get recent market volatility data
                        volatility_json = self.redis.get('market_volatility')
                        
                        if volatility_json:
                            volatility_data = json.loads(volatility_json)
                            if symbol in volatility_data:
                                # Higher volatility = shorter intervals
                                symbol_volatility = volatility_data[symbol]
                                if symbol_volatility > 2.0:  # High volatility
                                    volatility_factor = 1.0 - (0.3 * volatility_weight)  # Reduce by up to 30%
                                elif symbol_volatility < 0.5:  # Low volatility
                                    volatility_factor = 1.0 + (0.3 * volatility_weight)  # Increase by up to 30%
                    except Exception as e:
                        self.logger.error(f"Error calculating volatility factor: {str(e)}")
                
                # Calculate sentiment factor if available
                sentiment_factor = 1.0
                if sentiment_weight > 0:
                    try:
                        # Get social sentiment data
                        sentiment_key = f"enhanced_social_metrics:{symbol}"
                        sentiment_json = self.redis.get(sentiment_key)
                        
                        if not sentiment_json:
                            # Try the regular key for backward compatibility
                            sentiment_json = self.redis.hget('enhanced_social_metrics', symbol)
                        
                        if sentiment_json:
                            sentiment_data = json.loads(sentiment_json)
                            enhanced_sentiment = sentiment_data.get('enhanced_sentiment', {})
                            sentiment_value = enhanced_sentiment.get('enhanced_sentiment', 0.5)
                            
                            # Bearish sentiment = shorter intervals to accumulate more
                            if sentiment_value < 0.4:  # Bearish
                                sentiment_factor = 1.0 - (0.25 * sentiment_weight)  # Reduce by up to 25%
                            elif sentiment_value > 0.6:  # Bullish
                                sentiment_factor = 1.0 + (0.25 * sentiment_weight)  # Increase by up to 25%
                    except Exception as e:
                        self.logger.error(f"Error calculating sentiment factor: {str(e)}")
                
                # Calculate combined adjustment factor
                combined_factor = day_factor * volatility_factor * sentiment_factor
                
                # Apply to interval (limit to 50% change in either direction)
                interval_hours = max(interval_hours * 0.5, min(interval_hours * 1.5, interval_hours * combined_factor))
                
            except Exception as e:
                self.logger.error(f"Error applying weighted scheduling: {str(e)}")
        
        # Return the next purchase time
        return current_time + timedelta(hours=interval_hours)
    
    async def _process_dca(self, symbol):
        """Process DCA strategy for a symbol"""
        current_time = datetime.now()
        
        # Skip if symbol not initialized
        if symbol not in self.dca_positions:
            return
        
        # Update current price
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            self.dca_positions[symbol]['current_price'] = current_price
            
            # Update current value and P/L
            total_quantity = self.dca_positions[symbol]['total_quantity']
            current_value = total_quantity * current_price
            self.dca_positions[symbol]['current_value'] = current_value
            
            if self.dca_positions[symbol]['total_invested'] > 0:
                profit_loss = current_value - self.dca_positions[symbol]['total_invested']
                profit_loss_pct = (profit_loss / self.dca_positions[symbol]['total_invested']) * 100
                
                self.dca_positions[symbol]['profit_loss'] = profit_loss
                self.dca_positions[symbol]['profit_loss_pct'] = profit_loss_pct
        except Exception as e:
            self.logger.error(f"Error updating price for {symbol}: {str(e)}")
            return
        
        # Check if it's time for scheduled purchase
        if symbol in self.next_scheduled_buys:
            next_purchase = self.next_scheduled_buys[symbol]
            
            if current_time >= next_purchase:
                self.logger.info(f"Scheduled DCA purchase for {symbol}")
                
                # Check market conditions first
                if await self._check_market_conditions(symbol):
                    # Execute the purchase
                    await self._execute_dca_purchase(symbol, "scheduled")
                    
                    # Set next purchase time
                    self.next_scheduled_buys[symbol] = self._calculate_next_purchase_time(symbol)
                else:
                    self.logger.info(f"Skipping scheduled purchase for {symbol} due to market conditions")
                    # Still update the next scheduled time even if we skip
                    self.next_scheduled_buys[symbol] = self._calculate_next_purchase_time(symbol)
        
        # Update Redis with current position
        self._update_redis_position(symbol)
    
    async def _check_market_conditions(self, symbol):
        """Check market conditions to decide whether to execute a purchase"""
        # Default to proceeding with purchase
        proceed = True
        
        # Skip purchase in extreme volatility if configured
        if self.market_condition_rules.get('skip_in_extreme_volatility', False):
            threshold = self.market_condition_rules.get('extreme_volatility_threshold', 4.0)
            
            try:
                # Get recent market volatility data
                volatility_json = self.redis.get('market_volatility')
                
                if volatility_json:
                    volatility_data = json.loads(volatility_json)
                    if symbol in volatility_data:
                        symbol_volatility = volatility_data[symbol]
                        
                        if symbol_volatility > threshold:
                            self.logger.info(f"Skipping purchase for {symbol} due to extreme volatility: {symbol_volatility:.2f} > {threshold:.2f}")
                            proceed = False
            except Exception as e:
                self.logger.error(f"Error checking volatility conditions: {str(e)}")
        
        # Skip purchase in extreme greed if configured
        if proceed and self.market_condition_rules.get('reduce_in_extreme_greed', False):
            threshold = self.market_condition_rules.get('extreme_greed_threshold', 80)
            
            try:
                # Check sentiment data (could be fear/greed or custom metric)
                sentiment_json = self.redis.get('market_sentiment')
                
                if sentiment_json:
                    sentiment_data = json.loads(sentiment_json)
                    if 'value' in sentiment_data:
                        sentiment_value = sentiment_data['value']
                        
                        if sentiment_value > threshold:
                            self.logger.info(f"Skipping purchase for {symbol} due to extreme greed sentiment: {sentiment_value} > {threshold}")
                            proceed = False
            except Exception as e:
                self.logger.error(f"Error checking sentiment conditions: {str(e)}")
        
        return proceed
    
    async def _execute_dca_purchase(self, symbol, purchase_type="scheduled"):
        """Execute a DCA purchase for a symbol"""
        # Get current price
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        
        # Get symbol-specific settings
        symbol_config = self.symbols_config.get(symbol, {})
        base_order_size = symbol_config.get('base_order_size_usdc', self.base_order_size_usdc)
        
        # Calculate order size with adjustments
        order_size = self._calculate_adjusted_order_size(symbol, base_order_size, purchase_type)
        
        # Calculate quantity based on current price
        quantity = order_size / current_price
        
        # Get symbol info for precision
        symbol_info = None
        exchange_info = self.client.get_exchange_info()
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol:
                symbol_info = s
                break
        
        if not symbol_info:
            self.logger.error(f"Symbol information not found for {symbol}")
            return
        
        # Calculate quantity with proper precision
        quantity_precision = self._get_quantity_precision(symbol_info)
        rounded_quantity = self._round_to_precision(quantity, quantity_precision)
        
        # Execute order if not in simulation mode
        if not self.simulation_mode:
            try:
                order = self.client.create_order(
                    symbol=symbol,
                    side=Client.SIDE_BUY,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=rounded_quantity
                )
                
                self.logger.info(f"Executed DCA purchase for {symbol}: {rounded_quantity} at ~${current_price:.6f}")
                
                # Get the actual filled price and quantity
                filled_price = float(order['fills'][0]['price']) if 'fills' in order and order['fills'] else current_price
                filled_quantity = sum(float(fill['qty']) for fill in order['fills']) if 'fills' in order else rounded_quantity
                
                # Update position tracking
                self._update_position_after_purchase(symbol, filled_price, filled_quantity, purchase_type)
                
            except BinanceAPIException as e:
                self.logger.error(f"Error executing DCA purchase for {symbol}: {str(e)}")
        else:
            # Simulation mode - use current price
            self.logger.info(f"[SIMULATION] Executed DCA purchase for {symbol}: {rounded_quantity} at ${current_price:.6f}")
            
            # Update position tracking
            self._update_position_after_purchase(symbol, current_price, rounded_quantity, purchase_type)
        
        # Publish notification to Redis
        if self.notify_on_purchase:
            self.redis.publish(
                'dca_purchase_notifications',
                json.dumps({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'price': current_price,
                    'quantity': rounded_quantity,
                    'amount_usdc': order_size,
                    'purchase_type': purchase_type,
                    'simulation': self.simulation_mode
                })
            )
        
        # Store notification in a list for display
        self.redis.lpush(
            'dca_purchase_list',
            json.dumps({
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'price': current_price,
                'quantity': rounded_quantity,
                'amount_usdc': order_size,
                'purchase_type': purchase_type,
                'simulation': self.simulation_mode
            })
        )
        
        # Trim the list to keep most recent 50 purchases
        self.redis.ltrim('dca_purchase_list', 0, 49)
        
        # Update global stats
        self.total_purchases += 1
        self.total_invested += order_size
        
        # Update last purchase time
        self.last_purchases[symbol] = datetime.now()
        
        # If this was a dip purchase, update tracking
        if purchase_type == "dip":
            self.dip_purchases[symbol] = datetime.now()
    
    def _calculate_adjusted_order_size(self, symbol, base_order_size, purchase_type):
        """Calculate adjusted order size based on various factors"""
        adjusted_size = base_order_size
        
        # Start with the base order size
        if purchase_type == "dip" and self.price_dip_detection:
            # Increase order size for dip purchases
            dip_adjustment = 1.0 + self.additional_dip_allocation
            adjusted_size = base_order_size * dip_adjustment
            self.logger.info(f"Dip purchase adjustment for {symbol}: {dip_adjustment:.2f}x")
        
        # Apply volatility adjustment if enabled
        if self.volatility_adjustment_enabled:
            try:
                # Get recent market volatility data
                volatility_json = self.redis.get('market_volatility')
                
                if volatility_json:
                    volatility_data = json.loads(volatility_json)
                    if symbol in volatility_data:
                        symbol_volatility = volatility_data[symbol]
                        
                        # Adjust based on volatility
                        if symbol_volatility > 2.0:  # High volatility
                            volatility_factor = 0.8  # Reduce order size
                        elif symbol_volatility < 0.5:  # Low volatility
                            volatility_factor = 1.2  # Increase order size
                        else:
                            volatility_factor = 1.0  # No change
                        
                        adjusted_size *= volatility_factor
                        self.logger.info(f"Volatility adjustment for {symbol}: {volatility_factor:.2f}x")
            except Exception as e:
                self.logger.error(f"Error applying volatility adjustment: {str(e)}")
        
        # Apply sentiment adjustment if enabled
        if self.sentiment_adjustment_enabled:
            try:
                # Get social sentiment data
                sentiment_key = f"enhanced_social_metrics:{symbol}"
                sentiment_json = self.redis.get(sentiment_key)
                
                if not sentiment_json:
                    # Try the regular key for backward compatibility
                    sentiment_json = self.redis.hget('enhanced_social_metrics', symbol)
                
                if sentiment_json:
                    sentiment_data = json.loads(sentiment_json)
                    enhanced_sentiment = sentiment_data.get('enhanced_sentiment', {})
                    sentiment_value = enhanced_sentiment.get('enhanced_sentiment', 0.5)
                    
                    # Calculate sentiment adjustment factor (0.7 to 1.3)
                    # Bearish sentiment = increase order size
                    # Bullish sentiment = decrease order size
                    max_impact = self.max_sentiment_impact
                    sentiment_factor = 1.0 + (max_impact * (0.5 - sentiment_value) * 2)
                    
                    adjusted_size *= sentiment_factor
                    self.logger.info(f"Sentiment adjustment for {symbol}: {sentiment_factor:.2f}x")
            except Exception as e:
                self.logger.error(f"Error applying sentiment adjustment: {str(e)}")
        
        # Apply bear market adjustment if configured
        if self.market_condition_rules.get('increase_in_bear_market', False):
            try:
                # Check for bear market condition
                bear_threshold = self.market_condition_rules.get('bear_market_threshold', -20.0)
                
                # Get market regime data
                regime_json = self.redis.get('market_regime_history')
                
                if regime_json:
                    regime_history = json.loads(regime_json)
                    
                    if regime_history and len(regime_history) > 0:
                        # Get the most recent regime
                        current_regime = regime_history[-1]['regime']
                        
                        # Apply adjustment for bear market
                        if current_regime == 'bear':
                            bear_adjustment = self.market_condition_rules.get('bear_market_adjustment', 1.5)
                            adjusted_size *= bear_adjustment
                            self.logger.info(f"Bear market adjustment for {symbol}: {bear_adjustment:.2f}x")
            except Exception as e:
                self.logger.error(f"Error applying bear market adjustment: {str(e)}")
        
        # Ensure the order size is within configured limits
        adjusted_size = max(self.min_order_size_usdc, min(self.max_order_size_usdc, adjusted_size))
        
        return adjusted_size
    
    def _update_position_after_purchase(self, symbol, price, quantity, purchase_type):
        """Update position tracking after a purchase"""
        # Get current position data
        position = self.dca_positions.get(symbol, {
            'symbol': symbol,
            'total_invested': 0.0,
            'total_quantity': 0.0,
            'average_price': 0.0,
            'last_purchase_time': None,
            'last_purchase_price': 0.0,
            'current_price': price,
            'current_value': 0.0,
            'profit_loss': 0.0,
            'profit_loss_pct': 0.0,
            'purchases': []
        })
        
        # Calculate amount in USDC
        amount_usdc = price * quantity
        
        # Update position data
        current_value = (position['total_quantity'] + quantity) * price
        
        # Calculate new average price
        new_total_invested = position['total_invested'] + amount_usdc
        new_total_quantity = position['total_quantity'] + quantity
        new_avg_price = new_total_invested / new_total_quantity if new_total_quantity > 0 else 0
        
        # Calculate profit/loss
        profit_loss = current_value - new_total_invested
        profit_loss_pct = (profit_loss / new_total_invested) * 100 if new_total_invested > 0 else 0.0
        
        # Add purchase record
        purchase = {
            'timestamp': datetime.now().isoformat(),
            'price': price,
            'quantity': quantity,
            'amount_usdc': amount_usdc,
            'type': purchase_type
        }
        
        # Update position
        position.update({
            'total_invested': new_total_invested,
            'total_quantity': new_total_quantity,
            'average_price': new_avg_price,
            'last_purchase_time': datetime.now().isoformat(),
            'last_purchase_price': price,
            'current_value': current_value,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct
        })
        
        # Append purchase to history
        position['purchases'].append(purchase)
        
        # Update position in tracking dictionary
        self.dca_positions[symbol] = position
        
        # Update Redis
        self._update_redis_position(symbol)
    
    def _update_redis_position(self, symbol):
        """Update Redis with current position data for a symbol"""
        if symbol in self.dca_positions:
            self.redis.set(
                f'dca_position:{symbol}',
                json.dumps(self.dca_positions[symbol])
            )
    
    def _update_redis_positions(self):
        """Update Redis with all positions"""
        for symbol in self.dca_positions:
            self._update_redis_position(symbol)
    
    async def _check_for_dips(self):
        """Check for price dips that might trigger opportunistic purchases"""
        if not self.price_dip_detection:
            return
        
        current_time = datetime.now()
        
        for symbol in self.symbols:
            # Check if we've recently made a dip purchase to avoid too frequent purchases
            if symbol in self.dip_purchases:
                last_dip_purchase = self.dip_purchases[symbol]
                days_since_dip = (current_time - last_dip_purchase).days
                
                if days_since_dip < self.max_dip_frequency_days:
                    continue  # Skip if we've recently made a dip purchase
            
            try:
                # Get recent price data
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=Client.KLINE_INTERVAL_1DAY,
                    start_str="7 days ago UTC"
                )
                
                if not klines or len(klines) < 2:
                    continue
                
                # Get current price
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
                
                # Get recent high
                recent_high = max([float(kline[2]) for kline in klines])  # High price
                
                # Calculate price drop percentage
                price_drop_pct = ((recent_high - current_price) / recent_high) * 100
                
                # Check if price drop exceeds threshold
                if price_drop_pct >= self.dip_threshold_pct:
                    self.logger.info(f"Detected price dip for {symbol}: {price_drop_pct:.2f}% from recent high")
                    
                    # Execute dip purchase
                    await self._execute_dca_purchase(symbol, "dip")
                    
            except Exception as e:
                self.logger.error(f"Error checking for dips for {symbol}: {str(e)}")
    
    async def _rebalance_portfolio(self):
        """Rebalance portfolio allocations according to target allocations"""
        self.logger.info("Starting portfolio rebalancing")
        
        try:
            total_value = 0.0
            current_allocations = {}
            
            # Get user USDC balance
            account = self.client.get_account()
            usdc_balance = next(
                (float(asset['free']) for asset in account['balances'] 
                 if asset['asset'] == 'USDC'),
                0.0
            )
            
            # Calculate total portfolio value including USDC
            for symbol in self.dca_positions:
                total_value += self.dca_positions[symbol]['current_value']
            
            total_value += usdc_balance
            
            # Calculate current allocations
            for symbol in self.dca_positions:
                current_value = self.dca_positions[symbol]['current_value']
                current_allocations[symbol] = current_value / total_value if total_value > 0 else 0
            
            # Get target allocations
            target_allocations = {}
            for symbol in self.dca_positions:
                symbol_config = self.symbols_config.get(symbol, {})
                target_allocations[symbol] = symbol_config.get('allocation', 1.0 / len(self.symbols))
            
            # Check deviations and rebalance if necessary
            max_deviation_pct = self.dca_config.get('periodic_rebalancing', {}).get('max_deviation_pct', 5.0)
            
            rebalance_actions = []
            
            for symbol in self.dca_positions:
                current_alloc = current_allocations.get(symbol, 0)
                target_alloc = target_allocations.get(symbol, 0)
                
                # Calculate deviation
                deviation_pct = abs((current_alloc - target_alloc) / target_alloc) * 100 if target_alloc > 0 else 0
                
                # Check if rebalancing is needed
                if deviation_pct > max_deviation_pct:
                    self.logger.info(f"Rebalancing needed for {symbol}: Current {current_alloc:.2%}, Target {target_alloc:.2%}, Deviation {deviation_pct:.2f}%")
                    
                    # Calculate target value
                    target_value = total_value * target_alloc
                    current_value = self.dca_positions[symbol]['current_value']
                    
                    if current_value < target_value:
                        # Need to buy more
                        buy_value = target_value - current_value
                        self.logger.info(f"Need to buy more {symbol}: ${buy_value:.2f}")
                        
                        # Add to rebalance actions
                        rebalance_actions.append({
                            'symbol': symbol,
                            'action': 'buy',
                            'value': buy_value
                        })
                    else:
                        # Could sell some, but DCA typically only buys
                        self.logger.info(f"{symbol} is overweight but DCA strategy typically doesn't sell")
            
            # Execute buy actions if not in simulation mode
            if rebalance_actions and not self.simulation_mode:
                for action in rebalance_actions:
                    if action['action'] == 'buy':
                        symbol = action['symbol']
                        buy_value = action['value']
                        
                        # Get current price
                        ticker = self.client.get_symbol_ticker(symbol=symbol)
                        current_price = float(ticker['price'])
                        
                        # Calculate quantity
                        quantity = buy_value / current_price
                        
                        # Get symbol info for precision
                        symbol_info = None
                        exchange_info = self.client.get_exchange_info()
                        for s in exchange_info['symbols']:
                            if s['symbol'] == symbol:
                                symbol_info = s
                                break
                        
                        if not symbol_info:
                            self.logger.error(f"Symbol information not found for {symbol}")
                            continue
                        
                        # Calculate quantity with proper precision
                        quantity_precision = self._get_quantity_precision(symbol_info)
                        rounded_quantity = self._round_to_precision(quantity, quantity_precision)
                        
                        # Execute order
                        try:
                            order = self.client.create_order(
                                symbol=symbol,
                                side=Client.SIDE_BUY,
                                type=Client.ORDER_TYPE_MARKET,
                                quantity=rounded_quantity
                            )
                            
                            self.logger.info(f"Executed rebalance purchase for {symbol}: {rounded_quantity} at ~${current_price:.6f}")
                            
                            # Get the actual filled price and quantity
                            filled_price = float(order['fills'][0]['price']) if 'fills' in order and order['fills'] else current_price
                            filled_quantity = sum(float(fill['qty']) for fill in order['fills']) if 'fills' in order else rounded_quantity
                            
                            # Update position tracking
                            self._update_position_after_purchase(symbol, filled_price, filled_quantity, "rebalance")
                            
                        except BinanceAPIException as e:
                            self.logger.error(f"Error executing rebalance purchase for {symbol}: {str(e)}")
            
            # In simulation mode, log the rebalance actions
            elif rebalance_actions and self.simulation_mode:
                for action in rebalance_actions:
                    if action['action'] == 'buy':
                        symbol = action['symbol']
                        buy_value = action['value']
                        
                        # Get current price
                        ticker = self.client.get_symbol_ticker(symbol=symbol)
                        current_price = float(ticker['price'])
                        
                        # Calculate quantity
                        quantity = buy_value / current_price
                        
                        # Get symbol info for precision
                        symbol_info = None
                        exchange_info = self.client.get_exchange_info()
                        for s in exchange_info['symbols']:
                            if s['symbol'] == symbol:
                                symbol_info = s
                                break
                        
                        if not symbol_info:
                            self.logger.error(f"Symbol information not found for {symbol}")
                            continue
                        
                        # Calculate quantity with proper precision
                        quantity_precision = self._get_quantity_precision(symbol_info)
                        rounded_quantity = self._round_to_precision(quantity, quantity_precision)
                        
                        self.logger.info(f"[SIMULATION] Rebalance purchase for {symbol}: {rounded_quantity} at ${current_price:.6f}")
                        
                        # Update position tracking
                        self._update_position_after_purchase(symbol, current_price, rounded_quantity, "rebalance")
            
            self.logger.info("Portfolio rebalancing complete")
            
        except Exception as e:
            self.logger.error(f"Error during portfolio rebalancing: {str(e)}")
    
    async def _update_performance_metrics(self):
        """Update and publish performance metrics"""
        try:
            # Update position values first
            for symbol in self.dca_positions:
                try:
                    # Get current price
                    ticker = self.client.get_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    
                    # Update position data
                    self.dca_positions[symbol]['current_price'] = current_price
                    self.dca_positions[symbol]['current_value'] = self.dca_positions[symbol]['total_quantity'] * current_price
                    
                    if self.dca_positions[symbol]['total_invested'] > 0:
                        profit_loss = self.dca_positions[symbol]['current_value'] - self.dca_positions[symbol]['total_invested']
                        profit_loss_pct = (profit_loss / self.dca_positions[symbol]['total_invested']) * 100
                        
                        self.dca_positions[symbol]['profit_loss'] = profit_loss
                        self.dca_positions[symbol]['profit_loss_pct'] = profit_loss_pct
                    
                except Exception as e:
                    self.logger.error(f"Error updating price for {symbol}: {str(e)}")
            
            # Calculate total portfolio value and performance
            total_invested = 0.0
            total_value = 0.0
            
            for symbol in self.dca_positions:
                total_invested += self.dca_positions[symbol]['total_invested']
                total_value += self.dca_positions[symbol]['current_value']
            
            total_profit = total_value - total_invested
            total_profit_pct = (total_profit / total_invested) * 100 if total_invested > 0 else 0.0
            
            # Calculate average cost values for each position
            avg_costs = {}
            for symbol in self.dca_positions:
                if self.dca_positions[symbol]['total_quantity'] > 0:
                    avg_costs[symbol] = {
                        'average_price': self.dca_positions[symbol]['average_price'],
                        'total_quantity': self.dca_positions[symbol]['total_quantity'],
                        'current_price': self.dca_positions[symbol]['current_price'],
                        'profit_loss_pct': self.dca_positions[symbol]['profit_loss_pct']
                    }
            
            # Calculate portfolio statistics
            performance_data = {
                'timestamp': datetime.now().isoformat(),
                'start_time': self.start_time.isoformat(),
                'running_time_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                'total_purchases': self.total_purchases,
                'total_invested': total_invested,
                'total_value': total_value,
                'total_profit': total_profit,
                'total_profit_pct': total_profit_pct,
                'symbols': list(self.dca_positions.keys()),
                'avg_costs': avg_costs,
                'simulation_mode': self.simulation_mode,
                'next_scheduled_buys': {symbol: dt.isoformat() for symbol, dt in self.next_scheduled_buys.items()}
            }
            
            # Publish to Redis
            self.redis.set('dca_performance', json.dumps(performance_data))
            
            # Update Redis positions
            self._update_redis_positions()
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {str(e)}")
    
    def _get_quantity_precision(self, symbol_info):
        """Get quantity precision from symbol info"""
        for filter in symbol_info['filters']:
            if filter['filterType'] == 'LOT_SIZE':
                step_size = float(filter['stepSize'])
                return len(str(step_size).rstrip('0').split('.')[1]) if '.' in str(step_size) else 0
        return 6  # Default precision
    
    def _round_to_precision(self, value, precision):
        """Round value to specified decimal precision"""
        factor = 10 ** precision
        return math.floor(value * factor) / factor
    
    async def stop(self):
        """Stop the DCA strategy and clean up"""
        self.logger.info("Stopping DCA Strategy")
        
        # Update final performance
        await self._update_performance_metrics()
        
        # Update service status
        self.redis.set(
            'dca_strategy_status', 
            json.dumps({
                'status': 'stopped',
                'timestamp': datetime.now().isoformat(),
                'simulation_mode': self.simulation_mode
            })
        )