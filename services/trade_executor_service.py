import os
import json
import redis
import asyncio
from datetime import datetime
from binance.client import Client
from binance.enums import *
import logging as logger

# Configure logging
logger.basicConfig(
    level=logger.INFO,
    format='%(asctime)s - %(levelname)s - [TradeExecutor] %(message)s',
    handlers=[
        logger.FileHandler('logs/trade_executor.log'),
        logger.StreamHandler()
    ]
)

class TradeExecutorService:
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

        self.running = True
        self.active_trades = {}
        self.symbol_info = {}
        self.available_usdc = 0.0
        self.load_trading_rules()

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

            # Skip if confidence is too low
            if confidence < self.config['trading_params']['ai_confidence_threshold']:
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
                
                # Check minimum notional value
                if quantity * current_price < rules['min_notional']:
                    logger.error(f"Order does not meet minimum notional value")
                    return

                # Place market buy order
                order = self.client.create_order(
                    symbol=symbol,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )

                # Get fill price
                fill_price = float(order['fills'][0]['price'])
                filled_quantity = float(order['executedQty'])

                # Calculate stop loss and take profit prices
                stop_loss_price = fill_price * (1 - self.config['trading_params']['stop_loss_pct'] / 100)
                take_profit_price = fill_price * (1 + self.config['trading_params']['take_profit_pct'] / 100)

                # Place stop loss order
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
            logger.error(f"Error executing trade: {str(e)}")

    async def close_position(self, symbol: str, quantity: float, reason: str):
        """Close a specific trading position"""
        try:
            # Place market sell order
            order = self.client.create_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            
            logger.info(f"Closed position for {symbol} ({reason})")
            
            # Remove from active trades
            self.active_trades.pop(symbol, None)
            
            # Update balance
            self.update_usdc_balance()
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {str(e)}")

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
                    
                    # Check stop loss
                    if current_price <= trade['stop_loss_price']:
                        logger.info(f"Stop loss triggered for {symbol} at ${current_price:.8f} ({pnl_pct:.2f}%)")
                        await self.close_position(symbol, quantity, "Stop Loss")
                    
                    # Check take profit
                    elif current_price >= trade['take_profit_price']:
                        logger.info(f"Take profit triggered for {symbol} at ${current_price:.8f} ({pnl_pct:.2f}%)")
                        await self.close_position(symbol, quantity, "Take Profit")
                    
                except Exception as e:
                    logger.error(f"Error monitoring trade for {symbol}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error in trade monitoring: {str(e)}")

    async def process_trading_signals(self):
        """Process trading signals from Redis"""
        try:
            pubsub = self.redis.pubsub()
            await pubsub.subscribe('trading_signals')

            while self.running:
                message = await pubsub.get_message()
                if message and message['type'] == 'message':
                    signal = json.loads(message['data'])
                    await self.execute_trade(signal)

                # Monitor active trades
                await self.monitor_active_trades()
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error processing trading signals: {str(e)}")
        finally:
            await pubsub.unsubscribe()

    async def start(self):
        """Start the trade executor service"""
        try:
            logger.info("Starting Trade Executor Service...")
            self.update_usdc_balance()
            await self.process_trading_signals()
        except Exception as e:
            logger.error(f"Error in Trade Executor Service: {str(e)}")
        finally:
            self.stop()

    def stop(self):
        """Stop the trade executor service"""
        logger.info("Stopping Trade Executor Service...")
        self.running = False
        self.redis.close()

if __name__ == "__main__":
    service = TradeExecutorService()
    try:
        asyncio.run(service.start())
    except KeyboardInterrupt:
        service.stop()
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        service.stop()
