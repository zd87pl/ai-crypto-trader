import threading
import datetime
import json
import os
import time
import logging as logger
import asyncio
from queue import Queue
from binance.client import Client
from binance.streams import ThreadedWebsocketManager
from binance.enums import *
import pandas as pd
from typing import Dict
from binance_ml_strategy import CryptoScanner, TradingSignal, PositionSizer
from ai_trader import AITrader

# Configure logging to output to the console
logger.basicConfig(level=logger.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_event_loop():
    """Set up an event loop for the current thread if one doesn't exist"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

class MarketMonitor(threading.Thread):
    def __init__(self, client, config, opportunity_queue, market_data, twm):
        threading.Thread.__init__(self, name="MarketMonitor")
        self.client = client
        self.config = config
        self.opportunity_queue = opportunity_queue
        self.running = True
        self.daemon = True
        self.last_scan = {}
        self.market_data = market_data  # Shared market data
        self.twm = twm  # Use shared ThreadedWebsocketManager
        
    def process_message(self, msg):
        """Process incoming WebSocket message"""
        try:
            logger.debug(f"Received market data: {msg}")  # Debug log for all messages
            
            if isinstance(msg, dict) and msg.get('e') == '24hrMiniTicker':
                data = msg
            elif isinstance(msg, dict) and isinstance(msg.get('data'), dict):
                data = msg['data']
            else:
                return
                
            if data.get('e') == '24hrMiniTicker':
                symbol = data['s']
                if symbol.endswith('USDC'):
                    price = float(data['c'])  # Close price
                    volume = float(data['v']) * price  # Volume in USDC
                    price_change = ((price - float(data['o'])) / float(data['o'])) * 100  # Price change percentage
                    
                    self.market_data[symbol] = {
                        'price': price,
                        'volume': volume,
                        'price_change': price_change,
                        'timestamp': datetime.datetime.now()
                    }
                    
                    logger.info(f"Market update - {symbol}: ${price:.8f} (24h volume: ${volume:.2f}, change: {price_change:.2f}%)")
                    
                    # Check if we should create an opportunity
                    current_time = datetime.datetime.now()
                    if (symbol not in self.last_scan or 
                        (current_time - self.last_scan[symbol]).seconds >= 60):  # 1 min cooldown
                        
                        if volume >= self.config['trading_params']['min_volume_usdc']:
                            opportunity = {
                                'symbol': symbol,
                                'price': price,
                                'volume': volume,
                                'price_change': price_change,
                                'timestamp': current_time
                            }
                            self.opportunity_queue.put(opportunity)
                            self.last_scan[symbol] = current_time
                            logger.info(f"Found opportunity: {symbol} at ${price:.8f} (24h volume: ${volume:.2f})")
                    
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {str(e)}")
            logger.error(f"Message content: {msg}")
    
    def run(self):
        """Start WebSocket connection and monitor market"""
        try:
            setup_event_loop()  # Ensure event loop is set up
            
            logger.info("Starting market monitor...")
            
            # Subscribe to all mini ticker streams
            self.twm.start_miniticker_socket(
                callback=self.process_message
            )
            logger.info("Subscribed to mini ticker stream")
            
            # Keep thread alive
            while self.running:
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in market monitor: {str(e)}")
    
    def stop(self):
        """Stop the market monitor and cleanup resources"""
        logger.info("Stopping market monitor...")
        self.running = False
        # Close any open connections
        if hasattr(self, 'client'):
            try:
                self.client.close_connection()
            except:
                pass

class TradeExecutor(threading.Thread):
    def __init__(self, client, config, opportunity_queue, market_data, twm):
        threading.Thread.__init__(self, name="TradeExecutor")
        self.client = client
        self.config = config
        self.opportunity_queue = opportunity_queue
        self.running = True
        self.daemon = True
        self.active_trades = {}
        self.last_trade_time = {}
        self.symbol_info = {}  # Cache for symbol information
        self.market_data = market_data  # Shared market data
        self.twm = twm  # Use shared ThreadedWebsocketManager
        self.balances = {}
        self.available_usdc = 0.0  # Track available USDC balance
        self.ai_trader = AITrader(config)  # Initialize AI trader
        self.last_market_analysis = None
        self.last_market_analysis_time = None
        self.load_trading_rules()

    def load_trading_rules(self):
        """Load and cache trading rules for all USDC pairs"""
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

    def start_user_socket(self):
        """Start user data stream for account updates"""
        try:
            # Start user socket to track account updates
            self.twm.start_user_socket(
                callback=self.process_user_socket_message
            )
            logger.info("Started user data stream")
            
            # Initial balance update
            self.update_usdc_balance()
            
        except Exception as e:
            logger.error(f"Error starting user socket: {str(e)}")
            raise

    def process_user_socket_message(self, msg):
        """Process user data stream messages"""
        try:
            if msg.get('e') == 'outboundAccountPosition':
                # Update balances
                for balance in msg['B']:
                    asset = balance['a']
                    free = float(balance['f'])
                    locked = float(balance['l'])
                    self.balances[asset] = {
                        'free': free,
                        'locked': locked,
                        'total': free + locked
                    }
                    if asset == 'USDC':
                        self.available_usdc = free
                        logger.info(f"USDC balance updated: ${free:.2f} available")
                        
            elif msg.get('e') == 'executionReport':
                # Handle order execution updates
                symbol = msg['s']
                order_id = msg['i']
                status = msg['X']
                
                if symbol in self.active_trades:
                    if status == 'FILLED':
                        logger.info(f"Order {order_id} for {symbol} has been filled")
                        # Update trade status
                        self.active_trades[symbol]['status'] = 'FILLED'
                    elif status == 'CANCELED':
                        logger.info(f"Order {order_id} for {symbol} has been canceled")
                        # Remove from active trades
                        self.active_trades.pop(symbol, None)
                
        except Exception as e:
            logger.error(f"Error processing user socket message: {str(e)}")
            logger.error(f"Message content: {msg}")

    def update_usdc_balance(self):
        """Update available USDC balance"""
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == 'USDC':
                    free = float(balance['free'])
                    locked = float(balance['locked'])
                    self.balances['USDC'] = {
                        'free': free,
                        'locked': locked,
                        'total': free + locked
                    }
                    self.available_usdc = free
                    logger.info(f"USDC balance updated: ${free:.2f} available")
                    break
        except Exception as e:
            logger.error(f"Error updating USDC balance: {str(e)}")

    def liquidate_all_positions(self):
        """Close all active trading positions"""
        try:
            if not self.active_trades:
                logger.info("No active positions to liquidate")
                return

            logger.info(f"Liquidating {len(self.active_trades)} active positions...")
            
            for symbol, trade in list(self.active_trades.items()):
                try:
                    # Get current market price
                    ticker = self.client.get_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    
                    # Calculate quantity to sell (all of it)
                    quantity = trade['quantity']
                    
                    # Place market sell order
                    order = self.client.create_order(
                        symbol=symbol,
                        side=SIDE_SELL,
                        type=ORDER_TYPE_MARKET,
                        quantity=quantity
                    )
                    
                    logger.info(f"Liquidated position for {symbol} at market price ${current_price:.8f}")
                    
                    # Remove from active trades
                    self.active_trades.pop(symbol, None)
                    
                except Exception as e:
                    logger.error(f"Error liquidating position for {symbol}: {str(e)}")
            
            # Final balance update
            self.update_usdc_balance()
            logger.info("All positions have been liquidated")
            
        except Exception as e:
            logger.error(f"Error during position liquidation: {str(e)}")

    def monitor_active_trades(self):
        """Monitor and manage active trades"""
        try:
            for symbol, trade in list(self.active_trades.items()):
                try:
                    # Get current market price
                    ticker = self.client.get_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    
                    # Calculate current P&L
                    entry_price = trade['entry_price']
                    quantity = trade['quantity']
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    
                    # Check stop loss
                    if pnl_pct <= -trade['stop_loss_pct']:
                        logger.info(f"Stop loss triggered for {symbol} at ${current_price:.8f} ({pnl_pct:.2f}%)")
                        self.close_position(symbol, quantity, "Stop loss")
                    
                    # Check take profit
                    elif pnl_pct >= trade['take_profit_pct']:
                        logger.info(f"Take profit triggered for {symbol} at ${current_price:.8f} ({pnl_pct:.2f}%)")
                        self.close_position(symbol, quantity, "Take profit")
                    
                except Exception as e:
                    logger.error(f"Error monitoring trade for {symbol}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error in trade monitoring: {str(e)}")

    def close_position(self, symbol: str, quantity: float, reason: str):
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

    def round_step_size(self, quantity: float, step_size: float) -> float:
        """Round quantity to valid step size"""
        precision = len(str(step_size).split('.')[-1])
        return round(quantity - (quantity % step_size), precision)

    async def execute_trade(self, trade_setup: Dict):
        """Execute a trade based on the provided setup"""
        try:
            symbol = trade_setup['symbol']
            price = trade_setup['price']
            position_size = trade_setup['position_size']
            stop_loss_pct = trade_setup['stop_loss_pct']
            take_profit_pct = trade_setup['take_profit_pct']
            
            # Calculate quantity based on position size and current price
            quantity = position_size / price
            
            # Get symbol trading rules
            rules = self.symbol_info.get(symbol)
            if not rules:
                logger.error(f"No trading rules found for {symbol}")
                return
            
            # Round quantity to valid step size
            step_size = rules['step_size']
            quantity = self.round_step_size(quantity, step_size)
            
            # Check minimum notional value
            if quantity * price < rules['min_notional']:
                logger.error(f"Order for {symbol} does not meet minimum notional value")
                return
            
            # Place market buy order
            order = self.client.create_order(
                symbol=symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            
            # Get actual fill price from order
            fill_price = float(order['fills'][0]['price'])
            filled_quantity = float(order['executedQty'])
            
            # Calculate stop loss and take profit prices
            stop_loss_price = fill_price * (1 - stop_loss_pct / 100)
            take_profit_price = fill_price * (1 + take_profit_pct / 100)
            
            # Place stop loss order
            stop_loss_order = self.client.create_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_STOP_LOSS_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=filled_quantity,
                stopPrice=stop_loss_price,
                price=stop_loss_price * 0.99  # Slightly below stop price to ensure execution
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
            
            # Record the trade
            self.active_trades[symbol] = {
                'entry_price': fill_price,
                'quantity': filled_quantity,
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
                'stop_loss_order': stop_loss_order['orderId'],
                'take_profit_order': take_profit_order['orderId'],
                'entry_time': datetime.datetime.now(),
                'status': 'ACTIVE'
            }
            
            logger.info(f"Opened position for {symbol}:")
            logger.info(f"Entry Price: ${fill_price:.8f}")
            logger.info(f"Quantity: {filled_quantity}")
            logger.info(f"Stop Loss: ${stop_loss_price:.8f} ({stop_loss_pct}%)")
            logger.info(f"Take Profit: ${take_profit_price:.8f} ({take_profit_pct}%)")
            
            # Update available balance
            self.update_usdc_balance()
            
        except Exception as e:
            logger.error(f"Error executing trade for {trade_setup['symbol']}: {str(e)}")

    async def get_ai_analysis(self, market_data: Dict) -> Dict:
        """Get AI analysis for a trading opportunity"""
        try:
            # Get AI analysis
            analysis = await self.ai_trader.analyze_trade_opportunity(market_data)
            
            # If analysis is successful and confidence is high enough
            if self.ai_trader.should_take_trade(analysis):
                # Get risk analysis
                risk_setup = {
                    'symbol': market_data['symbol'],
                    'available_capital': self.available_usdc,
                    'volatility': market_data['volatility'],
                    'current_price': market_data['current_price'],
                    'trend_strength': market_data['trend_strength']
                }
                risk_analysis = await self.ai_trader.analyze_risk_setup(risk_setup)
                
                # Combine analyses
                return {
                    'trade_analysis': analysis,
                    'risk_analysis': risk_analysis
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting AI analysis: {str(e)}")
            return None

    async def get_market_analysis(self, volatile_pairs: list) -> Dict:
        """Get AI analysis of overall market conditions"""
        try:
            # Check if we need new market analysis
            current_time = datetime.datetime.now()
            if (self.last_market_analysis_time and 
                (current_time - self.last_market_analysis_time).seconds < self.config['trading_params']['ai_analysis_interval']):
                return self.last_market_analysis
            
            # Get market analysis
            analysis = await self.ai_trader.analyze_market_conditions(volatile_pairs)
            
            # Update last analysis
            self.last_market_analysis = analysis
            self.last_market_analysis_time = current_time
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting market analysis: {str(e)}")
            return None

    def should_execute_trade(self, technical_signal: TradingSignal, ai_analysis: Dict) -> Dict:
        """Determine if we should execute a trade based on both technical and AI analysis"""
        try:
            # Must have both analyses
            if not technical_signal or not ai_analysis:
                return {'execute': False}
            
            # Check AI confidence
            if ai_analysis['trade_analysis']['confidence'] < self.config['trading_params']['ai_confidence_threshold']:
                return {'execute': False}
            
            # Check technical signal strength
            if technical_signal.strength < 70:  # Minimum technical strength required
                return {'execute': False}
            
            # Check market sentiment if available
            if self.last_market_analysis:
                if self.last_market_analysis['market_sentiment'].lower() in ['bearish', 'very bearish']:
                    return {'execute': False}
            
            # Get the AI decision
            decision = ai_analysis['trade_analysis']['decision']
            
            # Both must agree on the direction (BUY or SELL)
            if technical_signal.signal != decision:
                return {'execute': False}
            
            # Return decision along with execute flag
            return {
                'execute': True,
                'decision': decision
            }
            
        except Exception as e:
            logger.error(f"Error in trade decision: {str(e)}")
            return {'execute': False}

    async def execute_trade_with_ai(self, opportunity: Dict, technical_signal: TradingSignal):
        """Execute trade with combined AI and technical analysis"""
        try:
            # Get AI analysis
            ai_analysis = await self.get_ai_analysis(opportunity)
            
            # Check if we should execute the trade
            trade_decision = self.should_execute_trade(technical_signal, ai_analysis)
            if not trade_decision['execute']:
                return
            
            # Calculate position size using both analyses
            technical_position = PositionSizer.calculate_position_size(
                self.available_usdc,
                opportunity['volatility'],
                opportunity['avg_volume']
            )
            
            # Combine position sizing from both sources
            final_position = self.ai_trader.adjust_position_size(
                ai_analysis['risk_analysis'],
                technical_position
            )
            
            # Execute the trade based on the decision
            if trade_decision['decision'] == 'SELL':
                # For SELL decisions, check if we have an existing position
                if opportunity['symbol'] in self.active_trades:
                    position = self.active_trades[opportunity['symbol']]
                    await self.close_position(
                        opportunity['symbol'],
                        position['quantity'],
                        "AI SELL Signal"
                    )
            else:  # BUY decision
                # Execute the buy trade
                await self.execute_trade({
                    'symbol': opportunity['symbol'],
                    'price': opportunity['current_price'],
                    'position_size': final_position['position_size'],
                    'stop_loss_pct': final_position['stop_loss_pct'],
                    'take_profit_pct': final_position['take_profit_pct']
                })
            
        except Exception as e:
            logger.error(f"Error executing AI trade: {str(e)}")

    async def run_async(self):
        """Async version of run method"""
        try:
            logger.info("Starting trade executor...")
            
            # Start user socket
            self.start_user_socket()
            
            # Initialize the CryptoScanner with the client instance
            scanner = CryptoScanner(self.client)
            
            while self.running:
                try:
                    # Monitor active trades first
                    self.monitor_active_trades()
                    
                    # Update USDC balance before scanning
                    self.update_usdc_balance()
                    
                    # Check if we have enough USDC and available positions
                    max_positions = self.config['trading_params']['max_positions']
                    if len(self.active_trades) >= max_positions:
                        logger.info(f"Maximum positions ({max_positions}) reached. Waiting...")
                        await asyncio.sleep(10)
                        continue
                    
                    if self.available_usdc < self.config['trading_params']['min_trade_amount']:
                        logger.info(f"Insufficient USDC balance (${self.available_usdc:.2f}). Waiting...")
                        await asyncio.sleep(10)
                        continue
                    
                    # Scan market for volatile USDC pairs
                    volatile_pairs = scanner.scan_market(min_volume_usdc=self.config['trading_params']['min_volume_usdc'])
                    
                    # Get market analysis
                    market_analysis = await self.get_market_analysis(volatile_pairs)
                    
                    for pair in volatile_pairs:
                        if not self.running or len(self.active_trades) >= max_positions:
                            break
                            
                        # Skip if we already have a position in this pair
                        if pair['symbol'] in self.active_trades:
                            continue
                            
                        # Create trading signal
                        signal = TradingSignal(
                            symbol=pair['symbol'],
                            price=pair['current_price'],
                            rsi=pair['rsi'],
                            stoch_k=pair['stoch_k'],
                            macd=pair['macd'],
                            volume=pair['avg_volume'],
                            volatility=pair['volatility'],
                            williams_r=pair['williams_r'],
                            trend=pair['trend'],
                            trend_strength=pair['trend_strength'],
                            bb_position=pair['bb_position']
                        )
                        
                        # Execute trade with AI analysis
                        await self.execute_trade_with_ai(pair, signal)
                    
                    if self.running:
                        await asyncio.sleep(30)  # Reduced scan interval
                    
                except Exception as e:
                    logger.error(f"Error in trade executor loop: {str(e)}")
                    if self.running:
                        await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"Error in trade executor: {str(e)}")

    def run(self):
        """Run the trade executor"""
        # Set up the event loop
        loop = setup_event_loop()
        
        # Run the async method
        loop.run_until_complete(self.run_async())

    def stop(self):
        """Stop the trade executor and cleanup resources"""
        logger.info("Stopping trade executor...")
        self.running = False
        # Cleanup any active trades
        self.liquidate_all_positions()
        # Close any open connections
        if hasattr(self, 'client'):
            try:
                self.client.close_connection()
            except:
                pass

class AutoTrader:
    def __init__(self, config_path: str = 'config.json'):
        """Initialize AutoTrader with configuration"""
        self.load_config(config_path)
        self.client = Client(self.config['api_key'], self.config['api_secret'])
        self.opportunity_queue = Queue()
        self.market_data = {}  # Shared market data
        self.initialize_trading_directory()
        self.running = True
        
        # Initialize shared ThreadedWebsocketManager
        self.twm = ThreadedWebsocketManager(api_key=self.config['api_key'], api_secret=self.config['api_secret'])
        
        # Initialize threads with shared ThreadedWebsocketManager
        self.market_monitor = MarketMonitor(self.client, self.config, self.opportunity_queue, self.market_data, self.twm)
        self.trade_executor = TradeExecutor(self.client, self.config, self.opportunity_queue, self.market_data, self.twm)
    
    def load_config(self, config_path: str):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    def initialize_trading_directory(self):
        """Initialize directory structure for logging and data"""
        os.makedirs('logs', exist_ok=True)
        os.makedirs('data', exist_ok=True)

    def start(self):
        """Start trading operations"""
        logger.info("Starting trading operations...")
        try:
            # Start the websocket manager first
            self.twm.start()
            time.sleep(1)  # Give the websocket manager time to initialize
            
            # Then start the threads
            self.market_monitor.start()
            self.trade_executor.start()
            
            # Keep main thread alive while running
            while self.running:
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in trading operations: {str(e)}")
            self.stop()

    def stop(self):
        """Stop trading operations and cleanup resources"""
        logger.info("Stopping trading operations...")
        self.running = False
        
        try:
            # Stop threads
            if hasattr(self, 'trade_executor'):
                self.trade_executor.stop()
            if hasattr(self, 'market_monitor'):
                self.market_monitor.stop()
            
            # Stop websocket manager
            if hasattr(self, 'twm'):
                try:
                    self.twm.stop()
                except:
                    pass
            
            # Close client connection
            if hasattr(self, 'client'):
                try:
                    self.client.close_connection()
                except:
                    pass
            
            # Wait for threads to finish with timeout
            if hasattr(self, 'market_monitor'):
                self.market_monitor.join(timeout=5)
            if hasattr(self, 'trade_executor'):
                self.trade_executor.join(timeout=5)
            
            logger.info("Trading bot stopped.")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")

def main():
    trader = None
    try:
        trader = AutoTrader()
        trader.start()
    except KeyboardInterrupt:
        logger.info("Shutting down trading bot...")
        if trader:
            trader.stop()
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        if trader:
            trader.stop()

if __name__ == "__main__":
    main()
