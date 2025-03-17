import os
import json
import asyncio
import logging
import redis
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from binance.client import Client
from services.utils.order_book_analyzer import OrderBookAnalyzer

class OrderBookAnalysisService:
    """
    Service for analyzing order book depth data to gain trading insights.
    
    This service:
    - Fetches order book data from Binance
    - Analyzes order book depth for market insights
    - Detects support/resistance levels, liquidity imbalances
    - Generates trading signals based on order book analysis
    - Monitors important metrics like buy/sell pressure
    - Stores analysis results for use by other services
    """
    
    def __init__(self):
        """Initialize the OrderBookAnalysisService."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Order Book Analysis Service")
        
        # Load configuration
        try:
            with open('config.json', 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            self.config = {}
        
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
        
        # Get order book analysis configuration
        self.order_book_config = self.config.get('order_book_analysis', {})
        
        # Get symbols to monitor
        self.symbols = self.order_book_config.get('symbols', ['BTCUSDC', 'ETHUSDC', 'BNBUSDC'])
        
        # Initialize analyzer
        self.analyzer = OrderBookAnalyzer(self.config)
        
        # Analysis interval
        self.analysis_interval = self.order_book_config.get('analysis_interval', 60)  # Default 60 seconds
        
        # Order book depth
        self.max_depth = self.order_book_config.get('max_depth', 20)
        
        # Aggregation parameters
        self.aggregation_enabled = self.order_book_config.get('aggregation_enabled', True)
        self.aggregation_interval = self.order_book_config.get('aggregation_interval', 300)  # 5 minutes
        
        # Visualization parameters
        self.create_visualizations = self.order_book_config.get('create_visualizations', True)
        
        # Service port
        self.service_port = self.order_book_config.get('service_port', 8013)
        
        # Last analysis timestamps
        self.last_analysis = {}
        
        # Aggregated metrics for each symbol
        self.aggregated_metrics = {}
        
        self.logger.info(f"Order Book Analysis Service initialized with {len(self.symbols)} symbols")
    
    async def run(self):
        """Run the Order Book Analysis Service."""
        self.logger.info("Starting Order Book Analysis Service")
        
        # Check if Binance client is available
        if not self.client:
            self.logger.error("Cannot start service: Binance client not available")
            return
        
        # Publish service status to Redis
        self.redis.set(
            'order_book_analysis_status', 
            json.dumps({
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'monitored_symbols': self.symbols,
                'analysis_interval': self.analysis_interval
            })
        )
        
        # Main service loop
        try:
            while True:
                start_time = time.time()
                
                # Process each symbol
                for symbol in self.symbols:
                    try:
                        await self.process_symbol(symbol)
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol}: {str(e)}")
                
                # Generate aggregated metrics if enabled
                if self.aggregation_enabled:
                    now = datetime.now()
                    for symbol in self.symbols:
                        agg_key = f"order_book_agg:{symbol}"
                        last_agg_json = self.redis.get(agg_key)
                        
                        if not last_agg_json:
                            await self.aggregate_metrics(symbol)
                            continue
                        
                        try:
                            last_agg = json.loads(last_agg_json)
                            last_agg_time = datetime.fromisoformat(last_agg.get('timestamp', '2000-01-01T00:00:00'))
                            
                            # Check if it's time to update
                            if (now - last_agg_time).total_seconds() >= self.aggregation_interval:
                                await self.aggregate_metrics(symbol)
                        except Exception as e:
                            self.logger.error(f"Error checking aggregation for {symbol}: {str(e)}")
                
                # Update overall service metrics
                await self.update_service_metrics()
                
                # Sleep for the remainder of the interval
                elapsed = time.time() - start_time
                sleep_time = max(0.1, self.analysis_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            self.logger.info("Order Book Analysis Service shutting down")
            # Update service status in Redis
            self.redis.set(
                'order_book_analysis_status', 
                json.dumps({
                    'status': 'stopped',
                    'timestamp': datetime.now().isoformat()
                })
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in Order Book Analysis Service: {str(e)}")
            # Update service status in Redis with error
            self.redis.set(
                'order_book_analysis_status', 
                json.dumps({
                    'status': 'error',
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                })
            )
    
    async def process_symbol(self, symbol):
        """Process a single symbol's order book data."""
        # Fetch current order book
        order_book = self.client.get_order_book(symbol=symbol, limit=self.max_depth)
        
        # Get current price (midpoint of best bid/ask)
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        
        # Extract bids and asks
        bids = order_book['bids']
        asks = order_book['asks']
        
        # Convert to proper format (price and quantity as floats)
        bids = [[float(price), float(qty)] for price, qty in bids]
        asks = [[float(price), float(qty)] for price, qty in asks]
        
        # Analyze order book
        analysis_results = self.analyzer.analyze_order_book(bids, asks, current_price)
        
        # Get trading signals
        trading_signals = self.analyzer.generate_trading_signals(analysis_results)
        
        # Add trading signals to results
        analysis_results['trading_signals'] = trading_signals
        
        # Record the timestamp of this analysis
        self.last_analysis[symbol] = datetime.now().isoformat()
        
        # Store results in Redis
        self.redis.set(f'order_book:{symbol}', json.dumps(analysis_results))
        
        # Store aggregated signals
        self.redis.set(f'order_book_signals:{symbol}', json.dumps(trading_signals))
        
        # Create visualizations if enabled
        if self.create_visualizations:
            await self.create_visualization(symbol, bids, asks, current_price, analysis_results)
        
        # Log analysis completion
        self.logger.debug(f"Completed order book analysis for {symbol}")
    
    async def aggregate_metrics(self, symbol):
        """Aggregate historical order book metrics for a symbol."""
        try:
            # Get historical data from Redis
            key_pattern = f'order_book:{symbol}'
            current_data_json = self.redis.get(key_pattern)
            
            if not current_data_json:
                self.logger.warning(f"No order book data available for {symbol}")
                return
            
            current_data = json.loads(current_data_json)
            
            # Get historical trends if available
            trends = self.analyzer.get_historical_trends()
            
            # Combine current metrics with trends
            aggregated = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'current_price': current_data.get('current_price', 0),
                'current_metrics': {
                    'bid_ask_ratio': current_data.get('bid_ask_ratio', 1.0),
                    'liquidity_imbalance': current_data.get('liquidity_imbalance', 0),
                    'spread_percentage': current_data.get('spread_percentage', 0),
                    'near_pressure': current_data.get('near_pressure', 0),
                    'wall_pressure': current_data.get('wall_pressure', 0)
                },
                'trends': trends
            }
            
            # Add trading signals
            key_pattern = f'order_book_signals:{symbol}'
            signals_json = self.redis.get(key_pattern)
            if signals_json:
                signals = json.loads(signals_json)
                aggregated['signals'] = signals.get('overall', {})
            
            # Store aggregated metrics in Redis
            self.redis.set(f'order_book_agg:{symbol}', json.dumps(aggregated))
            
            # Store locally for reporting
            self.aggregated_metrics[symbol] = aggregated
            
            self.logger.debug(f"Aggregated order book metrics for {symbol}")
        
        except Exception as e:
            self.logger.error(f"Error aggregating metrics for {symbol}: {str(e)}")
    
    async def create_visualization(self, symbol, bids, asks, current_price, analysis_results):
        """Create visualization of order book data."""
        try:
            plt.figure(figsize=(10, 6))
            
            # Convert to pandas for easier plotting
            bids_df = pd.DataFrame(bids, columns=['price', 'quantity'])
            asks_df = pd.DataFrame(asks, columns=['price', 'quantity'])
            
            # Calculate cumulative volumes
            bids_df['cumulative'] = bids_df['quantity'].cumsum()
            asks_df['cumulative'] = asks_df['quantity'].cumsum()
            
            # Plot cumulative volumes
            plt.step(bids_df['price'], bids_df['cumulative'], 'g-', where='post', label='Bids')
            plt.step(asks_df['price'], asks_df['cumulative'], 'r-', where='post', label='Asks')
            
            # Mark current price
            plt.axvline(x=current_price, color='blue', linestyle='--', label=f'Current Price: ${current_price:.4f}')
            
            # Mark support and resistance levels
            support_levels = analysis_results.get('support_levels', [])
            resistance_levels = analysis_results.get('resistance_levels', [])
            
            for level in support_levels[:3]:  # Plot top 3 support levels
                plt.axvline(x=level['price'], color='green', alpha=0.5, 
                            linestyle=':', label=f"Support: ${level['price']:.4f}")
            
            for level in resistance_levels[:3]:  # Plot top 3 resistance levels
                plt.axvline(x=level['price'], color='red', alpha=0.5, 
                            linestyle=':', label=f"Resistance: ${level['price']:.4f}")
            
            # Add annotations for walls if they exist
            bid_wall = analysis_results.get('bid_wall')
            ask_wall = analysis_results.get('ask_wall')
            
            if bid_wall and bid_wall.get('quantity') > analysis_results.get('average_bid_size', 0) * 2:
                plt.scatter(bid_wall['price'], bid_wall['quantity'], 
                            color='green', s=100, marker='^', label=f"Bid Wall: ${bid_wall['price']:.4f}")
            
            if ask_wall and ask_wall.get('quantity') > analysis_results.get('average_ask_size', 0) * 2:
                plt.scatter(ask_wall['price'], ask_wall['quantity'], 
                            color='red', s=100, marker='v', label=f"Ask Wall: ${ask_wall['price']:.4f}")
            
            # Title and labels
            plt.title(f"Order Book Depth: {symbol} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            plt.xlabel("Price")
            plt.ylabel("Cumulative Volume")
            
            # Add overall signal
            signals = analysis_results.get('trading_signals', {}).get('overall', {})
            signal_text = (
                f"Signal: {signals.get('signal', 'N/A').upper()}, "
                f"Confidence: {signals.get('confidence', 0):.2f}, "
                f"Buy Signals: {signals.get('buy_signals', 0)}, "
                f"Sell Signals: {signals.get('sell_signals', 0)}"
            )
            plt.figtext(0.5, 0.01, signal_text, ha='center', fontsize=10)
            
            # Add metrics
            metrics_text = (
                f"Bid/Ask Ratio: {analysis_results.get('bid_ask_ratio', 0):.2f}, "
                f"Liq. Imbalance: {analysis_results.get('liquidity_imbalance', 0):.2f}, "
                f"Near Pressure: {analysis_results.get('near_pressure', 0):.2f}"
            )
            plt.figtext(0.5, 0.03, metrics_text, ha='center', fontsize=10)
            
            # Legend with smaller fonts to accommodate more entries
            plt.legend(fontsize='small', loc='upper right')
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.08, 1, 0.98])
            
            # Save plot to bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            
            # Convert to base64 string for Redis storage
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Store in Redis
            self.redis.set(f'order_book_visualization:{symbol}', img_str)
            
            # Clean up
            plt.close()
            
            self.logger.debug(f"Created visualization for {symbol}")
        
        except Exception as e:
            self.logger.error(f"Error creating visualization for {symbol}: {str(e)}")
    
    async def update_service_metrics(self):
        """Update overall service metrics."""
        try:
            # Calculate processing metrics
            signals_by_symbol = {}
            for symbol in self.symbols:
                signals_key = f'order_book_signals:{symbol}'
                signals_json = self.redis.get(signals_key)
                
                if signals_json:
                    signals = json.loads(signals_json)
                    overall = signals.get('overall', {})
                    signals_by_symbol[symbol] = {
                        'signal': overall.get('signal', 'neutral'),
                        'confidence': overall.get('confidence', 0),
                        'buy_signals': overall.get('buy_signals', 0),
                        'sell_signals': overall.get('sell_signals', 0)
                    }
            
            # Compile summary report
            buy_signals = sum(1 for s in signals_by_symbol.values() if s.get('signal') == 'buy')
            sell_signals = sum(1 for s in signals_by_symbol.values() if s.get('signal') == 'sell')
            neutral_signals = sum(1 for s in signals_by_symbol.values() if s.get('signal') == 'neutral')
            
            # Find strongest signals
            strongest_buy = None
            strongest_sell = None
            
            for symbol, data in signals_by_symbol.items():
                if data.get('signal') == 'buy':
                    if not strongest_buy or data.get('confidence', 0) > strongest_buy[1]:
                        strongest_buy = (symbol, data.get('confidence', 0))
                elif data.get('signal') == 'sell':
                    if not strongest_sell or data.get('confidence', 0) > strongest_sell[1]:
                        strongest_sell = (symbol, data.get('confidence', 0))
            
            # Create summary report
            summary = {
                'timestamp': datetime.now().isoformat(),
                'monitored_symbols': len(self.symbols),
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'neutral_signals': neutral_signals,
                'strongest_buy': strongest_buy[0] if strongest_buy else None,
                'strongest_buy_confidence': strongest_buy[1] if strongest_buy else 0,
                'strongest_sell': strongest_sell[0] if strongest_sell else None,
                'strongest_sell_confidence': strongest_sell[1] if strongest_sell else 0,
                'signals_by_symbol': signals_by_symbol
            }
            
            # Store summary in Redis
            self.redis.set('order_book_analysis_summary', json.dumps(summary))
            
            self.logger.debug("Updated order book analysis service metrics")
        
        except Exception as e:
            self.logger.error(f"Error updating service metrics: {str(e)}")
    
    async def stop(self):
        """Stop the Order Book Analysis Service."""
        self.logger.info("Stopping Order Book Analysis Service")
        # Update service status in Redis
        self.redis.set(
            'order_book_analysis_status', 
            json.dumps({
                'status': 'stopped',
                'timestamp': datetime.now().isoformat()
            })
        )