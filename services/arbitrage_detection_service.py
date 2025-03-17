import os
import json
import logging
import asyncio
import time
import math
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set, Any
import redis
from collections import defaultdict

from services.utils.exchange_interface import ExchangeInterface, ExchangeFactory

class ArbitrageDetectionService:
    """
    Arbitrage Detection Service for finding and executing arbitrage opportunities.
    
    This service implements two types of arbitrage strategies:
    1. Triangle arbitrage - exploiting price differences between three related trading pairs
       on the same exchange (e.g., BTC/USDC, ETH/USDC, ETH/BTC)
    2. Cross-exchange arbitrage - exploiting price differences for the same trading pair
       across different exchanges (currently in simulation mode only)
    
    The service continuously scans for opportunities and can either notify about them
    or automatically execute trades if configured to do so.
    """
    
    def __init__(self):
        """Initialize the Arbitrage Detection Service"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Arbitrage Detection Service")
        
        # Load configuration
        try:
            with open('config.json', 'r') as f:
                self.config = json.load(f)
                self.arb_config = self.config.get('arbitrage_detection', {})
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            self.config = {}
            self.arb_config = {}
        
        # Set up Redis connection
        redis_host = os.environ.get('REDIS_HOST', 'localhost')
        redis_port = int(os.environ.get('REDIS_PORT', 6379))
        self.redis = redis.Redis(host=redis_host, port=redis_port, db=0)
        
        # Initialize exchanges
        self.exchanges = {}
        self.init_exchanges()
        
        # Service parameters
        self.enabled = self.arb_config.get('enabled', False)
        self.simulation_mode = self.arb_config.get('simulation_mode', True)
        self.service_port = self.arb_config.get('service_port', 8016)
        self.update_interval = self.arb_config.get('update_interval', 10)  # seconds
        self.min_profit_pct = self.arb_config.get('min_profit_pct', 0.5)
        
        # Trading parameters
        self.trade_enabled = self.arb_config.get('trade_enabled', False)
        self.min_notional_usdc = self.arb_config.get('min_notional_usdc', 100)
        self.max_notional_usdc = self.arb_config.get('max_notional_usdc', 10000)
        self.min_depth_ratio = self.arb_config.get('min_depth_ratio', 2.0)
        self.auto_adjust_notional = self.arb_config.get('auto_adjust_notional', True)
        self.symbol_blacklist = set(self.arb_config.get('symbol_blacklist', []))
        
        # Triangle arbitrage parameters
        self.triangle_enabled = self.arb_config.get('triangle_arbitrage', {}).get('enabled', True)
        self.triangle_base_currencies = self.arb_config.get('triangle_arbitrage', {}).get('base_currencies', ['USDC', 'USDT', 'BUSD', 'BTC', 'ETH'])
        self.triangle_min_profit_pct = self.arb_config.get('triangle_arbitrage', {}).get('min_profit_pct', 0.3)
        self.max_exchange_steps = self.arb_config.get('triangle_arbitrage', {}).get('max_exchange_steps', 3)
        self.check_balance = self.arb_config.get('triangle_arbitrage', {}).get('check_balance', True)
        self.max_order_book_age_ms = self.arb_config.get('triangle_arbitrage', {}).get('max_order_book_age_ms', 1000)
        self.min_volume_filter = self.arb_config.get('triangle_arbitrage', {}).get('min_volume_filter', 1000)
        
        # Cross-exchange parameters
        self.cross_exchange_enabled = self.arb_config.get('cross_exchange_arbitrage', {}).get('enabled', False)
        self.cross_exchange_base_currencies = self.arb_config.get('cross_exchange_arbitrage', {}).get('base_currencies', ['USDC', 'USDT', 'BTC'])
        self.cross_exchange_min_profit_pct = self.arb_config.get('cross_exchange_arbitrage', {}).get('min_profit_pct', 1.0)
        
        # Notification and reporting
        self.notification_enabled = self.arb_config.get('notification', {}).get('enabled', True)
        self.profit_threshold_pct = self.arb_config.get('notification', {}).get('profit_threshold_pct', 0.5)
        self.notification_interval = self.arb_config.get('notification', {}).get('interval_minutes', 5) * 60  # convert to seconds
        self.last_notification = defaultdict(lambda: datetime.min)
        
        # State tracking
        self.market_data = {}  # symbol -> ticker data
        self.order_books = {}  # symbol -> order book data
        self.graph = None  # for triangle arbitrage
        self.opportunities = []  # list of detected opportunities
        self.trading_pairs = {}  # map of trading pairs
        self.currency_pairs = {}  # available currency pairs by exchange
        self.initialized = False
        
        # Performance metrics
        self.total_opportunities = 0
        self.profitable_opportunities = 0
        self.executed_arbitrages = 0
        self.total_profit = 0.0
        self.start_time = datetime.now()
        
        self.logger.info("Arbitrage Detection Service initialized")
    
    def init_exchanges(self):
        """Initialize exchange interfaces"""
        for exchange_config in self.arb_config.get('exchanges', []):
            if exchange_config.get('enabled', False):
                exchange_name = exchange_config.get('name', '').lower()
                try:
                    exchange = ExchangeFactory.create_exchange(exchange_name)
                    self.exchanges[exchange_name] = exchange
                    self.logger.info(f"Initialized exchange: {exchange_name}")
                except Exception as e:
                    self.logger.error(f"Error initializing exchange {exchange_name}: {str(e)}")
    
    async def run(self):
        """Run the arbitrage detection service"""
        self.logger.info("Starting Arbitrage Detection Service")
        
        # Check if arbitrage detection is enabled
        if not self.enabled:
            self.logger.info("Arbitrage detection is disabled in configuration")
            return
        
        # Check if at least one exchange is available
        if not self.exchanges:
            self.logger.error("Cannot start service: No exchanges initialized")
            return
        
        # Publish service status to Redis
        self.redis.set(
            'arbitrage_detection_status',
            json.dumps({
                'status': 'starting',
                'timestamp': datetime.now().isoformat(),
                'simulation_mode': self.simulation_mode
            })
        )
        
        # Initialize exchange data
        await self._initialize_markets()
        
        # Update status to running
        self.redis.set(
            'arbitrage_detection_status',
            json.dumps({
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'exchanges': list(self.exchanges.keys()),
                'simulation_mode': self.simulation_mode
            })
        )
        
        # Main service loop
        try:
            while True:
                start_time = time.time()
                
                # Update market data
                await self._update_market_data()
                
                # Detect arbitrage opportunities
                opportunities = []
                
                # Triangle arbitrage on Binance
                if self.triangle_enabled and 'binance' in self.exchanges:
                    triangle_opportunities = self._detect_triangle_arbitrage('binance')
                    opportunities.extend(triangle_opportunities)
                
                # Cross-exchange arbitrage (simulation only for now)
                if self.cross_exchange_enabled and len(self.exchanges) > 1:
                    cross_opportunities = self._detect_cross_exchange_arbitrage()
                    opportunities.extend(cross_opportunities)
                
                # Process the detected opportunities
                if opportunities:
                    await self._process_opportunities(opportunities)
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Sleep for the remainder of the interval
                elapsed = time.time() - start_time
                sleep_time = max(0.1, self.update_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            self.logger.info("Arbitrage Detection Service shutting down")
            
            # Update service status in Redis
            self.redis.set(
                'arbitrage_detection_status',
                json.dumps({
                    'status': 'stopped',
                    'timestamp': datetime.now().isoformat()
                })
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in Arbitrage Detection Service: {str(e)}")
            # Update service status in Redis with error
            self.redis.set(
                'arbitrage_detection_status',
                json.dumps({
                    'status': 'error',
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'simulation_mode': self.simulation_mode
                })
            )
    
    async def _initialize_markets(self):
        """Initialize market data for all exchanges"""
        self.logger.info("Initializing market data")
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                # Get exchange information
                exchange_info = exchange.get_exchange_info()
                
                # Get trading pairs for this exchange
                trading_pairs = []
                for symbol_info in exchange_info.get('symbols', []):
                    symbol = symbol_info.get('symbol', '')
                    base_asset = symbol_info.get('baseAsset', '')
                    quote_asset = symbol_info.get('quoteAsset', '')
                    status = symbol_info.get('status', '')
                    
                    if status == 'TRADING' and symbol not in self.symbol_blacklist:
                        trading_pairs.append({
                            'symbol': symbol,
                            'base_asset': base_asset,
                            'quote_asset': quote_asset
                        })
                
                # Store trading pairs for this exchange
                self.currency_pairs[exchange_name] = trading_pairs
                
                # Map symbol to base/quote assets for easy lookup
                for pair in trading_pairs:
                    self.trading_pairs[pair['symbol']] = {
                        'base': pair['base_asset'],
                        'quote': pair['quote_asset']
                    }
                
                self.logger.info(f"Initialized {len(trading_pairs)} trading pairs for {exchange_name}")
                
            except Exception as e:
                self.logger.error(f"Error initializing market data for {exchange_name}: {str(e)}")
        
        # Build graph for triangle arbitrage if enabled
        if self.triangle_enabled:
            self._build_arbitrage_graph()
        
        self.initialized = True
        self.logger.info("Market data initialization complete")
    
    def _build_arbitrage_graph(self):
        """Build a graph representation of the market for triangle arbitrage detection"""
        self.logger.info("Building arbitrage graph")
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # For now, we'll focus on Binance only for triangle arbitrage
        exchange_name = 'binance'
        
        if exchange_name not in self.currency_pairs:
            self.logger.error(f"Exchange {exchange_name} not initialized")
            return
        
        # Add edges to the graph
        for pair in self.currency_pairs[exchange_name]:
            base = pair['base_asset']
            quote = pair['quote_asset']
            symbol = pair['symbol']
            
            # Only consider pairs with base currencies of interest
            if base in self.triangle_base_currencies or quote in self.triangle_base_currencies:
                # Add both directions (buy and sell)
                G.add_edge(quote, base, symbol=symbol, action='buy')
                G.add_edge(base, quote, symbol=symbol, action='sell')
        
        self.graph = G
        self.logger.info(f"Built arbitrage graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    async def _update_market_data(self):
        """Update market data for all exchanges"""
        for exchange_name, exchange in self.exchanges.items():
            try:
                # Get all ticker data
                tickers = exchange.get_ticker_all()
                
                # Update market data
                for symbol, price in tickers.items():
                    if symbol in self.trading_pairs:
                        self.market_data[f"{exchange_name}:{symbol}"] = {
                            'symbol': symbol,
                            'exchange': exchange_name,
                            'price': price,
                            'timestamp': datetime.now().isoformat()
                        }
            except Exception as e:
                self.logger.error(f"Error updating market data for {exchange_name}: {str(e)}")
    
    def _detect_triangle_arbitrage(self, exchange_name: str) -> List[Dict]:
        """
        Detect triangle arbitrage opportunities on a single exchange.
        
        This method uses a graph-based approach to find profitable cycles in the market.
        """
        if not self.graph:
            return []
        
        opportunities = []
        
        # Use only the base currencies as starting points
        for start_currency in self.triangle_base_currencies:
            if start_currency not in self.graph.nodes():
                continue
            
            # Find all simple cycles starting from this currency
            # Limit cycle length to max_exchange_steps + 1 (to return to starting currency)
            for cycle in nx.simple_cycles(self.graph):
                # Only consider cycles that start and end with our currency
                if cycle[0] != start_currency or len(cycle) > self.max_exchange_steps + 1:
                    continue
                
                # Ensure the cycle comes back to the start currency
                full_cycle = cycle + [cycle[0]]
                
                # Check if this is a valid arbitrage opportunity
                opportunity = self._evaluate_cycle(full_cycle, exchange_name)
                if opportunity and opportunity['profit_pct'] >= self.triangle_min_profit_pct:
                    opportunities.append(opportunity)
        
        return opportunities
    
    def _evaluate_cycle(self, cycle: List[str], exchange_name: str) -> Optional[Dict]:
        """
        Evaluate a potential arbitrage cycle to determine profitability.
        
        Args:
            cycle: A list of currencies representing a cycle (first and last currencies should be the same)
            exchange_name: The name of the exchange to use for pricing
            
        Returns:
            A dictionary with the opportunity details if profitable, otherwise None
        """
        if len(cycle) < 3:  # Need at least 3 nodes for a valid cycle (including duplicate start/end)
            return None
        
        steps = []
        rate_product = 1.0
        fee_product = 1.0
        
        for i in range(len(cycle) - 1):
            from_currency = cycle[i]
            to_currency = cycle[i + 1]
            
            # Find the edge in the graph
            edge_data = None
            for _, _, data in self.graph.edges(data=True):
                if data['symbol'] in self.trading_pairs:
                    symbol_data = self.trading_pairs[data['symbol']]
                    if (symbol_data['base'] == to_currency and symbol_data['quote'] == from_currency and data['action'] == 'buy') or \
                       (symbol_data['base'] == from_currency and symbol_data['quote'] == to_currency and data['action'] == 'sell'):
                        edge_data = data
                        break
            
            if not edge_data:
                return None  # No valid edge found
            
            symbol = edge_data['symbol']
            action = edge_data['action']
            
            # Get the current market price for this symbol
            market_key = f"{exchange_name}:{symbol}"
            if market_key not in self.market_data:
                return None  # No current market data
            
            price = self.market_data[market_key]['price']
            
            # Get the fee for this exchange
            fee = 0.001  # Default fee (0.1%)
            fee_data = self.arb_config.get('exchanges', [{}])
            for exchange_info in fee_data:
                if exchange_info.get('name') == exchange_name:
                    fee = exchange_info.get('taker_fee', 0.001)
                    break
            
            # Calculate the rate and apply the fee
            if action == 'buy':
                rate = 1.0 / price
                rate_with_fee = rate * (1.0 - fee)
            else:  # sell
                rate = price
                rate_with_fee = rate * (1.0 - fee)
            
            # Update the product
            rate_product *= rate_with_fee
            fee_product *= (1.0 - fee)
            
            # Add the step to our path
            steps.append({
                'from_currency': from_currency,
                'to_currency': to_currency,
                'symbol': symbol,
                'action': action,
                'rate': rate,
                'fee': fee
            })
        
        # Check if the cycle is profitable
        profit_pct = (rate_product - 1.0) * 100.0
        
        # If profitable, return the opportunity details
        if profit_pct > 0:
            return {
                'type': 'triangle',
                'exchange': exchange_name,
                'cycle': cycle,
                'steps': steps,
                'profit_pct': profit_pct,
                'fee_product': fee_product,
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    def _detect_cross_exchange_arbitrage(self) -> List[Dict]:
        """
        Detect cross-exchange arbitrage opportunities.
        
        This method compares prices across different exchanges for the same trading pair.
        """
        opportunities = []
        
        # We need at least 2 exchanges for cross-exchange arbitrage
        if len(self.exchanges) < 2:
            return []
        
        # Find common trading pairs across exchanges
        common_symbols = set()
        exchange_symbols = {}
        
        for exchange_name, pairs in self.currency_pairs.items():
            symbols = set(pair['symbol'] for pair in pairs)
            exchange_symbols[exchange_name] = symbols
            
            if not common_symbols:
                common_symbols = symbols
            else:
                common_symbols &= symbols
        
        self.logger.debug(f"Found {len(common_symbols)} common symbols across exchanges")
        
        # Check each common symbol for price differences
        for symbol in common_symbols:
            # Skip if symbol is blacklisted
            if symbol in self.symbol_blacklist:
                continue
            
            prices = {}
            for exchange_name in self.exchanges:
                market_key = f"{exchange_name}:{symbol}"
                if market_key in self.market_data:
                    prices[exchange_name] = self.market_data[market_key]['price']
            
            # Need at least 2 exchanges with prices
            if len(prices) < 2:
                continue
            
            # Find the best buy and sell prices
            buy_exchange = min(prices.items(), key=lambda x: x[1])[0]
            buy_price = prices[buy_exchange]
            
            sell_exchange = max(prices.items(), key=lambda x: x[1])[0]
            sell_price = prices[sell_exchange]
            
            # Skip if buy and sell exchanges are the same
            if buy_exchange == sell_exchange:
                continue
            
            # Calculate profit percentage
            profit_pct = ((sell_price / buy_price) - 1.0) * 100.0
            
            # Apply fees
            buy_fee = 0.001  # Default fee (0.1%)
            sell_fee = 0.001
            
            fee_data = self.arb_config.get('exchanges', [{}])
            for exchange_info in fee_data:
                if exchange_info.get('name') == buy_exchange:
                    buy_fee = exchange_info.get('taker_fee', 0.001)
                if exchange_info.get('name') == sell_exchange:
                    sell_fee = exchange_info.get('taker_fee', 0.001)
            
            # Adjusted profit after fees
            adjusted_profit_pct = ((sell_price * (1.0 - sell_fee)) / (buy_price * (1.0 + buy_fee)) - 1.0) * 100.0
            
            # Check if profitable after fees and meets minimum threshold
            if adjusted_profit_pct >= self.cross_exchange_min_profit_pct:
                opportunities.append({
                    'type': 'cross_exchange',
                    'symbol': symbol,
                    'buy_exchange': buy_exchange,
                    'buy_price': buy_price,
                    'sell_exchange': sell_exchange,
                    'sell_price': sell_price,
                    'profit_pct': profit_pct,
                    'adjusted_profit_pct': adjusted_profit_pct,
                    'buy_fee': buy_fee,
                    'sell_fee': sell_fee,
                    'timestamp': datetime.now().isoformat()
                })
        
        return opportunities
    
    async def _process_opportunities(self, opportunities: List[Dict]):
        """Process detected arbitrage opportunities"""
        # Sort opportunities by adjusted profit percentage
        sorted_opportunities = sorted(
            opportunities,
            key=lambda x: x.get('adjusted_profit_pct', x.get('profit_pct', 0)),
            reverse=True
        )
        
        # Keep track of processed opportunities
        processed = []
        
        for opportunity in sorted_opportunities:
            # Check if this is a profitable opportunity
            profit_pct = opportunity.get('adjusted_profit_pct', opportunity.get('profit_pct', 0))
            
            if profit_pct >= self.min_profit_pct:
                # Log the opportunity
                self.logger.info(f"Detected arbitrage opportunity: {opportunity['type']} with {profit_pct:.2f}% profit")
                
                # Store in processed list
                processed.append(opportunity)
                
                # Send notification if enabled and above threshold
                if self.notification_enabled and profit_pct >= self.profit_threshold_pct:
                    await self._send_notification(opportunity)
                
                # Execute the trade if enabled and in live mode
                if self.trade_enabled and not self.simulation_mode:
                    await self._execute_arbitrage(opportunity)
        
        # Update opportunities list
        self.opportunities = processed
        
        # Update statistics
        self.total_opportunities += len(opportunities)
        self.profitable_opportunities += len(processed)
        
        # Store opportunities in Redis
        opportunities_json = json.dumps({
            'timestamp': datetime.now().isoformat(),
            'count': len(processed),
            'opportunities': processed[:10]  # Limit to top 10 for Redis storage
        })
        
        self.redis.set('arbitrage_opportunities', opportunities_json)
        
        # If we have profitable opportunities, also store in a time-series list
        if processed:
            for opp in processed[:3]:  # Limit to top 3 most profitable
                self.redis.lpush('arbitrage_opportunity_history', json.dumps(opp))
                # Trim the list to keep only most recent entries
                self.redis.ltrim('arbitrage_opportunity_history', 0, 99)
    
    async def _send_notification(self, opportunity: Dict):
        """Send notification about an arbitrage opportunity"""
        # Check notification interval to avoid spamming
        opportunity_type = opportunity['type']
        now = datetime.now()
        last_time = self.last_notification[opportunity_type]
        
        if (now - last_time).total_seconds() < self.notification_interval:
            return
        
        # Update last notification time
        self.last_notification[opportunity_type] = now
        
        # Format notification message
        if opportunity_type == 'triangle':
            message = self._format_triangle_notification(opportunity)
        else:  # cross_exchange
            message = self._format_cross_exchange_notification(opportunity)
        
        # Publish notification to Redis
        self.redis.publish(
            'arbitrage_notifications',
            json.dumps({
                'timestamp': now.isoformat(),
                'type': opportunity_type,
                'profit_pct': opportunity.get('adjusted_profit_pct', opportunity.get('profit_pct', 0)),
                'message': message
            })
        )
    
    def _format_triangle_notification(self, opportunity: Dict) -> str:
        """Format notification message for triangle arbitrage"""
        cycle = opportunity['cycle']
        profit_pct = opportunity['profit_pct']
        
        # Format the path
        path = ' → '.join(cycle)
        
        # Format steps
        steps = []
        for step in opportunity['steps']:
            from_curr = step['from_currency']
            to_curr = step['to_currency']
            action = step['action'].upper()
            symbol = step['symbol']
            rate = step['rate']
            
            steps.append(f"{action} {symbol} @ {rate:.8f}")
        
        steps_str = ' | '.join(steps)
        
        return f"Triangle Arbitrage: {profit_pct:.2f}% profit\nPath: {path}\nSteps: {steps_str}"
    
    def _format_cross_exchange_notification(self, opportunity: Dict) -> str:
        """Format notification message for cross-exchange arbitrage"""
        symbol = opportunity['symbol']
        buy_exchange = opportunity['buy_exchange'].title()
        sell_exchange = opportunity['sell_exchange'].title()
        buy_price = opportunity['buy_price']
        sell_price = opportunity['sell_price']
        profit_pct = opportunity['adjusted_profit_pct']
        
        return f"Cross-Exchange Arbitrage: {profit_pct:.2f}% profit\nBuy {symbol} on {buy_exchange} @ {buy_price:.8f}\nSell on {sell_exchange} @ {sell_price:.8f}"
    
    async def _execute_arbitrage(self, opportunity: Dict):
        """Execute an arbitrage trade"""
        # This is a placeholder for actual trade execution
        # In a real implementation, this would place orders on the relevant exchanges
        
        self.logger.info(f"Would execute {opportunity['type']} arbitrage for {opportunity.get('adjusted_profit_pct', opportunity.get('profit_pct', 0)):.2f}% profit")
        
        # For now, we'll just log it and track it in our metrics
        self.executed_arbitrages += 1
        self.total_profit += opportunity.get('adjusted_profit_pct', opportunity.get('profit_pct', 0))
        
        # In a real implementation, we would:
        # 1. Calculate the optimal trade size based on order book depth
        # 2. Place the orders in the correct sequence
        # 3. Monitor the orders for execution
        # 4. Handle partial fills and errors
    
    async def _update_performance_metrics(self):
        """Update and publish performance metrics"""
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'start_time': self.start_time.isoformat(),
            'running_time_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'total_opportunities': self.total_opportunities,
            'profitable_opportunities': self.profitable_opportunities,
            'executed_arbitrages': self.executed_arbitrages,
            'total_profit': self.total_profit,
            'profit_per_day': self.total_profit / ((datetime.now() - self.start_time).total_seconds() / 86400) if (datetime.now() - self.start_time).total_seconds() > 0 else 0,
            'exchanges': list(self.exchanges.keys()),
            'simulation_mode': self.simulation_mode,
        }
        
        # Publish to Redis
        self.redis.set('arbitrage_performance', json.dumps(performance_data))
    
    def generate_arbitrage_graph_visualization(self, opportunity: Dict = None):
        """Generate a visualization of the arbitrage graph or a specific opportunity"""
        if not self.graph:
            return None
        
        # Create a new plot
        plt.figure(figsize=(12, 8))
        
        if opportunity and opportunity['type'] == 'triangle':
            # Highlight the specific cycle in the opportunity
            cycle = opportunity['cycle']
            
            # Create a subgraph with only the nodes and edges in the cycle
            G = nx.DiGraph()
            
            for i in range(len(cycle) - 1):
                from_currency = cycle[i]
                to_currency = cycle[i + 1]
                
                # Find the edge in the graph
                edge_data = None
                for _, _, data in self.graph.edges(data=True):
                    if data['symbol'] in self.trading_pairs:
                        symbol_data = self.trading_pairs[data['symbol']]
                        if (symbol_data['base'] == to_currency and symbol_data['quote'] == from_currency and data['action'] == 'buy') or \
                           (symbol_data['base'] == from_currency and symbol_data['quote'] == to_currency and data['action'] == 'sell'):
                            edge_data = data
                            break
                
                if edge_data:
                    G.add_edge(from_currency, to_currency, **edge_data)
            
            # Draw the subgraph
            pos = nx.spring_layout(G)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
            
            # Draw edges
            edges = G.edges(data=True)
            nx.draw_networkx_edges(G, pos, edgelist=edges, arrows=True, arrowsize=20)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10)
            
            # Draw edge labels
            edge_labels = {(u, v): f"{d['action']}\n{d['symbol']}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
            
            # Set title
            profit_pct = opportunity['profit_pct']
            plt.title(f"Triangle Arbitrage: {profit_pct:.2f}% Profit")
            
        else:
            # Draw the full graph
            pos = nx.spring_layout(self.graph)
            
            # Draw nodes
            base_nodes = [n for n in self.graph.nodes() if n in self.triangle_base_currencies]
            other_nodes = [n for n in self.graph.nodes() if n not in self.triangle_base_currencies]
            
            nx.draw_networkx_nodes(self.graph, pos, nodelist=base_nodes, node_color='lightgreen', node_size=700)
            nx.draw_networkx_nodes(self.graph, pos, nodelist=other_nodes, node_color='lightblue', node_size=500)
            
            # Draw edges
            nx.draw_networkx_edges(self.graph, pos, arrows=True, arrowsize=15)
            
            # Draw labels
            nx.draw_networkx_labels(self.graph, pos, font_size=10)
            
            # Set title
            plt.title("Arbitrage Market Graph")
        
        plt.axis('off')
        
        # Save the plot to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_type = "opportunity" if opportunity else "full"
        filename = f"data/arbitrage_graph_{graph_type}_{timestamp}.png"
        
        os.makedirs('data', exist_ok=True)
        plt.savefig(filename)
        plt.close()
        
        return filename
    
    async def stop(self):
        """Stop the arbitrage detection service and clean up"""
        self.logger.info("Stopping Arbitrage Detection Service")
        
        # Update final performance
        await self._update_performance_metrics()
        
        # Generate a final graph visualization if enabled
        if self.graph and self.arb_config.get('reporting', {}).get('graph_visualization', True):
            if self.opportunities:
                self.generate_arbitrage_graph_visualization(self.opportunities[0])
            else:
                self.generate_arbitrage_graph_visualization()
        
        # Update service status
        self.redis.set(
            'arbitrage_detection_status',
            json.dumps({
                'status': 'stopped',
                'timestamp': datetime.now().isoformat(),
                'simulation_mode': self.simulation_mode
            })
        )