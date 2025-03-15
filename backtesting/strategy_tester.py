import os
import json
import pandas as pd
import numpy as np
import logging as logger
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from pathlib import Path
import asyncio
import time

from binance_ml_strategy import TechnicalAnalyzer, TradingSignal, PositionSizer
from services.ai_trader import AITrader
from .data_manager import HistoricalDataManager
from .social_data_provider import SocialDataProvider

class StrategyTester:
    """Class for testing trading strategies on historical data"""
    
    def __init__(self, config_path: str = 'config.json'):
        """Initialize with configuration"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Initialize components
        self.data_manager = HistoricalDataManager(config_path)
        self.social_provider = SocialDataProvider(self.data_manager)
        
        # AI Trader requires asyncio event loop
        self.ai_trader = AITrader(self.config)
        
        # Trading parameters
        self.trading_params = self.config['trading_params']
        
        # Results directory
        self.results_dir = Path('backtesting/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance statistics
        self.reset_stats()
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.stats = {
            'initial_balance': 0.0,
            'final_balance': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'trades': [],
            'equity_curve': [],
            'drawdown_curve': []
        }
    
    def prepare_market_data(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Convert DataFrame to list of market data dictionaries"""
        market_data_list = []
        
        # Create technical analyzer
        analyzer = TechnicalAnalyzer(df)
        
        # Get all indicators
        indicators = analyzer.get_all_indicators()
        
        # Estimate volume in USDC
        avg_volume = df['volume'].mean() * df['close'].mean()
        
        # Get trend and volatility
        trend, trend_strength = analyzer.get_trend()
        volatility = analyzer.get_volatility()
        
        # Calculate price changes
        current_price = float(df['close'].iloc[-1])
        price_change_1m = 0
        price_change_3m = 0
        price_change_5m = 0
        price_change_15m = 0
        
        if len(df) > 1:
            price_change_1m = ((current_price - float(df['close'].iloc[-2])) / float(df['close'].iloc[-2])) * 100
            
        if len(df) > 3:
            price_change_3m = ((current_price - float(df['close'].iloc[-4])) / float(df['close'].iloc[-4])) * 100
            
        if len(df) > 5:
            price_change_5m = ((current_price - float(df['close'].iloc[-6])) / float(df['close'].iloc[-6])) * 100
            
        if len(df) > 15:
            price_change_15m = ((current_price - float(df['close'].iloc[-16])) / float(df['close'].iloc[-16])) * 100
        
        # Create market data for each row
        for idx, row in df.iterrows():
            market_data = {
                'symbol': symbol,
                'current_price': float(row['close']),
                'avg_volume': avg_volume,
                'timestamp': idx.isoformat(),
                'rsi': indicators['rsi'],
                'stoch_k': indicators['stoch_k'],
                'macd': indicators['macd'],
                'williams_r': indicators['williams_r'],
                'bb_position': indicators['bb_position'],
                'trend': indicators['trend'],
                'trend_strength': indicators['trend_strength'],
                'volatility': indicators['volatility'],
                'price_change_1m': price_change_1m,
                'price_change_3m': price_change_3m,
                'price_change_5m': price_change_5m,
                'price_change_15m': price_change_15m
            }
            
            # Enrich with social data
            market_data = self.social_provider.generate_market_update_with_social(market_data, idx)
            
            market_data_list.append(market_data)
            
        return market_data_list
    
    async def analyze_with_ai(self, market_data: Dict) -> Dict:
        """Analyze trade opportunity using AI"""
        # Create a copy to avoid modifying the original
        market_data_copy = market_data.copy()
        
        # Get AI analysis
        analysis = await self.ai_trader.analyze_trade_opportunity(market_data_copy)
        
        if self.ai_trader.should_take_trade(analysis):
            # Get risk analysis
            risk_setup = {
                'symbol': market_data_copy['symbol'],
                'available_capital': self.trading_params.get('position_size_pct', 0.4) * self.current_balance,
                'volatility': market_data_copy['volatility'],
                'current_price': market_data_copy['current_price'],
                'trend_strength': market_data_copy['trend_strength']
            }
            risk_analysis = await self.ai_trader.analyze_risk_setup(risk_setup)
            
            return {
                'trade_analysis': analysis,
                'risk_analysis': risk_analysis
            }
        
        return {
            'trade_analysis': analysis,
            'risk_analysis': None
        }
    
    async def backtest_strategy(self, symbol: str, interval: str, 
                              start_date: datetime, end_date: datetime = None,
                              initial_balance: float = 10000.0) -> Dict:
        """Backtest a trading strategy on historical data"""
        # Reset stats
        self.reset_stats()
        
        # Set initial balance
        self.current_balance = initial_balance
        self.stats['initial_balance'] = initial_balance
        self.stats['equity_curve'].append({
            'timestamp': start_date.isoformat(),
            'equity': initial_balance
        })
        
        # Load historical data
        market_data = self.data_manager.merge_market_and_social_data(
            symbol, interval, start_date, end_date
        )
        
        if market_data.empty:
            logger.error(f"No data available for {symbol} from {start_date} to {end_date}")
            return self.stats
        
        # Prepare market data for backtesting
        market_updates = self.prepare_market_data(market_data, symbol)
        
        # Track max equity for drawdown calculation
        max_equity = initial_balance
        
        # Track open positions
        open_positions = {}
        
        # Process each market update
        for i, update in enumerate(market_updates):
            # Skip the first few updates (need enough data for indicators)
            if i < 10:
                continue
                
            # Current timestamp
            timestamp = datetime.fromisoformat(update['timestamp'])
            
            # Current price
            current_price = update['current_price']
            
            # Check stop loss and take profit for open positions
            for pos_symbol, position in list(open_positions.items()):
                # Calculate current P&L
                entry_price = position['entry_price']
                quantity = position['quantity']
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                
                # Check stop loss
                if pnl_pct <= -position['stop_loss_pct']:
                    # Close position (stop loss hit)
                    self.close_position(pos_symbol, position, current_price, timestamp, "Stop Loss")
                    open_positions.pop(pos_symbol)
                
                # Check take profit
                elif pnl_pct >= position['take_profit_pct']:
                    # Close position (take profit hit)
                    self.close_position(pos_symbol, position, current_price, timestamp, "Take Profit")
                    open_positions.pop(pos_symbol)
            
            # Check if we already have this position
            if symbol in open_positions:
                continue
                
            # Check if we've reached max positions
            if len(open_positions) >= self.trading_params['max_positions']:
                continue
            
            # Create trading signal
            signal = TradingSignal(
                symbol=update['symbol'],
                price=update['current_price'],
                rsi=update['rsi'],
                stoch_k=update['stoch_k'],
                macd=update['macd'],
                volume=update['avg_volume'],
                volatility=update['volatility'],
                williams_r=update['williams_r'],
                trend=update['trend'],
                trend_strength=update['trend_strength'],
                bb_position=update['bb_position']
            )
            
            # Get AI analysis
            ai_analysis = await self.analyze_with_ai(update)
            
            # Check if we should execute trade
            should_execute = self.should_execute_trade(signal, ai_analysis)
            
            if should_execute['execute'] and should_execute['decision'] == 'BUY':
                # Calculate position size
                technical_position = PositionSizer.calculate_position_size(
                    self.current_balance,
                    update['volatility'],
                    update['avg_volume']
                )
                
                # Adjust position size with AI if available
                if ai_analysis['risk_analysis']:
                    position_params = self.ai_trader.adjust_position_size(
                        ai_analysis['risk_analysis'],
                        technical_position
                    )
                else:
                    position_params = technical_position
                
                # Open position
                self.open_position(symbol, position_params, current_price, timestamp)
                
                # Add to open positions
                open_positions[symbol] = {
                    'entry_price': current_price,
                    'quantity': position_params['position_size'] / current_price,
                    'stop_loss_pct': position_params['stop_loss_pct'],
                    'take_profit_pct': position_params['take_profit_pct'],
                    'entry_time': timestamp,
                    'entry_balance': self.current_balance
                }
            
            # Update equity curve
            self.stats['equity_curve'].append({
                'timestamp': timestamp.isoformat(),
                'equity': self.current_balance
            })
            
            # Update max equity and drawdown
            if self.current_balance > max_equity:
                max_equity = self.current_balance
            
            current_drawdown = max_equity - self.current_balance
            current_drawdown_pct = (current_drawdown / max_equity) * 100
            
            self.stats['drawdown_curve'].append({
                'timestamp': timestamp.isoformat(),
                'drawdown': current_drawdown,
                'drawdown_pct': current_drawdown_pct
            })
            
            if current_drawdown > self.stats['max_drawdown']:
                self.stats['max_drawdown'] = current_drawdown
                self.stats['max_drawdown_pct'] = current_drawdown_pct
        
        # Close any remaining positions at the end
        final_price = market_updates[-1]['current_price'] if market_updates else 0
        final_timestamp = datetime.fromisoformat(market_updates[-1]['timestamp']) if market_updates else end_date
        
        for pos_symbol, position in list(open_positions.items()):
            self.close_position(pos_symbol, position, final_price, final_timestamp, "End of Test")
        
        # Calculate final statistics
        self.calculate_final_stats()
        
        return self.stats
    
    def open_position(self, symbol: str, position_params: Dict, price: float, timestamp: datetime):
        """Open a new trading position"""
        position_size = position_params['position_size']
        quantity = position_size / price
        
        # Add trade to history
        self.stats['trades'].append({
            'symbol': symbol,
            'entry_price': price,
            'entry_time': timestamp.isoformat(),
            'quantity': quantity,
            'position_size': position_size,
            'stop_loss_pct': position_params['stop_loss_pct'],
            'take_profit_pct': position_params['take_profit_pct'],
            'exit_price': None,
            'exit_time': None,
            'pnl': None,
            'pnl_pct': None,
            'exit_reason': None
        })
        
        logger.info(f"Opened position for {symbol} at ${price:.8f} (Size: ${position_size:.2f})")
    
    def close_position(self, symbol: str, position: Dict, price: float, timestamp: datetime, reason: str):
        """Close a trading position"""
        entry_price = position['entry_price']
        quantity = position['quantity']
        
        # Calculate P&L
        pnl = (price - entry_price) * quantity
        pnl_pct = ((price - entry_price) / entry_price) * 100
        
        # Update current balance
        self.current_balance += pnl
        
        # Update trade history
        for trade in reversed(self.stats['trades']):
            if trade['symbol'] == symbol and trade['exit_price'] is None:
                trade['exit_price'] = price
                trade['exit_time'] = timestamp.isoformat()
                trade['pnl'] = pnl
                trade['pnl_pct'] = pnl_pct
                trade['exit_reason'] = reason
                break
        
        # Update trade statistics
        self.stats['total_trades'] += 1
        
        if pnl > 0:
            self.stats['winning_trades'] += 1
            self.stats['total_profit'] += pnl
        else:
            self.stats['losing_trades'] += 1
            self.stats['total_loss'] -= pnl  # Convert to positive value
        
        logger.info(f"Closed position for {symbol} at ${price:.8f} ({reason}) - PnL: ${pnl:.2f} ({pnl_pct:.2f}%)")
    
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
    
    def calculate_final_stats(self):
        """Calculate final performance statistics"""
        self.stats['final_balance'] = self.current_balance
        
        # Win rate
        if self.stats['total_trades'] > 0:
            self.stats['win_rate'] = (self.stats['winning_trades'] / self.stats['total_trades']) * 100
        
        # Profit factor
        if self.stats['total_loss'] > 0:
            self.stats['profit_factor'] = self.stats['total_profit'] / self.stats['total_loss']
        
        # Sharpe ratio (simplified)
        daily_returns = []
        prev_equity = self.stats['initial_balance']
        
        for point in self.stats['equity_curve']:
            equity = point['equity']
            daily_return = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            daily_returns.append(daily_return)
            prev_equity = equity
        
        if len(daily_returns) > 1:
            avg_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            
            if std_return > 0:
                self.stats['sharpe_ratio'] = (avg_return / std_return) * np.sqrt(252)  # Annualize
    
    def save_results(self, strategy_name: str, symbol: str, interval: str,
                   start_date: datetime, end_date: datetime = None) -> str:
        """Save backtest results to file"""
        if end_date is None:
            end_date = datetime.now()
            
        # Create filename
        filename = f"{strategy_name}_{symbol}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
        file_path = self.results_dir / filename
        
        # Create result object
        result = {
            'strategy': strategy_name,
            'symbol': symbol,
            'interval': interval,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'stats': self.stats
        }
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        logger.info(f"Saved backtest results to {file_path}")
        
        return str(file_path)
            
    async def run_multiple_backtests(self, symbols: List[str], intervals: List[str],
                                   start_date: datetime, end_date: datetime = None,
                                   initial_balance: float = 10000.0) -> Dict:
        """Run multiple backtests in parallel"""
        all_results = {}
        
        for symbol in symbols:
            symbol_results = {}
            
            for interval in intervals:
                try:
                    logger.info(f"Running backtest for {symbol} on {interval} timeframe...")
                    result = await self.backtest_strategy(
                        symbol, interval, start_date, end_date, initial_balance
                    )
                    
                    # Save results
                    strategy_name = f"AI_Social_Strategy"
                    self.save_results(strategy_name, symbol, interval, start_date, end_date)
                    
                    symbol_results[interval] = result
                    
                except Exception as e:
                    logger.error(f"Error in backtest for {symbol} on {interval}: {str(e)}")
                    symbol_results[interval] = {"error": str(e)}
            
            all_results[symbol] = symbol_results
        
        return all_results