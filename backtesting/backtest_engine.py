import os
import json
import asyncio
import logging as logger
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

from .data_manager import HistoricalDataManager
from .social_data_provider import SocialDataProvider
from .strategy_tester import StrategyTester
from .result_analyzer import ResultAnalyzer

class BacktestEngine:
    """Main engine for running and managing backtests"""
    
    def __init__(self, config_path: str = 'config.json'):
        """Initialize with configuration"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Initialize components
        self.data_manager = HistoricalDataManager(config_path)
        self.social_provider = SocialDataProvider(self.data_manager)
        self.strategy_tester = StrategyTester(config_path)
        self.result_analyzer = ResultAnalyzer()
        
        # Task queue for managing backtest operations
        self.task_queue = asyncio.Queue()
        self.running_tasks = set()
    
    async def fetch_data_for_backtest(self, symbol: str, intervals: List[str],
                                   start_date: datetime, end_date: datetime = None,
                                   include_social: bool = True) -> Dict[str, bool]:
        """Fetch all necessary data for backtesting"""
        if end_date is None:
            end_date = datetime.now()
            
        results = {}
        
        for interval in intervals:
            try:
                logger.info(f"Fetching {interval} data for {symbol} from {start_date} to {end_date}")
                market_success, social_success = await self.data_manager.fetch_and_save_data(
                    symbol, interval, start_date, end_date, include_social
                )
                
                results[interval] = {
                    'market_data': market_success,
                    'social_data': social_success if include_social else False
                }
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol} ({interval}): {str(e)}")
                results[interval] = {
                    'market_data': False,
                    'social_data': False,
                    'error': str(e)
                }
                
        return results
    
    async def run_backtest(self, symbol: str, interval: str,
                         start_date: datetime, end_date: datetime = None,
                         initial_balance: float = 10000.0,
                         save_results: bool = True) -> Dict:
        """Run a backtest for a specific symbol and interval"""
        try:
            if end_date is None:
                end_date = datetime.now()
                
            logger.info(f"Running backtest for {symbol} ({interval}) from {start_date} to {end_date}")
            
            # Check if we have data
            market_data = self.data_manager.load_market_data(symbol, interval, start_date, end_date)
            if market_data.empty:
                # Try to fetch data first
                fetch_results = await self.fetch_data_for_backtest(
                    symbol, [interval], start_date, end_date
                )
                
                if not fetch_results.get(interval, {}).get('market_data', False):
                    logger.error(f"No market data available for {symbol} ({interval}) and data fetch failed")
                    return {'error': 'No market data available and data fetch failed'}
                    
                # Try loading again
                market_data = self.data_manager.load_market_data(symbol, interval, start_date, end_date)
                if market_data.empty:
                    logger.error(f"No market data available for {symbol} ({interval}) even after fetch attempt")
                    return {'error': 'No market data available even after fetch attempt'}
            
            # Run backtest
            result = await self.strategy_tester.backtest_strategy(
                symbol, interval, start_date, end_date, initial_balance
            )
            
            # Save results if requested
            if save_results:
                strategy_name = f"AI_Social_Strategy"
                result_path = self.strategy_tester.save_results(
                    strategy_name, symbol, interval, start_date, end_date
                )
                
                # Generate equity curve plot
                plot_path = self.result_analyzer.plot_equity_curve(
                    {'strategy': strategy_name, 'symbol': symbol, 'interval': interval, 'stats': result}
                )
                
                # Generate trade analysis plot if there are trades
                if result.get('total_trades', 0) > 0:
                    trade_plot_path = self.result_analyzer.plot_trade_analysis(
                        {'strategy': strategy_name, 'symbol': symbol, 'interval': interval, 'stats': result}
                    )
                    
                    result['trade_plot_path'] = trade_plot_path
                
                result['result_path'] = result_path
                result['plot_path'] = plot_path
            
            return result
            
        except Exception as e:
            logger.error(f"Error running backtest for {symbol} ({interval}): {str(e)}")
            return {'error': str(e)}
    
    async def run_multiple_backtests(self, symbols: List[str], intervals: List[str],
                                   start_date: datetime, end_date: datetime = None,
                                   initial_balance: float = 10000.0) -> Dict:
        """Run multiple backtests for different symbols/intervals"""
        results = {}
        
        for symbol in symbols:
            symbol_results = {}
            
            for interval in intervals:
                try:
                    result = await self.run_backtest(
                        symbol, interval, start_date, end_date, initial_balance
                    )
                    
                    symbol_results[interval] = result
                    
                except Exception as e:
                    logger.error(f"Error in backtest for {symbol} on {interval}: {str(e)}")
                    symbol_results[interval] = {"error": str(e)}
            
            results[symbol] = symbol_results
        
        # Generate summary report
        all_results = []
        for symbol, sym_results in results.items():
            for interval, result in sym_results.items():
                if 'error' not in result:
                    all_results.append({
                        'strategy': 'AI_Social_Strategy',
                        'symbol': symbol,
                        'interval': interval,
                        'stats': result
                    })
        
        if all_results:
            summary = self.result_analyzer.generate_summary_report(all_results)
            summary_path = self.result_analyzer.save_summary_report(summary)
            
            # Add summary path to results
            results['summary'] = {
                'path': summary_path,
                'profitable_strategies': summary.get('profitable_strategies', 0),
                'total_results': summary.get('total_results', 0)
            }
            
            # Generate comparison chart
            comparison_path = self.result_analyzer.compare_results(all_results, 'return_pct')
            if comparison_path:
                results['summary']['comparison_chart'] = comparison_path
        
        return results
    
    def get_available_data(self) -> Dict:
        """Get information about available historical data"""
        symbols = self.data_manager.available_symbols()
        
        available_data = {}
        for symbol in symbols:
            intervals = self.data_manager.available_intervals(symbol)
            symbol_data = {'intervals': {}}
            
            for interval in intervals:
                start_date, end_date = self.data_manager.get_data_range(symbol, interval)
                if start_date and end_date:
                    symbol_data['intervals'][interval] = {
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat(),
                        'days': (end_date - start_date).days
                    }
            
            available_data[symbol] = symbol_data
        
        return available_data
    
    async def fetch_data_for_symbols(self, symbols: List[str], intervals: List[str],
                                   days_back: int = 30) -> Dict:
        """Fetch historical data for multiple symbols and intervals"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        results = {}
        for symbol in symbols:
            symbol_result = await self.fetch_data_for_backtest(
                symbol, intervals, start_date, end_date
            )
            results[symbol] = symbol_result
        
        return results
    
    async def add_backtest_task(self, task_type: str, params: Dict) -> int:
        """Add a backtest task to the queue"""
        task_id = len(self.running_tasks) + self.task_queue.qsize() + 1
        
        await self.task_queue.put({
            'id': task_id,
            'type': task_type,
            'params': params,
            'status': 'queued',
            'created_at': datetime.now().isoformat()
        })
        
        return task_id
    
    async def process_task_queue(self):
        """Process tasks from the queue"""
        while True:
            try:
                task = await self.task_queue.get()
                
                # Update task status
                task['status'] = 'running'
                task['started_at'] = datetime.now().isoformat()
                
                # Add to running tasks
                self.running_tasks.add(task['id'])
                
                # Process based on task type
                if task['type'] == 'fetch_data':
                    params = task['params']
                    result = await self.fetch_data_for_symbols(
                        params.get('symbols', []),
                        params.get('intervals', []),
                        params.get('days_back', 30)
                    )
                    
                elif task['type'] == 'run_backtest':
                    params = task['params']
                    result = await self.run_backtest(
                        params.get('symbol'),
                        params.get('interval'),
                        datetime.fromisoformat(params.get('start_date')),
                        datetime.fromisoformat(params.get('end_date')) if params.get('end_date') else None,
                        params.get('initial_balance', 10000.0)
                    )
                    
                elif task['type'] == 'run_multiple_backtests':
                    params = task['params']
                    result = await self.run_multiple_backtests(
                        params.get('symbols', []),
                        params.get('intervals', []),
                        datetime.fromisoformat(params.get('start_date')),
                        datetime.fromisoformat(params.get('end_date')) if params.get('end_date') else None,
                        params.get('initial_balance', 10000.0)
                    )
                    
                else:
                    result = {'error': f"Unknown task type: {task['type']}"}
                
                # Update task status
                task['status'] = 'completed'
                task['completed_at'] = datetime.now().isoformat()
                task['result'] = result
                
                # Remove from running tasks
                self.running_tasks.remove(task['id'])
                
                # Mark task as done
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing task: {str(e)}")
                
                # Update task status if possible
                if 'task' in locals():
                    task['status'] = 'failed'
                    task['error'] = str(e)
                    task['completed_at'] = datetime.now().isoformat()
                    
                    # Remove from running tasks
                    if task['id'] in self.running_tasks:
                        self.running_tasks.remove(task['id'])
                    
                    # Mark task as done
                    self.task_queue.task_done()
                
                # Wait before trying next task
                await asyncio.sleep(1)
    
    async def run(self):
        """Run the backtest engine"""
        # Start task processor
        task_processor = asyncio.create_task(self.process_task_queue())
        
        try:
            # Keep the engine running
            while True:
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            logger.info("Backtest engine shutting down...")
            
        finally:
            # Cancel task processor
            task_processor.cancel()
            try:
                await task_processor
            except asyncio.CancelledError:
                pass