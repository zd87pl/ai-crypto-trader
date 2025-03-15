#!/usr/bin/env python3
import os
import sys
import json
import argparse
import asyncio
import logging as logger
from datetime import datetime, timedelta
from typing import List, Dict, Any

from backtesting import BacktestEngine, ResultAnalyzer
from backtesting.data_manager import HistoricalDataManager

# Configure logging
logger.basicConfig(
    level=logger.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logger.FileHandler(f'logs/backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logger.StreamHandler(sys.stdout)
    ]
)

def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(description='Crypto Trading Backtesting CLI')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Fetch data command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch historical data')
    fetch_parser.add_argument('--symbols', type=str, nargs='+', required=True, help='Symbol(s) to fetch data for (e.g. BTCUSDC)')
    fetch_parser.add_argument('--intervals', type=str, nargs='+', default=['1h'], help='Timeframe interval(s) (e.g. 1m, 5m, 15m, 1h, 4h, 1d)')
    fetch_parser.add_argument('--days', type=int, default=30, help='Number of days to fetch')
    fetch_parser.add_argument('--no-social', action='store_true', help='Skip fetching social data')
    
    # Run backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run a backtest')
    backtest_parser.add_argument('--symbols', type=str, nargs='+', required=True, help='Symbol(s) to backtest (e.g. BTCUSDC)')
    backtest_parser.add_argument('--intervals', type=str, nargs='+', default=['1h'], help='Timeframe interval(s) (e.g. 1m, 5m, 15m, 1h, 4h, 1d)')
    backtest_parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    backtest_parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance for backtest')
    backtest_parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD) (overrides --days)')
    backtest_parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD) (defaults to today)')
    
    # List data command
    list_parser = subparsers.add_parser('list', help='List available data')
    list_parser.add_argument('--symbols', type=str, nargs='+', help='Filter by symbol(s)')
    list_parser.add_argument('--intervals', type=str, nargs='+', help='Filter by interval(s)')
    
    # Analyze results command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze backtest results')
    analyze_parser.add_argument('--results', type=str, nargs='+', help='Result file(s) to analyze')
    analyze_parser.add_argument('--symbols', type=str, nargs='+', help='Filter results by symbol(s)')
    analyze_parser.add_argument('--intervals', type=str, nargs='+', help='Filter results by interval(s)')
    analyze_parser.add_argument('--metric', type=str, default='return_pct', help='Metric to compare (return_pct, win_rate, sharpe_ratio, etc.)')
    
    return parser

async def fetch_data(args: argparse.Namespace) -> Dict:
    """Fetch historical data for the specified symbols and intervals"""
    logger.info(f"Fetching data for {args.symbols} ({', '.join(args.intervals)}) for the past {args.days} days")
    
    # Initialize components
    data_manager = HistoricalDataManager()
    backtest_engine = BacktestEngine()
    
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Fetch data for each symbol and interval
    results = {}
    for symbol in args.symbols:
        symbol_result = await backtest_engine.fetch_data_for_backtest(
            symbol, args.intervals, start_date, end_date, not args.no_social
        )
        results[symbol] = symbol_result
    
    return results

async def run_backtest(args: argparse.Namespace) -> Dict:
    """Run backtests for the specified symbols and intervals"""
    # Parse dates
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = datetime.now() - timedelta(days=args.days)
        
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()
    
    logger.info(f"Running backtest for {args.symbols} ({', '.join(args.intervals)}) "
               f"from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} "
               f"with initial balance ${args.balance}")
    
    # Initialize backtesting engine
    backtest_engine = BacktestEngine()
    
    # Run backtests
    results = await backtest_engine.run_multiple_backtests(
        args.symbols, args.intervals, start_date, end_date, args.balance
    )
    
    return results

def list_data(args: argparse.Namespace) -> Dict:
    """List available historical data"""
    logger.info("Listing available historical data")
    
    # Initialize data manager
    data_manager = HistoricalDataManager()
    
    # Get available data
    available_data = data_manager.available_symbols()
    
    # Filter by symbols if specified
    if args.symbols:
        available_data = [symbol for symbol in available_data if symbol in args.symbols]
    
    # Get detailed data for each symbol
    result = {}
    for symbol in available_data:
        intervals = data_manager.available_intervals(symbol)
        
        # Filter by intervals if specified
        if args.intervals:
            intervals = [interval for interval in intervals if interval in args.intervals]
        
        symbol_data = {'intervals': {}}
        for interval in intervals:
            start_date, end_date = data_manager.get_data_range(symbol, interval)
            if start_date and end_date:
                symbol_data['intervals'][interval] = {
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'days': (end_date - start_date).days
                }
        
        result[symbol] = symbol_data
    
    return result

def analyze_results(args: argparse.Namespace) -> Dict:
    """Analyze backtest results"""
    logger.info("Analyzing backtest results")
    
    # Initialize result analyzer
    result_analyzer = ResultAnalyzer()
    
    # Get all available results if not specified
    if args.results:
        # Load specific result files
        results = []
        for result_path in args.results:
            result = result_analyzer.load_results(result_path)
            if result:
                results.append(result)
    else:
        # Filter available results
        results = result_analyzer.filter_results(
            symbol=args.symbols[0] if args.symbols and len(args.symbols) == 1 else None,
            interval=args.intervals[0] if args.intervals and len(args.intervals) == 1 else None
        )
    
    # If no results found
    if not results:
        logger.error("No results found matching the criteria")
        return {'error': 'No results found'}
    
    # Generate summary report
    summary = result_analyzer.generate_summary_report(results)
    summary_path = result_analyzer.save_summary_report(summary)
    
    # Generate comparison chart
    comparison_path = result_analyzer.compare_results(results, args.metric)
    
    # Add paths to summary
    summary['summary_path'] = summary_path
    if comparison_path:
        summary['comparison_chart'] = comparison_path
    
    return summary

def print_json_result(result: Dict):
    """Print result as formatted JSON"""
    print(json.dumps(result, indent=2))

async def main():
    """Main entry point"""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Execute the selected command
        if args.command == 'fetch':
            result = await fetch_data(args)
            print_json_result(result)
            
        elif args.command == 'backtest':
            result = await run_backtest(args)
            print_json_result(result)
            
        elif args.command == 'list':
            result = list_data(args)
            print_json_result(result)
            
        elif args.command == 'analyze':
            result = analyze_results(args)
            print_json_result(result)
        
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run the main function
    asyncio.run(main())