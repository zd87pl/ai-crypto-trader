import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging as logger

class ResultAnalyzer:
    """Analyze backtesting results for strategy optimization"""
    
    def __init__(self, results_dir: str = 'backtesting/results'):
        """Initialize with results directory"""
        self.results_dir = Path(results_dir)
        
        # Create plots directory
        self.plots_dir = Path('backtesting/plots')
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def load_results(self, result_path: Union[str, Path]) -> Dict:
        """Load a specific backtest result file"""
        try:
            with open(result_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading result file {result_path}: {str(e)}")
            return {}
    
    def get_available_results(self) -> List[Dict]:
        """Get list of available backtest results with metadata"""
        results = []
        
        for file_path in self.results_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # Add file path to result
                    data['file_path'] = str(file_path)
                    
                    results.append(data)
            except Exception as e:
                logger.error(f"Error loading result file {file_path}: {str(e)}")
        
        return results
    
    def filter_results(self, strategy: str = None, symbol: str = None, 
                     interval: str = None, min_trades: int = 0) -> List[Dict]:
        """Filter results by various criteria"""
        all_results = self.get_available_results()
        filtered = []
        
        for result in all_results:
            if strategy and result.get('strategy') != strategy:
                continue
                
            if symbol and result.get('symbol') != symbol:
                continue
                
            if interval and result.get('interval') != interval:
                continue
                
            if min_trades and result.get('stats', {}).get('total_trades', 0) < min_trades:
                continue
                
            filtered.append(result)
        
        return filtered
    
    def plot_equity_curve(self, result: Dict, save_path: Optional[str] = None) -> str:
        """Plot equity curve and drawdown from a backtest result"""
        try:
            stats = result.get('stats', {})
            
            # Extract equity curve data
            equity_data = stats.get('equity_curve', [])
            drawdown_data = stats.get('drawdown_curve', [])
            
            if not equity_data:
                logger.error("No equity curve data found in result")
                return None
                
            # Convert to DataFrame
            equity_df = pd.DataFrame(equity_data)
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            equity_df.set_index('timestamp', inplace=True)
            
            drawdown_df = pd.DataFrame(drawdown_data)
            if not drawdown_df.empty:
                drawdown_df['timestamp'] = pd.to_datetime(drawdown_df['timestamp'])
                drawdown_df.set_index('timestamp', inplace=True)
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot equity curve
            equity_df['equity'].plot(ax=ax1, color='blue', label='Account Value')
            ax1.set_title(f"Backtest Results: {result.get('strategy')} on {result.get('symbol')} ({result.get('interval')})")
            ax1.set_ylabel("Account Value")
            ax1.grid(True)
            ax1.legend()
            
            # Plot drawdown
            if not drawdown_df.empty:
                drawdown_df['drawdown_pct'].plot(ax=ax2, color='red', label='Drawdown %')
                ax2.set_ylabel("Drawdown %")
                ax2.set_xlabel("Date")
                ax2.grid(True)
                ax2.legend()
            
            # Add key statistics
            stats_text = (
                f"Initial Balance: ${stats.get('initial_balance', 0):.2f}\n"
                f"Final Balance: ${stats.get('final_balance', 0):.2f}\n"
                f"Total Return: {((stats.get('final_balance', 0) / stats.get('initial_balance', 1)) - 1) * 100:.2f}%\n"
                f"Trades: {stats.get('total_trades', 0)}\n"
                f"Win Rate: {stats.get('win_rate', 0):.2f}%\n"
                f"Profit Factor: {stats.get('profit_factor', 0):.2f}\n"
                f"Max Drawdown: {stats.get('max_drawdown_pct', 0):.2f}%\n"
                f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}"
            )
            
            # Add text box with statistics
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            
            # Save or show
            if save_path:
                plt.savefig(save_path)
                plt.close()
                return save_path
            else:
                # Generate filename automatically
                filename = f"{result.get('strategy')}_{result.get('symbol')}_{result.get('interval')}_equity.png"
                file_path = self.plots_dir / filename
                plt.savefig(file_path)
                plt.close()
                return str(file_path)
                
        except Exception as e:
            logger.error(f"Error plotting equity curve: {str(e)}")
            return None
    
    def plot_trade_analysis(self, result: Dict, save_path: Optional[str] = None) -> str:
        """Plot trade analysis (win/loss, trade duration, etc.)"""
        try:
            stats = result.get('stats', {})
            trades = stats.get('trades', [])
            
            if not trades:
                logger.error("No trade data found in result")
                return None
                
            # Convert to DataFrame
            trades_df = pd.DataFrame(trades)
            
            # Convert timestamps to datetime
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            
            # Calculate trade duration
            trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600  # Hours
            
            # Create win/loss column
            trades_df['result'] = trades_df['pnl'].apply(lambda x: 'Win' if x > 0 else 'Loss')
            
            # Create plot
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Trade P&L
            trades_df['pnl'].plot(kind='bar', ax=axs[0, 0], color=trades_df['pnl'].apply(lambda x: 'green' if x > 0 else 'red'))
            axs[0, 0].set_title('Trade P&L')
            axs[0, 0].set_xlabel('Trade #')
            axs[0, 0].set_ylabel('P&L ($)')
            axs[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axs[0, 0].grid(True)
            
            # Plot 2: Win/Loss Distribution
            win_loss_counts = trades_df['result'].value_counts()
            if not win_loss_counts.empty:
                win_loss_counts.plot(kind='pie', ax=axs[0, 1], autopct='%1.1f%%', colors=['green', 'red'])
                axs[0, 1].set_title('Win/Loss Distribution')
                axs[0, 1].set_ylabel('')
            
            # Plot 3: Trade Duration
            if 'duration' in trades_df.columns:
                trades_df['duration'].plot(kind='hist', ax=axs[1, 0], bins=20, alpha=0.7)
                axs[1, 0].set_title('Trade Duration Distribution')
                axs[1, 0].set_xlabel('Duration (hours)')
                axs[1, 0].set_ylabel('Frequency')
                axs[1, 0].grid(True)
            
            # Plot 4: P&L by Exit Reason
            if 'exit_reason' in trades_df.columns and 'pnl' in trades_df.columns:
                trades_df.groupby('exit_reason')['pnl'].sum().plot(kind='bar', ax=axs[1, 1])
                axs[1, 1].set_title('P&L by Exit Reason')
                axs[1, 1].set_xlabel('Exit Reason')
                axs[1, 1].set_ylabel('Total P&L ($)')
                axs[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Save or show
            if save_path:
                plt.savefig(save_path)
                plt.close()
                return save_path
            else:
                # Generate filename automatically
                filename = f"{result.get('strategy')}_{result.get('symbol')}_{result.get('interval')}_trades.png"
                file_path = self.plots_dir / filename
                plt.savefig(file_path)
                plt.close()
                return str(file_path)
                
        except Exception as e:
            logger.error(f"Error plotting trade analysis: {str(e)}")
            return None
    
    def generate_summary_report(self, results: List[Dict]) -> Dict:
        """Generate performance summary across multiple backtest results"""
        if not results:
            return {}
            
        # Extract and compile statistics
        summary = {
            'strategies': set(),
            'symbols': set(),
            'intervals': set(),
            'total_results': len(results),
            'average_win_rate': 0,
            'average_profit_factor': 0,
            'average_sharpe_ratio': 0,
            'best_result': None,
            'worst_result': None,
            'total_trades': 0,
            'profitable_strategies': 0,
            'results': []
        }
        
        # Process each result
        total_win_rate = 0
        total_profit_factor = 0
        total_sharpe_ratio = 0
        best_return = -float('inf')
        worst_return = float('inf')
        best_result_index = -1
        worst_result_index = -1
        
        for i, result in enumerate(results):
            # Extract basic info
            strategy = result.get('strategy', 'Unknown')
            symbol = result.get('symbol', 'Unknown')
            interval = result.get('interval', 'Unknown')
            
            summary['strategies'].add(strategy)
            summary['symbols'].add(symbol)
            summary['intervals'].add(interval)
            
            # Extract statistics
            stats = result.get('stats', {})
            initial_balance = stats.get('initial_balance', 0)
            final_balance = stats.get('final_balance', 0)
            
            # Calculate return
            if initial_balance > 0:
                return_pct = ((final_balance / initial_balance) - 1) * 100
            else:
                return_pct = 0
                
            # Check if profitable
            if final_balance > initial_balance:
                summary['profitable_strategies'] += 1
                
            # Update best/worst result
            if return_pct > best_return:
                best_return = return_pct
                best_result_index = i
                
            if return_pct < worst_return:
                worst_return = return_pct
                worst_result_index = i
            
            # Accumulate statistics
            summary['total_trades'] += stats.get('total_trades', 0)
            total_win_rate += stats.get('win_rate', 0)
            total_profit_factor += stats.get('profit_factor', 0)
            total_sharpe_ratio += stats.get('sharpe_ratio', 0)
            
            # Add to results list
            summary['results'].append({
                'strategy': strategy,
                'symbol': symbol,
                'interval': interval,
                'trades': stats.get('total_trades', 0),
                'win_rate': stats.get('win_rate', 0),
                'profit_factor': stats.get('profit_factor', 0),
                'sharpe_ratio': stats.get('sharpe_ratio', 0),
                'return_pct': return_pct,
                'max_drawdown': stats.get('max_drawdown_pct', 0),
                'file_path': result.get('file_path', '')
            })
        
        # Calculate averages
        if summary['total_results'] > 0:
            summary['average_win_rate'] = total_win_rate / summary['total_results']
            summary['average_profit_factor'] = total_profit_factor / summary['total_results']
            summary['average_sharpe_ratio'] = total_sharpe_ratio / summary['total_results']
        
        # Set best and worst results
        if best_result_index >= 0:
            summary['best_result'] = results[best_result_index]
        
        if worst_result_index >= 0:
            summary['worst_result'] = results[worst_result_index]
        
        # Convert sets to lists for JSON serialization
        summary['strategies'] = list(summary['strategies'])
        summary['symbols'] = list(summary['symbols'])
        summary['intervals'] = list(summary['intervals'])
        
        return summary
    
    def compare_results(self, results: List[Dict], metric: str = 'return_pct', 
                       save_path: Optional[str] = None) -> str:
        """Compare backtest results by a specific metric"""
        try:
            if not results:
                logger.error("No results provided for comparison")
                return None
                
            # Create a DataFrame for comparison
            comparison_data = []
            
            for result in results:
                stats = result.get('stats', {})
                initial_balance = stats.get('initial_balance', 0)
                final_balance = stats.get('final_balance', 0)
                
                # Calculate return percentage
                if initial_balance > 0:
                    return_pct = ((final_balance / initial_balance) - 1) * 100
                else:
                    return_pct = 0
                
                comparison_data.append({
                    'strategy': result.get('strategy', 'Unknown'),
                    'symbol': result.get('symbol', 'Unknown'),
                    'interval': result.get('interval', 'Unknown'),
                    'win_rate': stats.get('win_rate', 0),
                    'profit_factor': stats.get('profit_factor', 0),
                    'sharpe_ratio': stats.get('sharpe_ratio', 0),
                    'max_drawdown': stats.get('max_drawdown_pct', 0),
                    'return_pct': return_pct,
                    'total_trades': stats.get('total_trades', 0)
                })
            
            # Convert to DataFrame
            df = pd.DataFrame(comparison_data)
            
            # Create comparison chart
            plt.figure(figsize=(14, 8))
            
            # Create label for x-axis based on strategy and symbol
            df['label'] = df.apply(lambda row: f"{row['strategy']}\n{row['symbol']}\n{row['interval']}", axis=1)
            
            # Plot the specified metric
            if metric in df.columns:
                colors = ['green' if x >= 0 else 'red' for x in df[metric]]
                bars = df[metric].plot(kind='bar', color=colors)
                plt.title(f'Comparison by {metric}')
                plt.xlabel('Strategy / Symbol / Interval')
                plt.ylabel(metric)
                plt.xticks(range(len(df)), df['label'], rotation=45)
                plt.grid(True, axis='y')
                
                # Add value labels on bars
                for bar in bars.patches:
                    height = bar.get_height()
                    if np.isnan(height):
                        continue
                    plt.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height + 0.02 * max(df[metric].max(), abs(df[metric].min())),
                        f'{height:.2f}',
                        ha='center', va='bottom', rotation=0
                    )
            else:
                logger.error(f"Metric {metric} not found in result data")
                return None
            
            plt.tight_layout()
            
            # Save or show
            if save_path:
                plt.savefig(save_path)
                plt.close()
                return save_path
            else:
                # Generate filename automatically
                filename = f"comparison_{metric}.png"
                file_path = self.plots_dir / filename
                plt.savefig(file_path)
                plt.close()
                return str(file_path)
                
        except Exception as e:
            logger.error(f"Error comparing results: {str(e)}")
            return None
    
    def save_summary_report(self, summary: Dict, filename: str = "backtest_summary.json") -> str:
        """Save summary report to file"""
        file_path = self.results_dir / filename
        
        try:
            with open(file_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            return str(file_path)
        except Exception as e:
            logger.error(f"Error saving summary report: {str(e)}")
            return None