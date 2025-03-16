"""
Strategy Evaluation System

This module provides tools for systematically evaluating trading strategies
across different market conditions, timeframes, and parameters.
It includes metrics calculation, statistical analysis, and visualization tools.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from copy import deepcopy

# Configure logging
logger = logging.getLogger("strategy_evaluation")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler('logs/strategy_evaluation.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [StrategyEval] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class StrategyPerformanceMetrics:
    """Calculate and analyze performance metrics for trading strategies."""
    
    @staticmethod
    def calculate_metrics(trades: List[Dict], initial_capital: float = 10000.0) -> Dict:
        """
        Calculate comprehensive performance metrics from a list of trades.
        
        Args:
            trades: List of trade dictionaries with keys: timestamp, symbol, side, price, quantity, fees, pnl
            initial_capital: Initial capital amount
            
        Returns:
            Dictionary of performance metrics
        """
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "average_profit": 0.0,
                "average_loss": 0.0,
                "largest_profit": 0.0,
                "largest_loss": 0.0,
                "total_profit": 0.0,
                "total_loss": 0.0,
                "net_profit": 0.0,
                "return_pct": 0.0,
                "avg_trade_duration": 0,
                "risk_reward_ratio": 0.0,
                "profitable_symbols": {},
                "unprofitable_symbols": {},
                "monthly_returns": {},
                "daily_returns": {}
            }
        
        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda x: x.get('timestamp', ''))
        
        # Check if there's enough data to calculate metrics
        if len(sorted_trades) < 2:
            logger.warning("Not enough trades to calculate meaningful metrics")
            return {
                "total_trades": len(sorted_trades),
                "win_rate": 0.0 if len(sorted_trades) == 0 else 1.0 if sorted_trades[0].get('pnl', 0) > 0 else 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "average_profit": sorted_trades[0].get('pnl', 0) if len(sorted_trades) > 0 and sorted_trades[0].get('pnl', 0) > 0 else 0.0,
                "average_loss": sorted_trades[0].get('pnl', 0) if len(sorted_trades) > 0 and sorted_trades[0].get('pnl', 0) < 0 else 0.0,
                "largest_profit": sorted_trades[0].get('pnl', 0) if len(sorted_trades) > 0 and sorted_trades[0].get('pnl', 0) > 0 else 0.0,
                "largest_loss": sorted_trades[0].get('pnl', 0) if len(sorted_trades) > 0 and sorted_trades[0].get('pnl', 0) < 0 else 0.0,
                "total_profit": sorted_trades[0].get('pnl', 0) if len(sorted_trades) > 0 and sorted_trades[0].get('pnl', 0) > 0 else 0.0,
                "total_loss": sorted_trades[0].get('pnl', 0) if len(sorted_trades) > 0 and sorted_trades[0].get('pnl', 0) < 0 else 0.0,
                "net_profit": sorted_trades[0].get('pnl', 0) if len(sorted_trades) > 0 else 0.0,
                "return_pct": (sorted_trades[0].get('pnl', 0) / initial_capital) * 100 if len(sorted_trades) > 0 else 0.0,
                "avg_trade_duration": 0,
                "risk_reward_ratio": 0.0,
                "profitable_symbols": {},
                "unprofitable_symbols": {},
                "monthly_returns": {},
                "daily_returns": {}
            }
        
        # Initialize metrics
        total_trades = len(sorted_trades)
        winning_trades = [t for t in sorted_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in sorted_trades if t.get('pnl', 0) < 0]
        
        # Calculate basic metrics
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t.get('pnl', 0) for t in winning_trades)
        total_loss = sum(t.get('pnl', 0) for t in losing_trades)
        net_profit = total_profit + total_loss
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        average_profit = total_profit / win_count if win_count > 0 else 0
        average_loss = total_loss / loss_count if loss_count > 0 else 0
        
        largest_profit = max([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.get('pnl', 0) for t in losing_trades]) if losing_trades else 0
        
        risk_reward_ratio = abs(average_profit / average_loss) if average_loss != 0 else float('inf')
        
        # Calculate equity curve and drawdown
        equity_curve = [initial_capital]
        drawdowns = []
        daily_returns = {}
        monthly_returns = {}
        
        # Track peak equity to calculate drawdowns
        peak_equity = initial_capital
        current_drawdown = 0
        
        for trade in sorted_trades:
            # Update equity
            current_equity = equity_curve[-1] + trade.get('pnl', 0)
            equity_curve.append(current_equity)
            
            # Update peak equity and calculate drawdown
            if current_equity > peak_equity:
                peak_equity = current_equity
                current_drawdown = 0
            else:
                current_drawdown = (peak_equity - current_equity) / peak_equity
                drawdowns.append(current_drawdown)
            
            # Calculate daily and monthly returns
            date_str = trade.get('timestamp', '')
            if date_str:
                try:
                    date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    day_key = date.strftime('%Y-%m-%d')
                    month_key = date.strftime('%Y-%m')
                    
                    # Update daily returns
                    if day_key not in daily_returns:
                        daily_returns[day_key] = trade.get('pnl', 0)
                    else:
                        daily_returns[day_key] += trade.get('pnl', 0)
                    
                    # Update monthly returns
                    if month_key not in monthly_returns:
                        monthly_returns[month_key] = trade.get('pnl', 0)
                    else:
                        monthly_returns[month_key] += trade.get('pnl', 0)
                except Exception as e:
                    logger.warning(f"Error parsing trade timestamp: {e}")
        
        # Calculate max drawdown
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        # Calculate average trade duration
        durations = []
        for i in range(0, len(sorted_trades), 2):
            if i + 1 < len(sorted_trades):
                try:
                    entry_time = datetime.fromisoformat(sorted_trades[i].get('timestamp', '').replace('Z', '+00:00'))
                    exit_time = datetime.fromisoformat(sorted_trades[i+1].get('timestamp', '').replace('Z', '+00:00'))
                    duration = (exit_time - entry_time).total_seconds() / 60  # Duration in minutes
                    durations.append(duration)
                except Exception as e:
                    logger.warning(f"Error calculating trade duration: {e}")
        
        avg_trade_duration = sum(durations) / len(durations) if durations else 0
        
        # Calculate daily returns for Sharpe ratio
        daily_return_values = list(daily_returns.values())
        if len(daily_return_values) > 1:
            avg_daily_return = np.mean(daily_return_values)
            std_daily_return = np.std(daily_return_values)
            # Annualized Sharpe ratio (assuming 252 trading days per year)
            sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate profitable/unprofitable symbols
        symbol_pnl = {}
        for trade in sorted_trades:
            symbol = trade.get('symbol', 'UNKNOWN')
            if symbol not in symbol_pnl:
                symbol_pnl[symbol] = 0
            symbol_pnl[symbol] += trade.get('pnl', 0)
        
        profitable_symbols = {s: pnl for s, pnl in symbol_pnl.items() if pnl > 0}
        unprofitable_symbols = {s: pnl for s, pnl in symbol_pnl.items() if pnl <= 0}
        
        # Calculate return percentage
        return_pct = (net_profit / initial_capital) * 100
        
        # Compile metrics into dictionary
        metrics = {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "average_profit": average_profit,
            "average_loss": average_loss,
            "largest_profit": largest_profit,
            "largest_loss": largest_loss,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "net_profit": net_profit,
            "return_pct": return_pct,
            "avg_trade_duration": avg_trade_duration,
            "risk_reward_ratio": risk_reward_ratio,
            "profitable_symbols": profitable_symbols,
            "unprofitable_symbols": unprofitable_symbols,
            "monthly_returns": monthly_returns,
            "daily_returns": daily_returns,
            "equity_curve": equity_curve
        }
        
        return metrics
    
    @staticmethod
    def calculate_advanced_metrics(metrics: Dict) -> Dict:
        """
        Calculate advanced performance metrics based on existing metrics.
        
        Args:
            metrics: Dictionary of basic performance metrics
            
        Returns:
            Dictionary with additional advanced metrics
        """
        advanced_metrics = deepcopy(metrics)
        
        # Calculate Calmar ratio (annualized return / max drawdown)
        if metrics.get('max_drawdown', 0) > 0:
            annualized_return = metrics.get('return_pct', 0) / 100  # Convert to decimal
            calmar_ratio = annualized_return / metrics.get('max_drawdown', 1)
            advanced_metrics['calmar_ratio'] = calmar_ratio
        else:
            advanced_metrics['calmar_ratio'] = float('inf')
        
        # Calculate Sortino ratio (similar to Sharpe but penalizes only downside volatility)
        daily_returns = list(metrics.get('daily_returns', {}).values())
        if daily_returns:
            avg_daily_return = np.mean(daily_returns)
            # Calculate downside deviation (standard deviation of negative returns only)
            negative_returns = [r for r in daily_returns if r < 0]
            downside_deviation = np.std(negative_returns) if negative_returns else 0
            
            # Annualized Sortino ratio
            sortino_ratio = (avg_daily_return / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else float('inf')
            advanced_metrics['sortino_ratio'] = sortino_ratio
        else:
            advanced_metrics['sortino_ratio'] = 0
        
        # Calculate Maximum Consecutive Wins and Losses
        if metrics.get('total_trades', 0) > 0:
            trades = metrics.get('trades', [])
            
            # If trades data is available
            if trades:
                # Sort trades by timestamp
                sorted_trades = sorted(trades, key=lambda x: x.get('timestamp', ''))
                
                # Track win/loss streaks
                current_win_streak = 0
                max_win_streak = 0
                current_loss_streak = 0
                max_loss_streak = 0
                
                for trade in sorted_trades:
                    if trade.get('pnl', 0) > 0:
                        # Winning trade
                        current_win_streak += 1
                        current_loss_streak = 0
                        max_win_streak = max(max_win_streak, current_win_streak)
                    else:
                        # Losing trade
                        current_loss_streak += 1
                        current_win_streak = 0
                        max_loss_streak = max(max_loss_streak, current_loss_streak)
                
                advanced_metrics['max_consecutive_wins'] = max_win_streak
                advanced_metrics['max_consecutive_losses'] = max_loss_streak
        
        # Calculate Recovery Factor (net profit / max drawdown)
        if metrics.get('max_drawdown', 0) > 0:
            recovery_factor = metrics.get('net_profit', 0) / (metrics.get('max_drawdown', 1) * metrics.get('initial_capital', 10000))
            advanced_metrics['recovery_factor'] = recovery_factor
        else:
            advanced_metrics['recovery_factor'] = float('inf')
        
        # Calculate Expectancy (average trade profit/loss)
        if metrics.get('total_trades', 0) > 0:
            win_rate = metrics.get('win_rate', 0)
            avg_win = metrics.get('average_profit', 0)
            avg_loss = abs(metrics.get('average_loss', 0))
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            advanced_metrics['expectancy'] = expectancy
        else:
            advanced_metrics['expectancy'] = 0
        
        # Calculate Profit Per Day
        if daily_returns:
            profit_per_day = np.mean(daily_returns)
            advanced_metrics['profit_per_day'] = profit_per_day
        else:
            advanced_metrics['profit_per_day'] = 0
        
        return advanced_metrics
    
    @staticmethod
    def generate_performance_report(metrics: Dict, filename: Optional[str] = None) -> str:
        """
        Generate a detailed performance report from metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            filename: If provided, save the report to this file
            
        Returns:
            Performance report as a string
        """
        report = "========== STRATEGY PERFORMANCE REPORT ==========\n\n"
        
        # Overall performance
        report += "OVERALL PERFORMANCE:\n"
        report += f"Total Trades: {metrics.get('total_trades', 0)}\n"
        report += f"Win Rate: {metrics.get('win_rate', 0)*100:.2f}%\n"
        report += f"Net Profit: ${metrics.get('net_profit', 0):.2f}\n"
        report += f"Return: {metrics.get('return_pct', 0):.2f}%\n"
        report += f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
        report += f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%\n\n"
        
        # Risk/Reward metrics
        report += "RISK & REWARD METRICS:\n"
        report += f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
        report += f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}\n"
        report += f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}\n"
        report += f"Risk/Reward Ratio: {metrics.get('risk_reward_ratio', 0):.2f}\n"
        report += f"Expectancy: ${metrics.get('expectancy', 0):.2f}\n\n"
        
        # Trade statistics
        report += "TRADE STATISTICS:\n"
        report += f"Average Profit: ${metrics.get('average_profit', 0):.2f}\n"
        report += f"Average Loss: ${metrics.get('average_loss', 0):.2f}\n"
        report += f"Largest Profit: ${metrics.get('largest_profit', 0):.2f}\n"
        report += f"Largest Loss: ${metrics.get('largest_loss', 0):.2f}\n"
        report += f"Average Trade Duration: {metrics.get('avg_trade_duration', 0):.1f} minutes\n\n"
        
        # Symbol performance
        report += "SYMBOL PERFORMANCE:\n"
        profitable_symbols = metrics.get('profitable_symbols', {})
        unprofitable_symbols = metrics.get('unprofitable_symbols', {})
        
        report += "Profitable Symbols:\n"
        for symbol, pnl in sorted(profitable_symbols.items(), key=lambda x: x[1], reverse=True):
            report += f"  {symbol}: ${pnl:.2f}\n"
        
        report += "\nUnprofitable Symbols:\n"
        for symbol, pnl in sorted(unprofitable_symbols.items(), key=lambda x: x[1]):
            report += f"  {symbol}: ${pnl:.2f}\n"
        
        report += "\n"
        
        # Monthly performance
        report += "MONTHLY PERFORMANCE:\n"
        monthly_returns = metrics.get('monthly_returns', {})
        
        for month, pnl in sorted(monthly_returns.items()):
            report += f"  {month}: ${pnl:.2f}\n"
        
        report += "\n========== END OF REPORT ==========\n"
        
        # Save to file if filename provided
        if filename:
            with open(filename, 'w') as f:
                f.write(report)
            logger.info(f"Performance report saved to {filename}")
        
        return report
    
    @staticmethod
    def visualize_performance(metrics: Dict, output_dir: str = "reports") -> None:
        """
        Generate visualizations of strategy performance.
        
        Args:
            metrics: Dictionary of performance metrics
            output_dir: Directory to save the visualizations
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot equity curve
        equity_curve = metrics.get('equity_curve', [10000])
        axes[0, 0].plot(equity_curve)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Trades')
        axes[0, 0].set_ylabel('Account Value ($)')
        axes[0, 0].grid(True)
        
        # Plot monthly returns
        monthly_returns = metrics.get('monthly_returns', {})
        months = list(monthly_returns.keys())
        returns = list(monthly_returns.values())
        
        if months:
            colors = ['green' if r > 0 else 'red' for r in returns]
            axes[0, 1].bar(months, returns, color=colors)
            axes[0, 1].set_title('Monthly Returns')
            axes[0, 1].set_xlabel('Month')
            axes[0, 1].set_ylabel('Profit/Loss ($)')
            axes[0, 1].grid(True)
            plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # Plot win/loss pie chart
        win_rate = metrics.get('win_rate', 0)
        lose_rate = 1 - win_rate
        
        axes[1, 0].pie([win_rate, lose_rate], labels=['Wins', 'Losses'], 
                      autopct='%1.1f%%', colors=['green', 'red'])
        axes[1, 0].set_title('Win/Loss Ratio')
        
        # Plot symbol performance
        profitable_symbols = metrics.get('profitable_symbols', {})
        unprofitable_symbols = metrics.get('unprofitable_symbols', {})
        
        all_symbols = {}
        all_symbols.update(profitable_symbols)
        all_symbols.update(unprofitable_symbols)
        
        symbols = list(all_symbols.keys())
        performance = list(all_symbols.values())
        
        if symbols:
            colors = ['green' if p > 0 else 'red' for p in performance]
            axes[1, 1].bar(symbols, performance, color=colors)
            axes[1, 1].set_title('Symbol Performance')
            axes[1, 1].set_xlabel('Symbol')
            axes[1, 1].set_ylabel('Profit/Loss ($)')
            axes[1, 1].grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_summary.png'))
        
        # Create additional visualizations
        
        # Daily returns histogram
        daily_returns = list(metrics.get('daily_returns', {}).values())
        if daily_returns:
            plt.figure(figsize=(10, 6))
            plt.hist(daily_returns, bins=20, alpha=0.7, color='blue')
            plt.axvline(x=0, color='red', linestyle='--')
            plt.title('Distribution of Daily Returns')
            plt.xlabel('Daily Return ($)')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'daily_returns_distribution.png'))
        
        # Drawdown chart
        if len(equity_curve) > 1:
            drawdowns = []
            peak = equity_curve[0]
            
            for equity in equity_curve:
                if equity > peak:
                    peak = equity
                    drawdowns.append(0)
                else:
                    drawdown = (peak - equity) / peak
                    drawdowns.append(drawdown)
            
            plt.figure(figsize=(10, 6))
            plt.plot(drawdowns, color='red')
            plt.fill_between(range(len(drawdowns)), drawdowns, 0, color='red', alpha=0.3)
            plt.title('Drawdown Over Time')
            plt.xlabel('Trades')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'drawdown_chart.png'))
        
        logger.info(f"Performance visualizations saved to {output_dir}")


class StrategyEvaluationSystem:
    """
    System for comprehensive evaluation of trading strategies.
    Handles backtesting, optimization, and comparison of strategies.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the strategy evaluation system.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Extract evaluation parameters from config
        self.optimization_goals = self.config.get('evolution', {}).get('optimization_goals', {})
        self.performance_metrics = self.config.get('evolution', {}).get('performance_metrics', {})
        
        # Set up data directories
        self.data_dir = "data/evaluation"
        self.reports_dir = "reports/evaluation"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        logger.info("Strategy Evaluation System initialized")
    
    def evaluate_strategy(self, strategy_id: str, trades: List[Dict], 
                          parameters: Dict, market_conditions: Dict) -> Dict:
        """
        Evaluate a strategy's performance based on trade history.
        
        Args:
            strategy_id: Identifier for the strategy
            trades: List of trade dictionaries
            parameters: Strategy parameters used
            market_conditions: Market context during evaluation
            
        Returns:
            Dictionary of evaluation results
        """
        # Calculate basic metrics
        metrics = StrategyPerformanceMetrics.calculate_metrics(trades)
        
        # Calculate advanced metrics
        advanced_metrics = StrategyPerformanceMetrics.calculate_advanced_metrics(metrics)
        
        # Generate performance report
        report_path = os.path.join(self.reports_dir, f"{strategy_id}_report.txt")
        report = StrategyPerformanceMetrics.generate_performance_report(advanced_metrics, report_path)
        
        # Generate visualizations
        viz_dir = os.path.join(self.reports_dir, strategy_id)
        StrategyPerformanceMetrics.visualize_performance(advanced_metrics, viz_dir)
        
        # Calculate strategy score based on optimization goals
        score = self._calculate_strategy_score(advanced_metrics)
        
        # Compile evaluation results
        evaluation_results = {
            "strategy_id": strategy_id,
            "score": score,
            "metrics": advanced_metrics,
            "parameters": parameters,
            "market_conditions": market_conditions,
            "timestamp": datetime.now().isoformat(),
            "report_path": report_path,
            "visualization_dir": viz_dir
        }
        
        # Save evaluation results
        results_path = os.path.join(self.data_dir, f"{strategy_id}_evaluation.json")
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        logger.info(f"Strategy {strategy_id} evaluated, score: {score:.4f}")
        return evaluation_results
    
    def _calculate_strategy_score(self, metrics: Dict) -> float:
        """
        Calculate an overall score for a strategy based on optimization goals.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Strategy score as a float
        """
        # Get primary metric
        primary_metric = self.optimization_goals.get('primary', 'sharpe_ratio')
        primary_value = metrics.get(primary_metric, 0)
        
        # Get secondary metrics
        secondary_metrics = self.optimization_goals.get('secondary', [])
        
        # Calculate primary score
        primary_score = primary_value
        
        # Apply adjustments from secondary metrics
        score = primary_score
        
        for metric in secondary_metrics:
            if metric == 'max_drawdown':
                # Lower drawdown is better, so penalize the score
                max_dd = metrics.get('max_drawdown', 0)
                score *= (1 - max_dd)
            elif metric == 'win_rate':
                # Higher win rate is better
                win_rate = metrics.get('win_rate', 0)
                score *= (1 + win_rate)
            elif metric == 'profit_factor':
                # Higher profit factor is better
                profit_factor = metrics.get('profit_factor', 1)
                score *= (profit_factor / 2)  # Scale down the impact
            elif metric == 'expectancy':
                # Higher expectancy is better
                expectancy = metrics.get('expectancy', 0)
                # Normalize expectancy to a reasonable range
                norm_expectancy = min(expectancy / 100, 1)
                score *= (1 + norm_expectancy)
        
        # Apply constraints
        constraints = self.optimization_goals.get('constraints', {})
        
        min_trades_per_day = constraints.get('min_trades_per_day', 0)
        trades_per_day = metrics.get('trades_per_day', min_trades_per_day)
        
        if trades_per_day < min_trades_per_day:
            # Penalize if below minimum trades per day
            penalty = trades_per_day / min_trades_per_day
            score *= penalty
        
        return score
        
    def cross_validate_strategy(self, strategy_id: str, parameters: Dict, 
                               market_data_periods: List[Dict], k_folds: int = 5, 
                               test_metric: str = 'sharpe_ratio',
                               normalize_results: bool = True) -> Dict:
        """
        Perform k-fold cross-validation on a strategy across different market conditions.
        
        Args:
            strategy_id: Identifier for the strategy
            parameters: Strategy parameters to test
            market_data_periods: List of market data periods to use for validation
            k_folds: Number of validation folds to use
            test_metric: Primary metric to use for evaluation
            normalize_results: Whether to normalize results across folds
            
        Returns:
            Dictionary of cross-validation results
        """
        logger.info(f"Performing {k_folds}-fold cross-validation for strategy {strategy_id}")
        
        if len(market_data_periods) < k_folds:
            logger.warning(f"Not enough market data periods ({len(market_data_periods)}) for {k_folds} folds")
            k_folds = len(market_data_periods)
        
        # Split data into k folds
        fold_size = len(market_data_periods) // k_folds
        folds = []
        
        for i in range(k_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < k_folds - 1 else len(market_data_periods)
            folds.append(market_data_periods[start_idx:end_idx])
        
        # Track results for each fold
        fold_results = []
        
        # Perform cross-validation
        for fold_idx, test_fold in enumerate(folds):
            logger.info(f"Evaluating fold {fold_idx+1}/{k_folds}")
            
            # Combine all other folds for training
            train_folds = []
            for i in range(k_folds):
                if i != fold_idx:
                    train_folds.extend(folds[i])
            
            # Perform backtesting on training data
            train_trades = self._simulate_trades(strategy_id, parameters, train_folds)
            train_metrics = StrategyPerformanceMetrics.calculate_metrics(train_trades)
            
            # Perform backtesting on test data
            test_trades = self._simulate_trades(strategy_id, parameters, test_fold)
            test_metrics = StrategyPerformanceMetrics.calculate_metrics(test_trades)
            
            # Calculate fold score
            train_score = self._calculate_strategy_score(train_metrics)
            test_score = self._calculate_strategy_score(test_metrics)
            
            # Store fold results
            fold_result = {
                "fold": fold_idx + 1,
                "train_metrics": {
                    "sharpe_ratio": train_metrics.get('sharpe_ratio', 0),
                    "win_rate": train_metrics.get('win_rate', 0),
                    "max_drawdown": train_metrics.get('max_drawdown', 0),
                    "profit_factor": train_metrics.get('profit_factor', 0),
                    "return_pct": train_metrics.get('return_pct', 0),
                    "test_metric": train_metrics.get(test_metric, 0)
                },
                "test_metrics": {
                    "sharpe_ratio": test_metrics.get('sharpe_ratio', 0),
                    "win_rate": test_metrics.get('win_rate', 0),
                    "max_drawdown": test_metrics.get('max_drawdown', 0),
                    "profit_factor": test_metrics.get('profit_factor', 0),
                    "return_pct": test_metrics.get('return_pct', 0),
                    "test_metric": test_metrics.get(test_metric, 0)
                },
                "train_score": train_score,
                "test_score": test_score,
                "market_conditions": self._summarize_market_conditions(test_fold)
            }
            
            fold_results.append(fold_result)
        
        # Calculate cross-validation summary statistics
        cv_summary = self._calculate_cv_summary(fold_results, test_metric, normalize_results)
        
        # Compile cross-validation results
        cv_results = {
            "strategy_id": strategy_id,
            "parameters": parameters,
            "k_folds": k_folds,
            "test_metric": test_metric,
            "cv_summary": cv_summary,
            "fold_results": fold_results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store cross-validation results
        results_path = os.path.join(self.data_dir, f"{strategy_id}_cv_results.json")
        with open(results_path, 'w') as f:
            json.dump(cv_results, f, indent=2, default=str)
        
        # Generate visualization of cross-validation results
        self._visualize_cv_results(cv_results)
        
        logger.info(f"Cross-validation complete for strategy {strategy_id}")
        logger.info(f"CV Score: {cv_summary['mean_test_score']:.4f} (±{cv_summary['std_test_score']:.4f})")
        
        return cv_results
    
    def _simulate_trades(self, strategy_id: str, parameters: Dict, market_data: List[Dict]) -> List[Dict]:
        """
        Simulate trades based on strategy parameters and market data.
        
        Args:
            strategy_id: Identifier for the strategy
            parameters: Strategy parameters to use
            market_data: Market data to simulate trades on
            
        Returns:
            List of simulated trades
        """
        # This would be a real backtesting simulation in a full implementation
        # For now, we'll generate synthetic trades based on parameters and market data
        
        trades = []
        initial_capital = 10000
        position_size_pct = min(parameters.get('max_position_size', 5), 20) / 100
        position_size = initial_capital * position_size_pct
        
        current_position = None
        entry_price = 0
        
        # Strategy parameters
        rsi_period = parameters.get('rsi_period', 14)
        rsi_overbought = parameters.get('rsi_overbought', 70)
        rsi_oversold = parameters.get('rsi_oversold', 30)
        take_profit = parameters.get('take_profit', 3) / 100
        stop_loss = parameters.get('stop_loss', 2) / 100
        
        # Simulate trades based on strategy rules
        for i, data_point in enumerate(market_data):
            timestamp = data_point.get('timestamp', f"2023-01-{i+1:02d}T00:00:00Z")
            symbol = data_point.get('symbol', 'BTCUSDT')
            price = data_point.get('price', 50000)
            rsi = data_point.get('rsi', 50)
            
            # Entry logic
            if current_position is None:
                if rsi < rsi_oversold:
                    # Buy signal
                    current_position = "long"
                    entry_price = price
                    quantity = position_size / price
                    trades.append({
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "side": "buy",
                        "price": price,
                        "quantity": quantity,
                        "fees": position_size * 0.001,  # 0.1% fees
                        "pnl": -position_size * 0.001  # Initial PnL is negative due to fees
                    })
                elif rsi > rsi_overbought:
                    # Sell signal
                    current_position = "short"
                    entry_price = price
                    quantity = position_size / price
                    trades.append({
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "side": "sell",
                        "price": price,
                        "quantity": quantity,
                        "fees": position_size * 0.001,  # 0.1% fees
                        "pnl": -position_size * 0.001  # Initial PnL is negative due to fees
                    })
            
            # Exit logic
            elif current_position == "long":
                profit_pct = (price - entry_price) / entry_price
                if profit_pct >= take_profit or profit_pct <= -stop_loss or rsi > rsi_overbought:
                    # Close long position
                    quantity = position_size / entry_price
                    pnl = quantity * (price - entry_price) - position_size * 0.002  # PnL minus fees
                    trades.append({
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "side": "sell",
                        "price": price,
                        "quantity": quantity,
                        "fees": position_size * 0.001,
                        "pnl": pnl
                    })
                    current_position = None
            
            elif current_position == "short":
                profit_pct = (entry_price - price) / entry_price
                if profit_pct >= take_profit or profit_pct <= -stop_loss or rsi < rsi_oversold:
                    # Close short position
                    quantity = position_size / entry_price
                    pnl = quantity * (entry_price - price) - position_size * 0.002  # PnL minus fees
                    trades.append({
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "side": "buy",
                        "price": price,
                        "quantity": quantity,
                        "fees": position_size * 0.001,
                        "pnl": pnl
                    })
                    current_position = None
        
        # Force close any open positions at the end
        if current_position == "long":
            price = market_data[-1].get('price', 50000)
            quantity = position_size / entry_price
            pnl = quantity * (price - entry_price) - position_size * 0.002
            trades.append({
                "timestamp": market_data[-1].get('timestamp', "2023-01-31T00:00:00Z"),
                "symbol": market_data[-1].get('symbol', 'BTCUSDT'),
                "side": "sell",
                "price": price,
                "quantity": quantity,
                "fees": position_size * 0.001,
                "pnl": pnl
            })
        
        elif current_position == "short":
            price = market_data[-1].get('price', 50000)
            quantity = position_size / entry_price
            pnl = quantity * (entry_price - price) - position_size * 0.002
            trades.append({
                "timestamp": market_data[-1].get('timestamp', "2023-01-31T00:00:00Z"),
                "symbol": market_data[-1].get('symbol', 'BTCUSDT'),
                "side": "buy",
                "price": price,
                "quantity": quantity,
                "fees": position_size * 0.001,
                "pnl": pnl
            })
        
        return trades
    
    def _summarize_market_conditions(self, market_data: List[Dict]) -> Dict:
        """
        Summarize market conditions from a period of market data.
        
        Args:
            market_data: List of market data points
            
        Returns:
            Dictionary summarizing market conditions
        """
        if not market_data:
            return {
                "trend": "unknown",
                "volatility": 0,
                "volume": 0,
                "period_start": None,
                "period_end": None
            }
        
        # Extract price and volume data
        prices = [data.get('price', 0) for data in market_data if data.get('price', 0) > 0]
        volumes = [data.get('volume', 0) for data in market_data if data.get('volume', 0) > 0]
        
        # Determine trend
        if len(prices) >= 2:
            price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
            if price_change > 0.05:  # 5% up
                trend = "uptrend"
            elif price_change < -0.05:  # 5% down
                trend = "downtrend"
            else:
                trend = "ranging"
        else:
            trend = "unknown"
        
        # Calculate volatility (standard deviation of price changes)
        if len(prices) >= 2:
            price_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = np.std(price_changes) if price_changes else 0
        else:
            volatility = 0
        
        # Calculate average volume
        avg_volume = np.mean(volumes) if volumes else 0
        
        # Get period timestamps
        period_start = market_data[0].get('timestamp', None) if market_data else None
        period_end = market_data[-1].get('timestamp', None) if market_data else None
        
        return {
            "trend": trend,
            "volatility": float(volatility),
            "volume": float(avg_volume),
            "period_start": period_start,
            "period_end": period_end
        }
    
    def _calculate_cv_summary(self, fold_results: List[Dict], test_metric: str, 
                             normalize: bool) -> Dict:
        """
        Calculate summary statistics from cross-validation results.
        
        Args:
            fold_results: List of results from each fold
            test_metric: Primary metric used for evaluation
            normalize: Whether to normalize results across folds
            
        Returns:
            Dictionary of summary statistics
        """
        train_scores = [fold['train_score'] for fold in fold_results]
        test_scores = [fold['test_score'] for fold in fold_results]
        
        train_metrics = {}
        test_metrics = {}
        
        # Extract all metrics from first fold to initialize
        if fold_results:
            for metric in fold_results[0]['train_metrics']:
                train_metrics[metric] = []
                test_metrics[metric] = []
        
        # Collect all metrics across folds
        for fold in fold_results:
            for metric, value in fold['train_metrics'].items():
                train_metrics[metric].append(value)
            
            for metric, value in fold['test_metrics'].items():
                test_metrics[metric].append(value)
        
        # Calculate summary statistics
        summary = {
            "mean_train_score": float(np.mean(train_scores)),
            "std_train_score": float(np.std(train_scores)),
            "mean_test_score": float(np.mean(test_scores)),
            "std_test_score": float(np.std(test_scores)),
            "train_test_gap": float(np.mean(train_scores) - np.mean(test_scores)),
            "relative_overfitting": float((np.mean(train_scores) - np.mean(test_scores)) / np.mean(train_scores)) if np.mean(train_scores) > 0 else 0
        }
        
        # Calculate mean and std for each metric
        for metric in train_metrics:
            summary[f"mean_train_{metric}"] = float(np.mean(train_metrics[metric]))
            summary[f"std_train_{metric}"] = float(np.std(train_metrics[metric]))
            
        for metric in test_metrics:
            summary[f"mean_test_{metric}"] = float(np.mean(test_metrics[metric]))
            summary[f"std_test_{metric}"] = float(np.std(test_metrics[metric]))
        
        # Calculate consistency score (how consistent results are across folds)
        summary["consistency"] = 1.0 - min(1.0, summary["std_test_score"] / summary["mean_test_score"]) if summary["mean_test_score"] > 0 else 0.0
        
        return summary
    
    def _visualize_cv_results(self, cv_results: Dict) -> None:
        """
        Generate visualizations for cross-validation results.
        
        Args:
            cv_results: Dictionary of cross-validation results
        """
        strategy_id = cv_results["strategy_id"]
        fold_results = cv_results["fold_results"]
        k_folds = cv_results["k_folds"]
        test_metric = cv_results["test_metric"]
        cv_summary = cv_results["cv_summary"]
        
        # Create visualization directory
        viz_dir = os.path.join(self.reports_dir, f"{strategy_id}_cv")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Extract data for plotting
        folds = list(range(1, k_folds + 1))
        train_scores = [fold['train_score'] for fold in fold_results]
        test_scores = [fold['test_score'] for fold in fold_results]
        
        train_metrics = {metric: [] for metric in fold_results[0]['train_metrics']} if fold_results else {}
        test_metrics = {metric: [] for metric in fold_results[0]['test_metrics']} if fold_results else {}
        
        for fold in fold_results:
            for metric, value in fold['train_metrics'].items():
                train_metrics[metric].append(value)
            
            for metric, value in fold['test_metrics'].items():
                test_metrics[metric].append(value)
        
        # 1. Train vs Test Score by Fold
        plt.figure(figsize=(12, 6))
        plt.plot(folds, train_scores, 'o-', label='Train Score')
        plt.plot(folds, test_scores, 'o-', label='Test Score')
        plt.axhline(y=cv_summary['mean_train_score'], color='b', linestyle='--', alpha=0.7)
        plt.axhline(y=cv_summary['mean_test_score'], color='r', linestyle='--', alpha=0.7)
        plt.title(f'Train vs Test Scores Across {k_folds}-Fold Cross-Validation')
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, 'train_test_score.png'))
        plt.close()
        
        # 2. Specific Metrics Across Folds
        for metric in ['sharpe_ratio', 'win_rate', 'max_drawdown', 'profit_factor', 'return_pct']:
            if metric in train_metrics and metric in test_metrics:
                plt.figure(figsize=(12, 6))
                plt.plot(folds, train_metrics[metric], 'o-', label=f'Train {metric}')
                plt.plot(folds, test_metrics[metric], 'o-', label=f'Test {metric}')
                plt.title(f'{metric.replace("_", " ").title()} Across {k_folds}-Fold Cross-Validation')
                plt.xlabel('Fold')
                plt.ylabel(metric.replace("_", " ").title())
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(viz_dir, f'{metric}_by_fold.png'))
                plt.close()
        
        # 3. Market Conditions vs Performance
        trends = [fold['market_conditions']['trend'] for fold in fold_results]
        volatilities = [fold['market_conditions']['volatility'] for fold in fold_results]
        
        # 3.1. Performance by Trend
        trend_categories = set(trends)
        trend_performance = {trend: [] for trend in trend_categories}
        
        for i, trend in enumerate(trends):
            trend_performance[trend].append(test_scores[i])
        
        plt.figure(figsize=(10, 6))
        for trend, scores in trend_performance.items():
            plt.bar(trend, np.mean(scores), yerr=np.std(scores) if len(scores) > 1 else 0,
                   alpha=0.7, capsize=10)
        plt.title(f'Performance by Market Trend')
        plt.xlabel('Market Trend')
        plt.ylabel('Test Score')
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(viz_dir, 'performance_by_trend.png'))
        plt.close()
        
        # 3.2. Performance vs Volatility
        plt.figure(figsize=(10, 6))
        plt.scatter(volatilities, test_scores, alpha=0.7)
        
        # Add trend line
        if len(volatilities) > 1:
            z = np.polyfit(volatilities, test_scores, 1)
            p = np.poly1d(z)
            plt.plot(sorted(volatilities), p(sorted(volatilities)), "r--", alpha=0.7)
        
        plt.title(f'Performance vs Market Volatility')
        plt.xlabel('Volatility')
        plt.ylabel('Test Score')
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, 'performance_vs_volatility.png'))
        plt.close()
        
        # 4. Summary Statistics Visualization
        plt.figure(figsize=(12, 8))
        
        # Extract key summary statistics
        stats = [
            cv_summary['mean_train_score'],
            cv_summary['mean_test_score'],
            cv_summary['train_test_gap'],
            cv_summary['consistency']
        ]
        
        labels = [
            'Mean Train Score',
            'Mean Test Score',
            'Train-Test Gap',
            'Consistency'
        ]
        
        plt.bar(labels, stats, alpha=0.7)
        plt.title('Cross-Validation Summary Statistics')
        plt.ylabel('Value')
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(viz_dir, 'summary_statistics.png'))
        plt.close()
        
        logger.info(f"Cross-validation visualizations saved to {viz_dir}")
        
    def cross_validate_across_market_conditions(self, strategy_id: str, parameters: Dict) -> Dict:
        """
        Perform cross-validation across different market conditions.
        
        Args:
            strategy_id: Identifier for the strategy
            parameters: Strategy parameters to test
            
        Returns:
            Dictionary of cross-validation results
        """
        # Define different market condition periods
        market_conditions = [
            {
                "name": "bull_market",
                "start_date": "2021-01-01",
                "end_date": "2021-04-01",
                "description": "Strong uptrend with high volume"
            },
            {
                "name": "bear_market",
                "start_date": "2022-05-01",
                "end_date": "2022-08-01",
                "description": "Strong downtrend with high volatility"
            },
            {
                "name": "sideways_market",
                "start_date": "2019-09-01",
                "end_date": "2019-12-01",
                "description": "Ranging market with low volatility"
            },
            {
                "name": "recovery_market",
                "start_date": "2023-01-01",
                "end_date": "2023-04-01",
                "description": "Recovery phase with increasing volume"
            },
            {
                "name": "high_volatility",
                "start_date": "2020-03-01",
                "end_date": "2020-06-01",
                "description": "Extremely volatile market conditions"
            }
        ]
        
        # Load market data for each period
        market_data_periods = []
        for condition in market_conditions:
            # In a real implementation, we would load historical data for each period
            # For now, we'll generate synthetic data
            market_data = self._generate_synthetic_market_data(
                condition["name"],
                condition["start_date"],
                condition["end_date"]
            )
            market_data_periods.append(market_data)
        
        # Perform cross-validation
        cv_results = self.cross_validate_strategy(
            strategy_id,
            parameters,
            market_data_periods,
            k_folds=len(market_conditions),
            test_metric='sharpe_ratio'
        )
        
        # Add market condition information to results
        for i, condition in enumerate(market_conditions):
            if i < len(cv_results["fold_results"]):
                cv_results["fold_results"][i]["market_condition_name"] = condition["name"]
                cv_results["fold_results"][i]["market_condition_description"] = condition["description"]
        
        # Generate additional visualizations specific to market conditions
        self._visualize_market_condition_performance(cv_results, market_conditions)
        
        return cv_results
    
    def _generate_synthetic_market_data(self, condition_name: str, start_date: str, end_date: str) -> List[Dict]:
        """
        Generate synthetic market data for a specific market condition.
        
        Args:
            condition_name: Name of the market condition
            start_date: Start date for the period
            end_date: End date for the period
            
        Returns:
            List of synthetic market data points
        """
        import pandas as pd
        
        # Parse dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Generate date range with daily intervals
        dates = pd.date_range(start=start, end=end, freq='D')
        
        # Generate synthetic data based on market condition
        data = []
        
        if condition_name == "bull_market":
            # Bull market: strong uptrend, moderate volatility, high volume
            base_price = 10000
            trend_factor = 1.001  # Small daily gain
            volatility = 0.02
            volume_factor = 1000000
        
        elif condition_name == "bear_market":
            # Bear market: strong downtrend, high volatility, moderate volume
            base_price = 20000
            trend_factor = 0.999  # Small daily loss
            volatility = 0.03
            volume_factor = 500000
        
        elif condition_name == "sideways_market":
            # Sideways market: no trend, low volatility, low volume
            base_price = 15000
            trend_factor = 1.0  # No trend
            volatility = 0.01
            volume_factor = 300000
        
        elif condition_name == "recovery_market":
            # Recovery market: moderate uptrend, decreasing volatility, increasing volume
            base_price = 5000
            trend_factor = 1.0015  # Moderate daily gain
            volatility = 0.015
            volume_factor = 700000
        
        elif condition_name == "high_volatility":
            # High volatility market: variable trend, very high volatility, high volume
            base_price = 12000
            trend_factor = 1.0  # No clear trend
            volatility = 0.05
            volume_factor = 1500000
        
        else:
            # Default market condition
            base_price = 10000
            trend_factor = 1.0
            volatility = 0.02
            volume_factor = 500000
        
        # Generate daily data
        price = base_price
        for date in dates:
            # Add random noise to trend
            daily_change = np.random.normal(trend_factor, volatility)
            price *= daily_change
            
            # Generate other indicator values
            rsi = np.random.normal(50, 15)  # Center around 50 with std 15
            rsi = max(0, min(100, rsi))  # Clamp to 0-100
            
            volume = np.random.normal(volume_factor, volume_factor * 0.2)
            volume = max(0, volume)
            
            # Create data point
            data_point = {
                "timestamp": date.isoformat(),
                "symbol": "BTCUSDT",
                "price": price,
                "volume": volume,
                "rsi": rsi,
                "macd": np.random.normal(0, 1),
                "macd_signal": np.random.normal(0, 1),
                "bb_upper": price * (1 + volatility * 2),
                "bb_lower": price * (1 - volatility * 2),
                "bb_middle": price,
                "ema_short": price * (1 + np.random.normal(0, 0.01)),
                "ema_long": price * (1 + np.random.normal(0, 0.02)),
                "market_condition": condition_name
            }
            
            data.append(data_point)
        
        return data
    
    def _visualize_market_condition_performance(self, cv_results: Dict, market_conditions: List[Dict]) -> None:
        """
        Generate visualizations comparing performance across market conditions.
        
        Args:
            cv_results: Dictionary of cross-validation results
            market_conditions: List of market condition definitions
        """
        strategy_id = cv_results["strategy_id"]
        fold_results = cv_results["fold_results"]
        
        # Create visualization directory
        viz_dir = os.path.join(self.reports_dir, f"{strategy_id}_market_conditions")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Extract market condition names and test scores
        condition_names = [fold.get("market_condition_name", f"Fold {i+1}") 
                          for i, fold in enumerate(fold_results)]
        test_scores = [fold['test_score'] for fold in fold_results]
        
        # 1. Performance across market conditions
        plt.figure(figsize=(12, 6))
        bars = plt.bar(condition_names, test_scores, alpha=0.7)
        
        # Color bars by performance
        for i, bar in enumerate(bars):
            if test_scores[i] > 1.0:
                bar.set_color('green')
            elif test_scores[i] > 0:
                bar.set_color('blue')
            else:
                bar.set_color('red')
        
        plt.title(f'Strategy Performance Across Market Conditions')
        plt.xlabel('Market Condition')
        plt.ylabel('Performance Score')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'performance_by_condition.png'))
        plt.close()
        
        # 2. Radar chart of key metrics across market conditions
        metrics = ['sharpe_ratio', 'win_rate', 'profit_factor', 'return_pct']
        
        # Number of variables
        N = len(metrics)
        
        # Create a figure
        plt.figure(figsize=(12, 10))
        
        # Create angles for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create subplot with polar projection
        ax = plt.subplot(111, polar=True)
        
        # Add labels for each metric
        plt.xticks(angles[:-1], metrics)
        
        # Plot each market condition
        for i, fold in enumerate(fold_results):
            condition_name = fold.get("market_condition_name", f"Fold {i+1}")
            
            # Extract values for each metric
            values = [fold['test_metrics'].get(metric, 0) for metric in metrics]
            
            # Close the loop
            values += values[:1]
            
            # Plot values
            ax.plot(angles, values, linewidth=2, label=condition_name)
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Strategy Metrics Across Market Conditions')
        plt.savefig(os.path.join(viz_dir, 'metrics_radar_chart.png'))
        plt.close()
        
        # 3. Create a summary table visualization
        fig, ax = plt.figure(figsize=(12, 8)), plt.subplot(111)
        ax.axis('off')
        
        # Create table data
        table_data = []
        table_data.append(["Market Condition", "Sharpe Ratio", "Win Rate", "Max Drawdown", "Return %", "Score"])
        
        for i, fold in enumerate(fold_results):
            condition_name = fold.get("market_condition_name", f"Fold {i+1}")
            metrics = fold['test_metrics']
            
            row = [
                condition_name,
                f"{metrics.get('sharpe_ratio', 0):.2f}",
                f"{metrics.get('win_rate', 0)*100:.1f}%",
                f"{metrics.get('max_drawdown', 0)*100:.1f}%",
                f"{metrics.get('return_pct', 0):.2f}%",
                f"{fold['test_score']:.2f}"
            ]
            table_data.append(row)
        
        # Add summary row
        summary = cv_results["cv_summary"]
        summary_row = [
            "AVERAGE",
            f"{summary.get('mean_test_sharpe_ratio', 0):.2f}",
            f"{summary.get('mean_test_win_rate', 0)*100:.1f}%",
            f"{summary.get('mean_test_max_drawdown', 0)*100:.1f}%",
            f"{summary.get('mean_test_return_pct', 0):.2f}%",
            f"{summary.get('mean_test_score', 0):.2f}"
        ]
        table_data.append(summary_row)
        
        # Create table
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        table.auto_set_column_width(col=list(range(len(table_data[0]))))
        
        # Style header row
        for j in range(len(table_data[0])):
            table[(0, j)].set_facecolor('#b4b4b4')
            table[(0, j)].set_text_props(weight='bold')
        
        # Style summary row
        for j in range(len(table_data[0])):
            table[(len(table_data)-1, j)].set_facecolor('#e0e0e0')
            table[(len(table_data)-1, j)].set_text_props(weight='bold')
        
        plt.title('Strategy Performance Summary Across Market Conditions', fontsize=14, pad=20)
        plt.savefig(os.path.join(viz_dir, 'performance_summary_table.png'), bbox_inches='tight')
        plt.close()
        
        logger.info(f"Market condition performance visualizations saved to {viz_dir}")
    
    def compare_strategies(self, strategy_ids: List[str], 
                          metrics_to_compare: List[str] = None,
                          market_condition: str = None,
                          time_period: str = None,
                          normalize_scores: bool = True) -> Dict:
        """
        Compare multiple strategies based on specified metrics.
        
        Args:
            strategy_ids: List of strategy IDs to compare
            metrics_to_compare: List of metrics to use for comparison
            market_condition: Specific market condition to filter results (optional)
            time_period: Specific time period to compare (e.g., "last_month", "last_year")
            normalize_scores: Whether to normalize scores for fair comparison
            
        Returns:
            Dictionary with comparison results
        """
        if not metrics_to_compare:
            # Default metrics to compare
            metrics_to_compare = [
                'sharpe_ratio', 'win_rate', 'profit_factor', 'max_drawdown',
                'return_pct', 'expectancy', 'calmar_ratio'
            ]
        
        # Load data sources based on input parameters
        strategy_data = {}
        
        if market_condition:
            # Load cross-validation results to compare strategies in specific market conditions
            for strategy_id in strategy_ids:
                try:
                    cv_file = os.path.join(self.data_dir, f"{strategy_id}_cv_results.json")
                    if os.path.exists(cv_file):
                        with open(cv_file, 'r') as f:
                            cv_data = json.load(f)
                            # Extract data for the specified market condition
                            for fold in cv_data["fold_results"]:
                                if fold.get("market_condition_name") == market_condition:
                                    strategy_data[strategy_id] = {
                                        "strategy_id": strategy_id,
                                        "metrics": fold["test_metrics"],
                                        "score": fold["test_score"],
                                        "market_condition": market_condition,
                                        "parameters": cv_data["parameters"]
                                    }
                                    break
                except Exception as e:
                    logger.warning(f"Could not load CV results for strategy {strategy_id}: {e}")
        elif time_period:
            # Load historical evaluation data for the specified time period
            time_filter = self._parse_time_period(time_period)
            for strategy_id in strategy_ids:
                try:
                    # Find all evaluation files for this strategy
                    eval_files = [f for f in os.listdir(self.data_dir) 
                                if f.startswith(f"{strategy_id}_") and f.endswith("_evaluation.json")]
                    # Sort by timestamp (newest first)
                    eval_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.data_dir, f)), reverse=True)
                    
                    for eval_file in eval_files:
                        file_time = datetime.fromtimestamp(os.path.getmtime(os.path.join(self.data_dir, eval_file)))
                        if file_time >= time_filter:
                            with open(os.path.join(self.data_dir, eval_file), 'r') as f:
                                eval_data = json.load(f)
                                strategy_data[strategy_id] = eval_data
                                break
                except Exception as e:
                    logger.warning(f"Could not load time-filtered evaluation for strategy {strategy_id}: {e}")
        else:
            # Load the most recent evaluation data for each strategy
            for strategy_id in strategy_ids:
                try:
                    eval_file = os.path.join(self.data_dir, f"{strategy_id}_evaluation.json")
                    if os.path.exists(eval_file):
                        with open(eval_file, 'r') as f:
                            strategy_data[strategy_id] = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load evaluation for strategy {strategy_id}: {e}")
        
        # If no data found for any strategy, try to run evaluations on missing strategies
        missing_strategies = [sid for sid in strategy_ids if sid not in strategy_data]
        if missing_strategies and not (market_condition or time_period):
            for strategy_id in missing_strategies:
                try:
                    # Check if we have parameters for this strategy
                    params_file = os.path.join(self.data_dir, f"{strategy_id}_parameters.json")
                    if os.path.exists(params_file):
                        with open(params_file, 'r') as f:
                            parameters = json.load(f)
                            # Run evaluation with these parameters
                            synthetic_data = self._generate_synthetic_market_data("default", 
                                                                              "2023-01-01", 
                                                                              "2023-01-31")
                            trades = self._simulate_trades(strategy_id, parameters, synthetic_data)
                            metrics = StrategyPerformanceMetrics.calculate_metrics(trades)
                            score = self._calculate_strategy_score(metrics)
                            
                            strategy_data[strategy_id] = {
                                "strategy_id": strategy_id,
                                "metrics": metrics,
                                "score": score,
                                "parameters": parameters,
                                "timestamp": datetime.now().isoformat(),
                                "note": "Generated from synthetic data for comparison"
                            }
                except Exception as e:
                    logger.warning(f"Could not generate evaluation for missing strategy {strategy_id}: {e}")
        
        if not strategy_data:
            logger.error("No valid strategies found for comparison")
            return {"error": "No valid strategies found for comparison"}
        
        # Extract comparison metrics
        comparison_data = {
            "strategy_ids": list(strategy_data.keys()),
            "metrics": {},
            "best_strategy": None,
            "best_strategy_score": -float('inf'),
            "comparison_timestamp": datetime.now().isoformat(),
            "market_condition": market_condition,
            "time_period": time_period,
            "strategy_parameters": {}
        }
        
        # Store parameters for each strategy
        for strategy_id, data in strategy_data.items():
            comparison_data["strategy_parameters"][strategy_id] = data.get("parameters", {})
        
        # Compare each metric
        for metric in metrics_to_compare:
            comparison_data["metrics"][metric] = {}
            
            metric_values = {}
            for strategy_id, data in strategy_data.items():
                metric_value = data.get("metrics", {}).get(metric, 0)
                comparison_data["metrics"][metric][strategy_id] = metric_value
                metric_values[strategy_id] = metric_value
            
            if not metric_values:
                continue
                
            # Normalize metric values if requested
            if normalize_scores:
                normalized_values = self._normalize_metric_values(metric_values, metric)
                comparison_data["metrics"][metric]["normalized"] = normalized_values
            
            # Determine best value for this metric
            if metric in ['max_drawdown']:  # Lower is better
                best_strategy = min(metric_values.items(), key=lambda x: x[1])
            else:  # Higher is better
                best_strategy = max(metric_values.items(), key=lambda x: x[1])
            
            comparison_data["metrics"][metric]["best"] = best_strategy[0]
        
        # Calculate overall scores based on weighted metrics
        overall_scores = {}
        metric_weights = self.optimization_goals.get("metric_weights", {})
        
        for strategy_id in strategy_data:
            # Default equal weighting if no weights specified
            if not metric_weights:
                # Equal weights for all metrics
                weights = {metric: 1.0 / len(metrics_to_compare) for metric in metrics_to_compare}
            else:
                weights = metric_weights
                
            # Calculate weighted score
            total_score = 0
            total_weight = 0
            
            for metric, weight in weights.items():
                if metric not in metrics_to_compare:
                    continue
                    
                metric_value = comparison_data["metrics"].get(metric, {}).get(strategy_id, 0)
                
                # For metrics where lower is better, invert the value
                if metric in ['max_drawdown']:
                    # Avoid division by zero
                    if metric_value > 0:
                        metric_value = 1.0 / metric_value
                    else:
                        metric_value = float('inf')
                
                total_score += metric_value * weight
                total_weight += weight
            
            # Calculate final weighted score
            if total_weight > 0:
                overall_scores[strategy_id] = total_score / total_weight
            else:
                overall_scores[strategy_id] = 0
        
        # Find best overall strategy
        if overall_scores:
            best_strategy = max(overall_scores.items(), key=lambda x: x[1])
            comparison_data["best_strategy"] = best_strategy[0]
            comparison_data["best_strategy_score"] = best_strategy[1]
            comparison_data["overall_scores"] = overall_scores
        
        # Generate comparison visualizations
        self._visualize_enhanced_comparison(comparison_data)
        
        # Store comparison results
        comparison_id = f"comparison_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        results_path = os.path.join(self.data_dir, f"{comparison_id}.json")
        with open(results_path, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        comparison_data["results_path"] = results_path
        return comparison_data
    
    def _parse_time_period(self, time_period: str) -> datetime:
        """
        Parse a time period string into a datetime object.
        
        Args:
            time_period: Time period string (e.g., "last_month", "last_year", "last_week")
            
        Returns:
            Datetime object representing the cutoff time
        """
        now = datetime.now()
        
        if time_period == "last_day":
            return now - timedelta(days=1)
        elif time_period == "last_week":
            return now - timedelta(days=7)
        elif time_period == "last_month":
            return now - timedelta(days=30)
        elif time_period == "last_quarter":
            return now - timedelta(days=90)
        elif time_period == "last_year":
            return now - timedelta(days=365)
        elif time_period == "last_3_months":
            return now - timedelta(days=90)
        elif time_period == "last_6_months":
            return now - timedelta(days=180)
        elif time_period == "ytd":  # Year to date
            return datetime(now.year, 1, 1)
        else:
            # Default to last 30 days if unknown period
            logger.warning(f"Unknown time period: {time_period}, defaulting to last_month")
            return now - timedelta(days=30)
    
    def _normalize_metric_values(self, metric_values: Dict[str, float], metric: str) -> Dict[str, float]:
        """
        Normalize metric values for fair comparison.
        
        Args:
            metric_values: Dictionary of metric values by strategy ID
            metric: The metric being normalized
            
        Returns:
            Dictionary of normalized values
        """
        values = list(metric_values.values())
        if not values:
            return {}
            
        # Special handling for metrics where lower is better
        if metric in ['max_drawdown']:
            # Invert values so higher is better
            inverted_values = {}
            for strategy_id, value in metric_values.items():
                if value > 0:
                    inverted_values[strategy_id] = 1.0 / value
                else:
                    inverted_values[strategy_id] = float('inf')
            
            values = list(inverted_values.values())
        
        # Min-max normalization
        min_val = min(values)
        max_val = max(values)
        
        range_val = max_val - min_val
        
        normalized = {}
        for strategy_id, value in metric_values.items():
            if range_val > 0:
                if metric in ['max_drawdown']:
                    # For metrics where lower is better, invert before normalizing
                    if value > 0:
                        inv_value = 1.0 / value
                        normalized[strategy_id] = (inv_value - min_val) / range_val
                    else:
                        normalized[strategy_id] = 1.0  # Perfect score for zero drawdown
                else:
                    # For metrics where higher is better
                    normalized[strategy_id] = (value - min_val) / range_val
            else:
                normalized[strategy_id] = 0.5  # All values are the same
        
        return normalized
    
    def _visualize_enhanced_comparison(self, comparison_data: Dict) -> None:
        """
        Generate enhanced visualizations for strategy comparison.
        
        Args:
            comparison_data: Dictionary of comparison results
        """
        strategy_ids = comparison_data["strategy_ids"]
        if not strategy_ids:
            return
            
        # Create visualization directory
        viz_dir = os.path.join(self.reports_dir, "strategy_comparison")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Get metrics
        metrics = list(comparison_data["metrics"].keys())
        
        # 1. Spider/Radar chart for all metrics
        self._create_strategy_radar_chart(comparison_data, viz_dir)
        
        # 2. Bar charts for individual metrics
        self._create_metric_bar_charts(comparison_data, viz_dir)
        
        # 3. Overall score comparison
        if "overall_scores" in comparison_data:
            plt.figure(figsize=(12, 6))
            
            scores = comparison_data["overall_scores"]
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            strategies = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]
            
            # Color bars based on score
            bars = plt.bar(strategies, values, alpha=0.7)
            for i, bar in enumerate(bars):
                if values[i] >= 0.8 * max(values):
                    bar.set_color('green')
                elif values[i] >= 0.5 * max(values):
                    bar.set_color('blue')
                else:
                    bar.set_color('orange')
            
            plt.title('Overall Strategy Score Comparison')
            plt.xlabel('Strategy')
            plt.ylabel('Score')
            plt.grid(True, axis='y')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'overall_scores.png'))
            plt.close()
        
        # 4. Strategy parameter comparison table
        if "strategy_parameters" in comparison_data:
            self._create_parameter_comparison_table(comparison_data, viz_dir)
        
        # 5. Rank summary table
        self._create_rank_summary_table(comparison_data, viz_dir)
        
        # 6. Summary report with market condition context
        self._create_comparison_summary_report(comparison_data, viz_dir)
        
        logger.info(f"Enhanced strategy comparison visualizations saved to {viz_dir}")
    
    def _create_strategy_radar_chart(self, comparison_data: Dict, viz_dir: str) -> None:
        """
        Create a radar chart comparing strategies across all metrics.
        
        Args:
            comparison_data: Dictionary of comparison results
            viz_dir: Directory to save visualizations
        """
        strategy_ids = comparison_data["strategy_ids"]
        metrics = list(comparison_data["metrics"].keys())
        
        # Create a radar chart for all metrics
        plt.figure(figsize=(12, 10))
        
        # Number of variables
        N = len(metrics)
        if N < 3:
            # Need at least 3 metrics for a radar chart
            return
            
        # Create angles for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create subplot with polar projection
        ax = plt.subplot(111, polar=True)
        
        # Format metric labels for display
        metric_labels = [m.replace('_', ' ').title() for m in metrics]
        plt.xticks(angles[:-1], metric_labels)
        
        # Plot each strategy
        for strategy_id in strategy_ids:
            # Get normalized values if available, otherwise use raw values
            values = []
            for metric in metrics:
                if "normalized" in comparison_data["metrics"][metric]:
                    value = comparison_data["metrics"][metric]["normalized"].get(strategy_id, 0)
                else:
                    # Get raw values and normalize them for display
                    all_values = [comparison_data["metrics"][metric].get(sid, 0) for sid in strategy_ids 
                                  if isinstance(comparison_data["metrics"][metric].get(sid, 0), (int, float))]
                    if all_values:
                        min_val = min(all_values)
                        max_val = max(all_values)
                        range_val = max_val - min_val
                        
                        raw_value = comparison_data["metrics"][metric].get(strategy_id, 0)
                        if isinstance(raw_value, (int, float)):
                            if range_val > 0:
                                if metric in ['max_drawdown']:  # Lower is better
                                    value = 1 - ((raw_value - min_val) / range_val)
                                else:  # Higher is better
                                    value = (raw_value - min_val) / range_val
                            else:
                                value = 0.5  # All values are the same
                        else:
                            value = 0  # Non-numeric value
                    else:
                        value = 0
                
                values.append(value)
            
            # Close the loop for the radar chart
            values += values[:1]
            
            # Plot strategy
            ax.plot(angles, values, linewidth=2, label=strategy_id)
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Set title based on market condition or time period
        title = 'Strategy Comparison Across Metrics'
        if comparison_data.get("market_condition"):
            title += f" ({comparison_data['market_condition']} Market)"
        elif comparison_data.get("time_period"):
            title += f" ({comparison_data['time_period']})"
            
        plt.title(title)
        plt.grid(True)
        plt.savefig(os.path.join(viz_dir, 'strategy_radar_chart.png'))
        plt.close()
    
    def _create_metric_bar_charts(self, comparison_data: Dict, viz_dir: str) -> None:
        """
        Create individual bar charts for each metric.
        
        Args:
            comparison_data: Dictionary of comparison results
            viz_dir: Directory to save visualizations
        """
        strategy_ids = comparison_data["strategy_ids"]
        
        # Create metric subfolder
        metrics_dir = os.path.join(viz_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Create a bar chart for each metric
        for metric, values in comparison_data["metrics"].items():
            if metric in ['best', 'normalized']:
                continue
                
            plt.figure(figsize=(10, 6))
            
            # Extract values for each strategy
            metric_values = {}
            for strategy_id in strategy_ids:
                if strategy_id in values and isinstance(values[strategy_id], (int, float)):
                    metric_values[strategy_id] = values[strategy_id]
            
            if not metric_values:
                continue
                
            # Sort strategies by metric value
            if metric in ['max_drawdown']:  # Lower is better
                sorted_items = sorted(metric_values.items(), key=lambda x: x[1])
            else:  # Higher is better
                sorted_items = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                
            sorted_strategies = [item[0] for item in sorted_items]
            sorted_values = [item[1] for item in sorted_items]
            
            # Create bar chart
            bars = plt.bar(sorted_strategies, sorted_values, alpha=0.7)
            
            # Color the best strategy bar
            best_strategy = values.get("best")
            for i, strategy in enumerate(sorted_strategies):
                if strategy == best_strategy:
                    bars[i].set_color('green')
                else:
                    bars[i].set_color('skyblue')
            
            # Format metric name for display
            metric_display = metric.replace('_', ' ').title()
            
            plt.title(f'{metric_display} Comparison')
            plt.xlabel('Strategy')
            plt.ylabel(metric_display)
            plt.grid(True, axis='y')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save metric-specific chart
            plt.savefig(os.path.join(metrics_dir, f'{metric}_comparison.png'))
            plt.close()
    
    def _create_parameter_comparison_table(self, comparison_data: Dict, viz_dir: str) -> None:
        """
        Create a visual table comparing strategy parameters.
        
        Args:
            comparison_data: Dictionary of comparison results
            viz_dir: Directory to save visualizations
        """
        # Extract parameters and strategies
        parameters = comparison_data.get("strategy_parameters", {})
        if not parameters:
            return
            
        # Find common parameters across all strategies
        common_params = set()
        for strategy_id, params in parameters.items():
            if not common_params:
                common_params = set(params.keys())
            else:
                common_params &= set(params.keys())
        
        if not common_params:
            return
            
        # Create figure for table
        fig, ax = plt.figure(figsize=(12, 8)), plt.subplot(111)
        ax.axis('off')
        
        # Create table data
        table_data = []
        
        # Header row with parameter names
        header = ['Strategy']
        header.extend([param.replace('_', ' ').title() for param in sorted(common_params)])
        table_data.append(header)
        
        # Add rows for each strategy
        for strategy_id, params in parameters.items():
            row = [strategy_id]
            for param in sorted(common_params):
                value = params.get(param, '')
                if isinstance(value, float):
                    row.append(f"{value:.2f}")
                else:
                    row.append(str(value))
            table_data.append(row)
        
        # Create and style table
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Style header row
        for j in range(len(table_data[0])):
            table[(0, j)].set_facecolor('#b4b4b4')
            table[(0, j)].set_text_props(weight='bold')
        
        # Highlight best strategy row
        best_strategy = comparison_data.get("best_strategy")
        if best_strategy:
            for i, row in enumerate(table_data):
                if i > 0 and row[0] == best_strategy:
                    for j in range(len(row)):
                        table[(i, j)].set_facecolor('#e0f0e0')
        
        plt.title('Strategy Parameter Comparison', fontsize=14, pad=20)
        plt.savefig(os.path.join(viz_dir, 'parameter_comparison.png'), bbox_inches='tight')
        plt.close()
    
    def _create_rank_summary_table(self, comparison_data: Dict, viz_dir: str) -> None:
        """
        Create a table showing the rank of each strategy for each metric.
        
        Args:
            comparison_data: Dictionary of comparison results
            viz_dir: Directory to save visualizations
        """
        strategy_ids = comparison_data["strategy_ids"]
        
        # Create ranking data for each metric
        rankings = {}
        for metric, values in comparison_data["metrics"].items():
            if metric in ['best', 'normalized']:
                continue
                
            # Create dictionary of strategy: value for ranking
            metric_values = {}
            for strategy_id in strategy_ids:
                if strategy_id in values and isinstance(values[strategy_id], (int, float)):
                    metric_values[strategy_id] = values[strategy_id]
            
            if not metric_values:
                continue
                
            # Sort strategies by metric value
            if metric in ['max_drawdown']:  # Lower is better
                sorted_items = sorted(metric_values.items(), key=lambda x: x[1])
            else:  # Higher is better
                sorted_items = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                
            # Assign ranks (1 is best)
            rankings[metric] = {}
            for rank, (strategy_id, _) in enumerate(sorted_items):
                rankings[metric][strategy_id] = rank + 1
        
        if not rankings:
            return
            
        # Calculate average rank for each strategy
        avg_ranks = {}
        for strategy_id in strategy_ids:
            ranks = [rankings[metric][strategy_id] for metric in rankings if strategy_id in rankings[metric]]
            if ranks:
                avg_ranks[strategy_id] = sum(ranks) / len(ranks)
        
        # Create figure for table
        fig, ax = plt.figure(figsize=(12, 8)), plt.subplot(111)
        ax.axis('off')
        
        # Create table data
        table_data = []
        
        # Header row with metric names
        metrics = list(rankings.keys())
        header = ['Strategy']
        header.extend([metric.replace('_', ' ').title() for metric in metrics])
        header.append('Avg Rank')
        table_data.append(header)
        
        # Sort strategies by average rank
        sorted_strategies = sorted(avg_ranks.items(), key=lambda x: x[1])
        
        # Add rows for each strategy
        for strategy_id, avg_rank in sorted_strategies:
            row = [strategy_id]
            for metric in metrics:
                row.append(str(rankings[metric].get(strategy_id, '-')))
            row.append(f"{avg_rank:.2f}")
            table_data.append(row)
        
        # Create and style table
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Style header row
        for j in range(len(table_data[0])):
            table[(0, j)].set_facecolor('#b4b4b4')
            table[(0, j)].set_text_props(weight='bold')
        
        # Style rank columns - color based on rank
        for i in range(1, len(table_data)):
            for j in range(1, len(metrics) + 1):
                try:
                    rank = int(table_data[i][j])
                    if rank == 1:
                        table[(i, j)].set_facecolor('#90ee90')  # Light green for 1st
                    elif rank == 2:
                        table[(i, j)].set_facecolor('#add8e6')  # Light blue for 2nd
                    elif rank == 3:
                        table[(i, j)].set_facecolor('#ffffe0')  # Light yellow for 3rd
                except (ValueError, TypeError):
                    pass
            
            # Style average rank column
            avg_rank_col = len(metrics) + 1
            table[(i, avg_rank_col)].set_facecolor('#e0e0e0')
        
        plt.title('Strategy Ranking by Metric', fontsize=14, pad=20)
        plt.savefig(os.path.join(viz_dir, 'strategy_rankings.png'), bbox_inches='tight')
        plt.close()
    
    def _create_comparison_summary_report(self, comparison_data: Dict, viz_dir: str) -> None:
        """
        Create an HTML summary report of the strategy comparison.
        
        Args:
            comparison_data: Dictionary of comparison results
            viz_dir: Directory to save visualizations
        """
        # Generate an HTML report summarizing the comparison
        strategy_ids = comparison_data["strategy_ids"]
        best_strategy = comparison_data.get("best_strategy")
        
        # Format timestamp
        timestamp = comparison_data.get("comparison_timestamp", datetime.now().isoformat())
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_time = timestamp
        else:
            formatted_time = timestamp
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Strategy Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333366; }}
                .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .best-strategy {{ color: #009900; font-weight: bold; }}
                .image-container {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .footer {{ margin-top: 30px; font-size: 0.8em; color: #666; }}
            </style>
        </head>
        <body>
            <h1>Strategy Comparison Report</h1>
            
            <div class="summary">
                <h2>Comparison Summary</h2>
                <p><strong>Date:</strong> {formatted_time}</p>
                <p><strong>Strategies Compared:</strong> {len(strategy_ids)}</p>
        """
        
        # Add market condition or time period if available
        if comparison_data.get("market_condition"):
            html_content += f"<p><strong>Market Condition:</strong> {comparison_data['market_condition']}</p>"
        elif comparison_data.get("time_period"):
            html_content += f"<p><strong>Time Period:</strong> {comparison_data['time_period']}</p>"
        
        # Add best strategy information
        if best_strategy:
            html_content += f"""
                <p><strong>Best Overall Strategy:</strong> <span class="best-strategy">{best_strategy}</span></p>
                <p><strong>Overall Score:</strong> {comparison_data.get("best_strategy_score", "N/A"):.4f}</p>
            """
        
        html_content += """
            </div>
            
            <h2>Performance Visualizations</h2>
            
            <div class="image-container">
                <h3>Strategy Comparison Across Metrics</h3>
                <img src="strategy_radar_chart.png" alt="Strategy Radar Chart" style="max-width: 100%;">
            </div>
            
            <div class="image-container">
                <h3>Overall Strategy Scores</h3>
                <img src="overall_scores.png" alt="Overall Scores" style="max-width: 100%;">
            </div>
            
            <div class="image-container">
                <h3>Strategy Parameters</h3>
                <img src="parameter_comparison.png" alt="Parameter Comparison" style="max-width: 100%;">
            </div>
            
            <div class="image-container">
                <h3>Strategy Rankings</h3>
                <img src="strategy_rankings.png" alt="Strategy Rankings" style="max-width: 100%;">
            </div>
            
            <h2>Individual Metric Comparisons</h2>
            <p>See the 'metrics' folder for detailed charts of each performance metric.</p>
            
            <div class="footer">
                <p>Generated at {formatted_time}</p>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(os.path.join(viz_dir, 'comparison_report.html'), 'w') as f:
            f.write(html_content)
        
        logger.info(f"Strategy comparison report saved to {os.path.join(viz_dir, 'comparison_report.html')}")
    
    def _visualize_comparison(self, comparison_data: Dict) -> None:
        """
        Generate basic visualizations comparing strategies.
        
        Args:
            comparison_data: Dictionary of comparison results
        """
        # This is the original visualization method, kept for backward compatibility
        # For new visualizations, use _visualize_enhanced_comparison
        
        strategy_ids = comparison_data["strategy_ids"]
        metrics = comparison_data["metrics"]
        
        # Create a bar chart for each metric
        for metric, values in metrics.items():
            if metric == 'best':
                continue
                
            plt.figure(figsize=(10, 6))
            
            # Extract values for each strategy
            strategies = []
            metric_values = []
            
            for strategy_id in strategy_ids:
                if strategy_id in values:
                    strategies.append(strategy_id)
                    metric_values.append(values[strategy_id])
            
            # Create bar chart
            plt.bar(strategies, metric_values)
            
            # Highlight best strategy
            best_strategy = values.get("best")
            if best_strategy in strategies:
                best_idx = strategies.index(best_strategy)
                plt.bar([strategies[best_idx]], [metric_values[best_idx]], color='green')
            
            plt.title(f'Comparison of {metric}')
            plt.xlabel('Strategy')
            plt.ylabel(metric)
            plt.grid(True, axis='y')
            plt.xticks(rotation=45)
            
            # Save the chart
            plt.tight_layout()
            plt.savefig(f"reports/comparison_{metric}.png")
            plt.close()
        
        logger.info("Basic strategy comparison charts saved to reports/ directory")
