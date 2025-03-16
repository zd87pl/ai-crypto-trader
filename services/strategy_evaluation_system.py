import os
import json
import redis
import numpy as np
import pandas as pd
import asyncio
import logging as logger
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from dotenv import load_dotenv
from io import StringIO, BytesIO
import base64

# Configure logging
logger.basicConfig(
    level=logger.INFO,
    format='%(asctime)s - %(levelname)s - [StrategyEvaluation] %(message)s',
    handlers=[
        logger.FileHandler('logs/strategy_evaluation.log'),
        logger.StreamHandler()
    ]
)

class StrategyPerformanceMetrics:
    """
    Calculates and tracks various performance metrics for trading strategies.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.trades = []
        self.wins = 0
        self.losses = 0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.calmar_ratio = 0.0
        self.recovery_factor = 0.0
        self.expectancy = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.largest_win = 0.0
        self.largest_loss = 0.0
        self.avg_trade_duration = 0.0
        self.equity_curve = []
        self.drawdown_curve = []
        
    def calculate_metrics(self, trades: List[Dict]) -> Dict:
        """
        Calculate performance metrics from a list of trades.
        
        Args:
            trades: List of trade dictionaries with at least 'profit', 'entry_time', 'exit_time'
            
        Returns:
            Dictionary of calculated metrics
        """
        self.reset()
        self.trades = trades
        
        if not trades:
            logger.warning("No trades provided for metrics calculation")
            return self._get_metrics_dict()
        
        # Basic metrics
        profits = [trade.get('profit', 0) for trade in trades]
        self.total_profit = sum(max(0, p) for p in profits)
        self.total_loss = abs(sum(min(0, p) for p in profits))
        
        self.wins = sum(1 for p in profits if p > 0)
        self.losses = sum(1 for p in profits if p <= 0)
        
        # Calculate win rate
        if len(trades) > 0:
            self.win_rate = self.wins / len(trades)
        
        # Calculate profit factor
        if self.total_loss > 0:
            self.profit_factor = self.total_profit / self.total_loss
        elif self.total_profit > 0:
            self.profit_factor = float('inf')  # No losses, all profit
        
        # Calculate averages
        if self.wins > 0:
            win_profits = [p for p in profits if p > 0]
            self.avg_win = sum(win_profits) / self.wins
            self.largest_win = max(win_profits)
        
        if self.losses > 0:
            loss_profits = [p for p in profits if p <= 0]
            self.avg_loss = sum(loss_profits) / self.losses
            self.largest_loss = min(loss_profits)
        
        # Calculate trade durations
        durations = []
        for trade in trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                try:
                    entry_time = datetime.fromisoformat(trade['entry_time'])
                    exit_time = datetime.fromisoformat(trade['exit_time'])
                    duration = (exit_time - entry_time).total_seconds() / 3600  # Hours
                    durations.append(duration)
                except (ValueError, TypeError):
                    pass
        
        if durations:
            self.avg_trade_duration = sum(durations) / len(durations)
        
        # Calculate equity curve and drawdown
        equity = 0
        peak = 0
        self.equity_curve = [equity]
        self.drawdown_curve = [0]
        
        for profit in profits:
            equity += profit
            self.equity_curve.append(equity)
            
            peak = max(peak, equity)
            drawdown = (peak - equity) / (peak if peak > 0 else 1)
            self.drawdown_curve.append(drawdown)
            
            self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Advanced metrics
        returns = np.diff(self.equity_curve)
        if len(returns) > 1:
            # Sharpe ratio (annualized)
            returns_mean = np.mean(returns)
            returns_std = np.std(returns)
            if returns_std > 0:
                self.sharpe_ratio = (returns_mean / returns_std) * np.sqrt(365)  # Annualized
            
            # Sortino ratio (only considers downside deviation)
            downside_returns = [r for r in returns if r < 0]
            if downside_returns:
                downside_std = np.std(downside_returns)
                if downside_std > 0:
                    self.sortino_ratio = (returns_mean / downside_std) * np.sqrt(365)  # Annualized
            
            # Calmar ratio (return / max drawdown)
            if self.max_drawdown > 0:
                self.calmar_ratio = (equity / len(trades)) / self.max_drawdown
        
        # Recovery factor
        if self.max_drawdown > 0 and equity > 0:
            self.recovery_factor = equity / self.max_drawdown
        
        # Expectancy
        if len(trades) > 0:
            self.expectancy = (self.win_rate * self.avg_win) + ((1 - self.win_rate) * self.avg_loss)
        
        return self._get_metrics_dict()
    
    def _get_metrics_dict(self) -> Dict:
        """Get all metrics as a dictionary."""
        return {
            'trade_count': len(self.trades),
            'win_count': self.wins,
            'loss_count': self.losses,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'net_profit': self.total_profit - self.total_loss,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'recovery_factor': self.recovery_factor,
            'expectancy': self.expectancy,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'avg_trade_duration': self.avg_trade_duration
        }
    
    def generate_performance_charts(self) -> Dict[str, str]:
        """
        Generate performance visualization charts.
        
        Returns:
            Dictionary of chart names and their base64-encoded PNG data
        """
        if not self.trades:
            return {}
        
        charts = {}
        
        # Set up the style
        plt.style.use('seaborn-darkgrid')
        
        # 1. Equity Curve
        plt.figure(figsize=(10, 6))
        plt.plot(self.equity_curve, label='Equity', color='#4CAF50')
        plt.title('Equity Curve')
        plt.xlabel('Trade Number')
        plt.ylabel('Equity')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Convert plot to base64 for embedding
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        charts['equity_curve'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # 2. Drawdown Chart
        plt.figure(figsize=(10, 6))
        plt.plot(self.drawdown_curve, label='Drawdown', color='#F44336')
        plt.title('Drawdown Chart')
        plt.xlabel('Trade Number')
        plt.ylabel('Drawdown %')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        charts['drawdown'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # 3. Profit Distribution
        profits = [trade.get('profit', 0) for trade in self.trades]
        plt.figure(figsize=(10, 6))
        sns.histplot(profits, kde=True, color='#2196F3')
        plt.title('Profit Distribution')
        plt.xlabel('Profit/Loss')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        charts['profit_distribution'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # 4. Win/Loss Pie Chart
        plt.figure(figsize=(8, 8))
        plt.pie([self.wins, self.losses], 
                labels=['Wins', 'Losses'], 
                autopct='%1.1f%%',
                colors=['#4CAF50', '#F44336'],
                startangle=90)
        plt.title('Win/Loss Ratio')
        plt.axis('equal')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        charts['win_loss_ratio'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return charts

class StrategyEvaluationSystem:
    """
    System for comprehensive evaluation of trading strategies across different market conditions.
    Includes backtesting, cross-validation, and performance visualization.
    """
    
    def __init__(self, config_path='config.json'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize Redis connection
        self.redis = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=True
        )
        
        # Initialize metrics calculator
        self.metrics_calculator = StrategyPerformanceMetrics()
        
        # Evaluation parameters
        self.k_folds = int(os.getenv('EVALUATION_K_FOLDS', '5'))
        self.min_trades_per_fold = int(os.getenv('MIN_TRADES_PER_FOLD', '10'))
        self.evaluation_metrics = self.config.get('evaluation', {}).get('metrics', [
            'sharpe_ratio', 'win_rate', 'profit_factor', 'max_drawdown'
        ])
        
        logger.info("Strategy Evaluation System initialized")
        logger.info(f"- K-Folds: {self.k_folds}")
        logger.info(f"- Min Trades Per Fold: {self.min_trades_per_fold}")
        logger.info(f"- Evaluation Metrics: {self.evaluation_metrics}")
    
    async def get_historical_market_data(self, symbol: str = None, 
                                      timeframe: str = None, 
                                      start_time: str = None, 
                                      end_time: str = None) -> List[Dict]:
        """
        Retrieve historical market data for backtesting.
        
        Args:
            symbol: Trading pair symbol (e.g. 'BTCUSDC')
            timeframe: Candlestick timeframe (e.g. '1h', '15m')
            start_time: ISO format datetime string
            end_time: ISO format datetime string
            
        Returns:
            List of market data points
        """
        try:
            # Try to get from Redis
            hist_data_key = 'historical_market_data'
            if symbol:
                hist_data_key += f"_{symbol}"
            if timeframe:
                hist_data_key += f"_{timeframe}"
            
            data = await self.redis.get(hist_data_key)
            
            if not data:
                logger.warning(f"No historical data found for {hist_data_key}")
                return []
            
            market_data = json.loads(data)
            
            # Filter by time range if provided
            if start_time or end_time:
                filtered_data = []
                
                for point in market_data:
                    if 'timestamp' not in point:
                        continue
                        
                    timestamp = point['timestamp']
                    
                    if start_time and timestamp < start_time:
                        continue
                        
                    if end_time and timestamp > end_time:
                        continue
                        
                    filtered_data.append(point)
                    
                return filtered_data
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting historical market data: {str(e)}")
            return []
    
    async def _simulate_trades(self, strategy_code: str, market_data: List[Dict]) -> List[Dict]:
        """
        Simulate trades using a strategy against historical market data.
        
        Args:
            strategy_code: JavaScript code of the strategy
            market_data: Historical market data to run simulation against
            
        Returns:
            List of simulated trades
        """
        # In a real implementation, this would execute the strategy code against the market data
        # For now, we'll use a simplified approach that generates synthetic trades based on the data
        
        # Placeholder implementation for demo purposes - in a real system this would:
        # 1. Set up a VM sandbox for JavaScript execution
        # 2. Run the strategy code against each data point
        # 3. Track entries, exits, and position management
        # 4. Return the resulting trades
        
        simulated_trades = []
        
        if not market_data:
            return []
            
        # Simplified trade simulation based on market data patterns
        # (This is just a placeholder - actual simulation would execute the strategy code)
        in_position = False
        entry_price = 0
        entry_time = None
        
        for i, data_point in enumerate(market_data):
            if i < 5:  # Skip first few points to allow for indicators
                continue
            
            # Extract price and timestamp
            price = data_point.get('current_price', 0)
            timestamp = data_point.get('timestamp')
            
            if not price or not timestamp:
                continue
            
            # Simple decision logic (just for placeholder simulation)
            rsi = data_point.get('rsi', 50)
            trend = data_point.get('trend', 'neutral')
            
            # Entry logic
            if not in_position:
                if (rsi < 30 and trend != 'downtrend') or (rsi > 70 and trend == 'uptrend'):
                    entry_price = price
                    entry_time = timestamp
                    in_position = True
            
            # Exit logic
            elif in_position:
                # Exit after 10 data points or on opposing signal
                if i > 10 and ((rsi > 70 and entry_price < price) or (rsi < 30 and entry_price > price)):
                    exit_price = price
                    profit_pct = ((exit_price - entry_price) / entry_price) * 100
                    
                    # Create trade record
                    trade = {
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit_pct,
                        'direction': 'long' if entry_price < exit_price else 'short'
                    }
                    
                    simulated_trades.append(trade)
                    in_position = False
        
        return simulated_trades
    
    async def cross_validate_strategy(self, strategy_id: str, strategy_code: str) -> Dict:
        """
        Perform k-fold cross-validation of a strategy across different market periods.
        
        Args:
            strategy_id: Unique identifier for the strategy
            strategy_code: JavaScript code of the strategy
            
        Returns:
            Dictionary with cross-validation results
        """
        try:
            # Get historical data
            market_data = await self.get_historical_market_data()
            
            if not market_data:
                logger.warning("No historical data available for cross-validation")
                return {
                    'status': 'error',
                    'message': 'No historical data available'
                }
            
            # Sort data by timestamp
            market_data.sort(key=lambda x: x.get('timestamp', ''))
            
            # Create k folds
            fold_size = len(market_data) // self.k_folds
            folds = []
            
            for i in range(self.k_folds):
                start_idx = i * fold_size
                end_idx = (i + 1) * fold_size if i < self.k_folds - 1 else len(market_data)
                fold_data = market_data[start_idx:end_idx]
                folds.append(fold_data)
            
            # Perform k-fold cross-validation
            fold_results = []
            
            for i, fold_data in enumerate(folds):
                # Simulate trades on this fold
                trades = await self._simulate_trades(strategy_code, fold_data)
                
                # Check if we have enough trades
                if len(trades) < self.min_trades_per_fold:
                    logger.warning(f"Fold {i+1} has insufficient trades: {len(trades)} < {self.min_trades_per_fold}")
                    continue
                
                # Calculate metrics
                metrics = self.metrics_calculator.calculate_metrics(trades)
                
                # Get market regime for this fold
                start_time = fold_data[0].get('timestamp')
                end_time = fold_data[-1].get('timestamp')
                
                # Determine market regime
                uptrend_count = sum(1 for point in fold_data if point.get('trend') == 'uptrend')
                downtrend_count = sum(1 for point in fold_data if point.get('trend') == 'downtrend')
                sideways_count = sum(1 for point in fold_data if point.get('trend') == 'sideways')
                
                # Calculate volatility
                if len(fold_data) > 1:
                    prices = [point.get('current_price', 0) for point in fold_data]
                    returns = np.diff(prices) / prices[:-1]
                    volatility = np.std(returns) * np.sqrt(24)  # Annualized to 24-hour scale
                else:
                    volatility = 0
                
                # Determine regime
                if volatility > 0.015:  # High volatility threshold
                    regime = "volatile"
                elif uptrend_count > max(downtrend_count, sideways_count):
                    regime = "bull"
                elif downtrend_count > max(uptrend_count, sideways_count):
                    regime = "bear"
                else:
                    regime = "ranging"
                
                # Add result for this fold
                fold_results.append({
                    'fold': i + 1,
                    'trades': len(trades),
                    'metrics': metrics,
                    'regime': regime,
                    'volatility': volatility,
                    'start_time': start_time,
                    'end_time': end_time
                })
            
            # Calculate overall and regime-specific metrics
            overall_metrics = self._aggregate_fold_metrics(fold_results)
            regime_metrics = self._calculate_regime_specific_metrics(fold_results)
            
            # Store results in Redis
            cross_validation_results = {
                'strategy_id': strategy_id,
                'timestamp': datetime.now().isoformat(),
                'fold_count': self.k_folds,
                'fold_results': fold_results,
                'overall_metrics': overall_metrics,
                'regime_metrics': regime_metrics
            }
            
            await self.redis.set(
                f"strategy_cv_results_{strategy_id}",
                json.dumps(cross_validation_results)
            )
            
            return cross_validation_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _aggregate_fold_metrics(self, fold_results: List[Dict]) -> Dict:
        """
        Aggregate metrics across all folds.
        
        Args:
            fold_results: List of results from each fold
            
        Returns:
            Dictionary of aggregated metrics
        """
        if not fold_results:
            return {}
        
        # Extract metrics from all folds
        metrics_list = [fold['metrics'] for fold in fold_results]
        
        # Calculate weighted average based on number of trades in each fold
        total_trades = sum(fold['trades'] for fold in fold_results)
        
        if total_trades == 0:
            return {}
        
        # Initialize aggregated metrics
        aggregated_metrics = {}
        
        # Calculate weighted average for each metric
        for metric in self.evaluation_metrics:
            weighted_sum = sum(
                fold['metrics'].get(metric, 0) * fold['trades']
                for fold in fold_results
            )
            aggregated_metrics[metric] = weighted_sum / total_trades
        
        # Add trade count
        aggregated_metrics['trade_count'] = total_trades
        
        return aggregated_metrics
    
    def _calculate_regime_specific_metrics(self, fold_results: List[Dict]) -> Dict:
        """
        Calculate performance metrics for each market regime.
        
        Args:
            fold_results: List of results from each fold
            
        Returns:
            Dictionary of regime-specific metrics
        """
        # Group folds by regime
        regime_folds = {
            'bull': [],
            'bear': [],
            'ranging': [],
            'volatile': []
        }
        
        for fold in fold_results:
            regime = fold.get('regime')
            if regime in regime_folds:
                regime_folds[regime].append(fold)
        
        # Calculate metrics for each regime
        regime_metrics = {}
        
        for regime, folds in regime_folds.items():
            if folds:
                regime_metrics[regime] = self._aggregate_fold_metrics(folds)
            else:
                regime_metrics[regime] = {}
        
        return regime_metrics
    
    async def compare_strategies(self, strategy_ids: List[str]) -> Dict:
        """
        Compare multiple strategies across different metrics and market regimes.
        
        Args:
            strategy_ids: List of strategy IDs to compare
            
        Returns:
            Dictionary with comparison results and visualizations
        """
        try:
            # Get cross-validation results for each strategy
            cv_results = []
            
            for strategy_id in strategy_ids:
                cv_result = await self.redis.get(f"strategy_cv_results_{strategy_id}")
                
                if cv_result:
                    cv_results.append(json.loads(cv_result))
                else:
                    logger.warning(f"No cross-validation results found for strategy {strategy_id}")
            
            if not cv_results:
                return {
                    'status': 'error',
                    'message': 'No cross-validation results found for the specified strategies'
                }
            
            # Prepare comparison data
            comparison_data = {
                'strategies': [],
                'overall_comparison': {},
                'regime_comparison': {
                    'bull': {},
                    'bear': {},
                    'ranging': {},
                    'volatile': {}
                },
                'rankings': {}
            }
            
            # Extract data for each strategy
            for cv_result in cv_results:
                strategy_id = cv_result.get('strategy_id')
                
                # Get strategy details from Redis
                strategy_details = await self.redis.get(f"worker_deployment_{strategy_id}")
                
                if strategy_details:
                    details = json.loads(strategy_details)
                    strategy_name = details.get('name', strategy_id)
                else:
                    strategy_name = strategy_id
                
                # Add to strategies list
                comparison_data['strategies'].append({
                    'id': strategy_id,
                    'name': strategy_name,
                    'overall_metrics': cv_result.get('overall_metrics', {}),
                    'regime_metrics': cv_result.get('regime_metrics', {})
                })
                
                # Add overall metrics to comparison
                for metric in self.evaluation_metrics:
                    if metric not in comparison_data['overall_comparison']:
                        comparison_data['overall_comparison'][metric] = []
                    
                    comparison_data['overall_comparison'][metric].append({
                        'strategy_id': strategy_id,
                        'strategy_name': strategy_name,
                        'value': cv_result.get('overall_metrics', {}).get(metric, 0)
                    })
                
                # Add regime-specific metrics to comparison
                for regime in ['bull', 'bear', 'ranging', 'volatile']:
                    regime_metrics = cv_result.get('regime_metrics', {}).get(regime, {})
                    
                    for metric in self.evaluation_metrics:
                        if metric not in comparison_data['regime_comparison'][regime]:
                            comparison_data['regime_comparison'][regime][metric] = []
                        
                        comparison_data['regime_comparison'][regime][metric].append({
                            'strategy_id': strategy_id,
                            'strategy_name': strategy_name,
                            'value': regime_metrics.get(metric, 0)
                        })
            
            # Calculate rankings for each metric
            for metric in self.evaluation_metrics:
                # Overall ranking
                overall_ranking = sorted(
                    comparison_data['overall_comparison'].get(metric, []),
                    key=lambda x: x['value'],
                    reverse=True  # Higher is better for most metrics
                )
                
                # Special case for drawdown, lower is better
                if metric == 'max_drawdown':
                    overall_ranking.reverse()
                
                comparison_data['rankings'][metric] = {
                    'overall': overall_ranking,
                    'regimes': {}
                }
                
                # Regime-specific rankings
                for regime in ['bull', 'bear', 'ranging', 'volatile']:
                    regime_ranking = sorted(
                        comparison_data['regime_comparison'][regime].get(metric, []),
                        key=lambda x: x['value'],
                        reverse=True  # Higher is better for most metrics
                    )
                    
                    # Special case for drawdown, lower is better
                    if metric == 'max_drawdown':
                        regime_ranking.reverse()
                    
                    comparison_data['rankings'][metric]['regimes'][regime] = regime_ranking
            
            # Generate comparison visualizations
            comparison_data['visualizations'] = self._generate_comparison_visualizations(comparison_data)
            
            # Store comparison results in Redis
            comparison_id = f"strategy_comparison_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            await self.redis.set(
                comparison_id,
                json.dumps(comparison_data)
            )
            
            return comparison_data
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _generate_comparison_visualizations(self, comparison_data: Dict) -> Dict:
        """
        Generate visualizations for strategy comparison.
        
        Args:
            comparison_data: Strategy comparison data
            
        Returns:
            Dictionary of visualization data
        """
        visualizations = {}
        
        # Set up the style
        plt.style.use('seaborn-darkgrid')
        
        # 1. Radar Chart for Overall Performance
        strategy_names = [s['name'] for s in comparison_data['strategies']]
        metrics = self.evaluation_metrics
        
        if not metrics or not strategy_names:
            return {}
        
        # Normalize metrics for radar chart
        normalized_metrics = {}
        
        for metric in metrics:
            metric_values = []
            
            for strategy in comparison_data['strategies']:
                value = strategy['overall_metrics'].get(metric, 0)
                metric_values.append(value)
            
            # Skip if all values are 0
            if all(v == 0 for v in metric_values):
                continue
            
            # Normalize values to [0, 1]
            min_val = min(metric_values)
            max_val = max(metric_values)
            
            if max_val == min_val:
                normalized = [1 for _ in metric_values]
            else:
                if metric == 'max_drawdown':  # Lower is better for drawdown
                    normalized = [1 - ((v - min_val) / (max_val - min_val)) for v in metric_values]
                else:  # Higher is better for other metrics
                    normalized = [(v - min_val) / (max_val - min_val) for v in metric_values]
            
            normalized_metrics[metric] = normalized
        
        if not normalized_metrics:
            return {}
        
        # Create radar chart
        metrics_for_radar = list(normalized_metrics.keys())
        num_metrics = len(metrics_for_radar)
        
        # Create angles for each metric
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        colors = ['#4CAF50', '#2196F3', '#FFC107', '#9C27B0', '#F44336', '#00BCD4', '#FF9800', '#795548']
        
        for i, strategy in enumerate(comparison_data['strategies']):
            strategy_values = []
            
            for metric in metrics_for_radar:
                idx = comparison_data['strategies'].index(strategy)
                strategy_values.append(normalized_metrics[metric][idx])
            
            # Close the loop
            strategy_values += strategy_values[:1]
            
            # Plot the strategy
            color = colors[i % len(colors)]
            ax.plot(angles, strategy_values, color=color, linewidth=2, label=strategy['name'])
            ax.fill(angles, strategy_values, color=color, alpha=0.1)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_for_radar)
        ax.set_title('Strategy Performance Comparison', fontsize=16)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        visualizations['radar_chart'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # 2. Bar Charts for Key Metrics
        for metric in metrics_for_radar:
            plt.figure(figsize=(12, 6))
            
            values = []
            for strategy in comparison_data['strategies']:
                values.append(strategy['overall_metrics'].get(metric, 0))
            
            bars = plt.bar(strategy_names, values, color=colors[:len(strategy_names)])
            
            plt.title(f'{metric.replace("_", " ").title()} Comparison', fontsize=16)
            plt.xlabel('Strategy')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}',
                        ha='center', va='bottom', rotation=0)
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            visualizations[f'bar_{metric}'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
        
        # 3. Regime Performance Table (as HTML)
        html_table = """
        <table class="table table-striped table-bordered">
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Metric</th>
                    <th>Overall</th>
                    <th>Bull Market</th>
                    <th>Bear Market</th>
                    <th>Ranging Market</th>
                    <th>Volatile Market</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for strategy in comparison_data['strategies']:
            for metric in metrics_for_radar:
                html_table += f"""
                <tr>
                    <td>{strategy['name']}</td>
                    <td>{metric.replace('_', ' ').title()}</td>
                    <td>{strategy['overall_metrics'].get(metric, 0):.4f}</td>
                    <td>{strategy['regime_metrics'].get('bull', {}).get(metric, 0):.4f}</td>
                    <td>{strategy['regime_metrics'].get('bear', {}).get(metric, 0):.4f}</td>
                    <td>{strategy['regime_metrics'].get('ranging', {}).get(metric, 0):.4f}</td>
                    <td>{strategy['regime_metrics'].get('volatile', {}).get(metric, 0):.4f}</td>
                </tr>
                """
        
        html_table += """
            </tbody>
        </table>
        """
        
        visualizations['regime_table'] = html_table
        
        return visualizations

    def generate_html_report(self, strategy_id: str, evaluation_results: Dict) -> str:
        """
        Generate an HTML report for strategy evaluation results.
        
        Args:
            strategy_id: Strategy ID
            evaluation_results: Evaluation results dictionary
            
        Returns:
            HTML report as string
        """
        # Get strategy name
        strategy_name = strategy_id
        
        # Start HTML report
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Strategy Evaluation Report - {strategy_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }}
                h1, h2, h3 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .metrics-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .metrics-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .chart-container {{ margin-bottom: 30px; }}
                .chart {{ max-width: 100%; height: auto; }}
                .regime-box {{ margin-bottom: 20px; padding: 15px; border-radius: 5px; }}
                .bull {{ background-color: rgba(76, 175, 80, 0.1); border: 1px solid #4CAF50; }}
                .bear {{ background-color: rgba(244, 67, 54, 0.1); border: 1px solid #F44336; }}
                .ranging {{ background-color: rgba(33, 150, 243, 0.1); border: 1px solid #2196F3; }}
                .volatile {{ background-color: rgba(255, 193, 7, 0.1); border: 1px solid #FFC107; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Strategy Evaluation Report</h1>
                <p><strong>Strategy ID:</strong> {strategy_id}</p>
                <p><strong>Strategy Name:</strong> {strategy_name}</p>
                <p><strong>Evaluation Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Overall Performance Metrics</h2>
        """
        
        # Add overall metrics table
        overall_metrics = evaluation_results.get('overall_metrics', {})
        if overall_metrics:
            html += """
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            """
            
            for metric, value in overall_metrics.items():
                # Format the value
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                
                html += f"""
                <tr>
                    <td>{metric.replace('_', ' ').title()}</td>
                    <td>{formatted_value}</td>
                </tr>
                """
            
            html += "</table>"
        
        # Add regime-specific performance sections
        regime_metrics = evaluation_results.get('regime_metrics', {})
        if regime_metrics:
            html += "<h2>Performance By Market Regime</h2>"
            
            for regime, metrics in regime_metrics.items():
                if not metrics:
                    continue
                
                html += f"""
                <div class="regime-box {regime}">
                    <h3>{regime.title()} Market Performance</h3>
                    
                    <table class="metrics-table">
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                """
                
                for metric, value in metrics.items():
                    # Format the value
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    
                    html += f"""
                    <tr>
                        <td>{metric.replace('_', ' ').title()}</td>
                        <td>{formatted_value}</td>
                    </tr>
                    """
                
                html += """
                    </table>
                </div>
                """
        
        # Add fold results section
        fold_results = evaluation_results.get('fold_results', [])
        if fold_results:
            html += "<h2>Cross-Validation Results</h2>"
            
            for i, fold in enumerate(fold_results):
                fold_num = fold.get('fold', i + 1)
                trades = fold.get('trades', 0)
                regime = fold.get('regime', 'unknown')
                start_time = fold.get('start_time', '')
                end_time = fold.get('end_time', '')
                
                html += f"""
                <div class="regime-box {regime}">
                    <h3>Fold {fold_num} ({regime.title()} Market)</h3>
                    <p><strong>Trades:</strong> {trades}</p>
                    <p><strong>Period:</strong> {start_time} to {end_time}</p>
                    
                    <table class="metrics-table">
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                """
                
                fold_metrics = fold.get('metrics', {})
                for metric, value in fold_metrics.items():
                    # Format the value
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    
                    html += f"""
                    <tr>
                        <td>{metric.replace('_', ' ').title()}</td>
                        <td>{formatted_value}</td>
                    </tr>
                    """
                
                html += """
                    </table>
                </div>
                """
        
        # Close HTML document
        html += """
            </div>
        </body>
        </html>
        """
        
        return html