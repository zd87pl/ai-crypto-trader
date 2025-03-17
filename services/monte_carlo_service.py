import os
import json
import socket
import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging as logger
from logging.handlers import RotatingFileHandler
from redis.asyncio import Redis
from redis.exceptions import ConnectionError
from typing import Dict, List, Any, Optional, Tuple
from binance.client import Client
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('reports/monte_carlo', exist_ok=True)

# Configure rotating file handler
rotating_handler = RotatingFileHandler(
    'logs/monte_carlo.log',
    maxBytes=10*1024*1024,  # 10MB per file
    backupCount=5,  # Keep 5 backup files
    encoding='utf-8'
)

# Configure logging with rotation
logger.basicConfig(
    level=logger.DEBUG,
    format='%(asctime)s - %(levelname)s - [MonteCarlo] %(message)s',
    handlers=[
        rotating_handler,
        logger.StreamHandler()
    ]
)

class MonteCarloService:
    """
    Service for running Monte Carlo simulations for portfolio risk projection
    Implements RISK-10: Add Monte Carlo simulations for risk projection
    """
    
    def __init__(self):
        """Initialize the Monte Carlo simulation service"""
        logger.debug("Initializing Monte Carlo Service...")
        
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        logger.debug("Loaded configuration")
        
        # Initialize Binance client for historical data
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        if not api_key or not api_secret:
            raise ValueError("Binance API credentials not set")
            
        self.client = Client(api_key, api_secret)
        logger.info("Initialized Binance client")
        
        # Redis configuration
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis = None
        
        # Service state
        self.running = True
        self.service_port = int(os.getenv('MONTE_CARLO_PORT', 8009))
        
        # Monte Carlo parameters
        self.mc_params = self.config.get('monte_carlo', {})
        if not self.mc_params:
            # Set default parameters if not found in config
            self.mc_params = {
                'num_simulations': 1000,
                'time_horizon_days': 30,
                'confidence_level': 0.95,
                'lookback_days': 60,
                'return_method': 'log',  # 'simple' or 'log'
                'simulation_method': 'geometric_brownian_motion',  # 'historical' or 'geometric_brownian_motion'
                'plot_chart': True,
                'generate_reports': True,
                'report_frequency': 'daily',  # 'hourly', 'daily', 'weekly', 'monthly'
                'scenarios': {
                    'base': {},
                    'bull': {'drift_factor': 1.5, 'volatility_factor': 0.8},
                    'bear': {'drift_factor': 0.5, 'volatility_factor': 1.2},
                    'volatile': {'drift_factor': 1.0, 'volatility_factor': 2.0},
                    'crab': {'drift_factor': 0.2, 'volatility_factor': 0.5}
                }
            }
            
        # Add Monte Carlo parameters to config if not present
        if 'monte_carlo' not in self.config:
            self.config['monte_carlo'] = self.mc_params
            with open('config.json', 'w') as f:
                json.dump(self.config, f, indent=4)
                
        # Simulation data storage
        self.historical_data = {}
        self.simulation_results = {}
        self.last_simulation_time = {}
        
        logger.debug("Monte Carlo Service initialization complete")
            
    async def connect_redis(self, max_retries=5, retry_delay=5):
        """Establish Redis connection with retries"""
        retries = 0
        while retries < max_retries:
            try:
                if self.redis is None:
                    self.redis = Redis(
                        host=self.redis_host,
                        port=self.redis_port,
                        decode_responses=True
                    )
                await self.redis.ping()
                logger.info("Successfully connected to Redis")
                return True
            except (ConnectionError, Exception) as e:
                retries += 1
                logger.error(f"Failed to connect to Redis (attempt {retries}/{max_retries}): {str(e)}")
                if retries < max_retries:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("Max retries reached. Could not connect to Redis.")
                    return False
    
    async def fetch_holdings(self) -> Dict:
        """Fetch current portfolio holdings from Redis"""
        try:
            if not self.redis:
                await self.connect_redis()
                
            holdings_json = await self.redis.get('holdings')
            if holdings_json:
                holdings = json.loads(holdings_json)
                logger.debug(f"Fetched holdings: {len(holdings['assets'])} assets, total value: ${holdings['total_value']:.2f}")
                return holdings
            else:
                logger.warning("No holdings data found in Redis")
                return {'assets': {}, 'total_value': 0, 'available_usdc': 0}
                
        except Exception as e:
            logger.error(f"Error fetching holdings: {str(e)}")
            return {'assets': {}, 'total_value': 0, 'available_usdc': 0}
    
    def fetch_historical_prices(self, symbol: str, days: int = 60, interval: str = '1d') -> pd.DataFrame:
        """Fetch historical price data for a symbol"""
        try:
            logger.debug(f"Fetching {days} days of historical data for {symbol} with interval {interval}")
            
            # Set the appropriate interval from Binance constants
            binance_interval = {
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '4h': Client.KLINE_INTERVAL_4HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY
            }.get(interval, Client.KLINE_INTERVAL_1DAY)
            
            # Get klines (candlestick data)
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=binance_interval,
                limit=days
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close'] = pd.to_numeric(df['close'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            # Calculate returns based on the specified method
            if self.mc_params['return_method'] == 'log':
                df['returns'] = np.log(df['close'] / df['close'].shift(1))
            else:  # simple returns
                df['returns'] = df['close'].pct_change()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical prices for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def run_monte_carlo_simulation(self, symbol: str, initial_price: float, 
                                   days: int = None, num_simulations: int = None,
                                   scenario: str = 'base') -> Dict:
        """
        Run Monte Carlo simulation for price projection
        
        Parameters:
        - symbol: Trading symbol
        - initial_price: Starting price for simulation
        - days: Number of days to project (default from config)
        - num_simulations: Number of paths to simulate (default from config)
        - scenario: Scenario to simulate ('base', 'bull', 'bear', 'volatile', 'crab')
        
        Returns:
        - Dictionary with simulation results
        """
        try:
            logger.info(f"Running Monte Carlo simulation for {symbol}")
            
            # Use defaults from config if not provided
            if days is None:
                days = self.mc_params['time_horizon_days']
            if num_simulations is None:
                num_simulations = self.mc_params['num_simulations']
                
            # Get historical data for volatility and drift calculation
            if symbol not in self.historical_data:
                self.historical_data[symbol] = self.fetch_historical_prices(
                    symbol, 
                    days=self.mc_params['lookback_days']
                )
            
            df = self.historical_data[symbol]
            
            if df.empty:
                logger.error(f"No historical data available for {symbol}")
                return {}
                
            # Calculate drift and volatility parameters
            returns = df['returns'].dropna()
            
            # Calculate annualized drift (mu) and volatility (sigma)
            # Number of trading periods in a year (approx. 252 trading days)
            periods_per_year = 252
            
            if self.mc_params['return_method'] == 'log':
                mu = returns.mean() * periods_per_year
                sigma = returns.std() * np.sqrt(periods_per_year)
            else:
                mu = returns.mean() * periods_per_year
                sigma = returns.std() * np.sqrt(periods_per_year)
            
            # Apply scenario modifications if applicable
            scenario_params = self.mc_params['scenarios'].get(scenario, {})
            drift_factor = scenario_params.get('drift_factor', 1.0)
            volatility_factor = scenario_params.get('volatility_factor', 1.0)
            
            # Modify drift and volatility based on scenario
            mu = mu * drift_factor
            sigma = sigma * volatility_factor
            
            # Time increment (in years)
            dt = 1/periods_per_year
            
            # Generate simulation paths
            simulation_method = self.mc_params['simulation_method']
            
            if simulation_method == 'geometric_brownian_motion':
                # Geometric Brownian Motion
                paths = np.zeros((days, num_simulations))
                paths[0] = initial_price
                
                for t in range(1, days):
                    # Generate random standard normal values
                    Z = np.random.standard_normal(num_simulations)
                    # Apply GBM formula
                    paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            
            elif simulation_method == 'historical':
                # Historical simulation (bootstrapping returns)
                paths = np.zeros((days, num_simulations))
                paths[0] = initial_price
                
                # Use returns resampling
                for i in range(num_simulations):
                    # Sample returns with replacement
                    sampled_returns = returns.sample(days-1, replace=True).values
                    
                    if self.mc_params['return_method'] == 'log':
                        # For log returns: P_t = P_{t-1} * exp(r_t)
                        path = np.zeros(days)
                        path[0] = initial_price
                        for t in range(1, days):
                            path[t] = path[t-1] * np.exp(sampled_returns[t-1])
                    else:
                        # For simple returns: P_t = P_{t-1} * (1 + r_t)
                        path = np.zeros(days)
                        path[0] = initial_price
                        for t in range(1, days):
                            path[t] = path[t-1] * (1 + sampled_returns[t-1])
                            
                    paths[:, i] = path
            
            else:
                logger.error(f"Unknown simulation method: {simulation_method}")
                return {}
            
            # Calculate statistics from simulations
            final_prices = paths[-1, :]
            
            # Calculate percentiles for confidence intervals
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            percentile_values = np.percentile(final_prices, percentiles)
            
            # Calculate percentage changes from initial price
            pct_changes = (final_prices / initial_price - 1) * 100
            
            # Calculate Value at Risk (VaR)
            confidence = self.mc_params['confidence_level']
            var_percentile = 100 * (1 - confidence)
            var = np.percentile(pct_changes, var_percentile)
            
            # Expected Shortfall (Conditional VaR)
            cvar = np.mean(pct_changes[pct_changes <= var])
            
            # Probability of profit/loss
            prob_profit = np.mean(final_prices > initial_price)
            prob_loss = 1 - prob_profit
            
            # Maximum drawdown analysis across all paths
            max_drawdowns = []
            for i in range(num_simulations):
                path = paths[:, i]
                # Calculate running maximum
                running_max = np.maximum.accumulate(path)
                # Calculate drawdown
                drawdown = (running_max - path) / running_max
                # Get maximum drawdown
                max_drawdown = drawdown.max()
                max_drawdowns.append(max_drawdown)
            
            # Generate simulation results
            results = {
                'symbol': symbol,
                'initial_price': initial_price,
                'time_horizon_days': days,
                'num_simulations': num_simulations,
                'mu': mu,
                'sigma': sigma,
                'drift_factor': drift_factor,
                'volatility_factor': volatility_factor,
                'simulation_method': simulation_method,
                'scenario': scenario,
                'timestamp': datetime.now().isoformat(),
                'percentiles': {
                    str(p): {
                        'price': float(v),
                        'pct_change': float((v / initial_price - 1) * 100)
                    }
                    for p, v in zip(percentiles, percentile_values)
                },
                'expected': {
                    'price': float(np.mean(final_prices)),
                    'pct_change': float(np.mean(pct_changes))
                },
                'risk_metrics': {
                    'var': float(abs(var)),
                    'cvar': float(abs(cvar)),
                    'prob_profit': float(prob_profit),
                    'prob_loss': float(prob_loss),
                    'max_drawdown': {
                        'mean': float(np.mean(max_drawdowns)),
                        'median': float(np.median(max_drawdowns)),
                        'max': float(np.max(max_drawdowns))
                    }
                },
                'paths': paths.tolist() if self.mc_params.get('store_all_paths', False) else None
            }
            
            # Store simulation results
            self.simulation_results[symbol] = results
            self.last_simulation_time[symbol] = datetime.now()
            
            # Generate visualization if enabled
            if self.mc_params['plot_chart']:
                self._generate_simulation_chart(symbol, paths, initial_price, results)
            
            logger.info(f"Completed Monte Carlo simulation for {symbol}")
            logger.info(f"Expected price after {days} days: ${results['expected']['price']:.2f} "
                       f"({results['expected']['pct_change']:.2f}%)")
            logger.info(f"95% VaR: {results['risk_metrics']['var']:.2f}%")
            logger.info(f"Average Max Drawdown: {results['risk_metrics']['max_drawdown']['mean'] * 100:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running Monte Carlo simulation for {symbol}: {str(e)}", exc_info=True)
            return {}
    
    def _generate_simulation_chart(self, symbol: str, paths: np.ndarray, 
                                  initial_price: float, results: Dict) -> str:
        """
        Generate chart visualization for Monte Carlo simulation
        
        Parameters:
        - symbol: Trading symbol
        - paths: Simulation paths array
        - initial_price: Starting price
        - results: Simulation results dictionary
        
        Returns:
        - Path to the generated chart file
        """
        try:
            # Create directory for charts if it doesn't exist
            chart_dir = 'reports/monte_carlo'
            os.makedirs(chart_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{chart_dir}/{symbol}_montecarlo_{timestamp}.png"
            
            # Setup figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Set style
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # Plot a subset of simulation paths (100 random paths)
            if paths.shape[1] > 100:
                indices = np.random.choice(paths.shape[1], 100, replace=False)
                subset_paths = paths[:, indices]
            else:
                subset_paths = paths
                
            # Time axis (days)
            days = np.arange(paths.shape[0])
            
            # Plot paths with low opacity
            for i in range(subset_paths.shape[1]):
                ax.plot(days, subset_paths[:, i], linewidth=0.5, alpha=0.1, color='blue')
            
            # Plot percentile lines
            percentiles = [5, 50, 95]
            percentile_paths = np.percentile(paths, percentiles, axis=1)
            
            colors = ['red', 'black', 'green']
            labels = ['5th Percentile', 'Median', '95th Percentile']
            
            for i, (percentile, color, label) in enumerate(zip(percentiles, colors, labels)):
                ax.plot(days, percentile_paths[i], linewidth=2, color=color, label=label)
            
            # Add initial price as horizontal line
            ax.axhline(y=initial_price, color='gray', linestyle='--', alpha=0.7, label='Initial Price')
            
            # Formatting
            ax.set_title(f'Monte Carlo Simulation for {symbol} - {results["scenario"].capitalize()} Scenario', fontsize=16)
            ax.set_xlabel('Days', fontsize=12)
            ax.set_ylabel('Price (USDC)', fontsize=12)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Add annotations with key statistics
            expected_price = results['expected']['price']
            expected_change = results['expected']['pct_change']
            var = results['risk_metrics']['var']
            prob_profit = results['risk_metrics']['prob_profit'] * 100
            avg_max_drawdown = results['risk_metrics']['max_drawdown']['mean'] * 100
            
            info_text = (
                f"Expected Price: ${expected_price:.2f} ({expected_change:.2f}%)\n"
                f"VaR (95%): {var:.2f}%\n"
                f"Prob. of Profit: {prob_profit:.1f}%\n"
                f"Avg. Max Drawdown: {avg_max_drawdown:.2f}%\n"
                f"Simulations: {paths.shape[1]}"
            )
            
            # Add text box with statistics
            props = dict(boxstyle='round', facecolor='white', alpha=0.7)
            ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', bbox=props)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Generated Monte Carlo simulation chart: {filename}")
            
            return filename
            
        except Exception as e:
            logger.error(f"Error generating simulation chart for {symbol}: {str(e)}")
            return ""
    
    async def run_portfolio_monte_carlo(self) -> Dict:
        """
        Run Monte Carlo simulations for the entire portfolio
        
        Returns:
        - Dictionary with portfolio simulation results
        """
        try:
            logger.info("Running Monte Carlo simulations for portfolio")
            
            # Fetch current holdings
            holdings = await self.fetch_holdings()
            
            # Exit if no holdings
            if not holdings or not holdings['assets']:
                logger.debug("No holdings to analyze")
                return {}
                
            # Get list of assets in portfolio (exclude USDC)
            assets = [asset for asset in holdings['assets'].keys() if asset != 'USDC']
            if not assets:
                logger.debug("No non-USDC assets in portfolio")
                return {}
                
            # Create dictionary to store simulation results
            portfolio_simulations = {}
            
            # Run simulations for each asset
            for asset in assets:
                symbol = f"{asset}USDC"
                current_price = holdings['assets'][asset].get('current_price', 0)
                
                if current_price == 0:
                    logger.warning(f"No current price available for {asset}, skipping simulation")
                    continue
                
                # Run simulations for different scenarios
                for scenario in self.mc_params['scenarios'].keys():
                    # Skip simulations that were run recently
                    scenario_key = f"{symbol}_{scenario}"
                    if (scenario_key in self.last_simulation_time and 
                        (datetime.now() - self.last_simulation_time[scenario_key]).total_seconds() < 3600):
                        logger.debug(f"Using recent simulation for {symbol} ({scenario} scenario)")
                        if scenario_key in self.simulation_results:
                            portfolio_simulations[scenario_key] = self.simulation_results[scenario_key]
                        continue
                    
                    # Run new simulation
                    results = self.run_monte_carlo_simulation(
                        symbol, 
                        current_price,
                        scenario=scenario
                    )
                    
                    # Store results
                    if results:
                        portfolio_simulations[scenario_key] = results
                        self.simulation_results[scenario_key] = results
                        self.last_simulation_time[scenario_key] = datetime.now()
            
            # Calculate portfolio-level statistics
            portfolio_stats = self._calculate_portfolio_stats(holdings, portfolio_simulations)
            
            # Store complete results
            complete_results = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': holdings['total_value'],
                'asset_simulations': portfolio_simulations,
                'portfolio_stats': portfolio_stats
            }
            
            # Store in Redis
            await self.redis.set('monte_carlo_results', json.dumps(complete_results))
            
            # Log summary
            logger.info(f"Completed Monte Carlo simulations for {len(assets)} assets")
            logger.info(f"Portfolio expected value change: {portfolio_stats['expected_change']:.2f}%")
            logger.info(f"Portfolio VaR (95%): {portfolio_stats['var']:.2f}%")
            
            return complete_results
            
        except Exception as e:
            logger.error(f"Error running portfolio Monte Carlo simulations: {str(e)}", exc_info=True)
            return {}
    
    def _calculate_portfolio_stats(self, holdings: Dict, simulations: Dict) -> Dict:
        """
        Calculate portfolio-level statistics from individual asset simulations
        
        Parameters:
        - holdings: Current portfolio holdings
        - simulations: Dictionary of simulation results for each asset
        
        Returns:
        - Dictionary with portfolio-level statistics
        """
        try:
            # Get assets in portfolio (exclude USDC)
            assets = [asset for asset in holdings['assets'].keys() if asset != 'USDC']
            
            # Calculate weights of each asset in portfolio
            weights = {}
            non_usdc_value = sum(holdings['assets'][asset]['value_usdc'] for asset in assets)
            
            if non_usdc_value == 0:
                logger.warning("No non-USDC assets with value in portfolio")
                return {}
                
            for asset in assets:
                weights[asset] = holdings['assets'][asset]['value_usdc'] / non_usdc_value
            
            # Collect expected returns and VaRs by scenario
            scenario_stats = {}
            
            for scenario in self.mc_params['scenarios'].keys():
                expected_returns = []
                vars_list = []
                cvars_list = []
                
                # Get weighted returns and risks for each asset
                for asset in assets:
                    symbol = f"{asset}USDC"
                    scenario_key = f"{symbol}_{scenario}"
                    
                    if scenario_key in simulations:
                        asset_weight = weights.get(asset, 0)
                        expected_return = simulations[scenario_key]['expected']['pct_change'] / 100  # Convert percentage to decimal
                        var_value = simulations[scenario_key]['risk_metrics']['var'] / 100  # Convert percentage to decimal
                        cvar_value = simulations[scenario_key]['risk_metrics']['cvar'] / 100  # Convert percentage to decimal
                        
                        expected_returns.append(expected_return * asset_weight)
                        vars_list.append(var_value * asset_weight)
                        cvars_list.append(cvar_value * asset_weight)
                
                # Calculate portfolio-level stats for this scenario
                if expected_returns:
                    scenario_stats[scenario] = {
                        'expected_return': sum(expected_returns),  # Simple weighted sum for expected return
                        'var': sum(vars_list),  # Simplified approach - ignores correlations
                        'cvar': sum(cvars_list)  # Simplified approach - ignores correlations
                    }
            
            # Compute consolidated portfolio statistics
            base_scenario = scenario_stats.get('base', {})
            expected_change = base_scenario.get('expected_return', 0) * 100  # Convert to percentage
            var_value = base_scenario.get('var', 0) * 100  # Convert to percentage
            cvar_value = base_scenario.get('cvar', 0) * 100  # Convert to percentage
            
            # Calculate potential portfolio value changes
            current_portfolio_value = holdings['total_value']
            expected_portfolio_value = current_portfolio_value * (1 + base_scenario.get('expected_return', 0))
            var_loss_value = current_portfolio_value * base_scenario.get('var', 0)
            
            portfolio_stats = {
                'current_value': current_portfolio_value,
                'expected_value': expected_portfolio_value,
                'expected_change': expected_change,
                'var': var_value,
                'cvar': cvar_value,
                'var_loss_value': var_loss_value,
                'scenario_stats': scenario_stats
            }
            
            return portfolio_stats
            
        except Exception as e:
            logger.error(f"Error calculating portfolio statistics: {str(e)}")
            return {}
    
    async def generate_monte_carlo_report(self, symbol: str = None) -> Dict:
        """
        Generate a detailed risk report using Monte Carlo simulations
        
        Parameters:
        - symbol: Optional symbol to generate report for (if None, generates for whole portfolio)
        
        Returns:
        - Report data dictionary
        """
        try:
            if symbol:
                logger.info(f"Generating Monte Carlo risk report for {symbol}")
                
                # Check if we have recent simulation data
                if symbol in self.simulation_results:
                    simulation = self.simulation_results[symbol]
                else:
                    # Fetch current price
                    price_data = self.client.get_symbol_ticker(symbol=symbol)
                    current_price = float(price_data['price'])
                    
                    # Run simulation
                    simulation = self.run_monte_carlo_simulation(symbol, current_price)
                
                # Generate report data
                report = {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'simulation': simulation,
                    'risk_assessment': {
                        'time_horizon': f"{simulation['time_horizon_days']} days",
                        'expected_return': f"{simulation['expected']['pct_change']:.2f}%",
                        'price_range': {
                            'low': f"${simulation['percentiles']['5']['price']:.4f}",
                            'median': f"${simulation['percentiles']['50']['price']:.4f}",
                            'high': f"${simulation['percentiles']['95']['price']:.4f}"
                        },
                        'risk_metrics': {
                            'var': f"{simulation['risk_metrics']['var']:.2f}%",
                            'cvar': f"{simulation['risk_metrics']['cvar']:.2f}%",
                            'probability_of_profit': f"{simulation['risk_metrics']['prob_profit']*100:.1f}%",
                            'probability_of_loss': f"{simulation['risk_metrics']['prob_loss']*100:.1f}%",
                            'max_drawdown': f"{simulation['risk_metrics']['max_drawdown']['mean']*100:.2f}%"
                        }
                    }
                }
                
                return report
                
            else:
                logger.info("Generating portfolio-wide Monte Carlo risk report")
                
                # Run portfolio simulation if not recent
                portfolio_results_json = await self.redis.get('monte_carlo_results')
                if portfolio_results_json:
                    portfolio_results = json.loads(portfolio_results_json)
                    last_timestamp = datetime.fromisoformat(portfolio_results['timestamp'])
                    
                    # Check if we need to run a new simulation (older than 1 hour)
                    if (datetime.now() - last_timestamp).total_seconds() > 3600:
                        portfolio_results = await self.run_portfolio_monte_carlo()
                else:
                    # Run new simulation
                    portfolio_results = await self.run_portfolio_monte_carlo()
                
                # Generate portfolio report
                portfolio_stats = portfolio_results.get('portfolio_stats', {})
                
                # Create portfolio risk summary
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'portfolio_value': portfolio_stats.get('current_value', 0),
                    'expected_value': portfolio_stats.get('expected_value', 0),
                    'expected_change': f"{portfolio_stats.get('expected_change', 0):.2f}%",
                    'value_at_risk': {
                        'var_percent': f"{portfolio_stats.get('var', 0):.2f}%",
                        'var_amount': f"${portfolio_stats.get('var_loss_value', 0):.2f}",
                        'cvar_percent': f"{portfolio_stats.get('cvar', 0):.2f}%"
                    },
                    'scenario_analysis': {}
                }
                
                # Add scenario analysis
                for scenario, stats in portfolio_stats.get('scenario_stats', {}).items():
                    report['scenario_analysis'][scenario] = {
                        'expected_return': f"{stats.get('expected_return', 0) * 100:.2f}%",
                        'var': f"{stats.get('var', 0) * 100:.2f}%",
                        'cvar': f"{stats.get('cvar', 0) * 100:.2f}%"
                    }
                
                # Add individual asset analysis
                report['asset_analysis'] = {}
                
                asset_simulations = portfolio_results.get('asset_simulations', {})
                for key, simulation in asset_simulations.items():
                    if '_base' in key:  # Only include base scenario in summary
                        symbol = key.split('_base')[0]
                        report['asset_analysis'][symbol] = {
                            'expected_return': f"{simulation['expected']['pct_change']:.2f}%",
                            'price_range': {
                                'low': f"${simulation['percentiles']['5']['price']:.4f}",
                                'median': f"${simulation['percentiles']['50']['price']:.4f}",
                                'high': f"${simulation['percentiles']['95']['price']:.4f}"
                            },
                            'var': f"{simulation['risk_metrics']['var']:.2f}%",
                            'prob_profit': f"{simulation['risk_metrics']['prob_profit']*100:.1f}%"
                        }
                
                return report
                
        except Exception as e:
            logger.error(f"Error generating Monte Carlo report: {str(e)}", exc_info=True)
            return {}
    
    async def handle_symbol_requests(self):
        """Handle requests for individual symbol simulations"""
        try:
            if not self.redis:
                await self.connect_redis()
                
            # Check for simulation requests
            request_json = await self.redis.get('monte_carlo_request')
            if request_json:
                try:
                    request = json.loads(request_json)
                    symbol = request.get('symbol')
                    
                    if symbol:
                        logger.info(f"Processing simulation request for {symbol}")
                        
                        # Get current price
                        price_data = self.client.get_symbol_ticker(symbol=symbol)
                        current_price = float(price_data['price'])
                        
                        # Run simulation with requested parameters
                        days = request.get('days', self.mc_params['time_horizon_days'])
                        num_simulations = request.get('num_simulations', self.mc_params['num_simulations'])
                        scenario = request.get('scenario', 'base')
                        
                        simulation = self.run_monte_carlo_simulation(
                            symbol,
                            current_price,
                            days=days,
                            num_simulations=num_simulations,
                            scenario=scenario
                        )
                        
                        # Generate report
                        report = await self.generate_monte_carlo_report(symbol)
                        
                        # Store in Redis
                        await self.redis.set(f'monte_carlo_{symbol}', json.dumps(simulation))
                        await self.redis.set(f'monte_carlo_report_{symbol}', json.dumps(report))
                        
                        # Clear request
                        await self.redis.delete('monte_carlo_request')
                        
                except Exception as e:
                    logger.error(f"Error processing simulation request: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error in handle_symbol_requests: {str(e)}")
    
    async def health_check_server(self):
        """Run a simple TCP server for health checks"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server.bind(('0.0.0.0', self.service_port))
            server.listen(1)
            server.setblocking(False)
            
            logger.info(f"Health check server listening on port {self.service_port}")
            
            while self.running:
                try:
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Health check server error: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to start health check server: {str(e)}")
        finally:
            server.close()
    
    async def run(self):
        """Run the Monte Carlo service"""
        try:
            logger.info("Starting Monte Carlo Service...")
            
            # Connect to Redis
            if not await self.connect_redis():
                raise Exception("Failed to establish initial Redis connection")
            
            # Start health check server
            asyncio.create_task(self.health_check_server())
            
            # Initial portfolio simulation
            await self.run_portfolio_monte_carlo()
            
            # Main service loop
            while self.running:
                try:
                    # Handle individual symbol requests
                    await self.handle_symbol_requests()
                    
                    # Run portfolio-wide simulations periodically (once per hour)
                    portfolio_results_json = await self.redis.get('monte_carlo_results')
                    if portfolio_results_json:
                        portfolio_results = json.loads(portfolio_results_json)
                        last_timestamp = datetime.fromisoformat(portfolio_results['timestamp'])
                        
                        # Check if we need to run a new simulation (older than 1 hour)
                        if (datetime.now() - last_timestamp).total_seconds() > 3600:
                            await self.run_portfolio_monte_carlo()
                    else:
                        # No previous results, run simulation
                        await self.run_portfolio_monte_carlo()
                    
                    # Generate periodical reports based on configuration
                    if self.mc_params['generate_reports']:
                        report_frequency = self.mc_params['report_frequency']
                        current_time = datetime.now()
                        
                        # Determine if report should be generated
                        generate_report = False
                        
                        if report_frequency == 'hourly':
                            # Generate report at the start of each hour
                            if current_time.minute < 5:
                                generate_report = True
                        elif report_frequency == 'daily':
                            # Generate report once a day (at midnight)
                            if current_time.hour == 0 and current_time.minute < 5:
                                generate_report = True
                        elif report_frequency == 'weekly':
                            # Generate report once a week (Monday midnight)
                            if current_time.weekday() == 0 and current_time.hour == 0 and current_time.minute < 5:
                                generate_report = True
                        
                        if generate_report:
                            # Generate portfolio report
                            report = await self.generate_monte_carlo_report()
                            await self.redis.set('monte_carlo_latest_report', json.dumps(report))
                            
                            # Save report to file
                            report_dir = 'reports/monte_carlo'
                            os.makedirs(report_dir, exist_ok=True)
                            
                            report_file = f"{report_dir}/portfolio_risk_{current_time.strftime('%Y%m%d_%H%M')}.json"
                            with open(report_file, 'w') as f:
                                json.dump(report, f, indent=4)
                                
                            logger.info(f"Generated periodic Monte Carlo report: {report_file}")
                    
                    # Wait before next cycle
                    await asyncio.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"Error in Monte Carlo service main loop: {str(e)}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logger.error(f"Error in Monte Carlo Service: {str(e)}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the Monte Carlo service"""
        logger.info("Stopping Monte Carlo Service...")
        self.running = False
        if self.redis:
            await self.redis.close()

if __name__ == "__main__":
    service = MonteCarloService()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        asyncio.run(service.stop())
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        asyncio.run(service.stop())