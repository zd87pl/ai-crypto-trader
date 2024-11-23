import os
import json
import redis
import asyncio
import aiohttp
import logging as logger
from datetime import datetime
from openai import AsyncOpenAI
from typing import Dict, List

# Configure logging
logger.basicConfig(
    level=logger.INFO,
    format='%(asctime)s - %(levelname)s - [StrategyEvolution] %(message)s',
    handlers=[
        logger.FileHandler('logs/strategy_evolution.log'),
        logger.StreamHandler()
    ]
)

class StrategyEvolutionService:
    def __init__(self):
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)

        # Initialize OpenAI client
        self.openai = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Initialize Redis connection
        self.redis = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=True
        )

        # Initialize Cloudflare API client
        self.cf_account_id = os.getenv('CLOUDFLARE_ACCOUNT_ID')
        self.cf_api_token = os.getenv('CLOUDFLARE_API_TOKEN')
        self.cf_api_base = f"https://api.cloudflare.com/client/v4/accounts/{self.cf_account_id}/workers"

        self.running = True
        self.active_strategies = {}

    async def generate_strategy(self, parameters: Dict) -> str:
        """Generate a new trading strategy using OpenAI"""
        try:
            prompt = self._create_strategy_prompt(parameters)
            response = await self.openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert algorithmic trader. Create a complete trading strategy in JavaScript for a Cloudflare Worker that implements the specified strategy type and parameters. Include proper risk management and position sizing."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            strategy_code = response.choices[0].message.content
            return self._validate_and_format_strategy(strategy_code)
            
        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            return None

    def _create_strategy_prompt(self, parameters: Dict) -> str:
        """Create a detailed prompt for strategy generation"""
        return f"""
        Create a trading strategy with the following parameters:
        - Strategy Type: {parameters.get('type', 'mean_reversion')}
        - Timeframe: {parameters.get('timeframe', '5m')}
        - Risk Limit: {parameters.get('risk_limit', 2)}%
        - Target Profit: {parameters.get('target_profit', 1.5)}%
        - Maximum Position Size: {parameters.get('max_position_size', 10)}%
        
        Include:
        1. Market data processing
        2. Signal generation
        3. Position sizing
        4. Risk management
        5. Entry/exit rules
        
        Return complete JavaScript code for a Cloudflare Worker.
        """

    async def backtest_strategy(self, strategy_code: str, market_data: List[Dict]) -> Dict:
        """Backtest a strategy using historical market data"""
        try:
            # Create temporary worker for backtesting
            worker_script = self._create_backtest_worker(strategy_code)
            
            # Deploy temporary worker
            worker_id = await self._deploy_worker(worker_script, is_test=True)
            
            # Run backtest
            results = await self._run_backtest(worker_id, market_data)
            
            # Clean up temporary worker
            await self._delete_worker(worker_id)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtesting: {str(e)}")
            return None

    async def evolve_strategy(self, strategy_id: str, performance_data: Dict) -> str:
        """Evolve an existing strategy based on its performance"""
        try:
            # Get current strategy
            current_strategy = self.active_strategies.get(strategy_id)
            if not current_strategy:
                return None

            # Create improvement prompt
            prompt = self._create_improvement_prompt(current_strategy, performance_data)
            
            # Get improvements from OpenAI
            response = await self.openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are improving a trading strategy based on its performance metrics. Suggest specific improvements while maintaining the core strategy logic."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            improved_strategy = response.choices[0].message.content
            return self._validate_and_format_strategy(improved_strategy)
            
        except Exception as e:
            logger.error(f"Error evolving strategy: {str(e)}")
            return None

    async def deploy_strategy(self, strategy_code: str, strategy_id: str = None) -> str:
        """Deploy a strategy as a Cloudflare Worker"""
        try:
            # Create worker script
            worker_script = self._create_production_worker(strategy_code)
            
            # Deploy to Cloudflare
            worker_id = await self._deploy_worker(worker_script)
            
            # Store strategy info
            self.active_strategies[worker_id] = {
                'code': strategy_code,
                'deployed_at': datetime.now().isoformat(),
                'performance': {}
            }
            
            return worker_id
            
        except Exception as e:
            logger.error(f"Error deploying strategy: {str(e)}")
            return None

    async def monitor_strategy(self, worker_id: str) -> Dict:
        """Monitor a deployed strategy's performance"""
        try:
            # Get worker metrics
            metrics = await self._get_worker_metrics(worker_id)
            
            # Get trading performance
            performance = await self._get_trading_performance(worker_id)
            
            # Update strategy performance data
            self.active_strategies[worker_id]['performance'] = {
                **metrics,
                **performance,
                'last_updated': datetime.now().isoformat()
            }
            
            return self.active_strategies[worker_id]['performance']
            
        except Exception as e:
            logger.error(f"Error monitoring strategy: {str(e)}")
            return None

    async def run(self):
        """Main service loop"""
        try:
            logger.info("Starting Strategy Evolution Service...")
            
            while self.running:
                # Monitor active strategies
                for worker_id in list(self.active_strategies.keys()):
                    # Get performance metrics
                    performance = await self.monitor_strategy(worker_id)
                    
                    # Check if improvement needed
                    if self._needs_improvement(performance):
                        # Evolve strategy
                        improved_strategy = await self.evolve_strategy(worker_id, performance)
                        if improved_strategy:
                            # Deploy new version
                            new_worker_id = await self.deploy_strategy(improved_strategy)
                            if new_worker_id:
                                # Transition traffic gradually
                                await self._transition_traffic(worker_id, new_worker_id)
                
                await asyncio.sleep(3600)  # Check hourly
                
        except Exception as e:
            logger.error(f"Error in Strategy Evolution Service: {str(e)}")
        finally:
            self.stop()

    def stop(self):
        """Stop the strategy evolution service"""
        logger.info("Stopping Strategy Evolution Service...")
        self.running = False
        self.redis.close()

    async def _deploy_worker(self, script: str, is_test: bool = False) -> str:
        """Deploy a worker to Cloudflare"""
        async with aiohttp.ClientSession() as session:
            headers = {
                'Authorization': f'Bearer {self.cf_api_token}',
                'Content-Type': 'application/javascript'
            }
            
            async with session.put(
                f"{self.cf_api_base}/scripts/{datetime.now().strftime('%Y%m%d%H%M%S')}",
                headers=headers,
                data=script
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['result']['id']
                else:
                    raise Exception(f"Failed to deploy worker: {await response.text()}")

    def _needs_improvement(self, performance: Dict) -> bool:
        """Determine if a strategy needs improvement"""
        if not performance:
            return False
            
        # Check against performance thresholds
        return (
            performance.get('sharpe_ratio', 0) < self.config['evolution']['min_sharpe_ratio'] or
            performance.get('drawdown', 100) > self.config['evolution']['max_drawdown'] or
            performance.get('win_rate', 0) < self.config['evolution']['min_win_rate']
        )

if __name__ == "__main__":
    service = StrategyEvolutionService()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        service.stop()
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        service.stop()
