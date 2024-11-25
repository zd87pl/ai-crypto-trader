import os
import json
import redis
import asyncio
import aiohttp
import logging as logger
from datetime import datetime
from openai import AsyncOpenAI
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

        # Initialize OpenAI client with API key from .env
        self.openai = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.gpt_config = self.config['openai']
        logger.info(f"Using GPT model: {self.gpt_config['model']}")

        # Initialize Redis connection
        self.redis = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=True
        )

        self.running = True
        self.active_strategies = {}
        
        # Load environment variables
        self.risk_level = os.getenv('RISK_LEVEL', 'MEDIUM')
        self.leverage_trading = os.getenv('LEVERAGE_TRADING', 'no').lower() == 'yes'
        self.monitor_frequency = int(os.getenv('STRATEGY_MONITOR_FREQUENCY', '3600'))
        
        logger.info(f"Strategy Evolution Configuration:")
        logger.info(f"- Risk Level: {self.risk_level}")
        logger.info(f"- Leverage Trading: {'Enabled' if self.leverage_trading else 'Disabled'}")
        logger.info(f"- Monitor Frequency: {self.monitor_frequency} seconds")
        
        # Load evolution configuration
        self.evolution_config = self.config['evolution']
        
        # Adjust parameter ranges based on leverage trading
        self.param_ranges = {
            'rsi_period': (5, 30),
            'rsi_overbought': (65, 85),
            'rsi_oversold': (15, 35),
            'macd_fast': (8, 20),
            'macd_slow': (20, 40),
            'macd_signal': (5, 15),
            'stop_loss': (1, 5) if not self.leverage_trading else (0.5, 2.5),  # Tighter stops for leverage
            'take_profit': (1, 10) if not self.leverage_trading else (2, 20)   # Higher targets for leverage
        }

        # Risk-based thresholds adjusted for leverage
        base_position_size = self.evolution_config['risk_management']['max_position_size']
        leverage_multiplier = 0.5 if self.leverage_trading else 1.0  # Reduce position size if using leverage
        
        self.risk_thresholds = {
            'LOW': {
                'min_win_rate': self.evolution_config['min_win_rate'] + 0.05,
                'max_drawdown': self.evolution_config['max_drawdown'] - 5,
                'min_sharpe_ratio': self.evolution_config['min_sharpe_ratio'] + 0.3,
                'position_size_pct': base_position_size * 0.5 * leverage_multiplier
            },
            'MEDIUM': {
                'min_win_rate': self.evolution_config['min_win_rate'],
                'max_drawdown': self.evolution_config['max_drawdown'],
                'min_sharpe_ratio': self.evolution_config['min_sharpe_ratio'],
                'position_size_pct': base_position_size * leverage_multiplier
            },
            'HIGH': {
                'min_win_rate': self.evolution_config['min_win_rate'] - 0.05,
                'max_drawdown': self.evolution_config['max_drawdown'] + 5,
                'min_sharpe_ratio': self.evolution_config['min_sharpe_ratio'] - 0.3,
                'position_size_pct': base_position_size * 1.5 * leverage_multiplier
            }
        }

    async def get_trade_executor_info(self) -> Dict:
        """Get current capital and trade information from TradeExecutor"""
        try:
            trade_info = await self.redis.get('trade_executor_info')
            if trade_info:
                return json.loads(trade_info)
            return None
        except Exception as e:
            logger.error(f"Error getting trade executor info: {str(e)}")
            return None

    async def get_market_conditions(self) -> Dict:
        """Get current market conditions from Redis"""
        try:
            market_data = await self.redis.get('market_conditions')
            if market_data:
                return json.loads(market_data)
            return None
        except Exception as e:
            logger.error(f"Error getting market conditions: {str(e)}")
            return None

    async def hot_swap_strategy(self, new_params: Dict):
        """Hot swap strategy parameters in running system"""
        try:
            # Store new parameters in Redis
            await self.redis.set('strategy_params', json.dumps(new_params))
            
            # Signal trade executor to reload parameters
            await self.redis.publish('strategy_update', 'reload')
            
            logger.info("Strategy parameters hot-swapped successfully")
            return True
        except Exception as e:
            logger.error(f"Error hot-swapping strategy: {str(e)}")
            return False

    async def optimize_with_gpt(self, current_params: Dict, performance: Dict, market_conditions: Dict, wallet_info: Dict) -> Dict:
        """Use GPT to optimize strategy parameters based on current conditions"""
        try:
            # Create detailed prompt using config template
            prompt = f"""
            Analyze the current trading strategy and suggest optimal parameter adjustments based on:

            Current Market Conditions:
            - Trend: {market_conditions.get('trend', 'unknown')}
            - Volatility: {market_conditions.get('volatility', 'unknown')}
            - Volume: {market_conditions.get('volume', 'unknown')}
            - Key Support/Resistance levels: {market_conditions.get('levels', [])}

            Wallet Information:
            - Available Capital: ${wallet_info.get('available_usdc', 0):,.2f}
            - Active Positions: {wallet_info.get('active_positions', 0)}
            - Current P&L: {wallet_info.get('current_pnl', '0')}%

            Trading Configuration:
            - Risk Level: {self.risk_level}
            - Leverage Trading: {'Enabled' if self.leverage_trading else 'Disabled'}

            Current Strategy Performance:
            {json.dumps(performance, indent=2)}

            Current Parameters:
            {json.dumps(current_params, indent=2)}

            Risk Thresholds:
            {json.dumps(self.risk_thresholds[self.risk_level], indent=2)}

            Parameter Ranges:
            {json.dumps(self.param_ranges, indent=2)}

            Strategy Requirements:
            {json.dumps(self.evolution_config['strategy_prompt_template']['strategy_requirements'], indent=2)}

            Optimization Goals:
            {json.dumps(self.evolution_config['optimization_goals'], indent=2)}

            Additional Requirements:
            - {'Consider leverage trading implications for risk management' if self.leverage_trading else 'No leverage trading allowed'}
            - Adjust position sizes and risk parameters accordingly

            Return ONLY a JSON object with the optimized parameters, no explanation needed.
            """

            # Get GPT's suggestion using config settings
            response = await self.openai.chat.completions.create(
                model=self.gpt_config['model'],
                messages=[
                    {"role": "system", "content": self.evolution_config['strategy_prompt_template']['system']},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.gpt_config['temperature'],
                max_tokens=self.gpt_config['max_tokens']
            )
            
            # Parse the response
            suggested_params = json.loads(response.choices[0].message.content)
            
            # Validate parameters are within ranges
            validated_params = {}
            for param, value in suggested_params.items():
                if param in self.param_ranges:
                    min_val, max_val = self.param_ranges[param]
                    validated_params[param] = max(min_val, min(max_val, value))
                else:
                    validated_params[param] = value
            
            return validated_params
            
        except Exception as e:
            logger.error(f"Error in GPT optimization: {str(e)}")
            return None

    async def monitor_strategy(self, strategy_id: str) -> Dict:
        """Monitor a deployed strategy's performance"""
        try:
            # Get performance metrics from Redis
            performance = await self.redis.get(f'strategy_performance_{strategy_id}')
            if performance:
                return json.loads(performance)
            return None
        except Exception as e:
            logger.error(f"Error monitoring strategy: {str(e)}")
            return None

    async def evolve_strategy(self, strategy_id: str, performance_data: Dict) -> Dict:
        """Evolve strategy using GPT optimization"""
        try:
            # Get current market conditions and wallet info
            market_conditions = await self.get_market_conditions()
            wallet_info = await self.get_trade_executor_info()
            
            if not market_conditions or not wallet_info:
                logger.error("Missing market or wallet data for strategy evolution")
                return None
            
            # Get current parameters
            current_params = self.active_strategies[strategy_id]['parameters']
            
            # Optimize parameters using GPT
            new_params = await self.optimize_with_gpt(
                current_params,
                performance_data,
                market_conditions,
                wallet_info
            )
            
            if not new_params:
                logger.warning("Could not optimize parameters, keeping current strategy")
                return None
            
            # Hot swap to new parameters
            success = await self.hot_swap_strategy(new_params)
            if success:
                # Update active strategy
                self.active_strategies[strategy_id]['parameters'] = new_params
                self.active_strategies[strategy_id]['last_updated'] = datetime.now().isoformat()
                
                # Store evolution history
                await self.redis.lpush(
                    f'strategy_evolution_history_{strategy_id}',
                    json.dumps({
                        'timestamp': datetime.now().isoformat(),
                        'market_conditions': market_conditions,
                        'wallet_info': wallet_info,
                        'old_params': current_params,
                        'new_params': new_params,
                        'performance': performance_data,
                        'risk_level': self.risk_level,
                        'leverage_trading': self.leverage_trading
                    })
                )
                
                logger.info(f"Strategy evolved successfully: {json.dumps(new_params, indent=2)}")
                return new_params
            
            return None
            
        except Exception as e:
            logger.error(f"Error evolving strategy: {str(e)}")
            return None

    def _needs_improvement(self, performance: Dict) -> bool:
        """Determine if a strategy needs improvement based on risk level"""
        if not performance:
            return False
            
        risk_params = self.risk_thresholds[self.risk_level]
        
        return (
            performance.get('sharpe_ratio', 0) < risk_params['min_sharpe_ratio'] or
            performance.get('drawdown', 100) > risk_params['max_drawdown'] or
            performance.get('win_rate', 0) < risk_params['min_win_rate']
        )

    async def run(self):
        """Main service loop"""
        try:
            logger.info(f"Starting Strategy Evolution Service:")
            logger.info(f"- Risk Level: {self.risk_level}")
            logger.info(f"- Leverage Trading: {'Enabled' if self.leverage_trading else 'Disabled'}")
            logger.info(f"- Monitor Frequency: {self.monitor_frequency} seconds")
            
            while self.running:
                # Monitor active strategies
                for strategy_id in list(self.active_strategies.keys()):
                    try:
                        # Get performance metrics
                        performance = await self.monitor_strategy(strategy_id)
                        
                        # Check if improvement needed based on risk level
                        if self._needs_improvement(performance):
                            logger.info(f"Strategy {strategy_id} needs improvement based on {self.risk_level} risk parameters")
                            
                            # Evolve strategy using GPT
                            new_params = await self.evolve_strategy(strategy_id, performance)
                            
                            if new_params:
                                logger.info("Strategy evolved successfully")
                                # Notify about evolution
                                await self.redis.publish(
                                    'strategy_evolution_updates',
                                    json.dumps({
                                        'strategy_id': strategy_id,
                                        'timestamp': datetime.now().isoformat(),
                                        'new_params': new_params,
                                        'performance': performance,
                                        'risk_level': self.risk_level,
                                        'leverage_trading': self.leverage_trading
                                    })
                                )
                    except Exception as e:
                        logger.error(f"Error processing strategy {strategy_id}: {str(e)}")
                        continue
                
                await asyncio.sleep(self.monitor_frequency)
                
        except Exception as e:
            logger.error(f"Error in Strategy Evolution Service: {str(e)}")
        finally:
            self.stop()

    def stop(self):
        """Stop the strategy evolution service"""
        logger.info("Stopping Strategy Evolution Service...")
        self.running = False
        self.redis.close()

if __name__ == "__main__":
    service = StrategyEvolutionService()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        service.stop()
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        service.stop()
