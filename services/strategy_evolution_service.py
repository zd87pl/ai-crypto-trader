import os
import json
import redis
import asyncio
import aiohttp
import logging as logger
import numpy as np
import uuid
from datetime import datetime
from openai import AsyncOpenAI
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dotenv import load_dotenv

# Import our custom modules
from services.genetic_algorithm import GeneticAlgorithm
from services.reinforcement_learning import TradingRLAgent
from services.strategy_evaluation import StrategyPerformanceMetrics, StrategyEvaluationSystem

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
        
        # Strategy evolution parameters
        self.enable_genetic_algorithm = os.getenv('ENABLE_GENETIC_ALGORITHM', 'yes').lower() == 'yes'
        self.enable_reinforcement_learning = os.getenv('ENABLE_REINFORCEMENT_LEARNING', 'yes').lower() == 'yes'
        self.evolution_method = os.getenv('EVOLUTION_METHOD', 'hybrid')  # 'gpt', 'genetic', 'rl', or 'hybrid'
        self.ga_population_size = int(os.getenv('GA_POPULATION_SIZE', '20'))
        self.ga_generations = int(os.getenv('GA_GENERATIONS', '10'))
        
        logger.info(f"Strategy Evolution Configuration:")
        logger.info(f"- Risk Level: {self.risk_level}")
        logger.info(f"- Leverage Trading: {'Enabled' if self.leverage_trading else 'Disabled'}")
        logger.info(f"- Monitor Frequency: {self.monitor_frequency} seconds")
        logger.info(f"- Evolution Method: {self.evolution_method}")
        logger.info(f"- Genetic Algorithm: {'Enabled' if self.enable_genetic_algorithm else 'Disabled'}")
        logger.info(f"- Reinforcement Learning: {'Enabled' if self.enable_reinforcement_learning else 'Disabled'}")
        
        # Load evolution configuration
        self.evolution_config = self.config['evolution']
        
        # Initialize strategy evaluation system
        self.evaluation_system = StrategyEvaluationSystem(config_path='config.json')
        
        # Adjust parameter ranges based on leverage trading
        self.param_ranges = {
            'rsi_period': (5, 30),
            'rsi_overbought': (65, 85),
            'rsi_oversold': (15, 35),
            'macd_fast': (8, 20),
            'macd_slow': (20, 40),
            'macd_signal': (5, 15),
            'bollinger_period': (10, 30),
            'bollinger_std': (1.5, 3.0),
            'atr_period': (7, 25),
            'atr_multiplier': (1.0, 4.0),
            'ema_short': (5, 20),
            'ema_long': (20, 100),
            'volume_ma_period': (5, 30),
            'social_sentiment_threshold': (50, 80),
            'social_volume_threshold': (5000, 50000),
            'social_engagement_threshold': (1000, 20000),
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
        
        # Initialize RL agent if enabled
        if self.enable_reinforcement_learning:
            self._initialize_rl_agent()
            
        # Store model version histories
        self.model_versions = {}
        self.strategy_history = []
        
        # Metadata about the latest evolution
        self.last_evolution = {
            'timestamp': None,
            'method': None,
            'old_params': None,
            'new_params': None,
            'improvement': None,
            'market_conditions': None
        }
    
    def _initialize_rl_agent(self):
        """Initialize the Reinforcement Learning agent"""
        try:
            # Define state features (technical indicators and market data)
            state_features = [
                'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'bb_middle', 
                'ema_short', 'ema_long', 'atr', 'volume', 'price_change_1h', 'price_change_24h',
                'social_sentiment', 'social_volume', 'social_engagement', 'market_trend'
            ]
            
            # Define possible actions
            actions = ["BUY", "HOLD", "SELL"]
            
            # Create RL agent
            self.rl_agent = TradingRLAgent(
                state_features=state_features,
                actions=actions,
                learning_rate=0.001,
                gamma=0.95,  # Discount factor for future rewards
                epsilon_start=1.0,  # Start with 100% exploration
                epsilon_end=0.01,  # End with 1% exploration
                epsilon_decay=0.995,  # Decay rate
                memory_size=10000,  # Experience replay buffer size
                batch_size=64,  # Batch size for training
                update_target_every=100  # Update target network every 100 steps
            )
            
            logger.info("Reinforcement Learning agent initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RL agent: {str(e)}")
            self.enable_reinforcement_learning = False

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

    async def optimize_with_genetic_algorithm(self, current_params: Dict, 
                                      performance_data: Dict, historical_trades: List[Dict]) -> Dict:
        """
        Optimize strategy parameters using a genetic algorithm.
        
        Args:
            current_params: Current strategy parameters
            performance_data: Performance metrics for the current strategy
            historical_trades: List of historical trades
            
        Returns:
            Optimized parameters dictionary or None if optimization failed
        """
        try:
            logger.info("Starting genetic algorithm optimization")
            
            # Define fitness function for the genetic algorithm
            def fitness_function(params):
                # In a real implementation, this would simulate trades with the given parameters
                # For now, we'll use a simplified approach based on some heuristics
                
                # Base fitness on historical performance
                base_fitness = performance_data.get('sharpe_ratio', 0)
                
                # Adjust fitness based on parameter differences from current params
                adjustment = 0
                
                # Penalize extreme parameter values
                for param, value in params.items():
                    if param in self.param_ranges:
                        min_val, max_val = self.param_ranges[param]
                        # Penalize values too close to the edges of the allowed range
                        range_size = max_val - min_val
                        distance_from_edge = min(value - min_val, max_val - value)
                        edge_penalty = 1 - (distance_from_edge / (range_size * 0.5))
                        adjustment -= edge_penalty * 0.1
                        
                    # Consider correlations between parameters
                    if param == 'rsi_period' and 'rsi_oversold' in params:
                        # Lower RSI periods work better with higher oversold thresholds
                        rsi_balance = (30 - params['rsi_period']) * (params['rsi_oversold'] - 20) / 300
                        adjustment += rsi_balance
                    
                    # Adjust for social sentiment integration
                    if param == 'social_sentiment_threshold' and performance_data.get('social_correlation', 0) > 0.3:
                        # If social metrics correlate well with performance, give bonus to strategies using them
                        adjustment += 0.2
                
                # Final fitness score
                return base_fitness + adjustment
            
            # Create genetic algorithm
            ga = GeneticAlgorithm(
                param_ranges=self.param_ranges,
                fitness_function=fitness_function,
                population_size=self.ga_population_size,
                generations=self.ga_generations,
                mutation_rate=0.2,
                crossover_rate=0.8,
                elitism_pct=0.1
            )
            
            # Run genetic algorithm with current params as seed
            optimized_params = ga.run(seeded_individuals=[current_params])
            
            # Get generation history for analysis
            generation_history = ga.get_generation_history()
            
            # Log the evolution process
            final_generation = generation_history[-1]
            initial_generation = generation_history[0]
            improvement = final_generation['best_fitness'] - initial_generation['best_fitness']
            
            logger.info(f"Genetic algorithm completed with improvement: {improvement:.4f}")
            logger.info(f"Initial best fitness: {initial_generation['best_fitness']:.4f}")
            logger.info(f"Final best fitness: {final_generation['best_fitness']:.4f}")
            
            # Store the generation history
            evolution_id = str(uuid.uuid4())
            evolution_data = {
                'id': evolution_id,
                'algorithm': 'genetic',
                'timestamp': datetime.now().isoformat(),
                'improvement': improvement,
                'initial_fitness': initial_generation['best_fitness'],
                'final_fitness': final_generation['best_fitness'],
                'generations': len(generation_history),
                'population_size': self.ga_population_size,
                'old_params': current_params,
                'new_params': optimized_params
            }
            
            # Store evolution data in Redis
            await self.redis.set(
                f'strategy_evolution_genetic_{evolution_id}',
                json.dumps(evolution_data)
            )
            
            return optimized_params
            
        except Exception as e:
            logger.error(f"Error in genetic algorithm optimization: {str(e)}")
            return None
    
    async def optimize_with_reinforcement_learning(self, current_params: Dict, 
                                                 historical_market_data: List[Dict]) -> Dict:
        """
        Optimize strategy parameters using reinforcement learning.
        
        Args:
            current_params: Current strategy parameters
            historical_market_data: List of historical market data
            
        Returns:
            Optimized parameters dictionary or None if optimization failed
        """
        try:
            if not self.enable_reinforcement_learning:
                logger.warning("Reinforcement learning is disabled")
                return None
                
            logger.info("Starting reinforcement learning optimization")
            
            # In a real implementation, we would train the RL agent on historical data
            # For now, we'll simulate this process
            
            # Create a simplified state from historical market data
            if not historical_market_data:
                logger.warning("No historical market data for RL training")
                return None
            
            # Extract recent market conditions
            recent_data = historical_market_data[-20:]  # Last 20 data points
            
            # Track rewards for current training session
            rewards = []
            
            # Train the agent on historical data
            for i, market_data in enumerate(recent_data):
                # Convert market data to state
                state = self._convert_market_data_to_state(market_data)
                
                # Get action from agent
                action = self.rl_agent.act(state)
                
                # Calculate reward (in real implementation, this would be based on trading results)
                # For now, use a simplified reward function
                next_state = self._convert_market_data_to_state(
                    recent_data[i+1] if i < len(recent_data) - 1 else recent_data[-1]
                )
                
                reward = self._calculate_reward(action, state, next_state)
                rewards.append(reward)
                
                # Update agent memory
                done = (i == len(recent_data) - 1)
                self.rl_agent.remember(state, action, reward, next_state, done)
                
                # Train the agent
                if len(self.rl_agent.memory) >= self.rl_agent.batch_size:
                    self.rl_agent.replay()
            
            # Get the latest market data for current state
            current_state = self._convert_market_data_to_state(recent_data[-1])
            
            # Get Q-values for current state
            q_values = self.rl_agent.get_q_values(current_state)
            
            # Use Q-values to adjust parameters
            adjusted_params = self._adjust_params_from_q_values(current_params, q_values)
            
            # Log the RL training results
            avg_reward = sum(rewards) / len(rewards) if rewards else 0
            logger.info(f"RL training completed with average reward: {avg_reward:.4f}")
            logger.info(f"Q-values for current state: {q_values}")
            
            # Store the RL training results
            rl_training_id = str(uuid.uuid4())
            training_data = {
                'id': rl_training_id,
                'algorithm': 'reinforcement_learning',
                'timestamp': datetime.now().isoformat(),
                'avg_reward': avg_reward,
                'q_values': q_values,
                'old_params': current_params,
                'new_params': adjusted_params,
                'epsilon': self.rl_agent.epsilon
            }
            
            # Store training data in Redis
            await self.redis.set(
                f'strategy_evolution_rl_{rl_training_id}',
                json.dumps(training_data)
            )
            
            return adjusted_params
            
        except Exception as e:
            logger.error(f"Error in reinforcement learning optimization: {str(e)}")
            return None
    
    def _convert_market_data_to_state(self, market_data: Dict) -> Dict:
        """Convert market data to state for RL agent"""
        # Extract and normalize state features
        state = {}
        
        # Technical indicators
        state['rsi'] = market_data.get('rsi', 50) / 100
        state['macd'] = (market_data.get('macd', 0) + 1) / 2  # Normalize to [0, 1]
        state['macd_signal'] = (market_data.get('macd_signal', 0) + 1) / 2
        
        # Price action
        current_price = market_data.get('price', 0)
        if current_price > 0:
            # Bollinger Bands
            bb_middle = market_data.get('bb_middle', current_price)
            bb_upper = market_data.get('bb_upper', current_price * 1.05)
            bb_lower = market_data.get('bb_lower', current_price * 0.95)
            
            # Normalize Bollinger Bands
            bb_range = bb_upper - bb_lower
            state['bb_upper'] = (bb_upper - current_price) / bb_range if bb_range > 0 else 0.5
            state['bb_lower'] = (current_price - bb_lower) / bb_range if bb_range > 0 else 0.5
            state['bb_middle'] = (current_price - bb_middle) / current_price
        else:
            state['bb_upper'] = 0.5
            state['bb_lower'] = 0.5
            state['bb_middle'] = 0.5
        
        # EMAs
        state['ema_short'] = market_data.get('ema_short', current_price) / current_price if current_price > 0 else 1
        state['ema_long'] = market_data.get('ema_long', current_price) / current_price if current_price > 0 else 1
        
        # ATR
        state['atr'] = market_data.get('atr', 0) / current_price if current_price > 0 else 0
        
        # Volume and price changes
        state['volume'] = min(market_data.get('volume', 0) / 1000000, 1)  # Normalize volume
        state['price_change_1h'] = (market_data.get('price_change_1h', 0) + 10) / 20  # Normalize to [0, 1]
        state['price_change_24h'] = (market_data.get('price_change_24h', 0) + 30) / 60
        
        # Social metrics
        state['social_sentiment'] = market_data.get('social_sentiment', 50) / 100
        state['social_volume'] = min(market_data.get('social_volume', 0) / 50000, 1)
        state['social_engagement'] = min(market_data.get('social_engagement', 0) / 20000, 1)
        
        # Market trend
        trend = market_data.get('trend', 'neutral')
        if trend == 'uptrend':
            state['market_trend'] = 0.8
        elif trend == 'downtrend':
            state['market_trend'] = 0.2
        else:
            state['market_trend'] = 0.5
        
        return state
    
    def _calculate_reward(self, action: str, state: Dict, next_state: Dict) -> float:
        """Calculate reward for RL agent based on action and state transition"""
        # In a real implementation, this would be based on PnL and other factors
        # For now, use a simplified reward function
        
        # Price change as base for reward calculation
        price_change = (next_state.get('price_change_1h', 0.5) - 0.5) * 2  # Convert back to percentage
        
        if action == "BUY":
            # Reward for buying before price increase, penalty for buying before price decrease
            reward = price_change
            
            # Additional reward for buying at:
            # - Low RSI (oversold)
            # - Positive social sentiment
            # - Price at lower Bollinger Band
            if state.get('rsi', 0.5) < 0.3:  # RSI < 30
                reward += 0.2
            if state.get('social_sentiment', 0.5) > 0.6:  # High social sentiment
                reward += 0.1
            if state.get('bb_lower', 0.5) < 0.2:  # Price near lower BB
                reward += 0.2
                
        elif action == "SELL":
            # Reward for selling before price decrease, penalty for selling before price increase
            reward = -price_change
            
            # Additional reward for selling at:
            # - High RSI (overbought)
            # - Negative social sentiment
            # - Price at upper Bollinger Band
            if state.get('rsi', 0.5) > 0.7:  # RSI > 70
                reward += 0.2
            if state.get('social_sentiment', 0.5) < 0.4:  # Low social sentiment
                reward += 0.1
            if state.get('bb_upper', 0.5) > 0.8:  # Price near upper BB
                reward += 0.2
                
        else:  # HOLD
            # Small positive reward for holding during appropriate times
            # - Small price changes
            # - Neutral RSI
            if abs(price_change) < 0.01:  # Small price change
                reward = 0.05
            elif 0.4 <= state.get('rsi', 0.5) <= 0.6:  # Neutral RSI
                reward = 0.05
            else:
                # Small penalty for holding during significant price moves
                reward = -0.05
        
        return reward
    
    def _adjust_params_from_q_values(self, current_params: Dict, q_values: Dict) -> Dict:
        """
        Adjust strategy parameters based on Q-values from RL agent.
        
        Args:
            current_params: Current strategy parameters
            q_values: Q-values for current state
            
        Returns:
            Adjusted parameters dictionary
        """
        # Copy current parameters
        adjusted_params = current_params.copy()
        
        # Determine if the agent favors buying, selling, or holding
        max_action = max(q_values.items(), key=lambda x: x[1])[0]
        buy_confidence = q_values.get("BUY", 0)
        sell_confidence = q_values.get("SELL", 0)
        hold_confidence = q_values.get("HOLD", 0)
        
        # Adjust parameters based on agent's preferred action
        if max_action == "BUY":
            # Make strategy more aggressive for buying
            if 'rsi_oversold' in adjusted_params:
                # Increase RSI oversold threshold for more buy signals
                adjusted_params['rsi_oversold'] = min(
                    adjusted_params['rsi_oversold'] + 5,
                    self.param_ranges['rsi_oversold'][1]  # Don't exceed max value
                )
            
            if 'social_sentiment_threshold' in adjusted_params:
                # Lower sentiment threshold for more buy signals
                adjusted_params['social_sentiment_threshold'] = max(
                    adjusted_params['social_sentiment_threshold'] - 5,
                    self.param_ranges['social_sentiment_threshold'][0]  # Don't go below min value
                )
                
        elif max_action == "SELL":
            # Make strategy more defensive for selling
            if 'rsi_overbought' in adjusted_params:
                # Decrease RSI overbought threshold for more sell signals
                adjusted_params['rsi_overbought'] = max(
                    adjusted_params['rsi_overbought'] - 5,
                    self.param_ranges['rsi_overbought'][0]  # Don't go below min value
                )
            
            if 'take_profit' in adjusted_params:
                # Lower take profit for quicker exits
                adjusted_params['take_profit'] = max(
                    adjusted_params['take_profit'] * 0.9,
                    self.param_ranges['take_profit'][0]  # Don't go below min value
                )
                
        elif max_action == "HOLD":
            # Adjust for more conservative strategy
            if 'rsi_period' in adjusted_params:
                # Increase RSI period for smoother signals
                adjusted_params['rsi_period'] = min(
                    adjusted_params['rsi_period'] + 2,
                    self.param_ranges['rsi_period'][1]  # Don't exceed max value
                )
        
        # Make larger adjustments for higher confidence
        confidence_factor = max(buy_confidence, sell_confidence, hold_confidence)
        
        # Adjust stop loss based on confidence
        if 'stop_loss' in adjusted_params:
            # Tighter stop loss for higher confidence
            base_stop = adjusted_params['stop_loss']
            adjusted_params['stop_loss'] = max(
                base_stop * (1 - confidence_factor * 0.2),
                self.param_ranges['stop_loss'][0]  # Don't go below min value
            )
        
        return adjusted_params
    
    async def evolve_strategy(self, strategy_id: str, performance_data: Dict) -> Dict:
        """Evolve strategy using the selected evolution method"""
        try:
            # Get current market conditions and wallet info
            market_conditions = await self.get_market_conditions()
            wallet_info = await self.get_trade_executor_info()
            
            if not market_conditions or not wallet_info:
                logger.error("Missing market or wallet data for strategy evolution")
                return None
            
            # Get current parameters
            current_params = self.active_strategies[strategy_id]['parameters']
            
            # Get historical data for advanced methods
            historical_trades = await self._get_historical_trades(strategy_id)
            historical_market_data = await self._get_historical_market_data()
            
            # Choose evolution method based on configuration
            new_params = None
            evolution_method = self.evolution_method
            
            if evolution_method == 'hybrid':
                # Choose method dynamically based on data availability and market conditions
                market_volatility = market_conditions.get('volatility', 0.5)
                history_length = len(historical_trades)
                
                if market_volatility > 0.7 and self.enable_reinforcement_learning:
                    # Use RL in high volatility
                    logger.info("Hybrid mode selected RL due to high volatility")
                    evolution_method = 'rl'
                elif history_length > 50 and self.enable_genetic_algorithm:
                    # Use GA when we have enough historical data
                    logger.info("Hybrid mode selected GA due to sufficient historical data")
                    evolution_method = 'genetic'
                else:
                    # Default to GPT-based evolution
                    logger.info("Hybrid mode selected GPT as default method")
                    evolution_method = 'gpt'
            
            # Apply the selected evolution method
            if evolution_method == 'genetic' and self.enable_genetic_algorithm:
                logger.info("Using genetic algorithm for strategy evolution")
                new_params = await self.optimize_with_genetic_algorithm(
                    current_params, performance_data, historical_trades
                )
                
            elif evolution_method == 'rl' and self.enable_reinforcement_learning:
                logger.info("Using reinforcement learning for strategy evolution")
                new_params = await self.optimize_with_reinforcement_learning(
                    current_params, historical_market_data
                )
                
            else:
                # Default to GPT-based evolution
                logger.info("Using GPT for strategy evolution")
                new_params = await self.optimize_with_gpt(
                    current_params, performance_data, market_conditions, wallet_info
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
                evolution_data = {
                    'timestamp': datetime.now().isoformat(),
                    'market_conditions': market_conditions,
                    'wallet_info': wallet_info,
                    'old_params': current_params,
                    'new_params': new_params,
                    'performance': performance_data,
                    'risk_level': self.risk_level,
                    'leverage_trading': self.leverage_trading,
                    'evolution_method': evolution_method
                }
                
                # Update last evolution record
                self.last_evolution = {
                    'timestamp': datetime.now().isoformat(),
                    'method': evolution_method,
                    'old_params': current_params,
                    'new_params': new_params,
                    'improvement': None,  # Would be calculated in a real implementation
                    'market_conditions': market_conditions
                }
                
                # Store in Redis
                await self.redis.lpush(
                    f'strategy_evolution_history_{strategy_id}',
                    json.dumps(evolution_data)
                )
                
                # Store locally for reporting
                self.strategy_history.append(evolution_data)
                
                # Register a new model version if it's different enough
                await self._register_model_version(strategy_id, new_params, evolution_method)
                
                logger.info(f"Strategy evolved successfully using {evolution_method}: {json.dumps(new_params, indent=2)}")
                return new_params
            
            return None
            
        except Exception as e:
            logger.error(f"Error evolving strategy: {str(e)}")
            return None
    
    async def _get_historical_trades(self, strategy_id: str) -> List[Dict]:
        """Get historical trades for a strategy"""
        try:
            trades_data = await self.redis.get(f'strategy_trades_{strategy_id}')
            if trades_data:
                return json.loads(trades_data)
            return []
        except Exception as e:
            logger.error(f"Error getting historical trades: {str(e)}")
            return []
    
    async def _get_historical_market_data(self) -> List[Dict]:
        """Get historical market data"""
        try:
            market_data = await self.redis.get('historical_market_data')
            if market_data:
                return json.loads(market_data)
            return []
        except Exception as e:
            logger.error(f"Error getting historical market data: {str(e)}")
            return []
    
    async def _register_model_version(self, strategy_id: str, parameters: Dict, 
                                     method: str) -> bool:
        """
        Register a new model version if it's different enough from existing versions.
        
        Args:
            strategy_id: Strategy ID
            parameters: Strategy parameters
            method: Evolution method used
            
        Returns:
            True if a new version was registered, False otherwise
        """
        try:
            # Generate a version ID
            version_id = f"{strategy_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Compare with existing versions to avoid minor variations
            if strategy_id in self.model_versions:
                versions = self.model_versions[strategy_id]
                
                for version in versions:
                    # Check if the new version is similar to an existing one
                    similarity = self._calculate_parameter_similarity(
                        parameters, version['parameters']
                    )
                    
                    if similarity > 0.9:  # 90% similar
                        logger.info(f"New version is too similar to existing version {version['version_id']}")
                        return False
            else:
                self.model_versions[strategy_id] = []
            
            # Create version metadata
            version_metadata = {
                'version_id': version_id,
                'strategy_id': strategy_id,
                'parameters': parameters,
                'evolution_method': method,
                'creation_timestamp': datetime.now().isoformat(),
                'performance_metrics': None  # Will be updated after evaluation
            }
            
            # Store version metadata
            self.model_versions[strategy_id].append(version_metadata)
            
            # Store in Redis
            await self.redis.set(
                f'model_version_{version_id}',
                json.dumps(version_metadata)
            )
            
            logger.info(f"Registered new model version: {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering model version: {str(e)}")
            return False
    
    def _calculate_parameter_similarity(self, params1: Dict, params2: Dict) -> float:
        """
        Calculate similarity between two parameter sets.
        
        Args:
            params1: First parameter set
            params2: Second parameter set
            
        Returns:
            Similarity score between 0 and 1
        """
        # Get common parameters
        common_params = set(params1.keys()) & set(params2.keys())
        
        if not common_params:
            return 0
        
        similarities = []
        
        for param in common_params:
            val1 = params1[param]
            val2 = params2[param]
            
            # Skip non-numeric parameters
            if not isinstance(val1, (int, float)) or not isinstance(val2, (int, float)):
                continue
                
            # Calculate normalized difference
            if param in self.param_ranges:
                min_val, max_val = self.param_ranges[param]
                range_size = max_val - min_val
                
                if range_size > 0:
                    # Normalize difference to [0, 1]
                    normalized_diff = abs(val1 - val2) / range_size
                    similarity = 1 - normalized_diff
                    similarities.append(similarity)
            else:
                # For parameters without defined ranges, use relative difference
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    relative_diff = abs(val1 - val2) / max_val
                    similarity = 1 - min(relative_diff, 1)
                    similarities.append(similarity)
        
        # Average similarity
        return sum(similarities) / len(similarities) if similarities else 0
        
    async def generate_strategy(self, parameters: Dict) -> str:
        """
        Generate a trading strategy based on parameters using OpenAI.
        
        Args:
            parameters: Dictionary of strategy parameters
            
        Returns:
            Generated strategy code as a string
        """
        try:
            # Create detailed prompt using config template
            template = self.evolution_config['strategy_prompt_template']
            
            system_prompt = template['system']
            
            user_prompt = f"""
            Please generate a complete Cloudflare Worker trading strategy with the following parameters:
            
            Strategy Type: {parameters.get('type', 'mean_reversion')}
            Timeframe: {parameters.get('timeframe', '5m')}
            Risk Limit: {parameters.get('risk_limit', 1.5)}
            Target Profit: {parameters.get('target_profit', 1.0)}
            Maximum Position Size: {parameters.get('max_position_size', 5)}%
            
            Requirements:
            {json.dumps(template['strategy_requirements'], indent=2)}
            
            Implementation Guidelines:
            - Use ES6+ JavaScript
            - Follow Cloudflare Worker standards (export default with async fetch handler)
            - Implement error handling and logging
            - Include complete implementation of technical indicators
            - Add support for social metrics from LunarCrush API
            - Implement risk management with proper position sizing
            
            Respond with ONLY the complete code, no explanations or markdown.
            """
            
            # Get response from OpenAI
            response = await self.openai.chat.completions.create(
                model=self.gpt_config['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.gpt_config['temperature'],
                max_tokens=self.gpt_config['max_tokens']
            )
            
            # Extract the generated code
            strategy_code = response.choices[0].message.content.strip()
            
            # Log the generation (truncated)
            code_preview = strategy_code[:200] + "..." if len(strategy_code) > 200 else strategy_code
            logger.info(f"Generated strategy code: {code_preview}")
            
            return strategy_code
            
        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            return None
    
    async def deploy_strategy(self, strategy_code: str) -> str:
        """
        Deploy a strategy as a Cloudflare Worker.
        
        Args:
            strategy_code: JavaScript code for the strategy
            
        Returns:
            Worker ID if successful, None otherwise
        """
        try:
            # Generate a unique ID for the worker
            worker_id = f"strategy-{datetime.now().strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:8]}"
            
            # In a real implementation, this would deploy to Cloudflare
            # For now, we'll simulate deployment by storing the code
            
            # Create metadata for the deployment
            deployment = {
                "worker_id": worker_id,
                "strategy_code": strategy_code,
                "deployment_time": datetime.now().isoformat(),
                "status": "active",
                "metrics": {}
            }
            
            # Store in active strategies
            self.active_strategies[worker_id] = {
                "parameters": self.extract_parameters_from_code(strategy_code),
                "deployment_time": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "status": "active"
            }
            
            # Store complete deployment in Redis
            await self.redis.set(
                f'worker_deployment_{worker_id}',
                json.dumps(deployment, default=str)
            )
            
            logger.info(f"Deployed strategy as worker: {worker_id}")
            return worker_id
            
        except Exception as e:
            logger.error(f"Error deploying strategy: {str(e)}")
            return None
            
    def extract_parameters_from_code(self, code: str) -> Dict:
        """
        Extract strategy parameters from code.
        
        Args:
            code: Strategy code
            
        Returns:
            Dictionary of parameters
        """
        # This is a simplified implementation
        # In a real system, we would parse the JavaScript code
        
        parameters = {}
        
        # Look for common parameter patterns
        patterns = {
            'rsi_period': r'rsi(?:Period|_period)\s*[=:]\s*(\d+)',
            'rsi_overbought': r'rsi(?:Overbought|_overbought)\s*[=:]\s*(\d+)',
            'rsi_oversold': r'rsi(?:Oversold|_oversold)\s*[=:]\s*(\d+)',
            'macd_fast': r'macd(?:Fast|_fast)\s*[=:]\s*(\d+)',
            'macd_slow': r'macd(?:Slow|_slow)\s*[=:]\s*(\d+)',
            'macd_signal': r'macd(?:Signal|_signal)\s*[=:]\s*(\d+)',
            'bollinger_period': r'bollinger(?:Period|_period)\s*[=:]\s*(\d+)',
            'bollinger_std': r'bollinger(?:Std|_std)\s*[=:]\s*([\d\.]+)',
            'ema_short': r'ema(?:Short|_short)\s*[=:]\s*(\d+)',
            'ema_long': r'ema(?:Long|_long)\s*[=:]\s*(\d+)',
            'stop_loss': r'stop(?:Loss|_loss)\s*[=:]\s*([\d\.]+)',
            'take_profit': r'take(?:Profit|_profit)\s*[=:]\s*([\d\.]+)',
            'social_sentiment_threshold': r'sentiment(?:Threshold|_threshold)\s*[=:]\s*([\d\.]+)',
        }
        
        import re
        
        for param, pattern in patterns.items():
            match = re.search(pattern, code)
            if match:
                value = match.group(1)
                # Convert to appropriate type
                try:
                    if '.' in value:
                        parameters[param] = float(value)
                    else:
                        parameters[param] = int(value)
                except ValueError:
                    pass
        
        # Fall back to some defaults if we couldn't extract all parameters
        for param in self.param_ranges:
            if param not in parameters:
                min_val, max_val = self.param_ranges[param]
                # Set to midpoint of range
                if isinstance(min_val, int) and isinstance(max_val, int):
                    parameters[param] = (min_val + max_val) // 2
                else:
                    parameters[param] = (min_val + max_val) / 2
        
        return parameters

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
