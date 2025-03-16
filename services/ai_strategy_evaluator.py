import os
import json
import redis
import asyncio
import logging as logger
import numpy as np
import uuid
from datetime import datetime
from openai import AsyncOpenAI
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dotenv import load_dotenv

# Import custom modules
from services.strategy_evaluation_system import StrategyEvaluationSystem
from services.strategy_evolution_service import StrategyEvolutionService
from services.market_regime_service import MarketRegimeService

# Load environment variables
load_dotenv()

# Configure logging
logger.basicConfig(
    level=logger.INFO,
    format='%(asctime)s - %(levelname)s - [AIStrategyEvaluator] %(message)s',
    handlers=[
        logger.FileHandler('logs/ai_strategy_evaluation.log'),
        logger.StreamHandler()
    ]
)

class AIStrategyEvaluator:
    """
    Service for systematic evaluation of AI-generated trading strategies.
    Provides comprehensive testing, validation, and feedback for AI-generated strategies.
    """
    
    def __init__(self):
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Initialize OpenAI client
        self.openai = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.gpt_config = self.config.get('openai', {})
        
        # Initialize Redis connection
        self.redis = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=True
        )
        
        # Initialize evaluation system
        self.evaluation_system = StrategyEvaluationSystem(config_path='config.json')
        
        # Initialize strategy evolution service for strategy generation
        self.evolution_service = StrategyEvolutionService()
        
        # Initialize market regime service
        self.market_regime_service = MarketRegimeService()
        
        # Evaluation parameters
        self.min_trades = int(os.getenv('MIN_TRADES_FOR_EVALUATION', '30'))
        self.min_sharpe_ratio = float(os.getenv('MIN_SHARPE_RATIO', '0.5'))
        self.min_win_rate = float(os.getenv('MIN_WIN_RATE', '0.48'))
        self.max_drawdown = float(os.getenv('MAX_DRAWDOWN', '0.25'))
        self.min_profit_factor = float(os.getenv('MIN_PROFIT_FACTOR', '1.2'))
        
        # Quality scoring weights
        self.quality_weights = {
            'sharpe_ratio': 0.25,
            'win_rate': 0.15,
            'profit_factor': 0.20,
            'max_drawdown': 0.15,
            'trade_count': 0.05,
            'regime_consistency': 0.20  # Consistency across different market regimes
        }
        
        # Code quality metrics
        self.code_quality_checks = [
            'error_handling',
            'resource_usage',
            'security_practices',
            'modularity',
            'readability',
            'comments',
            'testability'
        ]
        
        # Strategy evaluation history
        self.evaluation_history = {}
        
        logger.info("AI Strategy Evaluator initialized")
        logger.info(f"Minimum required trades: {self.min_trades}")
        logger.info(f"Minimum Sharpe ratio: {self.min_sharpe_ratio}")
        logger.info(f"Minimum win rate: {self.min_win_rate}")
        logger.info(f"Maximum drawdown: {self.max_drawdown}")
        logger.info(f"Minimum profit factor: {self.min_profit_factor}")
    
    async def generate_strategy(self, parameters: Dict) -> Tuple[str, str]:
        """
        Generate a new trading strategy using OpenAI.
        
        Args:
            parameters: Dictionary of strategy parameters
            
        Returns:
            Tuple of (strategy_id, strategy_code)
        """
        try:
            # Use the strategy evolution service to generate strategy
            strategy_code = await self.evolution_service.generate_strategy(parameters)
            
            if not strategy_code:
                logger.error("Failed to generate strategy")
                return None, None
            
            # Generate a unique ID for the strategy
            strategy_id = f"ai_strategy_{datetime.now().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:6]}"
            
            # Store the generated strategy in Redis
            await self.redis.set(
                f"strategy_code_{strategy_id}",
                strategy_code
            )
            
            # Store metadata about the strategy
            metadata = {
                'id': strategy_id,
                'timestamp': datetime.now().isoformat(),
                'parameters': parameters,
                'evaluated': False,
                'deployed': False
            }
            
            await self.redis.set(
                f"strategy_metadata_{strategy_id}",
                json.dumps(metadata)
            )
            
            logger.info(f"Generated new strategy: {strategy_id}")
            return strategy_id, strategy_code
            
        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            return None, None
    
    async def evaluate_strategy_code(self, strategy_id: str, strategy_code: str) -> Dict:
        """
        Evaluate code quality of a strategy using OpenAI.
        
        Args:
            strategy_id: The strategy ID
            strategy_code: The JavaScript code of the strategy
            
        Returns:
            Dictionary with code evaluation results
        """
        try:
            # Create a detailed prompt for code evaluation
            prompt = f"""
            Please evaluate the following trading strategy code for quality and best practices.
            
            TRADING STRATEGY CODE:
            ```javascript
            {strategy_code}
            ```
            
            Please analyze the code for the following specific criteria:
            
            1. Error Handling: Is there proper error handling for API calls, data processing, and trading operations?
            2. Resource Usage: Is the code efficient in terms of memory and computational resources?
            3. Security Practices: Are there any security issues or vulnerabilities?
            4. Modularity: Is the code well-structured with proper separation of concerns?
            5. Readability: Is the code easy to read and understand?
            6. Comments: Are there sufficient and helpful comments?
            7. Testability: How easily can the code be tested?
            
            For each criterion, please give a rating from 1-10 (10 being best) and a brief explanation.
            
            Then provide an overall assessment and suggestions for improvement. Return your evaluation as a JSON object with the following structure:
            
            {{
                "criteria": {{
                    "error_handling": {{
                        "score": [1-10],
                        "comments": "brief explanation"
                    }},
                    "resource_usage": {{
                        "score": [1-10],
                        "comments": "brief explanation"
                    }},
                    "security_practices": {{
                        "score": [1-10],
                        "comments": "brief explanation"
                    }},
                    "modularity": {{
                        "score": [1-10],
                        "comments": "brief explanation"
                    }},
                    "readability": {{
                        "score": [1-10],
                        "comments": "brief explanation"
                    }},
                    "comments": {{
                        "score": [1-10],
                        "comments": "brief explanation"
                    }},
                    "testability": {{
                        "score": [1-10],
                        "comments": "brief explanation"
                    }}
                }},
                "overall_score": [1-10],
                "strengths": ["strength1", "strength2", ...],
                "weaknesses": ["weakness1", "weakness2", ...],
                "improvement_suggestions": ["suggestion1", "suggestion2", ...]
            }}
            
            Please be thorough but concise.
            """
            
            # Get evaluation from OpenAI
            response = await self.openai.chat.completions.create(
                model=self.gpt_config.get('model', 'gpt-4o'),
                messages=[
                    {"role": "system", "content": "You are a senior trading strategy code reviewer with expertise in JavaScript and algorithmic trading."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent evaluation
                max_tokens=2000
            )
            
            # Extract and parse the response
            evaluation_text = response.choices[0].message.content
            evaluation = json.loads(evaluation_text)
            
            # Store evaluation results in Redis
            evaluation_record = {
                'strategy_id': strategy_id,
                'timestamp': datetime.now().isoformat(),
                'code_evaluation': evaluation
            }
            
            await self.redis.set(
                f"strategy_code_evaluation_{strategy_id}",
                json.dumps(evaluation_record)
            )
            
            logger.info(f"Completed code evaluation for strategy {strategy_id}")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating strategy code: {str(e)}")
            return {
                'error': str(e),
                'overall_score': 0,
                'criteria': {check: {'score': 0, 'comments': 'Evaluation failed'} for check in self.code_quality_checks}
            }
    
    async def evaluate_strategy_performance(self, strategy_id: str, strategy_code: str) -> Dict:
        """
        Evaluate the performance of a strategy using backtesting and cross-validation.
        
        Args:
            strategy_id: The strategy ID
            strategy_code: The JavaScript code of the strategy
            
        Returns:
            Dictionary with performance evaluation results
        """
        try:
            # Perform k-fold cross-validation
            cv_results = await self.evaluation_system.cross_validate_strategy(strategy_id, strategy_code)
            
            if 'status' in cv_results and cv_results['status'] == 'error':
                logger.error(f"Cross-validation failed: {cv_results['message']}")
                return cv_results
            
            # Check if we have sufficient trades for evaluation
            overall_trades = cv_results.get('overall_metrics', {}).get('trade_count', 0)
            
            if overall_trades < self.min_trades:
                logger.warning(f"Insufficient trades for evaluation: {overall_trades} < {self.min_trades}")
                return {
                    'status': 'rejected',
                    'reason': f"Insufficient trades for evaluation: {overall_trades} < {self.min_trades}",
                    'cv_results': cv_results
                }
            
            # Calculate a quality score for the strategy
            quality_score = self._calculate_quality_score(cv_results)
            
            # Get regime-specific performance
            regime_scores = self._calculate_regime_scores(cv_results)
            
            # Determine if the strategy meets our minimum requirements
            meets_requirements = self._check_minimum_requirements(cv_results)
            
            # Create evaluation result
            evaluation_result = {
                'strategy_id': strategy_id,
                'timestamp': datetime.now().isoformat(),
                'cv_results': cv_results,
                'quality_score': quality_score,
                'regime_scores': regime_scores,
                'meets_requirements': meets_requirements,
                'status': 'accepted' if meets_requirements else 'rejected'
            }
            
            # Store evaluation result in Redis
            await self.redis.set(
                f"strategy_performance_evaluation_{strategy_id}",
                json.dumps(evaluation_result)
            )
            
            # Update strategy metadata
            metadata = await self.redis.get(f"strategy_metadata_{strategy_id}")
            
            if metadata:
                metadata_dict = json.loads(metadata)
                metadata_dict['evaluated'] = True
                metadata_dict['evaluation_timestamp'] = datetime.now().isoformat()
                metadata_dict['quality_score'] = quality_score
                metadata_dict['meets_requirements'] = meets_requirements
                
                await self.redis.set(
                    f"strategy_metadata_{strategy_id}",
                    json.dumps(metadata_dict)
                )
            
            logger.info(f"Completed performance evaluation for strategy {strategy_id}")
            logger.info(f"Quality score: {quality_score:.4f}")
            logger.info(f"Status: {'Accepted' if meets_requirements else 'Rejected'}")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating strategy performance: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _calculate_quality_score(self, cv_results: Dict) -> float:
        """
        Calculate a quality score for a strategy based on its performance metrics.
        
        Args:
            cv_results: Cross-validation results dictionary
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            # Extract overall metrics
            metrics = cv_results.get('overall_metrics', {})
            
            if not metrics:
                return 0.0
            
            # Initialize score components
            score_components = {}
            
            # Sharpe ratio (higher is better)
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            sharpe_score = min(sharpe_ratio / 3.0, 1.0)  # Normalize: 3.0 is excellent
            score_components['sharpe_ratio'] = sharpe_score
            
            # Win rate (higher is better)
            win_rate = metrics.get('win_rate', 0)
            win_rate_score = min(win_rate / 0.7, 1.0)  # Normalize: 70% is excellent
            score_components['win_rate'] = win_rate_score
            
            # Profit factor (higher is better)
            profit_factor = metrics.get('profit_factor', 0)
            profit_factor_score = min(profit_factor / 2.5, 1.0)  # Normalize: 2.5 is excellent
            score_components['profit_factor'] = profit_factor_score
            
            # Max drawdown (lower is better)
            max_drawdown = metrics.get('max_drawdown', 1.0)
            drawdown_score = 1.0 - min(max_drawdown / 0.5, 1.0)  # Normalize: 0% is best, 50% is worst
            score_components['max_drawdown'] = drawdown_score
            
            # Trade count (more trades provide more statistical significance)
            trade_count = metrics.get('trade_count', 0)
            trade_count_score = min(trade_count / 200, 1.0)  # Normalize: 200+ trades is excellent
            score_components['trade_count'] = trade_count_score
            
            # Regime consistency (how consistent the strategy performs across different regimes)
            regime_metrics = cv_results.get('regime_metrics', {})
            regime_scores = []
            
            for regime, regime_metrics in regime_metrics.items():
                if not regime_metrics:
                    continue
                
                regime_sharpe = regime_metrics.get('sharpe_ratio', 0)
                if regime_sharpe > 0:
                    regime_scores.append(min(regime_sharpe / 2.0, 1.0))
            
            # Calculate consistency score as the minimum of regime scores
            if regime_scores:
                regime_consistency_score = min(regime_scores)
            else:
                regime_consistency_score = 0
            
            score_components['regime_consistency'] = regime_consistency_score
            
            # Calculate weighted average quality score
            weighted_score = 0
            
            for component, score in score_components.items():
                weight = self.quality_weights.get(component, 0)
                weighted_score += score * weight
            
            return weighted_score
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {str(e)}")
            return 0.0
    
    def _calculate_regime_scores(self, cv_results: Dict) -> Dict:
        """
        Calculate performance scores for each market regime.
        
        Args:
            cv_results: Cross-validation results dictionary
            
        Returns:
            Dictionary of regime-specific scores
        """
        try:
            regime_metrics = cv_results.get('regime_metrics', {})
            regime_scores = {}
            
            for regime, metrics in regime_metrics.items():
                if not metrics:
                    regime_scores[regime] = 0.0
                    continue
                
                # Calculate a score for this regime similar to overall quality score
                # but simplified to key metrics
                
                # Sharpe ratio (higher is better)
                sharpe_ratio = metrics.get('sharpe_ratio', 0)
                sharpe_score = min(sharpe_ratio / 3.0, 1.0)  # Normalize: 3.0 is excellent
                
                # Win rate (higher is better)
                win_rate = metrics.get('win_rate', 0)
                win_rate_score = min(win_rate / 0.7, 1.0)  # Normalize: 70% is excellent
                
                # Profit factor (higher is better)
                profit_factor = metrics.get('profit_factor', 0)
                profit_factor_score = min(profit_factor / 2.5, 1.0)  # Normalize: 2.5 is excellent
                
                # Simple weighted average
                regime_score = (
                    sharpe_score * 0.5 +       # Higher weight for Sharpe ratio
                    win_rate_score * 0.25 +     # Medium weight for win rate
                    profit_factor_score * 0.25  # Medium weight for profit factor
                )
                
                regime_scores[regime] = regime_score
            
            return regime_scores
            
        except Exception as e:
            logger.error(f"Error calculating regime scores: {str(e)}")
            return {}
    
    def _check_minimum_requirements(self, cv_results: Dict) -> bool:
        """
        Check if a strategy meets minimum performance requirements.
        
        Args:
            cv_results: Cross-validation results dictionary
            
        Returns:
            True if the strategy meets requirements, False otherwise
        """
        try:
            # Extract overall metrics
            metrics = cv_results.get('overall_metrics', {})
            
            if not metrics:
                return False
            
            # Check trade count
            trade_count = metrics.get('trade_count', 0)
            if trade_count < self.min_trades:
                return False
            
            # Check Sharpe ratio
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            if sharpe_ratio < self.min_sharpe_ratio:
                return False
            
            # Check win rate
            win_rate = metrics.get('win_rate', 0)
            if win_rate < self.min_win_rate:
                return False
            
            # Check drawdown
            max_drawdown = metrics.get('max_drawdown', 1.0)
            if max_drawdown > self.max_drawdown:
                return False
            
            # Check profit factor
            profit_factor = metrics.get('profit_factor', 0)
            if profit_factor < self.min_profit_factor:
                return False
            
            # Check performance in at least one market regime
            regime_metrics = cv_results.get('regime_metrics', {})
            has_valid_regime = False
            
            for regime, metrics in regime_metrics.items():
                if not metrics:
                    continue
                
                regime_sharpe = metrics.get('sharpe_ratio', 0)
                regime_win_rate = metrics.get('win_rate', 0)
                
                if regime_sharpe >= self.min_sharpe_ratio and regime_win_rate >= self.min_win_rate:
                    has_valid_regime = True
                    break
            
            return has_valid_regime
            
        except Exception as e:
            logger.error(f"Error checking minimum requirements: {str(e)}")
            return False
    
    async def generate_improvement_suggestions(self, strategy_id: str, strategy_code: str, 
                                             cv_results: Dict) -> Dict:
        """
        Generate improvement suggestions for a strategy based on its performance evaluation.
        
        Args:
            strategy_id: The strategy ID
            strategy_code: The JavaScript code of the strategy
            cv_results: Cross-validation results
            
        Returns:
            Dictionary with improvement suggestions
        """
        try:
            # Extract metrics and regime-specific performance
            overall_metrics = cv_results.get('overall_metrics', {})
            regime_metrics = cv_results.get('regime_metrics', {})
            
            # Create a detailed prompt for improvement suggestions
            prompt = f"""
            Please analyze this trading strategy and provide specific improvement suggestions based on its performance metrics.
            
            TRADING STRATEGY CODE:
            ```javascript
            {strategy_code}
            ```
            
            PERFORMANCE METRICS:
            Overall Performance:
            {json.dumps(overall_metrics, indent=2)}
            
            Performance by Market Regime:
            {json.dumps(regime_metrics, indent=2)}
            
            Based on these metrics, please provide:
            
            1. Analysis of the strategy's strengths and weaknesses
            2. Specific technical improvements to address weaknesses
            3. Suggestions for better performance in each market regime
            4. Risk management improvements
            5. Social sentiment integration improvements
            
            Return your suggestions as a JSON object with the following structure:
            
            {{
                "strengths": ["strength1", "strength2", ...],
                "weaknesses": ["weakness1", "weakness2", ...],
                "technical_improvements": [
                    {{
                        "issue": "description of issue",
                        "suggestion": "implementation suggestion",
                        "expected_impact": "expected improvement"
                    }},
                    ...
                ],
                "regime_specific_improvements": {{
                    "bull": ["suggestion1", "suggestion2", ...],
                    "bear": ["suggestion1", "suggestion2", ...],
                    "ranging": ["suggestion1", "suggestion2", ...],
                    "volatile": ["suggestion1", "suggestion2", ...]
                }},
                "risk_management_improvements": ["suggestion1", "suggestion2", ...],
                "social_sentiment_improvements": ["suggestion1", "suggestion2", ...]
            }}
            
            Please be specific, actionable, and code-oriented in your suggestions.
            """
            
            # Get suggestions from OpenAI
            response = await self.openai.chat.completions.create(
                model=self.gpt_config.get('model', 'gpt-4o'),
                messages=[
                    {"role": "system", "content": "You are an expert trading strategy developer with deep knowledge of algorithmic trading, technical analysis, and market regimes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            # Extract and parse the response
            suggestions_text = response.choices[0].message.content
            suggestions = json.loads(suggestions_text)
            
            # Store suggestions in Redis
            suggestions_record = {
                'strategy_id': strategy_id,
                'timestamp': datetime.now().isoformat(),
                'improvement_suggestions': suggestions
            }
            
            await self.redis.set(
                f"strategy_improvement_suggestions_{strategy_id}",
                json.dumps(suggestions_record)
            )
            
            logger.info(f"Generated improvement suggestions for strategy {strategy_id}")
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {str(e)}")
            return {
                'error': str(e),
                'strengths': [],
                'weaknesses': ["Failed to analyze strategy"],
                'technical_improvements': []
            }
    
    async def apply_improvements(self, strategy_id: str, strategy_code: str, suggestions: Dict) -> str:
        """
        Apply suggested improvements to a strategy.
        
        Args:
            strategy_id: The strategy ID
            strategy_code: The original JavaScript code of the strategy
            suggestions: Improvement suggestions
            
        Returns:
            Improved strategy code
        """
        try:
            # Create a detailed prompt for applying improvements
            prompt = f"""
            Please improve this trading strategy based on the provided suggestions.
            
            ORIGINAL STRATEGY CODE:
            ```javascript
            {strategy_code}
            ```
            
            IMPROVEMENT SUGGESTIONS:
            {json.dumps(suggestions, indent=2)}
            
            Please implement these improvements in the strategy code. Make sure to maintain the overall structure and functionality while addressing the issues identified in the suggestions.
            
            Return ONLY the improved code, no explanations or markdown.
            """
            
            # Get improved code from OpenAI
            response = await self.openai.chat.completions.create(
                model=self.gpt_config.get('model', 'gpt-4o'),
                messages=[
                    {"role": "system", "content": "You are an expert trading strategy developer tasked with improving trading algorithms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4000
            )
            
            # Extract the improved code
            improved_code = response.choices[0].message.content
            
            # Remove markdown code blocks if present
            if improved_code.startswith("```javascript"):
                improved_code = improved_code.split("```javascript", 1)[1]
            if improved_code.startswith("```js"):
                improved_code = improved_code.split("```js", 1)[1]
            if improved_code.endswith("```"):
                improved_code = improved_code.rsplit("```", 1)[0]
            
            improved_code = improved_code.strip()
            
            # Generate a new strategy ID for the improved version
            improved_strategy_id = f"{strategy_id}_improved_{datetime.now().strftime('%Y%m%d%H%M')}"
            
            # Store the improved strategy
            await self.redis.set(
                f"strategy_code_{improved_strategy_id}",
                improved_code
            )
            
            # Store metadata about the improvement
            improvement_metadata = {
                'original_strategy_id': strategy_id,
                'improved_strategy_id': improved_strategy_id,
                'timestamp': datetime.now().isoformat(),
                'suggestions_applied': True,
                'evaluated': False
            }
            
            await self.redis.set(
                f"strategy_metadata_{improved_strategy_id}",
                json.dumps(improvement_metadata)
            )
            
            # Store relationship between original and improved strategy
            await self.redis.set(
                f"strategy_improvement_{strategy_id}",
                improved_strategy_id
            )
            
            logger.info(f"Applied improvements to strategy {strategy_id}, created improved version {improved_strategy_id}")
            return improved_code
            
        except Exception as e:
            logger.error(f"Error applying improvements: {str(e)}")
            return None
    
    async def systematic_evaluate_and_improve(self, strategy_id: str = None) -> Dict:
        """
        Perform a systematic evaluation and improvement of a strategy.
        If no strategy_id is provided, generates a new strategy.
        
        Args:
            strategy_id: Optional strategy ID to evaluate
            
        Returns:
            Comprehensive evaluation results
        """
        try:
            # Generate a new strategy if ID not provided
            if not strategy_id:
                # Generate parameters for a new strategy
                parameters = {
                    'type': 'adaptive',  # 'mean_reversion', 'trend_following', 'adaptive'
                    'timeframe': '5m',   # '1m', '5m', '15m', '1h', '4h'
                    'risk_limit': 2.0,
                    'target_profit': 1.5,
                    'max_position_size': 5.0
                }
                
                strategy_id, strategy_code = await self.generate_strategy(parameters)
                
                if not strategy_id or not strategy_code:
                    logger.error("Failed to generate strategy")
                    return {
                        'status': 'error',
                        'message': 'Failed to generate strategy'
                    }
            else:
                # Get existing strategy code
                strategy_code = await self.redis.get(f"strategy_code_{strategy_id}")
                
                if not strategy_code:
                    logger.error(f"Strategy code not found for {strategy_id}")
                    return {
                        'status': 'error',
                        'message': f"Strategy code not found for {strategy_id}"
                    }
            
            logger.info(f"Starting systematic evaluation for strategy {strategy_id}")
            
            # Step 1: Evaluate code quality
            code_evaluation = await self.evaluate_strategy_code(strategy_id, strategy_code)
            
            # Step 2: Evaluate performance
            performance_evaluation = await self.evaluate_strategy_performance(strategy_id, strategy_code)
            
            # Check if strategy meets minimum requirements
            if performance_evaluation.get('meets_requirements', False):
                logger.info(f"Strategy {strategy_id} meets minimum requirements")
                
                # Step 3: Generate improvement suggestions
                improvement_suggestions = await self.generate_improvement_suggestions(
                    strategy_id, 
                    strategy_code,
                    performance_evaluation.get('cv_results', {})
                )
                
                # Step 4: Apply improvements
                improved_code = await self.apply_improvements(
                    strategy_id,
                    strategy_code,
                    improvement_suggestions
                )
                
                # Get the improved strategy ID
                improved_strategy_id = await self.redis.get(f"strategy_improvement_{strategy_id}")
                
                # Step 5: Evaluate improved strategy
                if improved_code and improved_strategy_id:
                    improved_code_evaluation = await self.evaluate_strategy_code(
                        improved_strategy_id, 
                        improved_code
                    )
                    
                    improved_performance_evaluation = await self.evaluate_strategy_performance(
                        improved_strategy_id,
                        improved_code
                    )
                    
                    # Compare original and improved strategy
                    original_quality = performance_evaluation.get('quality_score', 0)
                    improved_quality = improved_performance_evaluation.get('quality_score', 0)
                    
                    improvement_percentage = ((improved_quality - original_quality) / max(0.0001, original_quality)) * 100
                    
                    comparison = {
                        'original_strategy_id': strategy_id,
                        'improved_strategy_id': improved_strategy_id,
                        'original_quality_score': original_quality,
                        'improved_quality_score': improved_quality,
                        'improvement_percentage': improvement_percentage,
                        'improvement_successful': improved_quality > original_quality
                    }
                    
                    # Update evaluation history
                    self.evaluation_history[strategy_id] = {
                        'timestamp': datetime.now().isoformat(),
                        'code_evaluation': code_evaluation,
                        'performance_evaluation': performance_evaluation,
                        'improvement_suggestions': improvement_suggestions,
                        'improved_strategy_id': improved_strategy_id,
                        'improvement_comparison': comparison
                    }
                    
                    # Store the comprehensive evaluation
                    comprehensive_evaluation = {
                        'strategy_id': strategy_id,
                        'timestamp': datetime.now().isoformat(),
                        'code_evaluation': code_evaluation,
                        'performance_evaluation': performance_evaluation,
                        'improvement_suggestions': improvement_suggestions,
                        'improved_strategy': {
                            'id': improved_strategy_id,
                            'code_evaluation': improved_code_evaluation,
                            'performance_evaluation': improved_performance_evaluation
                        },
                        'comparison': comparison,
                        'final_recommendation': 'deploy_improved' if comparison['improvement_successful'] else 'deploy_original'
                    }
                    
                    await self.redis.set(
                        f"comprehensive_evaluation_{strategy_id}",
                        json.dumps(comprehensive_evaluation)
                    )
                    
                    logger.info(f"Completed systematic evaluation for strategy {strategy_id}")
                    logger.info(f"Improved strategy quality score: {improved_quality:.4f} ({improvement_percentage:.2f}% change)")
                    
                    return comprehensive_evaluation
                    
                else:
                    # Just return the original evaluations
                    comprehensive_evaluation = {
                        'strategy_id': strategy_id,
                        'timestamp': datetime.now().isoformat(),
                        'code_evaluation': code_evaluation,
                        'performance_evaluation': performance_evaluation,
                        'improvement_suggestions': improvement_suggestions,
                        'final_recommendation': 'deploy_original' if performance_evaluation.get('meets_requirements', False) else 'reject'
                    }
                    
                    await self.redis.set(
                        f"comprehensive_evaluation_{strategy_id}",
                        json.dumps(comprehensive_evaluation)
                    )
                    
                    return comprehensive_evaluation
                    
            else:
                logger.info(f"Strategy {strategy_id} does not meet minimum requirements")
                
                # Return evaluation results with rejection
                comprehensive_evaluation = {
                    'strategy_id': strategy_id,
                    'timestamp': datetime.now().isoformat(),
                    'code_evaluation': code_evaluation,
                    'performance_evaluation': performance_evaluation,
                    'final_recommendation': 'reject'
                }
                
                await self.redis.set(
                    f"comprehensive_evaluation_{strategy_id}",
                    json.dumps(comprehensive_evaluation)
                )
                
                return comprehensive_evaluation
                
        except Exception as e:
            logger.error(f"Error in systematic evaluation: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def generate_html_report(self, strategy_id: str) -> str:
        """
        Generate a comprehensive HTML report for strategy evaluation.
        
        Args:
            strategy_id: The strategy ID
            
        Returns:
            HTML report as a string
        """
        try:
            # Get evaluation data
            comprehensive_evaluation = await self.redis.get(f"comprehensive_evaluation_{strategy_id}")
            
            if not comprehensive_evaluation:
                return f"<p>No comprehensive evaluation found for strategy {strategy_id}</p>"
            
            evaluation = json.loads(comprehensive_evaluation)
            
            # Get strategy metadata
            metadata = await self.redis.get(f"strategy_metadata_{strategy_id}")
            metadata_dict = json.loads(metadata) if metadata else {}
            
            # Start HTML report
            html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>AI Strategy Evaluation Report - {strategy_id}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }}
                    h1, h2, h3, h4 {{ color: #333; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .metrics-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                    .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    .metrics-table th {{ background-color: #f2f2f2; }}
                    .metrics-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .score-box {{ display: inline-block; padding: 8px 12px; border-radius: 4px; color: white; margin-right: 10px; }}
                    .high {{ background-color: #4CAF50; }}
                    .medium {{ background-color: #FFC107; }}
                    .low {{ background-color: #F44336; }}
                    .code-section {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow: auto; }}
                    .code {{ font-family: monospace; white-space: pre; }}
                    .improvement-item {{ margin-bottom: 10px; padding-left: 20px; }}
                    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 3px; color: white; font-size: 0.8em; margin-right: 5px; }}
                    .accepted {{ background-color: #4CAF50; }}
                    .rejected {{ background-color: #F44336; }}
                    .pending {{ background-color: #FFC107; }}
                    .chart-container {{ margin-bottom: 30px; }}
                    .chart {{ max-width: 100%; height: auto; }}
                    .comparison-container {{ display: flex; justify-content: space-between; }}
                    .comparison-box {{ width: 48%; padding: 15px; border-radius: 5px; border: 1px solid #ddd; }}
                    .criteria-box {{ margin-bottom: 15px; }}
                    .criteria-score {{ float: right; font-weight: bold; }}
                    .positive-change {{ color: #4CAF50; }}
                    .negative-change {{ color: #F44336; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>AI Strategy Evaluation Report</h1>
                    <p><strong>Strategy ID:</strong> {strategy_id}</p>
                    <p><strong>Created:</strong> {metadata_dict.get('timestamp', 'Unknown')}</p>
                    <p><strong>Evaluation Date:</strong> {evaluation.get('timestamp', 'Unknown')}</p>
                    
                    <h2>Evaluation Summary</h2>
                    <p><strong>Status:</strong> 
                        <span class="badge {'accepted' if evaluation.get('performance_evaluation', {}).get('meets_requirements', False) else 'rejected'}">
                            {'ACCEPTED' if evaluation.get('performance_evaluation', {}).get('meets_requirements', False) else 'REJECTED'}
                        </span>
                    </p>
                    <p><strong>Quality Score:</strong> 
                        <span class="score-box {'high' if evaluation.get('performance_evaluation', {}).get('quality_score', 0) > 0.7 else 'medium' if evaluation.get('performance_evaluation', {}).get('quality_score', 0) > 0.4 else 'low'}">
                            {evaluation.get('performance_evaluation', {}).get('quality_score', 0):.4f}
                        </span>
                    </p>
                    <p><strong>Final Recommendation:</strong> {evaluation.get('final_recommendation', 'Unknown').replace('_', ' ').title()}</p>
            """
            
            # Code evaluation section
            code_evaluation = evaluation.get('code_evaluation', {})
            if code_evaluation:
                html += """
                <h2>Code Quality Evaluation</h2>
                <div class="criteria-box">
                    <h3>Overall Code Quality Score: 
                        <span class="score-box """
                
                overall_score = code_evaluation.get('overall_score', 0)
                html += f"{'high' if overall_score >= 7 else 'medium' if overall_score >= 5 else 'low'}"
                html += f"""">
                            {overall_score} / 10
                        </span>
                    </h3>
                </div>
                
                <h3>Criteria Scores</h3>
                <table class="metrics-table">
                    <tr>
                        <th>Criterion</th>
                        <th>Score</th>
                        <th>Comments</th>
                    </tr>
                """
                
                for criterion, details in code_evaluation.get('criteria', {}).items():
                    score = details.get('score', 0)
                    comments = details.get('comments', '')
                    
                    html += f"""
                    <tr>
                        <td>{criterion.replace('_', ' ').title()}</td>
                        <td>
                            <span class="badge {'high' if score >= 7 else 'medium' if score >= 5 else 'low'}">
                                {score} / 10
                            </span>
                        </td>
                        <td>{comments}</td>
                    </tr>
                    """
                
                html += """
                </table>
                
                <h3>Strengths & Weaknesses</h3>
                <div style="display: flex; gap: 20px;">
                    <div style="flex: 1;">
                        <h4>Strengths</h4>
                        <ul>
                """
                
                for strength in code_evaluation.get('strengths', []):
                    html += f"<li>{strength}</li>"
                
                html += """
                        </ul>
                    </div>
                    <div style="flex: 1;">
                        <h4>Weaknesses</h4>
                        <ul>
                """
                
                for weakness in code_evaluation.get('weaknesses', []):
                    html += f"<li>{weakness}</li>"
                
                html += """
                        </ul>
                    </div>
                </div>
                """
            
            # Performance evaluation section
            performance_evaluation = evaluation.get('performance_evaluation', {})
            cv_results = performance_evaluation.get('cv_results', {})
            
            if performance_evaluation:
                html += """
                <h2>Performance Evaluation</h2>
                """
                
                overall_metrics = cv_results.get('overall_metrics', {})
                if overall_metrics:
                    html += """
                    <h3>Overall Performance Metrics</h3>
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
                
                # Add regime-specific performance
                regime_metrics = cv_results.get('regime_metrics', {})
                regime_scores = performance_evaluation.get('regime_scores', {})
                
                if regime_metrics:
                    html += """
                    <h3>Performance By Market Regime</h3>
                    <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                    """
                    
                    for regime, metrics in regime_metrics.items():
                        if not metrics:
                            continue
                        
                        regime_score = regime_scores.get(regime, 0)
                        
                        html += f"""
                        <div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px;">
                            <h4>{regime.title()} Market 
                                <span class="score-box {'high' if regime_score > 0.7 else 'medium' if regime_score > 0.4 else 'low'}">
                                    {regime_score:.2f}
                                </span>
                            </h4>
                            
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
                    
                    html += "</div>"
            
            # Improvement suggestions
            if 'improvement_suggestions' in evaluation:
                suggestions = evaluation['improvement_suggestions']
                
                html += """
                <h2>Improvement Suggestions</h2>
                
                <div style="display: flex; gap: 20px;">
                    <div style="flex: 1;">
                        <h3>Strengths</h3>
                        <ul>
                """
                
                for strength in suggestions.get('strengths', []):
                    html += f"<li>{strength}</li>"
                
                html += """
                        </ul>
                    </div>
                    <div style="flex: 1;">
                        <h3>Weaknesses</h3>
                        <ul>
                """
                
                for weakness in suggestions.get('weaknesses', []):
                    html += f"<li>{weakness}</li>"
                
                html += """
                        </ul>
                    </div>
                </div>
                
                <h3>Technical Improvements</h3>
                <table class="metrics-table">
                    <tr>
                        <th>Issue</th>
                        <th>Suggestion</th>
                        <th>Expected Impact</th>
                    </tr>
                """
                
                for improvement in suggestions.get('technical_improvements', []):
                    issue = improvement.get('issue', '')
                    suggestion = improvement.get('suggestion', '')
                    impact = improvement.get('expected_impact', '')
                    
                    html += f"""
                    <tr>
                        <td>{issue}</td>
                        <td>{suggestion}</td>
                        <td>{impact}</td>
                    </tr>
                    """
                
                html += """
                </table>
                
                <h3>Regime-Specific Improvements</h3>
                <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                """
                
                regime_improvements = suggestions.get('regime_specific_improvements', {})
                for regime, improvements in regime_improvements.items():
                    html += f"""
                    <div style="flex: 1; min-width: 300px; border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px;">
                        <h4>{regime.title()} Market</h4>
                        <ul>
                    """
                    
                    for improvement in improvements:
                        html += f"<li>{improvement}</li>"
                    
                    html += """
                        </ul>
                    </div>
                    """
                
                html += """
                </div>
                
                <h3>Risk Management Improvements</h3>
                <ul>
                """
                
                for improvement in suggestions.get('risk_management_improvements', []):
                    html += f"<li>{improvement}</li>"
                
                html += """
                </ul>
                
                <h3>Social Sentiment Improvements</h3>
                <ul>
                """
                
                for improvement in suggestions.get('social_sentiment_improvements', []):
                    html += f"<li>{improvement}</li>"
                
                html += """
                </ul>
                """
            
            # Improved strategy comparison
            if 'improved_strategy' in evaluation and 'comparison' in evaluation:
                improved_strategy = evaluation['improved_strategy']
                comparison = evaluation['comparison']
                
                improved_id = comparison.get('improved_strategy_id', '')
                improvement_percentage = comparison.get('improvement_percentage', 0)
                
                html += f"""
                <h2>Improved Strategy Comparison</h2>
                <p><strong>Improved Strategy ID:</strong> {improved_id}</p>
                <p><strong>Improvement:</strong> 
                    <span class="{'positive-change' if improvement_percentage > 0 else 'negative-change'}">
                        {improvement_percentage:.2f}%
                    </span>
                </p>
                
                <div class="comparison-container">
                    <div class="comparison-box">
                        <h3>Original Strategy</h3>
                        <p><strong>Quality Score:</strong> {comparison.get('original_quality_score', 0):.4f}</p>
                """
                
                if 'performance_evaluation' in evaluation:
                    original_metrics = cv_results.get('overall_metrics', {})
                    if original_metrics:
                        html += """
                        <h4>Key Metrics</h4>
                        <table class="metrics-table">
                        """
                        
                        for metric in ['sharpe_ratio', 'win_rate', 'profit_factor', 'max_drawdown']:
                            if metric in original_metrics:
                                value = original_metrics[metric]
                                html += f"""
                                <tr>
                                    <td>{metric.replace('_', ' ').title()}</td>
                                    <td>{value:.4f}</td>
                                </tr>
                                """
                        
                        html += """
                        </table>
                        """
                
                html += """
                    </div>
                    <div class="comparison-box">
                        <h3>Improved Strategy</h3>
                """
                
                improved_quality = comparison.get('improved_quality_score', 0)
                html += f"""
                        <p><strong>Quality Score:</strong> {improved_quality:.4f}</p>
                """
                
                if 'performance_evaluation' in improved_strategy:
                    improved_metrics = improved_strategy['performance_evaluation'].get('cv_results', {}).get('overall_metrics', {})
                    if improved_metrics:
                        html += """
                        <h4>Key Metrics</h4>
                        <table class="metrics-table">
                        """
                        
                        for metric in ['sharpe_ratio', 'win_rate', 'profit_factor', 'max_drawdown']:
                            if metric in improved_metrics:
                                improved_value = improved_metrics[metric]
                                original_value = original_metrics.get(metric, 0)
                                change = improved_value - original_value
                                change_pct = (change / max(0.0001, abs(original_value))) * 100
                                
                                html += f"""
                                <tr>
                                    <td>{metric.replace('_', ' ').title()}</td>
                                    <td>{improved_value:.4f} <span class="{'positive-change' if (metric != 'max_drawdown' and change > 0) or (metric == 'max_drawdown' and change < 0) else 'negative-change'}">({change_pct:.1f}%)</span></td>
                                </tr>
                                """
                        
                        html += """
                        </table>
                        """
                
                html += """
                    </div>
                </div>
                """
            
            # Close HTML document
            html += """
                </div>
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            return f"""
            <html>
                <body>
                    <h1>Error Generating Report</h1>
                    <p>An error occurred while generating the report for strategy {strategy_id}:</p>
                    <p>{str(e)}</p>
                </body>
            </html>
            """
    
    async def batch_evaluate_strategies(self, strategy_ids: List[str]) -> Dict:
        """
        Evaluate multiple strategies in batch and compare their performance.
        
        Args:
            strategy_ids: List of strategy IDs to evaluate
            
        Returns:
            Dictionary with batch evaluation results
        """
        try:
            batch_results = []
            
            # Evaluate each strategy
            for strategy_id in strategy_ids:
                strategy_code = await self.redis.get(f"strategy_code_{strategy_id}")
                
                if not strategy_code:
                    logger.warning(f"Strategy code not found for {strategy_id}")
                    continue
                
                evaluation = await self.systematic_evaluate_and_improve(strategy_id)
                
                if evaluation and evaluation.get('status') != 'error':
                    batch_results.append({
                        'strategy_id': strategy_id,
                        'quality_score': evaluation.get('performance_evaluation', {}).get('quality_score', 0),
                        'meets_requirements': evaluation.get('performance_evaluation', {}).get('meets_requirements', False),
                        'final_recommendation': evaluation.get('final_recommendation', 'reject')
                    })
            
            # Rank strategies
            if batch_results:
                ranked_strategies = sorted(
                    batch_results,
                    key=lambda x: x['quality_score'],
                    reverse=True
                )
                
                # Determine top performers
                top_strategies = [s for s in ranked_strategies if s['meets_requirements']]
                
                # Create batch evaluation record
                batch_evaluation = {
                    'timestamp': datetime.now().isoformat(),
                    'strategy_count': len(strategy_ids),
                    'evaluated_count': len(batch_results),
                    'accepted_count': len(top_strategies),
                    'ranked_strategies': ranked_strategies,
                    'top_strategies': top_strategies[:3] if len(top_strategies) > 3 else top_strategies
                }
                
                # Store batch evaluation
                batch_id = f"batch_evaluation_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                await self.redis.set(
                    batch_id,
                    json.dumps(batch_evaluation)
                )
                
                return batch_evaluation
            
            return {
                'status': 'error',
                'message': 'No strategies were successfully evaluated'
            }
            
        except Exception as e:
            logger.error(f"Error in batch evaluation: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def run(self):
        """Main service loop"""
        try:
            logger.info("Starting AI Strategy Evaluator...")
            
            # Initialize the necessary services
            
            while True:
                # Check for pending strategy evaluations
                pending_evals = await self.redis.lrange('pending_strategy_evaluations', 0, -1)
                
                for strategy_id in pending_evals:
                    logger.info(f"Processing pending evaluation for strategy {strategy_id}")
                    
                    # Perform evaluation
                    await self.systematic_evaluate_and_improve(strategy_id)
                    
                    # Remove from pending list
                    await self.redis.lrem('pending_strategy_evaluations', 1, strategy_id)
                
                # Generate strategies periodically if configured to do so
                generate_interval = int(os.getenv('STRATEGY_GENERATION_INTERVAL', '0'))
                
                if generate_interval > 0:
                    # Check last generation time
                    last_gen_time = await self.redis.get('last_strategy_generation_time')
                    
                    current_time = datetime.now()
                    should_generate = False
                    
                    if last_gen_time:
                        last_time = datetime.fromisoformat(last_gen_time)
                        if (current_time - last_time).total_seconds() > generate_interval:
                            should_generate = True
                    else:
                        should_generate = True
                    
                    if should_generate:
                        # Generate new strategy
                        logger.info("Generating new strategy as part of regular interval")
                        await self.systematic_evaluate_and_improve()
                        
                        # Update last generation time
                        await self.redis.set('last_strategy_generation_time', current_time.isoformat())
                
                # Sleep before checking again
                await asyncio.sleep(60)  # Check every minute
                
        except Exception as e:
            logger.error(f"Error in AI Strategy Evaluator service: {str(e)}")
        finally:
            logger.info("Stopping AI Strategy Evaluator...")

if __name__ == "__main__":
    evaluator = AIStrategyEvaluator()
    try:
        asyncio.run(evaluator.run())
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")