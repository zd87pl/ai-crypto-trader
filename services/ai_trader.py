import os
import json
import logging as logger
from typing import Dict, List, Optional
from openai import AsyncOpenAI
from datetime import datetime
import uuid

class AITrader:
    def __init__(self, config: Dict):
        """Initialize AITrader with configuration"""
        self.config = config
        
        # Get API key from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = config['openai']['model']
        self.temperature = config['openai']['temperature']
        self.max_tokens = config['openai']['max_tokens']
        
        # Model versioning attributes
        self.version_id = str(uuid.uuid4())[:8]  # Generate unique version ID
        self.version_name = f"ai_trader_{self.model.replace('-', '_')}_{self.version_id}"
        self.creation_date = datetime.now().isoformat()
        self.performance_metrics = {
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "average_confidence": 0,
            "cumulative_confidence": 0
        }
        
    async def analyze_trade_opportunity(self, market_data: Dict) -> Dict:
        """Analyze a single trading opportunity using OpenAI with explainability"""
        try:
            # Log incoming market data
            logger.debug(f"Analyzing trade opportunity with market data: {json.dumps(market_data, indent=2)}")
            
            # Verify all required fields are present
            required_fields = [
                'symbol', 'current_price', 'avg_volume', 'rsi', 'stoch_k',
                'macd', 'williams_r', 'bb_position', 'trend', 'trend_strength',
                'price_change_1m', 'price_change_3m', 'price_change_5m', 'price_change_15m'
            ]
            
            missing_fields = [field for field in required_fields if field not in market_data]
            if missing_fields:
                raise KeyError(f"Missing required fields: {', '.join(missing_fields)}")
            
            # Format the analysis prompt with market data
            try:
                # Use enhanced prompt that includes explainability requirements
                enhanced_prompt = self.config['openai'].get('explainable_analysis_prompt', self.config['openai']['analysis_prompt'])
                
                prompt = enhanced_prompt.format(
                    symbol=market_data['symbol'],
                    price=market_data['current_price'],
                    volume=market_data['avg_volume'],
                    rsi=market_data['rsi'],
                    stoch=market_data['stoch_k'],
                    macd=market_data['macd'],
                    williams_r=market_data['williams_r'],
                    bb_position=market_data['bb_position'],
                    trend=market_data['trend'],
                    trend_strength=market_data['trend_strength'],
                    price_change_1m=market_data['price_change_1m'],
                    price_change_3m=market_data['price_change_3m'],
                    price_change_5m=market_data['price_change_5m'],
                    price_change_15m=market_data['price_change_15m'],
                    social_volume=market_data.get('social_volume', 0),
                    social_engagement=market_data.get('social_engagement', 0),
                    social_contributors=market_data.get('social_contributors', 0),
                    social_sentiment=market_data.get('social_sentiment', 0.5),
                    recent_news=market_data.get('recent_news', 'No recent news available'),
                    market_context=market_data.get('market_context', 'Market context unavailable')
                )
            except KeyError as e:
                logger.error(f"Error formatting prompt: {str(e)}")
                logger.error(f"Market data keys: {list(market_data.keys())}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error formatting prompt: {str(e)}")
                raise
            
            logger.debug(f"Formatted prompt: {prompt}")
            
            # Get analysis from OpenAI
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an experienced cryptocurrency trader focused on technical analysis, risk management, and providing transparent explanations of your trading decisions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={ "type": "json_object" }
                )
            except Exception as e:
                logger.error(f"OpenAI API error: {str(e)}")
                raise
            
            # Parse the response
            try:
                analysis = json.loads(response.choices[0].message.content)
            except Exception as e:
                logger.error(f"Error parsing OpenAI response: {str(e)}")
                logger.error(f"Raw response: {response.choices[0].message.content}")
                raise
            
            # Add model metadata and explainability metrics
            analysis['timestamp'] = datetime.now().isoformat()
            analysis['model_version'] = self.version_name
            analysis['model_id'] = self.version_id
            
            # Add factor weights (if not already provided by the model)
            if 'factor_weights' not in analysis:
                analysis['factor_weights'] = {
                    'technical_indicators': {},
                    'price_action': {},
                    'social_metrics': {},
                    'market_context': 0.0
                }
                
                # Try to extract these from the reasoning if possible
                # For now, we'll use placeholder values but in a real implementation,
                # these would be extracted from the reasoning or explicitly returned by the model
            
            # Add explanation components if not already present
            if 'explanation' not in analysis:
                analysis['explanation'] = {
                    'summary': analysis.get('reasoning', 'No explanation provided'),
                    'technical_factors': 'Technical analysis factors not specified',
                    'social_factors': 'Social analysis factors not specified',
                    'key_indicators': [],
                    'risk_assessment': 'Risk not explicitly assessed'
                }
            
            # Log the analysis
            logger.info(f"AI Analysis for {market_data['symbol']}:")
            logger.info(f"Decision: {analysis['decision']}")
            logger.info(f"Confidence: {analysis['confidence']}")
            logger.info(f"Model: {self.version_name}")
            logger.info(f"Reasoning: {analysis['reasoning']}")
            
            # Update model performance metrics
            self.performance_metrics['total_trades'] += 1
            self.performance_metrics['cumulative_confidence'] += analysis['confidence']
            self.performance_metrics['average_confidence'] = self.performance_metrics['cumulative_confidence'] / self.performance_metrics['total_trades']
            
            # Track whether this was a successful analysis
            if analysis['decision'] != 'ERROR' and analysis['confidence'] > 0:
                self.performance_metrics['successful_trades'] += 1
            else:
                self.performance_metrics['failed_trades'] += 1
                
            # Add performance tracking to the analysis
            analysis['model_performance'] = {
                'success_rate': self.performance_metrics['successful_trades'] / self.performance_metrics['total_trades'],
                'avg_confidence': self.performance_metrics['average_confidence'],
                'total_trades': self.performance_metrics['total_trades']
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}", exc_info=True)
            # Update error stats
            self.performance_metrics['total_trades'] += 1
            self.performance_metrics['failed_trades'] += 1
            
            return {
                'decision': 'ERROR',
                'confidence': 0,
                'reasoning': f"Error during analysis: {str(e)}",
                'timestamp': datetime.now().isoformat(),
                'model_version': self.version_name,
                'model_id': self.version_id,
                'explanation': {
                    'summary': f"Error during analysis: {str(e)}",
                    'technical_factors': 'Analysis failed',
                    'social_factors': 'Analysis failed',
                    'key_indicators': [],
                    'risk_assessment': 'Unable to assess risk due to error'
                }
            }
    
    async def analyze_risk_setup(self, trade_setup: Dict) -> Dict:
        """Analyze risk and suggest position sizing"""
        try:
            # Format the risk analysis prompt
            prompt = self.config['openai']['risk_prompt'].format(
                symbol=trade_setup['symbol'],
                capital=trade_setup['available_capital'],
                volatility=trade_setup['volatility'],
                price=trade_setup['current_price'],
                trend_strength=trade_setup['trend_strength']
            )
            
            # Get analysis from OpenAI
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a risk management expert in cryptocurrency trading."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={ "type": "json_object" }
            )
            
            # Parse the response
            analysis = json.loads(response.choices[0].message.content)
            
            # Log the analysis
            logger.info(f"AI Risk Analysis for {trade_setup['symbol']}:")
            logger.info(f"Position Size: {analysis['position_size']}")
            logger.info(f"Stop Loss: {analysis['stop_loss_pct']}%")
            logger.info(f"Take Profit: {analysis['take_profit_pct']}%")
            logger.info(f"Reasoning: {analysis['reasoning']}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in risk analysis: {str(e)}")
            return {
                'position_size': 0,
                'stop_loss_pct': 0,
                'take_profit_pct': 0,
                'reasoning': f"Error during analysis: {str(e)}"
            }
    
    async def analyze_market_conditions(self, market_data: List[Dict]) -> Dict:
        """Analyze overall market conditions with explainability"""
        try:
            # Format market data for the prompt
            market_summary = []
            for data in market_data:
                summary = (
                    f"{data['symbol']}:\n"
                    f"Price: ${data['current_price']}\n"
                    f"Volume: ${data['avg_volume']:,.2f}\n"
                    f"RSI: {data['rsi']:.2f}\n"
                    f"Trend: {data['trend']}\n"
                    f"1m Change: {data['price_change_1m']:.2f}%\n"
                    f"3m Change: {data['price_change_3m']:.2f}%\n"
                    f"5m Change: {data['price_change_5m']:.2f}%\n"
                    f"15m Change: {data['price_change_15m']:.2f}%\n"
                    f"---"
                )
                market_summary.append(summary)
            
            # Format the market analysis prompt
            enhanced_prompt = self.config['openai'].get('explainable_market_prompt', self.config['openai']['market_prompt'])
            prompt = enhanced_prompt.format(
                market_data="\n".join(market_summary)
            )
            
            # Get analysis from OpenAI
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a cryptocurrency market analyst focused on identifying trading opportunities and risks with clear explanations of your reasoning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={ "type": "json_object" }
            )
            
            # Parse the response
            analysis = json.loads(response.choices[0].message.content)
            
            # Add model metadata and timing information
            analysis['timestamp'] = datetime.now().isoformat()
            analysis['model_version'] = self.version_name
            analysis['model_id'] = self.version_id
            
            # Add explainability components if not already present
            if 'explanation' not in analysis:
                analysis['explanation'] = {
                    'summary': analysis.get('reasoning', 'No explanation provided'),
                    'market_factors': 'Market factors not explicitly specified',
                    'key_trends': [],
                    'risk_factors': analysis.get('risks', []),
                    'sentiment_indicators': [],
                    'recommendation_rationale': 'Rationale not explicitly provided'
                }
                
            # Add factor influence weights if not already present
            if 'factor_weights' not in analysis:
                analysis['factor_weights'] = {
                    'price_action': 0.0,
                    'technical_indicators': 0.0,
                    'volume_analysis': 0.0,
                    'social_sentiment': 0.0,
                    'market_trends': 0.0
                }
            
            # Log the analysis
            logger.info("AI Market Analysis:")
            logger.info(f"Sentiment: {analysis['market_sentiment']}")
            logger.info(f"Top Opportunities: {analysis['top_opportunities']}")
            logger.info(f"Risks: {analysis['risks']}")
            logger.info(f"Model: {self.version_name}")
            logger.info(f"Reasoning: {analysis['reasoning']}")
            
            # Update model performance metrics
            self.performance_metrics['total_trades'] += 1
            
            # No direct way to measure "success" of market analysis, 
            # but we can track that it completed without error
            self.performance_metrics['successful_trades'] += 1
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            # Update error stats
            self.performance_metrics['total_trades'] += 1
            self.performance_metrics['failed_trades'] += 1
            
            return {
                'market_sentiment': 'ERROR',
                'top_opportunities': [],
                'risks': [str(e)],
                'reasoning': f"Error during analysis: {str(e)}",
                'timestamp': datetime.now().isoformat(),
                'model_version': self.version_name,
                'model_id': self.version_id,
                'explanation': {
                    'summary': f"Error during analysis: {str(e)}",
                    'market_factors': 'Analysis failed',
                    'key_trends': [],
                    'risk_factors': [str(e)],
                    'sentiment_indicators': [],
                    'recommendation_rationale': 'Unable to provide recommendations due to error'
                }
            }
            
    def get_model_info(self) -> Dict:
        """Get information about the current model version"""
        return {
            'version_id': self.version_id,
            'version_name': self.version_name,
            'model': self.model,
            'creation_date': self.creation_date,
            'performance_metrics': self.performance_metrics,
            'config': {
                'temperature': self.temperature,
                'max_tokens': self.max_tokens
            }
        }
        
    def reset_performance_metrics(self) -> None:
        """Reset performance metrics for the model"""
        self.performance_metrics = {
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "average_confidence": 0,
            "cumulative_confidence": 0
        }
    
    def should_take_trade(self, analysis: Dict) -> bool:
        """Determine if we should take a trade based on AI analysis"""
        try:
            # Check if analysis is valid
            if analysis.get('decision') == 'ERROR':
                return False
            
            # Check confidence threshold
            if analysis.get('confidence', 0) < self.config['trading_params']['ai_confidence_threshold']:
                return False
            
            # Only take BUY decisions
            if analysis.get('decision') != 'BUY':
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in trade decision: {str(e)}")
            return False
    
    def adjust_position_size(self, ai_position: Dict, technical_position: Dict) -> Dict:
        """Combine AI and technical analysis position sizing"""
        try:
            # Get position sizes
            ai_size = float(ai_position.get('position_size', 0))
            tech_size = float(technical_position.get('position_size', 0))
            
            # Average the position sizes
            position_size = (ai_size + tech_size) / 2
            
            # Get the more conservative stop loss
            ai_stop = float(ai_position.get('stop_loss_pct', 0))
            tech_stop = float(technical_position.get('stop_loss_pct', 0))
            stop_loss_pct = max(ai_stop, tech_stop)
            
            # Get the more conservative take profit
            ai_profit = float(ai_position.get('take_profit_pct', 0))
            tech_profit = float(technical_position.get('take_profit_pct', 0))
            take_profit_pct = min(ai_profit, tech_profit)
            
            return {
                'position_size': position_size,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'reasoning': f"Combined AI ({ai_size:.2f}) and Technical ({tech_size:.2f}) analysis"
            }
            
        except Exception as e:
            logger.error(f"Error adjusting position size: {str(e)}")
            return technical_position  # Fall back to technical analysis
