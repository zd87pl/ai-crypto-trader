import json
import logging as logger
from typing import Dict, List, Optional
from openai import AsyncOpenAI
from datetime import datetime

class AITrader:
    def __init__(self, config: Dict):
        """Initialize AITrader with configuration"""
        self.config = config
        self.client = AsyncOpenAI(api_key=config['openai']['api_key'])
        self.model = config['openai']['model']
        self.temperature = config['openai']['temperature']
        self.max_tokens = config['openai']['max_tokens']
        
    async def analyze_trade_opportunity(self, market_data: Dict) -> Dict:
        """Analyze a single trading opportunity using OpenAI"""
        try:
            # Format the analysis prompt with market data
            prompt = self.config['openai']['analysis_prompt'].format(
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
                price_change_5m=market_data['price_change_5m'],
                price_change_15m=market_data['price_change_15m']
            )
            
            # Get analysis from OpenAI
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an experienced cryptocurrency trader focused on technical analysis and risk management."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={ "type": "json_object" }
            )
            
            # Parse the response
            analysis = json.loads(response.choices[0].message.content)
            
            # Add timestamp to analysis
            analysis['timestamp'] = datetime.now().isoformat()
            
            # Log the analysis
            logger.info(f"AI Analysis for {market_data['symbol']}:")
            logger.info(f"Decision: {analysis['decision']}")
            logger.info(f"Confidence: {analysis['confidence']}")
            logger.info(f"Reasoning: {analysis['reasoning']}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}")
            return {
                'decision': 'ERROR',
                'confidence': 0,
                'reasoning': f"Error during analysis: {str(e)}",
                'timestamp': datetime.now().isoformat()
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
        """Analyze overall market conditions"""
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
                    f"5m Change: {data['price_change_5m']:.2f}%\n"
                    f"15m Change: {data['price_change_15m']:.2f}%\n"
                    f"---"
                )
                market_summary.append(summary)
            
            # Format the market analysis prompt
            prompt = self.config['openai']['market_prompt'].format(
                market_data="\n".join(market_summary)
            )
            
            # Get analysis from OpenAI
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a cryptocurrency market analyst focused on identifying trading opportunities and risks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={ "type": "json_object" }
            )
            
            # Parse the response
            analysis = json.loads(response.choices[0].message.content)
            
            # Add timestamp to analysis
            analysis['timestamp'] = datetime.now().isoformat()
            
            # Log the analysis
            logger.info("AI Market Analysis:")
            logger.info(f"Sentiment: {analysis['market_sentiment']}")
            logger.info(f"Top Opportunities: {analysis['top_opportunities']}")
            logger.info(f"Risks: {analysis['risks']}")
            logger.info(f"Reasoning: {analysis['reasoning']}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            return {
                'market_sentiment': 'ERROR',
                'top_opportunities': [],
                'risks': [str(e)],
                'reasoning': f"Error during analysis: {str(e)}",
                'timestamp': datetime.now().isoformat()
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
