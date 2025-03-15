import os
import json
import asyncio
import logging as logger
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import redis.asyncio as redis_async
from redis.exceptions import ConnectionError
from openai import AsyncOpenAI

# Configure logging
logger.basicConfig(
    level=logger.INFO,
    format='%(asctime)s - %(levelname)s - [AIExplain] %(message)s',
    handlers=[
        logger.FileHandler('logs/ai_explainability.log'),
        logger.StreamHandler()
    ]
)

class AIExplainabilityService:
    """
    Service for providing explanations of AI trading decisions.
    Implements AI-09: Add explainable AI features for trading decisions.
    """
    
    def __init__(self):
        """Initialize the AI explainability service"""
        # Ensure directories exist
        os.makedirs('explanations', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # OpenAI configuration
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = self.config['openai']['model']
        self.temperature = self.config['openai']['temperature']
        self.max_tokens = self.config['openai']['max_tokens']
        
        # Redis configuration
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis = None
        self.pubsub = None
        
        # Service state
        self.running = True
        
        # Explanation storage paths
        self.explanations_path = Path('explanations')
        self.explanations_path.mkdir(exist_ok=True)
        
        logger.info("AI Explainability Service initialized")

    async def connect_redis(self, max_retries=10, retry_delay=5):
        """Connect to Redis with retries"""
        retries = 0
        while retries < max_retries and self.running:
            try:
                if self.redis is None:
                    logger.debug(f"Attempting Redis connection (attempt {retries + 1}/{max_retries})")
                    self.redis = redis_async.Redis(
                        host=self.redis_host,
                        port=self.redis_port,
                        decode_responses=True,
                        socket_connect_timeout=5.0,
                        socket_keepalive=True,
                        health_check_interval=15
                    )
                await self.redis.ping()
                logger.info(f"Successfully connected to Redis at {self.redis_host}:{self.redis_port}")
                return True
            except (ConnectionError, OSError) as e:
                retries += 1
                logger.error(f"Failed to connect to Redis (attempt {retries}/{max_retries}): {str(e)}")
                if self.redis:
                    await self.redis.close()
                    self.redis = None
                if retries < max_retries:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("Max retries reached. Could not connect to Redis.")
                    return False
            except Exception as e:
                logger.error(f"Unexpected error connecting to Redis: {str(e)}")
                if self.redis:
                    await self.redis.close()
                    self.redis = None
                await asyncio.sleep(retry_delay)
                retries += 1
                
    async def setup_pubsub(self):
        """Set up Redis pubsub for trading signals"""
        try:
            # Verify Redis connection
            if not self.redis or not await self.redis.ping():
                logger.error("Redis connection not available for pubsub setup")
                return False
                
            # Close existing pubsub if any
            if self.pubsub:
                logger.debug("Closing existing pubsub connection")
                await self.pubsub.close()
                self.pubsub = None
                
            # Create new pubsub instance
            logger.debug("Creating new pubsub instance")
            self.pubsub = self.redis.pubsub()
            
            # Subscribe to channels
            logger.debug("Subscribing to trading_signals channel")
            await self.pubsub.subscribe('trading_signals')
            
            # Get first message to confirm subscription
            logger.debug("Waiting for subscription confirmation message")
            message = await self.pubsub.get_message(timeout=1.0)
            if message and message['type'] == 'subscribe':
                logger.info("Successfully subscribed to trading_signals channel")
                return True
            else:
                logger.error("Failed to subscribe to channel")
                return False
                
        except Exception as e:
            logger.error(f"Error in pubsub setup: {str(e)}", exc_info=True)
            if self.pubsub:
                await self.pubsub.close()
                self.pubsub = None
            return False

    async def explain_trade_decision(self, analysis_data: Dict) -> Dict:
        """
        Generate a detailed explanation for a trading decision
        """
        try:
            # Check if explanation is already provided
            if 'explanation' in analysis_data and isinstance(analysis_data['explanation'], dict):
                logger.info("Using existing explanation from analysis data")
                return analysis_data['explanation']
                
            # Prepare prompt for explanation
            prompt = f"""
            Please explain the following cryptocurrency trading decision in detail:
            
            Trading Signal:
            - Symbol: {analysis_data.get('symbol', 'Unknown')}
            - Decision: {analysis_data.get('decision', 'Unknown')}
            - Confidence: {analysis_data.get('confidence', '0')}
            - Timestamp: {analysis_data.get('timestamp', 'Unknown')}
            
            Technical Analysis Factors:
            - RSI: {analysis_data.get('market_data', {}).get('rsi', 'N/A')}
            - MACD: {analysis_data.get('market_data', {}).get('macd', 'N/A')}
            - Bollinger Bands Position: {analysis_data.get('market_data', {}).get('bb_position', 'N/A')}
            - Price Change (1m): {analysis_data.get('market_data', {}).get('price_change_1m', 'N/A')}%
            - Price Change (5m): {analysis_data.get('market_data', {}).get('price_change_5m', 'N/A')}%
            - Trend: {analysis_data.get('market_data', {}).get('trend', 'N/A')}
            
            Social Metrics:
            - Social Sentiment: {analysis_data.get('market_data', {}).get('social_sentiment', 'N/A')}
            - Social Volume: {analysis_data.get('market_data', {}).get('social_volume', 'N/A')}
            - Social Engagement: {analysis_data.get('market_data', {}).get('social_engagement', 'N/A')}
            
            Original Reasoning:
            {analysis_data.get('reasoning', 'No reasoning provided')}
            
            Provide a detailed explanation of this trading decision, including:
            1. A clear, concise summary
            2. The key technical factors that influenced the decision
            3. How social metrics affected the decision
            4. The most important indicators and their impact
            5. A risk assessment based on the data
            
            Return your explanation as a JSON object with keys: 'summary', 'technical_factors', 'social_factors', 'key_indicators', and 'risk_assessment'.
            """
            
            # Get explanation from LLM
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in explaining trading decisions and cryptocurrency analysis in clear, educational terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower for more consistency
                max_tokens=self.max_tokens,
                response_format={ "type": "json_object" }
            )
            
            # Parse the explanation
            explanation = json.loads(response.choices[0].message.content)
            
            # Add timestamp to explanation
            explanation['generated_at'] = datetime.now().isoformat()
            
            # Save the explanation
            self._save_explanation(analysis_data['symbol'], explanation, analysis_data.get('timestamp'))
            
            # Return the explanation
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return {
                'summary': "Error generating explanation",
                'technical_factors': f"Error: {str(e)}",
                'social_factors': "Unavailable due to error",
                'key_indicators': [],
                'risk_assessment': "Could not assess due to error",
                'generated_at': datetime.now().isoformat()
            }
            
    def _save_explanation(self, symbol: str, explanation: Dict, timestamp=None) -> None:
        """Save an explanation to disk"""
        try:
            # Create a directory for this symbol if it doesn't exist
            symbol_dir = self.explanations_path / symbol
            symbol_dir.mkdir(exist_ok=True)
            
            # Format timestamp for filename
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        # Try to parse ISO format timestamp
                        dt = datetime.fromisoformat(timestamp)
                    else:
                        dt = timestamp
                    timestamp_str = dt.strftime('%Y%m%d_%H%M%S')
                except:
                    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            else:
                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                
            # Create filename
            filename = f"{symbol}_{timestamp_str}_explanation.json"
            filepath = symbol_dir / filename
            
            # Save explanation to file
            with open(filepath, 'w') as f:
                json.dump(explanation, f, indent=4)
                
            logger.debug(f"Saved explanation to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving explanation: {str(e)}")
            
    async def visualize_decision_factors(self, analysis_data: Dict) -> Dict:
        """
        Generate visualization data for decision factors
        """
        try:
            # Extract factor weights or generate them
            factor_weights = analysis_data.get('factor_weights', {})
            
            # If factor weights are missing or incomplete, infer them from the reasoning
            if not factor_weights or all(not isinstance(v, dict) for v in factor_weights.values()):
                # Use simple placeholders for now
                # In a real implementation, we'd use NLP to extract these from the reasoning
                technical_weight = 0.6
                social_weight = 0.3
                market_context_weight = 0.1
                
                # Technical indicators (placeholder weights)
                technical_indicators = {
                    'rsi': 0.3,
                    'macd': 0.2,
                    'bollinger_bands': 0.15,
                    'price_action': 0.25,
                    'other': 0.1
                }
                
                # Social metrics (placeholder weights)
                social_metrics = {
                    'sentiment': 0.5,
                    'volume': 0.3,
                    'engagement': 0.2
                }
                
                # Create factor weights structure
                factor_weights = {
                    'technical_indicators': technical_indicators,
                    'social_metrics': social_metrics,
                    'market_context': market_context_weight
                }
            
            # Generate visualization data
            visualization_data = {
                'decision': analysis_data.get('decision', 'UNKNOWN'),
                'confidence': analysis_data.get('confidence', 0),
                'timestamp': analysis_data.get('timestamp', datetime.now().isoformat()),
                'symbol': analysis_data.get('symbol', 'UNKNOWN'),
                'factor_weights': factor_weights,
                'visualization_type': 'decision_factors',
                'generated_at': datetime.now().isoformat()
            }
            
            return visualization_data
            
        except Exception as e:
            logger.error(f"Error generating visualization data: {str(e)}")
            return {
                'error': f"Failed to generate visualization: {str(e)}"
            }
            
    async def explain_model_output(self, trading_signal: Dict) -> Dict:
        """
        Process a trading signal to add explainability
        """
        try:
            # Check if signal already has explanation
            if 'explanation' in trading_signal and trading_signal['explanation']:
                logger.info(f"Trading signal for {trading_signal.get('symbol')} already has explanation")
                return trading_signal
                
            # Generate explanation
            explanation = await self.explain_trade_decision(trading_signal)
            
            # Generate visualization data
            visualization = await self.visualize_decision_factors(trading_signal)
            
            # Add explanation and visualization to signal
            enriched_signal = trading_signal.copy()
            enriched_signal['explanation'] = explanation
            enriched_signal['visualization'] = visualization
            
            # Publish enriched signal to Redis
            if self.redis and await self.redis.ping():
                await self.redis.publish(
                    'explained_trading_signals',
                    json.dumps(enriched_signal)
                )
                
                # Store in Redis for later access
                symbol = trading_signal.get('symbol', 'unknown')
                timestamp = datetime.now().isoformat()
                key = f"explanation:{symbol}:{timestamp}"
                await self.redis.set(key, json.dumps(enriched_signal))
                await self.redis.expire(key, 60 * 60 * 24 * 7)  # Keep for 7 days
                
                # Track in time-ordered list
                await self.redis.lpush(f"explanations:{symbol}", key)
                await self.redis.ltrim(f"explanations:{symbol}", 0, 99)  # Keep last 100
                
            return enriched_signal
            
        except Exception as e:
            logger.error(f"Error explaining model output: {str(e)}")
            return trading_signal  # Return original signal on error
            
    async def process_trading_signals(self):
        """Process trading signals from Redis"""
        try:
            if not self.pubsub:
                if not await self.setup_pubsub():
                    logger.error("Failed to setup pubsub for trading signals")
                    return
                    
            # Get messages from pubsub
            message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if not message:
                return
                
            # Process the message
            if message['type'] == 'message':
                try:
                    # Parse the trading signal
                    trading_signal = json.loads(message['data'])
                    
                    # Log the signal
                    logger.info(f"Processing trading signal for {trading_signal.get('symbol', 'unknown')}")
                    
                    # Explain the trading signal
                    enriched_signal = await self.explain_model_output(trading_signal)
                    
                    logger.info(f"Generated explanation for {trading_signal.get('symbol', 'unknown')}")
                    
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in message: {message['data']}")
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error in process_trading_signals: {str(e)}")
            
    async def maintain_redis(self):
        """Maintain Redis connection"""
        while self.running:
            try:
                if self.redis:
                    await self.redis.ping()
                else:
                    await self.connect_redis()
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Redis connection error: {str(e)}")
                self.redis = None
                await asyncio.sleep(5)
                
    async def run(self):
        """Run the AI explainability service"""
        try:
            logger.info("Starting AI Explainability Service...")
            
            # First establish Redis connection
            if not await self.connect_redis(max_retries=15, retry_delay=2):
                raise Exception("Failed to establish initial Redis connection")
                
            # Setup pubsub
            if not await self.setup_pubsub():
                logger.warning("Failed to set up pubsub, will retry later")
                
            # Start Redis maintenance task
            redis_task = asyncio.create_task(self.maintain_redis())
            
            # Main service loop
            while self.running:
                try:
                    # Process trading signals
                    await self.process_trading_signals()
                    
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error in main service loop: {str(e)}")
                    await asyncio.sleep(5)
                    
        except Exception as e:
            logger.error(f"Error in AI Explainability Service: {str(e)}")
            
        finally:
            # Clean up
            if self.pubsub:
                await self.pubsub.close()
                
            if self.redis:
                await self.redis.close()
                
            # Cancel maintenance task
            if 'redis_task' in locals():
                redis_task.cancel()
                try:
                    await redis_task
                except asyncio.CancelledError:
                    pass
                    
            logger.info("AI Explainability Service stopped")
            
    async def stop(self):
        """Stop the AI explainability service"""
        logger.info("Stopping AI Explainability Service...")
        self.running = False
        
        # Close Redis connections
        if self.pubsub:
            await self.pubsub.close()
            
        if self.redis:
            await self.redis.close()

# Run the service if executed directly
if __name__ == "__main__":
    service = AIExplainabilityService()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        asyncio.run(service.stop())
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        asyncio.run(service.stop())