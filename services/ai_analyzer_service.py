import os
import json
import redis
import asyncio
from datetime import datetime
import logging as logger
from openai import AsyncOpenAI

# Configure logging
logger.basicConfig(
    level=logger.INFO,
    format='%(asctime)s - %(levelname)s - [AIAnalyzer] %(message)s',
    handlers=[
        logger.FileHandler('logs/ai_analyzer.log'),
        logger.StreamHandler()
    ]
)

class AIAnalyzerService:
    def __init__(self):
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)

        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Initialize Redis connection
        self.redis = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=True
        )

        self.running = True
        self.market_data = {}
        self.last_analysis_time = {}

    async def analyze_market_data(self, market_update):
        """Analyze market data using OpenAI API"""
        try:
            symbol = market_update['symbol']
            current_time = datetime.now()

            # Check if we need to analyze this symbol
            if (symbol in self.last_analysis_time and 
                (current_time - self.last_analysis_time[symbol]).seconds < self.config['trading_params']['ai_analysis_interval']):
                return

            # Prepare market data for analysis
            prompt = self.config['openai']['analysis_prompt'].format(
                symbol=symbol,
                price=market_update['price'],
                volume=market_update['volume'],
                price_change=market_update['price_change']
            )

            # Get analysis from OpenAI
            response = await self.client.chat.completions.create(
                model=self.config['openai']['model'],
                messages=[
                    {"role": "system", "content": "You are an experienced cryptocurrency trader focused on technical analysis and risk management."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['openai']['temperature'],
                max_tokens=self.config['openai']['max_tokens'],
                response_format={ "type": "json_object" }
            )

            # Parse the response
            analysis = json.loads(response.choices[0].message.content)

            # Add timestamp
            analysis['timestamp'] = current_time.isoformat()
            analysis['symbol'] = symbol

            # Log the analysis
            logger.info(f"AI Analysis for {symbol}:")
            logger.info(f"Decision: {analysis['decision']}")
            logger.info(f"Confidence: {analysis['confidence']}")
            logger.info(f"Reasoning: {analysis['reasoning']}")

            # Publish analysis to Redis
            self.redis.publish('trading_signals', json.dumps(analysis))

            # Update last analysis time
            self.last_analysis_time[symbol] = current_time

        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}")

    async def process_market_updates(self):
        """Process market updates from Redis"""
        try:
            pubsub = self.redis.pubsub()
            await pubsub.subscribe('market_updates')

            while self.running:
                message = await pubsub.get_message()
                if message and message['type'] == 'message':
                    market_update = json.loads(message['data'])
                    await self.analyze_market_data(market_update)

                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error processing market updates: {str(e)}")
        finally:
            await pubsub.unsubscribe()

    async def start(self):
        """Start the AI analyzer service"""
        try:
            logger.info("Starting AI Analyzer Service...")
            await self.process_market_updates()
        except Exception as e:
            logger.error(f"Error in AI Analyzer Service: {str(e)}")
        finally:
            self.stop()

    def stop(self):
        """Stop the AI analyzer service"""
        logger.info("Stopping AI Analyzer Service...")
        self.running = False
        self.redis.close()

if __name__ == "__main__":
    service = AIAnalyzerService()
    try:
        asyncio.run(service.start())
    except KeyboardInterrupt:
        service.stop()
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        service.stop()
