import os
import json
import socket
import asyncio
from datetime import datetime
import logging as logger
from logging.handlers import RotatingFileHandler
from openai import AsyncOpenAI
from redis.asyncio import Redis
from redis.exceptions import ConnectionError
from ai_trader import AITrader

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure rotating file handler
rotating_handler = RotatingFileHandler(
    'logs/ai_analyzer.log',
    maxBytes=10*1024*1024,  # 10MB per file
    backupCount=5,  # Keep 5 backup files
    encoding='utf-8'
)

# Configure logging with rotation
logger.basicConfig(
    level=logger.DEBUG,
    format='%(asctime)s - %(levelname)s - [AIAnalyzer] %(message)s',
    handlers=[
        rotating_handler,
        logger.StreamHandler()
    ]
)

class AIAnalyzerService:
    def __init__(self):
        logger.debug("Initializing AI Analyzer Service...")
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        logger.debug(f"Loaded configuration: {json.dumps(self.config, indent=2)}")

        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = AsyncOpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI client with model: {self.config['openai']['model']}")

        # Initialize AITrader
        logger.debug("Initializing AITrader...")
        self.ai_trader = AITrader(self.config)
        logger.debug("AITrader initialized successfully")

        # Redis configuration
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        logger.debug(f"Redis configuration - Host: {self.redis_host}, Port: {self.redis_port}")

        # Redis will be initialized in connect_redis
        self.redis = None
        self.pubsub = None
        self.running = True
        self.market_data = {}
        self.social_data = {}
        self.last_analysis_time = {}
        
        # Get service port from environment variable
        self.service_port = int(os.getenv('SERVICE_PORT', 8003))
        logger.debug(f"Service port configured as: {self.service_port}")
        logger.debug("AI Analyzer Service initialization complete")

    async def connect_redis(self, max_retries=10, retry_delay=5):
        """Establish Redis connection with retries"""
        retries = 0
        while retries < max_retries and self.running:
            try:
                if self.redis is None:
                    logger.debug(f"Attempting Redis connection (attempt {retries + 1}/{max_retries})")
                    self.redis = Redis(
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
        """Set up Redis pubsub connection"""
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
            logger.debug("Subscribing to market_updates and social_updates channels")
            await self.pubsub.subscribe('market_updates', 'social_updates')
            
            # Get first messages to confirm subscriptions
            logger.debug("Waiting for subscription confirmation messages")
            subscribed_count = 0
            while subscribed_count < 2:  # Wait for both subscriptions
                message = await self.pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'subscribe':
                    subscribed_count += 1
            
            if subscribed_count == 2:
                logger.info("Successfully subscribed to all channels")
                return True
            else:
                logger.error("Failed to subscribe to all channels")
                return False

        except Exception as e:
            logger.error(f"Error in pubsub setup: {str(e)}", exc_info=True)
            if self.pubsub:
                await self.pubsub.close()
                self.pubsub = None
            return False

    def get_market_context(self, symbol: str) -> str:
        """Generate market context description"""
        try:
            # Get overall market sentiment
            market_sentiment = "neutral"  # Default
            if symbol in self.market_data:
                data = self.market_data[symbol]
                
                # Determine sentiment based on multiple factors
                bullish_signals = 0
                bearish_signals = 0
                
                # Technical indicators
                if data['rsi'] > 70: bearish_signals += 1
                elif data['rsi'] < 30: bullish_signals += 1
                
                if data['macd'] > 0: bullish_signals += 1
                elif data['macd'] < 0: bearish_signals += 1
                
                if data['williams_r'] > -20: bearish_signals += 1
                elif data['williams_r'] < -80: bullish_signals += 1
                
                # Price action
                for timeframe in ['1m', '3m', '5m']:
                    change_key = f'price_change_{timeframe}'
                    if data[change_key] > 0: bullish_signals += 1
                    elif data[change_key] < 0: bearish_signals += 1
                
                # Check for combined indicators if available
                if 'combined_indicators' in data:
                    combined = data['combined_indicators']
                    
                    # Trend confirmation indicator
                    if 'trend_confirmation' in combined:
                        trend_conf = combined['trend_confirmation']
                        if trend_conf > 0.5: bullish_signals += 1
                        elif trend_conf < -0.5: bearish_signals += 1
                    
                    # Oscillator consensus
                    if 'oscillator_consensus' in combined:
                        consensus = combined['oscillator_consensus']
                        if consensus['signal'] == 'oversold' and consensus['agreement'] > 0.5: 
                            bullish_signals += 1
                        elif consensus['signal'] == 'overbought' and consensus['agreement'] > 0.5:
                            bearish_signals += 1
                    
                    # Market regime
                    if 'market_regime_indicator' in combined:
                        regime = combined['market_regime_indicator']
                        # Add context about market regime but don't affect signals directly
                        
                    # Reversal probability
                    if 'reversal_probability' in combined:
                        reversal = combined['reversal_probability']
                        if reversal['probability'] > 0.7:
                            # High probability of reversal
                            # Counter the current trend
                            if data['trend'] == 'uptrend':
                                bearish_signals += 1
                            elif data['trend'] == 'downtrend':
                                bullish_signals += 1
                
                # Check for volume profile signals if available
                if 'volume_profile' in data:
                    vp = data['volume_profile']
                    
                    # Add signals based on volume profile
                    if vp['signal'] == 'potential_resistance':
                        bearish_signals += 1
                    elif vp['signal'] == 'potential_support':
                        bullish_signals += 1
                    elif vp['signal'] == 'overbought':
                        bearish_signals += 1
                    elif vp['signal'] == 'oversold':
                        bullish_signals += 1
                    
                    # Volume pressure signals
                    if 'volume_pressure' in vp:
                        if vp['volume_pressure'] in ['strong_buying', 'moderate_buying']:
                            bullish_signals += 1
                        elif vp['volume_pressure'] in ['strong_selling', 'moderate_selling']:
                            bearish_signals += 1
                    
                    # Volume divergence signals
                    if 'volume_divergence' in vp:
                        if vp['volume_divergence'] == 'bullish_divergence':
                            bullish_signals += 1
                        elif vp['volume_divergence'] == 'bearish_divergence':
                            bearish_signals += 1
                    
                    # Volume anomalies
                    if vp.get('volume_anomalies', False) and vp.get('recent_anomaly_percentage', 0) > 20:
                        # High volume anomalies often precede significant price movements
                        # Add a signal in the direction of the current trend
                        if data['trend'] == 'uptrend':
                            bullish_signals += 1
                        elif data['trend'] == 'downtrend':
                            bearish_signals += 1
                
                # Determine overall sentiment
                if bullish_signals > bearish_signals + 2:
                    market_sentiment = "strongly bullish"
                elif bullish_signals > bearish_signals:
                    market_sentiment = "moderately bullish"
                elif bearish_signals > bullish_signals + 2:
                    market_sentiment = "strongly bearish"
                elif bearish_signals > bullish_signals:
                    market_sentiment = "moderately bearish"
                
            # Get market regime information (if available)
            market_regime_info = ""
            if symbol in self.market_data and 'combined_indicators' in self.market_data[symbol]:
                combined = self.market_data[symbol]['combined_indicators']
                
                # Market regime context
                if 'market_regime_indicator' in combined:
                    regime = combined['market_regime_indicator']
                    if regime['confidence'] > 0.6:
                        market_regime_info = f"Market is in a {regime['regime']} regime. "
                
                # Add reversal probability if high
                if 'reversal_probability' in combined:
                    reversal = combined['reversal_probability']
                    if reversal['probability'] > 0.7:
                        direction = "downward" if self.market_data[symbol]['trend'] == 'uptrend' else "upward"
                        market_regime_info += f"High probability ({int(reversal['probability']*100)}%) of {direction} reversal. "
                
                # Add breakout information if detected
                if 'breakout_confirmation' in combined:
                    breakout = combined['breakout_confirmation']
                    if breakout['direction'] != 0 and breakout['confirmation'] > 0.6:
                        direction = "upward" if breakout['direction'] > 0 else "downward"
                        market_regime_info += f"Potential {direction} breakout detected. "
            
            # Add volume profile information if available
            volume_profile_info = ""
            if symbol in self.market_data and 'volume_profile' in self.market_data[symbol]:
                vp = self.market_data[symbol]['volume_profile']
                
                # Add summary if available
                if 'summary' in vp:
                    volume_profile_info = vp['summary'] + " "
                    
                # Or build our own summary based on key metrics
                else:
                    # POC and value area
                    current_price = self.market_data[symbol]['current_price']
                    
                    if 'poc' in vp and 'value_area_high' in vp and 'value_area_low' in vp:
                        poc = vp['poc']
                        vah = vp['value_area_high']
                        val = vp['value_area_low']
                        
                        # Check price relation to value area
                        if current_price > vah:
                            volume_profile_info += "Price is above the value area. "
                        elif current_price < val:
                            volume_profile_info += "Price is below the value area. "
                        else:
                            volume_profile_info += "Price is inside the value area. "
                        
                        # Add POC relation
                        if abs(current_price - poc) / current_price < 0.005:  # Within 0.5%
                            volume_profile_info += "Price is at the point of control (highest volume level). "
                    
                    # Add volume pressure if available
                    if 'volume_pressure' in vp:
                        if vp['volume_pressure'] == 'strong_buying':
                            volume_profile_info += "Strong buying pressure detected in volume analysis. "
                        elif vp['volume_pressure'] == 'strong_selling':
                            volume_profile_info += "Strong selling pressure detected in volume analysis. "
                    
                    # Add volume divergence
                    if 'volume_divergence' in vp:
                        if vp['volume_divergence'] == 'bullish_divergence':
                            volume_profile_info += "Bullish price-volume divergence detected. "
                        elif vp['volume_divergence'] == 'bearish_divergence':
                            volume_profile_info += "Bearish price-volume divergence detected. "
                    
                    # Add volume anomalies info
                    if vp.get('volume_anomalies', False) and vp.get('recent_anomaly_percentage', 0) > 15:
                        volume_profile_info += f"Unusual volume activity detected ({vp.get('recent_anomaly_percentage', 0):.0f}% anomalies). "
            
            # Get social context
            social_context = "No significant social activity"
            if symbol in self.social_data:
                social = self.social_data[symbol]
                if social['metrics']['social_sentiment'] > 0.6:
                    social_context = "Very positive social sentiment"
                elif social['metrics']['social_sentiment'] > 0.5:
                    social_context = "Positive social sentiment"
                elif social['metrics']['social_sentiment'] < 0.4:
                    social_context = "Negative social sentiment"
                elif social['metrics']['social_sentiment'] < 0.3:
                    social_context = "Very negative social sentiment"
                
                # Add engagement context
                if social['metrics']['social_engagement'] > self.config['lunarcrush']['min_engagement']:
                    social_context += f" with high engagement ({social['metrics']['social_engagement']} interactions)"
            
            # Get news analysis context
            news_context = ""
            if symbol in self.market_data:
                data = self.market_data[symbol]
                if 'social_metrics' in data and 'news_analysis' in data['social_metrics']:
                    news_analysis = data['social_metrics']['news_analysis']
                    
                    # Add sentiment from advanced news analysis
                    news_sentiment = news_analysis.get('sentiment', 'neutral')
                    if news_sentiment == 'very_positive':
                        news_context = "News sentiment is very positive."
                    elif news_sentiment == 'positive':
                        news_context = "News sentiment is positive."
                    elif news_sentiment == 'negative':
                        news_context = "News sentiment is negative."
                    elif news_sentiment == 'very_negative':
                        news_context = "News sentiment is very negative."
                    
                    # Add topics if available
                    topics = news_analysis.get('topics', [])
                    if topics:
                        news_context += f" Key topics in the news: {', '.join(topics[:3])}."
            
            return f"Current market sentiment appears {market_sentiment}. {market_regime_info}{volume_profile_info}{social_context}. {news_context}"
            
        except Exception as e:
            logger.error(f"Error generating market context: {str(e)}")
            return "Market context unavailable"

    async def analyze_market_data(self, market_update):
        """Analyze market data using AITrader"""
        try:
            symbol = market_update['symbol']
            current_time = datetime.now()

            # Check if we need to analyze this symbol
            if symbol in self.last_analysis_time:
                time_since_last = (current_time - self.last_analysis_time[symbol]).seconds
                if time_since_last < self.config['trading_params']['ai_analysis_interval']:
                    logger.debug(f"Skipping analysis for {symbol}, last analysis was {time_since_last}s ago")
                    return

            # Update market data cache
            self.market_data[symbol] = market_update

            # Get social data if available
            social_metrics = {
                'social_volume': 0,
                'social_engagement': 0,
                'social_contributors': 0,
                'social_sentiment': 0.5,
                'recent_news': "No recent news available"
            }
            
            if symbol in self.social_data:
                social = self.social_data[symbol]
                social_metrics.update({
                    'social_volume': social['metrics']['social_volume'],
                    'social_engagement': social['metrics']['social_engagement'],
                    'social_contributors': social['metrics']['social_contributors'],
                    'social_sentiment': social['metrics']['social_sentiment']
                })
                
                # Format recent news
                if social['recent_news']:
                    news_summary = []
                    for news in social['recent_news'][:3]:  # Top 3 news items
                        sentiment = "neutral"
                        if news['sentiment'] > 0.6: sentiment = "positive"
                        elif news['sentiment'] < 0.4: sentiment = "negative"
                        news_summary.append(f"- {news['title']} ({sentiment} sentiment)")
                    social_metrics['recent_news'] = "\n".join(news_summary)
                
            # Get advanced news analysis if available
            news_analysis_data = {}
            try:
                news_analysis_json = await self.redis.hget('news_analysis', symbol)
                if news_analysis_json:
                    news_analysis = json.loads(news_analysis_json)
                    
                    # Check if analysis is recent (less than 6 hours old)
                    analysis_time = datetime.fromisoformat(news_analysis.get('timestamp', '2000-01-01T00:00:00'))
                    if (datetime.now() - analysis_time).total_seconds() < 21600:  # 6 hours
                        # Extract key data
                        news_sentiment = news_analysis.get('sentiment', {}).get('overall', 'neutral')
                        sentiment_score = news_analysis.get('sentiment', {}).get('score', 0.0)
                        
                        # Convert sentiment to numeric score (-1 to 1)
                        sentiment_numeric = 0.0
                        if news_sentiment == 'very_positive': sentiment_numeric = 0.9
                        elif news_sentiment == 'positive': sentiment_numeric = 0.5
                        elif news_sentiment == 'neutral': sentiment_numeric = 0.0
                        elif news_sentiment == 'negative': sentiment_numeric = -0.5
                        elif news_sentiment == 'very_negative': sentiment_numeric = -0.9
                        
                        # Create news analysis summary
                        news_info = []
                        
                        # Add topics if available
                        topics = news_analysis.get('topics', [])
                        if topics:
                            news_info.append(f"Key topics: {', '.join(topics[:5])}")
                        
                        # Add sentiment summary
                        news_info.append(f"News sentiment: {news_sentiment.replace('_', ' ')} ({sentiment_score:.2f})")
                        
                        # Add entity summary if available
                        entities = news_analysis.get('entities', [])
                        if entities:
                            entity_info = []
                            for entity in entities[:3]:
                                if 'text' in entity and 'type' in entity:
                                    entity_info.append(f"{entity['text']} ({entity['type']})")
                            if entity_info:
                                news_info.append(f"Key entities: {', '.join(entity_info)}")
                        
                        # Add headlines
                        headlines = []
                        for item in news_analysis.get('news_items', [])[:3]:
                            if 'title' in item and 'sentiment' in item:
                                headlines.append(f"- {item['title']} ({item['sentiment']})")
                        
                        if headlines:
                            news_info.append("Recent headlines:")
                            news_info.extend(headlines)
                        
                        # Update social metrics with news analysis
                        social_metrics['news_analysis'] = {
                            'sentiment': news_sentiment,
                            'sentiment_score': sentiment_score,
                            'sentiment_numeric': sentiment_numeric,
                            'topics': topics[:5] if topics else [],
                            'entities': entities[:5] if entities else []
                        }
                        
                        # Update recent news with more detailed info
                        if news_info:
                            social_metrics['recent_news'] = "\n".join(news_info)
            except Exception as e:
                logger.error(f"Error processing news analysis for {symbol}: {str(e)}")

            # Generate market context
            market_context = self.get_market_context(symbol)
            
            # Extract key combined indicators if available
            combined_indicators_data = {}
            if 'combined_indicators' in market_update:
                combined = market_update['combined_indicators']
                
                # Select most relevant combined indicators for AI analysis
                key_indicators = [
                    'trend_confirmation',
                    'oscillator_consensus',
                    'market_regime_indicator',
                    'reversal_probability',
                    'trend_strength_index',
                    'breakout_confirmation'
                ]
                
                for indicator in key_indicators:
                    if indicator in combined:
                        combined_indicators_data[indicator] = combined[indicator]
            
            # Extract volume profile data if available
            volume_profile_data = {}
            if 'volume_profile' in market_update:
                vp = market_update['volume_profile']
                
                # Include all volume profile metrics
                volume_profile_data = vp
                
                # Add description to explain what each value means
                volume_profile_data['description'] = {
                    'poc': 'Point of Control - price level with highest trading volume',
                    'value_area_high': 'Upper bound of the Value Area (70% of volume)',
                    'value_area_low': 'Lower bound of the Value Area (70% of volume)',
                    'signal': 'Trading signal derived from volume profile analysis',
                }
                
                # If we have volume pressure info, add it to the description
                if 'volume_pressure' in vp:
                    volume_profile_data['description']['volume_pressure'] = 'Direction of volume pressure (buying vs selling)'
                
                # If we have volume divergence, add it to the description
                if 'volume_divergence' in vp:
                    volume_profile_data['description']['volume_divergence'] = 'Divergence between price and volume movement'
                
                # Create a summary of combined indicators for the prompt
                combined_indicators_summary = []
                
                # Add trend confirmation
                if 'trend_confirmation' in combined:
                    conf = combined['trend_confirmation']
                    if conf > 0.5:
                        combined_indicators_summary.append(f"Strong uptrend confirmation ({conf:.2f})")
                    elif conf < -0.5:
                        combined_indicators_summary.append(f"Strong downtrend confirmation ({abs(conf):.2f})")
                
                # Add oscillator consensus
                if 'oscillator_consensus' in combined:
                    consensus = combined['oscillator_consensus']
                    if consensus['signal'] != 'neutral' and consensus['strength'] > 0.5:
                        combined_indicators_summary.append(
                            f"Oscillator consensus: {consensus['signal']} (strength: {consensus['strength']:.2f}, agreement: {consensus['agreement']:.2f})"
                        )
                
                # Add market regime
                if 'market_regime_indicator' in combined:
                    regime = combined['market_regime_indicator']
                    if regime['confidence'] > 0.6:
                        combined_indicators_summary.append(
                            f"Market regime: {regime['regime']} (confidence: {regime['confidence']:.2f})"
                        )
                
                # Add reversal probability
                if 'reversal_probability' in combined:
                    reversal = combined['reversal_probability']
                    if reversal['probability'] > 0.5:
                        signals_str = ", ".join(reversal['signals'][:3])  # Limit to top 3 signals
                        combined_indicators_summary.append(
                            f"Reversal probability: {reversal['probability']:.2f} [{signals_str}]"
                        )
                
                # Add trend strength
                if 'trend_strength_index' in combined:
                    tsi = combined['trend_strength_index']
                    direction_text = "bullish" if tsi['direction'] > 0 else ("bearish" if tsi['direction'] < 0 else "neutral")
                    combined_indicators_summary.append(
                        f"Trend strength: {tsi['strength']:.2f} ({direction_text}, confidence: {tsi['confidence']:.2f})"
                    )
                
                # Add breakout confirmation
                if 'breakout_confirmation' in combined:
                    breakout = combined['breakout_confirmation']
                    if breakout['direction'] != 0 and breakout['confirmation'] > 0.5:
                        combined_indicators_summary.append(
                            f"Breakout: {breakout['status']} (confirmation: {breakout['confirmation']:.2f})"
                        )
                
                # Add summary to analysis data
                if combined_indicators_summary:
                    combined_indicators_data['summary'] = "\n".join(combined_indicators_summary)

            # Combine data for analysis
            analysis_data = {
                **market_update, 
                **social_metrics, 
                'market_context': market_context,
                'combined_indicators': combined_indicators_data,
                'volume_profile': volume_profile_data
            }

            logger.info(f"Starting analysis for {symbol}")
            logger.debug(f"Analysis data: {json.dumps(analysis_data, indent=2)}")

            # Use AITrader to analyze the trading opportunity
            logger.debug("Calling AITrader.analyze_trade_opportunity...")
            analysis = await self.ai_trader.analyze_trade_opportunity(analysis_data)
            logger.debug(f"Received analysis from AITrader: {json.dumps(analysis, indent=2)}")

            # Add metadata
            analysis['timestamp'] = current_time.isoformat()
            analysis['symbol'] = symbol
            analysis['market_data'] = analysis_data

            # Log the analysis
            logger.info(f"AI Analysis for {symbol}:")
            logger.info(f"Decision: {analysis['decision']}")
            logger.info(f"Confidence: {analysis['confidence']}")
            logger.info(f"Reasoning: {analysis['reasoning']}")

            # Publish analysis to Redis
            if self.redis and await self.redis.ping():
                logger.info(f"Publishing trading signal for {symbol}")
                await self.redis.publish('trading_signals', json.dumps(analysis))
                logger.info(f"Published trading signal for {symbol}: {json.dumps(analysis)}")
            else:
                logger.error("Redis connection lost during analysis publishing")
                await self.connect_redis()

            # Update last analysis time
            self.last_analysis_time[symbol] = current_time

        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}", exc_info=True)

    async def process_updates(self):
        """Process market and social updates from Redis"""
        logger.debug("Starting updates processing...")
        pubsub_retry_count = 0
        max_pubsub_retries = 10
        
        while self.running:
            try:
                # Ensure Redis connection
                if not self.redis or not await self.redis.ping():
                    logger.debug("Redis connection not available, attempting to connect...")
                    if not await self.connect_redis():
                        await asyncio.sleep(5)
                        continue

                # Setup pubsub if needed
                if not self.pubsub:
                    logger.debug(f"Setting up pubsub (attempt {pubsub_retry_count + 1}/{max_pubsub_retries})")
                    if await self.setup_pubsub():
                        logger.info("Pubsub setup successful")
                        pubsub_retry_count = 0  # Reset counter on success
                    else:
                        pubsub_retry_count += 1
                        if pubsub_retry_count >= max_pubsub_retries:
                            logger.error("Max pubsub retry attempts reached")
                            raise Exception("Failed to set up pubsub after maximum retries")
                        await asyncio.sleep(5)
                        continue

                # Process messages
                try:
                    logger.debug("Waiting for message from Redis...")
                    message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    
                    if message:
                        logger.debug(f"Received message type: {message['type']}")
                        if message['type'] == 'message':
                            try:
                                logger.debug(f"Raw message data: {message['data']}")
                                data = json.loads(message['data'])
                                
                                # Handle different update types
                                if message['channel'] == 'market_updates':
                                    logger.info(f"Processing market update for {data['symbol']}")
                                    await self.analyze_market_data(data)
                                elif message['channel'] == 'social_updates':
                                    logger.info(f"Processing social update for {data['symbol']}")
                                    self.social_data[data['symbol']] = data['data']
                                    # Trigger reanalysis with new social data
                                    if data['symbol'] in self.market_data:
                                        await self.analyze_market_data(self.market_data[data['symbol']])
                                
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse update: {e}")
                                logger.error(f"Invalid JSON data: {message['data']}")
                            except KeyError as e:
                                logger.error(f"Missing required field in update: {e}")
                                logger.error(f"Update data: {data}")
                            except Exception as e:
                                logger.error(f"Error processing update: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error getting message from pubsub: {str(e)}")
                    self.pubsub = None  # Force pubsub reconnection
                    continue

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in process_updates: {str(e)}", exc_info=True)
                if self.pubsub:
                    await self.pubsub.close()
                    self.pubsub = None
                await asyncio.sleep(5)

    async def maintain_redis(self):
        """Maintain Redis connection"""
        logger.debug("Starting Redis connection maintenance...")
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

    async def health_check_server(self):
        """Run a simple TCP server for health checks"""
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('0.0.0.0', self.service_port))
            server.listen(1)
            server.setblocking(False)
            
            logger.info(f"Health check server listening on port {self.service_port}")
            
            while self.running:
                try:
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Health check server loop error: {str(e)}")
                    break
                    
        except Exception as e:
            logger.error(f"Failed to start health check server: {str(e)}")
            raise  # Re-raise the exception to trigger service restart
        finally:
            try:
                server.close()
            except Exception:
                pass

    async def run(self):
        """Run the AI analyzer service"""
        try:
            logger.info("Starting AI Analyzer Service...")
            
            # First establish Redis connection with increased retries
            if not await self.connect_redis(max_retries=15, retry_delay=2):
                raise Exception("Failed to establish initial Redis connection")
            
            # Create tasks for updates processing, Redis maintenance, and health check
            tasks = [
                asyncio.create_task(self.process_updates()),
                asyncio.create_task(self.maintain_redis()),
                asyncio.create_task(self.health_check_server())
            ]
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Error in AI Analyzer Service: {str(e)}")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the AI analyzer service"""
        logger.info("Stopping AI Analyzer Service...")
        self.running = False
        if self.pubsub:
            await self.pubsub.close()
        if self.redis:
            await self.redis.close()

if __name__ == "__main__":
    service = AIAnalyzerService()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        asyncio.run(service.stop())
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        asyncio.run(service.stop())
