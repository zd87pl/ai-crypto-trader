import os
import json
import socket
import asyncio
from datetime import datetime, timedelta
import logging as logger
from logging.handlers import RotatingFileHandler
from redis.asyncio import Redis
from redis.exceptions import ConnectionError
from services.utils.news_analyzer import NewsAnalyzer

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure rotating file handler
rotating_handler = RotatingFileHandler(
    'logs/news_analysis.log',
    maxBytes=10*1024*1024,  # 10MB per file
    backupCount=5,  # Keep 5 backup files
    encoding='utf-8'
)

# Configure logging with rotation
logger.basicConfig(
    level=logger.DEBUG,
    format='%(asctime)s - %(levelname)s - [NewsAnalysis] %(message)s',
    handlers=[
        rotating_handler,
        logger.StreamHandler()
    ]
)

class NewsAnalysisService:
    """Service for analyzing cryptocurrency news using NLP techniques"""
    
    def __init__(self):
        """Initialize the News Analysis Service"""
        logger.debug("Initializing News Analysis Service...")
        
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Initialize news analyzer
        self.news_analyzer = NewsAnalyzer(self.config)
        
        # Redis configuration
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis = None
        
        # Service state
        self.running = True
        self.monitored_symbols = set()
        self.last_update = {}
        
        # Configure update intervals
        news_config = self.config.get('news_analysis', {})
        self.update_interval = news_config.get('update_interval', 1800)  # 30 minutes
        self.retry_interval = news_config.get('retry_interval', 300)  # 5 minutes
        self.batch_size = news_config.get('batch_size', 3)
        self.use_transformers = news_config.get('use_transformers', True)
        
        # Get service port from environment variable
        self.service_port = int(os.getenv('NEWS_ANALYSIS_PORT', 8006))
        logger.debug(f"Service port configured as: {self.service_port}")
        logger.debug("News Analysis Service initialization complete")
    
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
    
    async def process_market_updates(self):
        """Process market updates to track active trading symbols"""
        while self.running:
            try:
                if not self.redis or not await self.redis.ping():
                    if not await self.connect_redis():
                        await asyncio.sleep(5)
                        continue
                
                # Subscribe to market updates to track active symbols
                pubsub = self.redis.pubsub()
                await pubsub.subscribe('market_updates')
                
                while self.running:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if message:
                        try:
                            market_update = json.loads(message['data'])
                            symbol = market_update['symbol']
                            
                            # Add symbol to monitored list
                            self.monitored_symbols.add(symbol)
                            logger.debug(f"Added {symbol} to monitored symbols, total: {len(self.monitored_symbols)}")
                            
                        except Exception as e:
                            logger.error(f"Error processing market update: {str(e)}")
                    
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in process_market_updates: {str(e)}")
                await asyncio.sleep(5)
    
    async def analyze_news_for_symbol(self, symbol: str, force_update: bool = False):
        """
        Analyze news for a specific symbol
        
        Args:
            symbol: Cryptocurrency symbol
            force_update: Force update even if recent analysis exists
        """
        try:
            current_time = datetime.now()
            
            # Check if we need to update
            if not force_update and symbol in self.last_update:
                time_since_last = (current_time - self.last_update[symbol]).total_seconds()
                if time_since_last < self.update_interval:
                    logger.debug(f"Skipping news analysis for {symbol}, last update {time_since_last:.0f}s ago")
                    return
            
            logger.info(f"Analyzing news for {symbol}")
            
            # Analyze news
            news_analysis = await self.news_analyzer.analyze_news(symbol, force_update)
            
            if news_analysis['status'] != 'success' and news_analysis['status'] != 'no_news':
                logger.warning(f"News analysis failed for {symbol}: {news_analysis.get('error', 'Unknown error')}")
                return
            
            # Save analysis timestamp
            self.last_update[symbol] = current_time
            
            # Create deep copy before modifications
            analysis_to_publish = news_analysis.copy()
            
            # Clean up large fields for publishing
            if 'news_items' in analysis_to_publish:
                # Keep only essential fields for each news item
                cleaned_items = []
                for item in analysis_to_publish['news_items'][:5]:  # Only include top 5 items
                    cleaned_item = {
                        'title': item.get('title', ''),
                        'source': item.get('source', ''),
                        'url': item.get('url', ''),
                        'published_at': item.get('published_at', ''),
                        'sentiment': item.get('sentiment_analysis', {}).get('sentiment', 'neutral'),
                        'relevance': item.get('relevance', 0.0)
                    }
                    
                    # Include summary if available
                    if 'summary' in item:
                        cleaned_item['summary'] = item['summary']
                    
                    cleaned_items.append(cleaned_item)
                
                analysis_to_publish['news_items'] = cleaned_items
            
            # Publish analysis to Redis
            await self.redis.publish(
                'news_analysis_updates',
                json.dumps({
                    'symbol': symbol,
                    'analysis': analysis_to_publish
                })
            )
            
            # Store latest analysis
            await self.redis.hset(
                'news_analysis',
                symbol,
                json.dumps(analysis_to_publish)
            )
            
            # Log success
            sentiment = news_analysis.get('sentiment', {}).get('overall', 'neutral')
            topics = news_analysis.get('topics', [])
            topic_str = ', '.join(topics[:3]) if topics else 'none'
            
            logger.info(f"Published news analysis for {symbol}: Sentiment={sentiment}, Topics={topic_str}")
            
        except Exception as e:
            logger.error(f"Error analyzing news for {symbol}: {str(e)}")
    
    async def update_news_analysis(self):
        """Update news analysis for monitored symbols"""
        while self.running:
            try:
                if not self.redis or not await self.redis.ping():
                    if not await self.connect_redis():
                        await asyncio.sleep(5)
                        continue
                
                # Process symbols in batches
                symbols = list(self.monitored_symbols)
                if not symbols:
                    await asyncio.sleep(10)  # Wait for symbols to be added
                    continue
                
                # Process in batches to avoid overloading resources
                for i in range(0, len(symbols), self.batch_size):
                    batch = symbols[i:i+self.batch_size]
                    
                    # Process batch in parallel
                    tasks = [self.analyze_news_for_symbol(symbol) for symbol in batch]
                    await asyncio.gather(*tasks)
                    
                    # Short delay between batches
                    await asyncio.sleep(1)
                
                # Wait before next full update cycle
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in update_news_analysis: {str(e)}")
                await asyncio.sleep(5)
    
    async def generate_market_news_summary(self):
        """Generate and publish a summary of market-wide news"""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                if not self.redis or not await self.redis.ping():
                    continue
                
                # Get all recent news analyses
                all_analyses = {}
                for symbol in self.monitored_symbols:
                    analysis_json = await self.redis.hget('news_analysis', symbol)
                    if analysis_json:
                        try:
                            analysis = json.loads(analysis_json)
                            # Only include recent analyses (last 6 hours)
                            analysis_time = datetime.fromisoformat(analysis['timestamp'])
                            if datetime.now() - analysis_time < timedelta(hours=6):
                                all_analyses[symbol] = analysis
                        except Exception as e:
                            logger.error(f"Error parsing analysis for {symbol}: {str(e)}")
                
                if not all_analyses:
                    logger.debug("No recent news analyses available for summary")
                    continue
                
                # Aggregate sentiment
                sentiment_counts = {
                    'very_positive': 0,
                    'positive': 0,
                    'neutral': 0,
                    'negative': 0,
                    'very_negative': 0
                }
                
                for symbol, analysis in all_analyses.items():
                    sentiment = analysis.get('sentiment', {}).get('overall', 'neutral')
                    if sentiment in sentiment_counts:
                        sentiment_counts[sentiment] += 1
                
                # Determine overall market sentiment
                total = sum(sentiment_counts.values())
                if total == 0:
                    overall_sentiment = 'neutral'
                else:
                    positive_ratio = (sentiment_counts['very_positive'] + sentiment_counts['positive']) / total
                    negative_ratio = (sentiment_counts['very_negative'] + sentiment_counts['negative']) / total
                    
                    if positive_ratio > 0.6:
                        overall_sentiment = 'very_positive'
                    elif positive_ratio > 0.4:
                        overall_sentiment = 'positive'
                    elif negative_ratio > 0.6:
                        overall_sentiment = 'very_negative'
                    elif negative_ratio > 0.4:
                        overall_sentiment = 'negative'
                    else:
                        overall_sentiment = 'neutral'
                
                # Get top topics across all assets
                topic_counts = {}
                for symbol, analysis in all_analyses.items():
                    for topic in analysis.get('topics', []):
                        if topic in topic_counts:
                            topic_counts[topic] += 1
                        else:
                            topic_counts[topic] = 1
                
                # Get top 10 topics
                top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                
                # Collect trending news
                trending_news = []
                for symbol, analysis in all_analyses.items():
                    for item in analysis.get('news_items', [])[:2]:  # Top 2 news per symbol
                        trending_news.append({
                            'symbol': symbol,
                            'title': item.get('title', ''),
                            'source': item.get('source', ''),
                            'sentiment': item.get('sentiment', 'neutral'),
                            'url': item.get('url', '')
                        })
                
                # Limit to top 10 trending news
                trending_news = sorted(trending_news, key=lambda x: x.get('relevance', 0), reverse=True)[:10]
                
                # Prepare market summary
                market_summary = {
                    'timestamp': datetime.now().isoformat(),
                    'overall_sentiment': overall_sentiment,
                    'sentiment_distribution': sentiment_counts,
                    'top_topics': [topic for topic, count in top_topics],
                    'trending_news': trending_news,
                    'assets_analyzed': len(all_analyses)
                }
                
                # Publish summary
                await self.redis.set('news_market_summary', json.dumps(market_summary))
                logger.info(f"Published market news summary, sentiment: {overall_sentiment}")
                
            except Exception as e:
                logger.error(f"Error generating market news summary: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def maintain_redis(self):
        """Maintain Redis connection"""
        while self.running:
            try:
                if self.redis:
                    await self.redis.ping()
                else:
                    await self.connect_redis()
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Redis connection error: {str(e)}")
                self.redis = None
                await asyncio.sleep(5)
    
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
        """Run the news analysis service"""
        try:
            logger.info("Starting News Analysis Service...")
            
            # First establish Redis connection
            if not await self.connect_redis():
                raise Exception("Failed to establish initial Redis connection")
            
            # Create tasks
            tasks = [
                asyncio.create_task(self.process_market_updates()),
                asyncio.create_task(self.update_news_analysis()),
                asyncio.create_task(self.generate_market_news_summary()),
                asyncio.create_task(self.maintain_redis()),
                asyncio.create_task(self.health_check_server())
            ]
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Error in News Analysis Service: {str(e)}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the news analysis service"""
        logger.info("Stopping News Analysis Service...")
        self.running = False
        if self.redis:
            await self.redis.close()

if __name__ == "__main__":
    service = NewsAnalysisService()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        asyncio.run(service.stop())
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        asyncio.run(service.stop())