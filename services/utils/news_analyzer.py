import os
import json
import aiohttp
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [NewsAnalyzer] %(message)s',
    handlers=[
        logging.FileHandler('logs/news_analyzer.log'),
        logging.StreamHandler()
    ]
)

# Initialize NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class NewsAnalyzer:
    """
    Class for analyzing crypto news using natural language processing.
    
    Features:
    1. News sentiment analysis (positive/negative/neutral)
    2. Entity recognition for crypto assets
    3. Topic extraction
    4. Impact assessment
    5. News clustering and summarization
    6. Relevant news filtering
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the NewsAnalyzer
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Get news analysis config
        news_config = self.config.get('news_analysis', {})
        
        # Initialize NLP components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vader = SentimentIntensityAnalyzer()
        
        # Set up transformers pipeline for more advanced NLP tasks
        self.use_transformers = news_config.get('use_transformers', True)
        if self.use_transformers:
            try:
                # Load or download sentiment model
                model_name = news_config.get('sentiment_model', 'finiteautomata/bertweet-base-sentiment-analysis')
                self.sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)
                
                # Load or download entity recognition model
                ner_model = news_config.get('ner_model', 'dbmdz/bert-large-cased-finetuned-conll03-english')
                self.ner_analyzer = pipeline("ner", model=ner_model, aggregation_strategy="simple")
                
                # Load or download text summarization model
                summarization_model = news_config.get('summarization_model', 'sshleifer/distilbart-cnn-12-6')
                self.summarizer = pipeline("summarization", model=summarization_model)
                
                self.logger.info("Successfully loaded transformer models")
            except Exception as e:
                self.logger.error(f"Error loading transformer models: {str(e)}")
                self.use_transformers = False
        
        # Crypto keywords for asset detection
        self.crypto_keywords = news_config.get('crypto_keywords', {})
        
        # Add default crypto keywords if none provided
        if not self.crypto_keywords:
            self.crypto_keywords = {
                "BTC": ["bitcoin", "btc", "xbt"],
                "ETH": ["ethereum", "eth", "ether"],
                "BNB": ["binance coin", "bnb", "binance"],
                "XRP": ["ripple", "xrp"],
                "ADA": ["cardano", "ada"],
                "SOL": ["solana", "sol"],
                "DOGE": ["dogecoin", "doge"],
                "DOT": ["polkadot", "dot"],
                "AVAX": ["avalanche", "avax"],
                "MATIC": ["polygon", "matic"],
            }
        
        # News sources for API fetching
        self.news_sources = news_config.get('news_sources', [
            "cryptopanic", "lunarcrush", "coindesk", "cointelegraph"
        ])
        
        # News caching parameters
        self.cache_duration = news_config.get('cache_duration', 1800)  # 30 minutes
        self.max_news_age = news_config.get('max_news_age', 86400)  # 24 hours
        self.min_news_relevance = news_config.get('min_news_relevance', 0.5)
        
        # Sentiment thresholds
        self.sentiment_thresholds = news_config.get('sentiment_thresholds', {
            'positive': 0.5,
            'negative': -0.3,
            'very_positive': 0.7,
            'very_negative': -0.6
        })
        
        # Storage for analyzed news
        self.news_cache = {}
        self.last_update = {}
        
        self.logger.info("NewsAnalyzer initialized successfully")
    
    async def fetch_news(self, symbol: str, sources: List[str] = None) -> List[Dict]:
        """
        Fetch news from various sources for a specific symbol
        
        Args:
            symbol: Cryptocurrency symbol
            sources: List of news sources to fetch from
            
        Returns:
            List of news items
        """
        if sources is None:
            sources = self.news_sources
        
        all_news = []
        
        for source in sources:
            fetcher = getattr(self, f"_fetch_from_{source.lower()}", None)
            if fetcher:
                try:
                    news_items = await fetcher(symbol)
                    if news_items:
                        all_news.extend(news_items)
                except Exception as e:
                    self.logger.error(f"Error fetching news from {source} for {symbol}: {str(e)}")
        
        # Deduplicate news by URL
        unique_news = {}
        for item in all_news:
            if 'url' in item and item['url'] not in unique_news:
                unique_news[item['url']] = item
        
        return list(unique_news.values())
    
    async def _fetch_from_cryptopanic(self, symbol: str) -> List[Dict]:
        """Fetch news from CryptoPanic API"""
        try:
            api_key = os.getenv('CRYPTOPANIC_API_KEY', '')
            if not api_key:
                self.logger.warning("CRYPTOPANIC_API_KEY not set")
                return []
            
            url = "https://cryptopanic.com/api/v1/posts/"
            params = {
                'auth_token': api_key,
                'currencies': symbol.replace('USDC', ''),
                'kind': 'news',
                'public': 'true',
                'filter': 'important'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        news_items = []
                        
                        if 'results' in data:
                            for item in data['results']:
                                news_items.append({
                                    'title': item.get('title', ''),
                                    'url': item.get('url', ''),
                                    'source': 'CryptoPanic',
                                    'published_at': item.get('published_at', ''),
                                    'content': item.get('body', '')
                                })
                        
                        return news_items
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching from CryptoPanic: {str(e)}")
            return []
    
    async def _fetch_from_lunarcrush(self, symbol: str) -> List[Dict]:
        """Fetch news from LunarCrush API"""
        try:
            # Load from config.json
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            api_key = os.getenv('LUNARCRUSH_API_KEY', config.get('lunarcrush', {}).get('api_key', ''))
            if not api_key:
                self.logger.warning("LUNARCRUSH_API_KEY not set")
                return []
            
            base_url = config.get('lunarcrush', {}).get('base_url', 'https://lunarcrush.com/api/v4')
            url = f"{base_url}/feeds"
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Accept': 'application/json'
            }
            
            params = {
                'symbol': symbol.replace('USDC', ''),
                'limit': 10,
                'sources': 'news'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        news_items = []
                        
                        if 'data' in data:
                            for item in data['data']:
                                news_items.append({
                                    'title': item.get('title', ''),
                                    'url': item.get('url', ''),
                                    'source': 'LunarCrush',
                                    'published_at': datetime.fromtimestamp(item.get('time', 0)).isoformat(),
                                    'content': item.get('body', ''),
                                    'sentiment': item.get('sentiment', 0)
                                })
                        
                        return news_items
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching from LunarCrush: {str(e)}")
            return []
    
    async def _fetch_from_coindesk(self, symbol: str) -> List[Dict]:
        """Fetch news from CoinDesk website"""
        try:
            # Extract the ticker without USDC
            ticker = symbol.replace('USDC', '')
            
            # Search CoinDesk for the ticker
            url = f"https://www.coindesk.com/search?s={ticker}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # Extract article titles and URLs using regex
                        # This is a simplified approach - a proper implementation would use BeautifulSoup
                        title_pattern = r'<h4[^>]*class="[^"]*title[^"]*"[^>]*>([^<]+)</h4>'
                        url_pattern = r'<a[^>]*href="([^"]+)"[^>]*>'
                        date_pattern = r'<time[^>]*datetime="([^"]+)"[^>]*>'
                        
                        titles = re.findall(title_pattern, html)
                        urls = re.findall(url_pattern, html)
                        dates = re.findall(date_pattern, html)
                        
                        news_items = []
                        
                        # Use first 5 results
                        for i in range(min(5, len(titles))):
                            if i < len(urls):
                                url = urls[i]
                                if not url.startswith('http'):
                                    url = f"https://www.coindesk.com{url}"
                                
                                published_at = dates[i] if i < len(dates) else datetime.now().isoformat()
                                
                                news_items.append({
                                    'title': titles[i].strip(),
                                    'url': url,
                                    'source': 'CoinDesk',
                                    'published_at': published_at,
                                    'content': ''  # Would need to fetch the article content
                                })
                        
                        return news_items
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching from CoinDesk: {str(e)}")
            return []
    
    async def _fetch_from_cointelegraph(self, symbol: str) -> List[Dict]:
        """Fetch news from CoinTelegraph website"""
        try:
            # Extract the ticker without USDC
            ticker = symbol.replace('USDC', '')
            
            # Search CoinTelegraph for the ticker
            url = f"https://cointelegraph.com/tags/{ticker.lower()}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # Extract article titles and URLs using regex
                        title_pattern = r'<a[^>]*class="[^"]*post-card__title-link[^"]*"[^>]*>([^<]+)</a>'
                        url_pattern = r'<a[^>]*class="[^"]*post-card__title-link[^"]*"[^>]*href="([^"]+)"[^>]*>'
                        date_pattern = r'<time[^>]*datetime="([^"]+)"[^>]*>'
                        
                        titles = re.findall(title_pattern, html)
                        urls = re.findall(url_pattern, html)
                        dates = re.findall(date_pattern, html)
                        
                        news_items = []
                        
                        # Use first 5 results
                        for i in range(min(5, len(titles))):
                            if i < len(urls):
                                url = urls[i]
                                if not url.startswith('http'):
                                    url = f"https://cointelegraph.com{url}"
                                
                                published_at = dates[i] if i < len(dates) else datetime.now().isoformat()
                                
                                news_items.append({
                                    'title': titles[i].strip(),
                                    'url': url,
                                    'source': 'CoinTelegraph',
                                    'published_at': published_at,
                                    'content': ''  # Would need to fetch the article content
                                })
                        
                        return news_items
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching from CoinTelegraph: {str(e)}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for analysis
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(cleaned_tokens)
    
    def analyze_sentiment_vader(self, text: str) -> Dict:
        """
        Analyze sentiment using VADER lexicon
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not text:
            return {
                'compound': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 1,
                'sentiment': 'neutral'
            }
        
        # Get sentiment scores
        scores = self.vader.polarity_scores(text)
        
        # Determine sentiment category
        if scores['compound'] >= self.sentiment_thresholds['very_positive']:
            sentiment = 'very_positive'
        elif scores['compound'] >= self.sentiment_thresholds['positive']:
            sentiment = 'positive'
        elif scores['compound'] <= self.sentiment_thresholds['very_negative']:
            sentiment = 'very_negative'
        elif scores['compound'] <= self.sentiment_thresholds['negative']:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Add sentiment category to results
        scores['sentiment'] = sentiment
        
        return scores
    
    def analyze_sentiment_transformer(self, text: str) -> Dict:
        """
        Analyze sentiment using transformer model
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not text or not self.use_transformers:
            return {
                'label': 'NEUTRAL',
                'score': 0.5,
                'sentiment': 'neutral'
            }
        
        try:
            # Truncate text if too long (most models have a max length)
            if len(text) > 1024:
                text = text[:1024]
            
            # Get sentiment from model
            result = self.sentiment_analyzer(text)[0]
            
            # Standardize output
            label = result['label']
            score = result['score']
            
            # Convert to more standard format
            sentiment_dict = {
                'label': label,
                'score': score
            }
            
            # Map to standard sentiment categories
            if label == 'POSITIVE' and score >= 0.8:
                sentiment_dict['sentiment'] = 'very_positive'
            elif label == 'POSITIVE':
                sentiment_dict['sentiment'] = 'positive'
            elif label == 'NEGATIVE' and score >= 0.8:
                sentiment_dict['sentiment'] = 'very_negative'
            elif label == 'NEGATIVE':
                sentiment_dict['sentiment'] = 'negative'
            else:
                sentiment_dict['sentiment'] = 'neutral'
            
            return sentiment_dict
            
        except Exception as e:
            self.logger.error(f"Error in transformer sentiment analysis: {str(e)}")
            # Fall back to VADER sentiment
            return self.analyze_sentiment_vader(text)
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of entities with types
        """
        if not text:
            return []
        
        entities = []
        
        # Use transformer NER if available
        if self.use_transformers:
            try:
                # Truncate text if too long
                if len(text) > 1024:
                    text = text[:1024]
                
                # Get entities from model
                ner_results = self.ner_analyzer(text)
                
                # Process results
                for entity in ner_results:
                    entities.append({
                        'text': entity['word'],
                        'type': entity['entity_group'],
                        'score': entity['score']
                    })
                
                return entities
            except Exception as e:
                self.logger.error(f"Error in transformer entity extraction: {str(e)}")
                # Fall back to keyword matching
        
        # Simple keyword matching for crypto assets
        for symbol, keywords in self.crypto_keywords.items():
            text_lower = text.lower()
            for keyword in keywords:
                if keyword in text_lower:
                    entities.append({
                        'text': keyword,
                        'type': 'CRYPTO',
                        'symbol': symbol,
                        'score': 1.0
                    })
        
        return entities
    
    def calculate_relevance(self, news_item: Dict, symbol: str) -> float:
        """
        Calculate relevance score for a news item to a specific symbol
        
        Args:
            news_item: News item dictionary
            symbol: Cryptocurrency symbol
            
        Returns:
            Relevance score between 0 and 1
        """
        relevance = 0.0
        
        # Get base symbol without USDC
        base_symbol = symbol.replace('USDC', '')
        
        # Check if symbol is directly mentioned in title
        if base_symbol.lower() in news_item.get('title', '').lower():
            relevance += 0.5
        
        # Check for keywords in title and content
        symbol_keywords = self.crypto_keywords.get(base_symbol, [])
        symbol_keywords.append(base_symbol.lower())
        
        title = news_item.get('title', '').lower()
        content = news_item.get('content', '').lower()
        
        for keyword in symbol_keywords:
            if keyword in title:
                relevance += 0.3
            if keyword in content:
                relevance += 0.2
        
        # Check entities
        entities = news_item.get('entities', [])
        for entity in entities:
            if entity.get('symbol') == base_symbol or entity.get('text') in symbol_keywords:
                relevance += 0.2 * entity.get('score', 1.0)
        
        # Normalize to 0-1 range
        return min(1.0, relevance)
    
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """
        Generate a summary of the text
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length
            
        Returns:
            Summarized text
        """
        if not text or len(text) < max_length:
            return text
        
        # Use transformer summarizer if available
        if self.use_transformers:
            try:
                # Truncate input if too long (most models have max lengths)
                if len(text) > 1024:
                    text = text[:1024]
                
                # Generate summary
                summary = self.summarizer(
                    text, 
                    max_length=max_length,
                    min_length=min(30, max_length // 2),
                    do_sample=False
                )[0]['summary_text']
                
                return summary
            except Exception as e:
                self.logger.error(f"Error in transformer summarization: {str(e)}")
                # Fall back to extractive summary
        
        # Simple extractive summary
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= 3:
            return text
        
        # Use first and last sentences for summary
        summary = ' '.join([sentences[0], sentences[-1]])
        
        # Add a middle sentence if needed
        if len(summary) < max_length and len(sentences) > 2:
            summary = ' '.join([sentences[0], sentences[len(sentences)//2], sentences[-1]])
        
        return summary
    
    def extract_topics(self, news_items: List[Dict]) -> List[str]:
        """
        Extract common topics from a list of news items
        
        Args:
            news_items: List of news items
            
        Returns:
            List of common topics
        """
        # Combine all text
        all_text = ' '.join([
            item.get('title', '') + ' ' + item.get('content', '')
            for item in news_items
        ])
        
        # Clean text
        cleaned_text = self._clean_text(all_text)
        
        # Count word frequencies
        words = cleaned_text.split()
        word_freq = {}
        
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        
        # Get top 10 words as topics
        topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return [topic[0] for topic in topics]
    
    def filter_relevant_news(self, news_items: List[Dict], symbol: str) -> List[Dict]:
        """
        Filter news items by relevance to a symbol
        
        Args:
            news_items: List of news items
            symbol: Cryptocurrency symbol
            
        Returns:
            Filtered list of relevant news items
        """
        relevant_news = []
        
        for item in news_items:
            # Calculate relevance
            relevance = self.calculate_relevance(item, symbol)
            item['relevance'] = relevance
            
            # Keep only relevant news
            if relevance >= self.min_news_relevance:
                relevant_news.append(item)
        
        # Sort by relevance
        relevant_news.sort(key=lambda x: x['relevance'], reverse=True)
        
        return relevant_news
    
    async def analyze_news(self, symbol: str, force_update: bool = False) -> Dict:
        """
        Analyze news for a specific symbol
        
        Args:
            symbol: Cryptocurrency symbol
            force_update: Force update even if cached data is available
            
        Returns:
            Dictionary with news analysis results
        """
        # Check cache first
        if not force_update and symbol in self.news_cache:
            cache_time = self.last_update.get(symbol, datetime.min)
            if (datetime.now() - cache_time).total_seconds() < self.cache_duration:
                return self.news_cache[symbol]
        
        try:
            # Fetch news from sources
            news_items = await self.fetch_news(symbol)
            
            if not news_items:
                self.logger.warning(f"No news found for {symbol}")
                return {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'news_items': [],
                    'sentiment': {
                        'overall': 'neutral',
                        'score': 0.0
                    },
                    'topics': [],
                    'entities': [],
                    'status': 'no_news'
                }
            
            # Process each news item
            for item in news_items:
                # Clean and analyze title and content
                title = item.get('title', '')
                content = item.get('content', '')
                combined_text = f"{title} {content}"
                
                # Analyze sentiment
                if self.use_transformers:
                    item['sentiment_analysis'] = self.analyze_sentiment_transformer(combined_text)
                else:
                    item['sentiment_analysis'] = self.analyze_sentiment_vader(combined_text)
                
                # Extract entities
                item['entities'] = self.extract_entities(combined_text)
                
                # Generate summary if content is available
                if content and len(content) > 150:
                    item['summary'] = self.summarize_text(content)
            
            # Filter relevant news
            relevant_news = self.filter_relevant_news(news_items, symbol)
            
            # Extract topics from all relevant news
            topics = self.extract_topics(relevant_news) if relevant_news else []
            
            # Calculate overall sentiment
            overall_sentiment = self._calculate_overall_sentiment(relevant_news)
            
            # Prepare result
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'news_items': relevant_news[:10],  # Limit to top 10 most relevant
                'sentiment': overall_sentiment,
                'topics': topics,
                'entities': self._aggregate_entities(relevant_news),
                'status': 'success'
            }
            
            # Cache the result
            self.news_cache[symbol] = result
            self.last_update[symbol] = datetime.now()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing news for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
    
    def _calculate_overall_sentiment(self, news_items: List[Dict]) -> Dict:
        """
        Calculate overall sentiment from news items
        
        Args:
            news_items: List of news items
            
        Returns:
            Dictionary with overall sentiment
        """
        if not news_items:
            return {
                'overall': 'neutral',
                'score': 0.0
            }
        
        # Collect all sentiment scores
        scores = []
        for item in news_items:
            sentiment_analysis = item.get('sentiment_analysis', {})
            if 'compound' in sentiment_analysis:
                # VADER score
                scores.append(sentiment_analysis['compound'])
            elif 'score' in sentiment_analysis:
                # Transformer score (adjust from 0-1 to -1 to 1 range)
                label = sentiment_analysis.get('label', 'NEUTRAL')
                score = sentiment_analysis.get('score', 0.5)
                
                if label == 'POSITIVE':
                    scores.append(score)
                elif label == 'NEGATIVE':
                    scores.append(-score)
                else:
                    scores.append(0)
        
        # Calculate weighted average, giving more weight to higher relevance
        weighted_scores = []
        weights = []
        
        for i, item in enumerate(news_items):
            relevance = item.get('relevance', 0.5)
            if i < len(scores):
                weighted_scores.append(scores[i] * relevance)
                weights.append(relevance)
        
        if not weighted_scores:
            return {
                'overall': 'neutral',
                'score': 0.0
            }
        
        total_weight = sum(weights) if weights else 1
        avg_score = sum(weighted_scores) / total_weight
        
        # Determine sentiment category
        if avg_score >= self.sentiment_thresholds['very_positive']:
            overall = 'very_positive'
        elif avg_score >= self.sentiment_thresholds['positive']:
            overall = 'positive'
        elif avg_score <= self.sentiment_thresholds['very_negative']:
            overall = 'very_negative'
        elif avg_score <= self.sentiment_thresholds['negative']:
            overall = 'negative'
        else:
            overall = 'neutral'
        
        return {
            'overall': overall,
            'score': avg_score
        }
    
    def _aggregate_entities(self, news_items: List[Dict]) -> List[Dict]:
        """
        Aggregate entities from multiple news items
        
        Args:
            news_items: List of news items
            
        Returns:
            List of unique entities with frequency
        """
        entity_map = defaultdict(lambda: {'count': 0, 'score': 0, 'type': ''})
        
        for item in news_items:
            for entity in item.get('entities', []):
                text = entity.get('text', '').lower()
                if text:
                    entity_map[text]['count'] += 1
                    entity_map[text]['score'] += entity.get('score', 1.0)
                    entity_map[text]['type'] = entity.get('type', '')
                    entity_map[text]['symbol'] = entity.get('symbol', '')
        
        # Convert to list and sort by count
        entities = []
        for text, data in entity_map.items():
            entities.append({
                'text': text,
                'count': data['count'],
                'avg_score': data['score'] / data['count'],
                'type': data['type'],
                'symbol': data['symbol']
            })
        
        entities.sort(key=lambda x: x['count'], reverse=True)
        
        return entities[:10]  # Return top 10 entities

# Create a standalone function for direct use
async def analyze_news(symbol: str, config: Dict = None) -> Dict:
    """
    Analyze news for a specific cryptocurrency symbol
    
    Args:
        symbol: Cryptocurrency symbol
        config: Optional configuration dictionary
        
    Returns:
        Dictionary with news analysis results
    """
    analyzer = NewsAnalyzer(config)
    return await analyzer.analyze_news(symbol)