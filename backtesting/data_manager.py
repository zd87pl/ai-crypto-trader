import os
import json
import pandas as pd
import numpy as np
import aiohttp
import asyncio
import logging as logger
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

# Configure logging
logger.basicConfig(
    level=logger.INFO,
    format='%(asctime)s - %(levelname)s - [Backtesting] %(message)s'
)

class HistoricalDataManager:
    """Manager for historical market and social data for backtesting"""
    def __init__(self, config_path: str = 'config.json'):
        """Initialize with configuration"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Data storage paths
        self.data_dir = Path('backtesting/data')
        self.market_data_dir = self.data_dir / 'market'
        self.social_data_dir = self.data_dir / 'social'
        
        # Create directories if they don't exist
        self.market_data_dir.mkdir(parents=True, exist_ok=True)
        self.social_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Binance API settings
        self.binance_api_url = "https://api.binance.com/api/v3"
        
        # LunarCrush API settings
        self.lunarcrush_api_key = os.getenv('LUNARCRUSH_API_KEY', '')
        self.lunarcrush_base_url = self.config['lunarcrush']['base_url']
        self.lunarcrush_endpoints = self.config['lunarcrush']['endpoints']
        
        # Cache for loaded datasets
        self.market_data_cache = {}
        self.social_data_cache = {}
        
    async def fetch_historical_klines(self, symbol: str, interval: str, 
                                    start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
        """Fetch historical klines/candlestick data from Binance API"""
        if end_date is None:
            end_date = datetime.now()
            
        start_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        
        klines = []
        current_timestamp = start_timestamp
        
        async with aiohttp.ClientSession() as session:
            while current_timestamp < end_timestamp:
                try:
                    params = {
                        'symbol': symbol,
                        'interval': interval,
                        'startTime': current_timestamp,
                        'endTime': end_timestamp,
                        'limit': 1000
                    }
                    
                    async with session.get(f"{self.binance_api_url}/klines", params=params) as response:
                        if response.status == 200:
                            response_data = await response.json()
                            
                            if not response_data:
                                break
                                
                            klines.extend(response_data)
                            
                            # Update current timestamp for next request
                            current_timestamp = response_data[-1][0] + 1
                        else:
                            error_text = await response.text()
                            logger.error(f"Failed to fetch klines: {response.status}, {error_text}")
                            raise Exception(f"Failed to fetch klines: {response.status}, {error_text}")
                            
                except Exception as e:
                    logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
                    raise
                
                await asyncio.sleep(0.1)  # Rate limiting
        
        if not klines:
            logger.warning(f"No historical data found for {symbol} from {start_date} to {end_date}")
            return pd.DataFrame()
            
        # Create DataFrame from klines data
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convert numeric columns to float
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        return df
        
    async def fetch_historical_social_data(self, symbol: str, 
                                         start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
        """Fetch historical social data from LunarCrush API"""
        if not self.lunarcrush_api_key:
            logger.error("LunarCrush API key not set")
            return pd.DataFrame()
            
        if end_date is None:
            end_date = datetime.now()
            
        # Calculate days for interval
        days = (end_date - start_date).days + 1
        
        try:
            # Prepare API request
            headers = {
                'Authorization': f'Bearer {self.lunarcrush_api_key}',
                'Accept': 'application/json'
            }
            
            params = {
                'symbol': symbol,
                'interval': '1d',  # Daily data
                'days': min(days, 90)  # API limit is 90 days
            }
            
            url = f"{self.lunarcrush_base_url}{self.lunarcrush_endpoints['assets']}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'data' in data and data['data']:
                            # Extract timeseries data if available
                            asset_data = data['data'][0]
                            timeseries = asset_data.get('timeSeries', [])
                            
                            if timeseries:
                                # Create DataFrame from timeseries
                                df = pd.DataFrame(timeseries)
                                
                                # Convert time to datetime
                                df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                                df.set_index('timestamp', inplace=True)
                                
                                # Filter to requested date range
                                df = df[(df.index >= start_date) & (df.index <= end_date)]
                                
                                return df
                    
                    logger.error(f"Failed to fetch social data for {symbol}: {response.status}")
                    return pd.DataFrame()
                    
        except Exception as e:
            logger.error(f"Error fetching historical social data for {symbol}: {str(e)}")
            return pd.DataFrame()
            
    async def fetch_and_save_data(self, symbol: str, interval: str, 
                                start_date: datetime, end_date: datetime = None,
                                include_social: bool = True) -> Tuple[bool, bool]:
        """Fetch and save both market and social data"""
        market_success = False
        social_success = False
        
        # Create symbol specific directories
        symbol_market_dir = self.market_data_dir / symbol
        symbol_social_dir = self.social_data_dir / symbol
        symbol_market_dir.mkdir(exist_ok=True)
        symbol_social_dir.mkdir(exist_ok=True)
        
        # Fetch and save market data
        try:
            market_data = await self.fetch_historical_klines(symbol, interval, start_date, end_date)
            if not market_data.empty:
                file_name = f"{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                file_path = symbol_market_dir / file_name
                market_data.to_csv(file_path)
                logger.info(f"Saved market data for {symbol} to {file_path}")
                market_success = True
        except Exception as e:
            logger.error(f"Failed to fetch and save market data for {symbol}: {str(e)}")
        
        # Fetch and save social data if requested
        if include_social:
            try:
                social_data = await self.fetch_historical_social_data(symbol, start_date, end_date)
                if not social_data.empty:
                    file_name = f"social_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                    file_path = symbol_social_dir / file_name
                    social_data.to_csv(file_path)
                    logger.info(f"Saved social data for {symbol} to {file_path}")
                    social_success = True
            except Exception as e:
                logger.error(f"Failed to fetch and save social data for {symbol}: {str(e)}")
        
        return market_success, social_success
    
    def load_market_data(self, symbol: str, interval: str, 
                       start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
        """Load market data from stored files or cache"""
        # Check if already in cache
        cache_key = f"{symbol}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d') if end_date else 'now'}"
        if cache_key in self.market_data_cache:
            return self.market_data_cache[cache_key]
        
        # Find matching files
        symbol_dir = self.market_data_dir / symbol
        if not symbol_dir.exists():
            logger.warning(f"No data directory found for {symbol}")
            return pd.DataFrame()
        
        # Get list of available data files
        data_files = list(symbol_dir.glob(f"{interval}_*.csv"))
        if not data_files:
            logger.warning(f"No {interval} data files found for {symbol}")
            return pd.DataFrame()
        
        # Load and concatenate relevant files
        dfs = []
        for file_path in data_files:
            try:
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Filter by date range
                if end_date is None:
                    end_date = datetime.now()
                
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {str(e)}")
        
        if not dfs:
            logger.warning(f"No data found for {symbol} in specified date range")
            return pd.DataFrame()
        
        # Concatenate and sort
        result = pd.concat(dfs).sort_index()
        
        # Remove duplicates
        result = result[~result.index.duplicated(keep='first')]
        
        # Store in cache
        self.market_data_cache[cache_key] = result
        
        return result
    
    def load_social_data(self, symbol: str, start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
        """Load social data from stored files or cache"""
        # Check if already in cache
        cache_key = f"{symbol}_social_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d') if end_date else 'now'}"
        if cache_key in self.social_data_cache:
            return self.social_data_cache[cache_key]
        
        # Find matching files
        symbol_dir = self.social_data_dir / symbol
        if not symbol_dir.exists():
            logger.warning(f"No social data directory found for {symbol}")
            return pd.DataFrame()
        
        # Get list of available data files
        data_files = list(symbol_dir.glob("social_*.csv"))
        if not data_files:
            logger.warning(f"No social data files found for {symbol}")
            return pd.DataFrame()
        
        # Load and concatenate relevant files
        dfs = []
        for file_path in data_files:
            try:
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Filter by date range
                if end_date is None:
                    end_date = datetime.now()
                
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading social data file {file_path}: {str(e)}")
        
        if not dfs:
            logger.warning(f"No social data found for {symbol} in specified date range")
            return pd.DataFrame()
        
        # Concatenate and sort
        result = pd.concat(dfs).sort_index()
        
        # Remove duplicates
        result = result[~result.index.duplicated(keep='first')]
        
        # Store in cache
        self.social_data_cache[cache_key] = result
        
        return result
    
    def available_symbols(self) -> List[str]:
        """Get list of symbols with available data"""
        symbols = set()
        
        # Check market data directory
        for path in self.market_data_dir.iterdir():
            if path.is_dir():
                symbols.add(path.name)
        
        return sorted(list(symbols))
    
    def available_intervals(self, symbol: str) -> List[str]:
        """Get list of available intervals for a symbol"""
        intervals = set()
        
        symbol_dir = self.market_data_dir / symbol
        if not symbol_dir.exists():
            return []
        
        for file_path in symbol_dir.glob("*.csv"):
            try:
                interval = file_path.name.split('_')[0]
                intervals.add(interval)
            except:
                pass
        
        return sorted(list(intervals))
    
    def get_data_range(self, symbol: str, interval: str) -> Tuple[datetime, datetime]:
        """Get the available date range for a symbol and interval"""
        symbol_dir = self.market_data_dir / symbol
        if not symbol_dir.exists():
            return None, None
        
        start_dates = []
        end_dates = []
        
        for file_path in symbol_dir.glob(f"{interval}_*.csv"):
            try:
                # Extract dates from filename
                parts = file_path.stem.split('_')
                if len(parts) >= 3:
                    start_date = datetime.strptime(parts[1], '%Y%m%d')
                    end_date = datetime.strptime(parts[2], '%Y%m%d')
                    start_dates.append(start_date)
                    end_dates.append(end_date)
            except:
                pass
        
        if not start_dates or not end_dates:
            return None, None
            
        return min(start_dates), max(end_dates)
    
    def merge_market_and_social_data(self, symbol: str, interval: str, 
                                   start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
        """Merge market and social data into a single DataFrame"""
        # Load market data
        market_data = self.load_market_data(symbol, interval, start_date, end_date)
        if market_data.empty:
            logger.warning(f"No market data available for {symbol}")
            return pd.DataFrame()
        
        # Load social data
        social_data = self.load_social_data(symbol, start_date, end_date)
        
        # If no social data, return just market data
        if social_data.empty:
            logger.warning(f"No social data available for {symbol}, returning market data only")
            return market_data
        
        # Resample daily social data to match market data frequency
        freq_map = {
            '1m': '1T', '3m': '3T', '5m': '5T', '15m': '15T', 
            '30m': '30T', '1h': '1H', '2h': '2H', '4h': '4H',
            '6h': '6H', '8h': '8H', '12h': '12H', '1d': '1D',
            '3d': '3D', '1w': '1W', '1M': '1M'
        }
        
        # Forward fill social data to match market data frequency
        if interval in freq_map:
            resampled_social = social_data.resample(freq_map[interval]).ffill()
        else:
            # Default to 1-day resampling if interval not recognized
            resampled_social = social_data.resample('1D').ffill()
        
        # Merge the datasets
        merged_data = pd.merge_asof(
            market_data.reset_index(), 
            resampled_social.reset_index(),
            on='timestamp',
            direction='nearest'
        )
        
        # Set timestamp as index again
        merged_data.set_index('timestamp', inplace=True)
        
        return merged_data