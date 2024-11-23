import numpy as np
import pandas as pd
from binance.client import Client
from sklearn.ensemble import RandomForestClassifier
from ta.trend import SMAIndicator, EMAIndicator, MACD, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
import warnings
import datetime
import concurrent.futures
warnings.filterwarnings('ignore')

class TechnicalAnalyzer:
    def __init__(self, data):
        """Initialize with DataFrame containing OHLCV data"""
        self.data = data.copy()  # Create a copy to avoid modifying original data
        
        # Convert only numeric columns to float
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        self._calculate_all_indicators()
        self._handle_nan_values()  # Handle any NaN values after calculation
    
    def _handle_nan_values(self):
        """Handle NaN values in indicators"""
        # Get all columns except timestamp
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        
        # Forward fill NaN values for numeric columns only
        self.data[numeric_columns] = self.data[numeric_columns].fillna(method='ffill')
        # Back fill any remaining NaN values at the start
        self.data[numeric_columns] = self.data[numeric_columns].fillna(method='bfill')
        # If still any NaN values, fill with 0
        self.data[numeric_columns] = self.data[numeric_columns].fillna(0)
    
    def _calculate_all_indicators(self):
        """Calculate all technical indicators"""
        try:
            # Trend Indicators
            self._calculate_moving_averages()
            self._calculate_macd()
            self._calculate_ichimoku()
            
            # Momentum Indicators
            self._calculate_rsi()
            self._calculate_stochastic()
            self._calculate_williams_r()
            
            # Volatility Indicators
            self._calculate_bollinger_bands()
            self._calculate_atr()
            
            # Volume Indicators
            self._calculate_vwap()
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            raise
    
    def _calculate_moving_averages(self):
        """Calculate various moving averages"""
        try:
            # Short-term SMA (20 periods)
            sma20 = SMAIndicator(close=self.data['close'], window=20)
            self.data['sma_20'] = sma20.sma_indicator()
            
            # Medium-term SMA (50 periods)
            sma50 = SMAIndicator(close=self.data['close'], window=50)
            self.data['sma_50'] = sma50.sma_indicator()
            
            # Long-term SMA (200 periods)
            sma200 = SMAIndicator(close=self.data['close'], window=200)
            self.data['sma_200'] = sma200.sma_indicator()
            
            # Exponential Moving Averages
            ema12 = EMAIndicator(close=self.data['close'], window=12)
            self.data['ema_12'] = ema12.ema_indicator()
            
            ema26 = EMAIndicator(close=self.data['close'], window=26)
            self.data['ema_26'] = ema26.ema_indicator()
        except Exception as e:
            print(f"Error calculating moving averages: {str(e)}")
            raise
    
    def _calculate_macd(self):
        """Calculate MACD indicator"""
        try:
            macd = MACD(close=self.data['close'])
            self.data['macd'] = macd.macd()
            self.data['macd_signal'] = macd.macd_signal()
            self.data['macd_diff'] = macd.macd_diff()
        except Exception as e:
            print(f"Error calculating MACD: {str(e)}")
            raise
    
    def _calculate_ichimoku(self):
        """Calculate Ichimoku Cloud indicator"""
        try:
            ichimoku = IchimokuIndicator(high=self.data['high'], low=self.data['low'])
            self.data['ichimoku_a'] = ichimoku.ichimoku_a()
            self.data['ichimoku_b'] = ichimoku.ichimoku_b()
        except Exception as e:
            print(f"Error calculating Ichimoku: {str(e)}")
            raise
    
    def _calculate_rsi(self):
        """Calculate RSI indicator"""
        try:
            rsi = RSIIndicator(close=self.data['close'])
            self.data['rsi'] = rsi.rsi()
        except Exception as e:
            print(f"Error calculating RSI: {str(e)}")
            raise
    
    def _calculate_stochastic(self):
        """Calculate Stochastic Oscillator"""
        try:
            stoch = StochasticOscillator(
                high=self.data['high'],
                low=self.data['low'],
                close=self.data['close']
            )
            self.data['stoch_k'] = stoch.stoch()
            self.data['stoch_d'] = stoch.stoch_signal()
        except Exception as e:
            print(f"Error calculating Stochastic: {str(e)}")
            raise
    
    def _calculate_williams_r(self):
        """Calculate Williams %R indicator"""
        try:
            williams = WilliamsRIndicator(
                high=self.data['high'],
                low=self.data['low'],
                close=self.data['close']
            )
            self.data['williams_r'] = williams.williams_r()
        except Exception as e:
            print(f"Error calculating Williams %R: {str(e)}")
            raise
    
    def _calculate_bollinger_bands(self):
        """Calculate Bollinger Bands"""
        try:
            bb = BollingerBands(close=self.data['close'])
            self.data['bb_high'] = bb.bollinger_hband()
            self.data['bb_mid'] = bb.bollinger_mavg()
            self.data['bb_low'] = bb.bollinger_lband()
            self.data['bb_width'] = (self.data['bb_high'] - self.data['bb_low']) / self.data['bb_mid']
            # Calculate BB position with error handling
            bb_range = self.data['bb_high'] - self.data['bb_low']
            bb_range = bb_range.replace(0, np.nan)  # Replace zero range with NaN
            self.data['bb_position'] = (self.data['close'] - self.data['bb_low']) / bb_range
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {str(e)}")
            raise
    
    def _calculate_atr(self):
        """Calculate Average True Range"""
        try:
            atr = AverageTrueRange(high=self.data['high'], low=self.data['low'], close=self.data['close'])
            self.data['atr'] = atr.average_true_range()
        except Exception as e:
            print(f"Error calculating ATR: {str(e)}")
            raise
    
    def _calculate_vwap(self):
        """Calculate Volume Weighted Average Price"""
        try:
            vwap = VolumeWeightedAveragePrice(
                high=self.data['high'],
                low=self.data['low'],
                close=self.data['close'],
                volume=self.data['volume']
            )
            self.data['vwap'] = vwap.volume_weighted_average_price()
        except Exception as e:
            print(f"Error calculating VWAP: {str(e)}")
            raise
    
    def get_trend(self):
        """Determine current market trend"""
        try:
            last_close = float(self.data['close'].iloc[-1])
            sma20 = float(self.data['sma_20'].iloc[-1])
            sma50 = float(self.data['sma_50'].iloc[-1])
            
            # Calculate trend strength based on price distance from moving averages
            strength = ((last_close - sma20) / sma20 * 100 +
                       (last_close - sma50) / sma50 * 100) / 2
            
            if last_close > sma20 and sma20 > sma50:
                return 'uptrend', abs(strength)
            elif last_close < sma20 and sma20 < sma50:
                return 'downtrend', abs(strength)
            else:
                return 'sideways', abs(strength)
        except Exception as e:
            print(f"Error calculating trend: {str(e)}")
            return 'sideways', 0
    
    def get_volatility(self):
        """Calculate current volatility"""
        try:
            return float(self.data['atr'].iloc[-1]) / float(self.data['close'].iloc[-1])
        except Exception as e:
            print(f"Error calculating volatility: {str(e)}")
            return 0
    
    def get_support_resistance(self):
        """Calculate support and resistance levels"""
        try:
            pivot = (self.data['high'].iloc[-1] + self.data['low'].iloc[-1] + self.data['close'].iloc[-1]) / 3
            support1 = 2 * pivot - self.data['high'].iloc[-1]
            support2 = pivot - (self.data['high'].iloc[-1] - self.data['low'].iloc[-1])
            resistance1 = 2 * pivot - self.data['low'].iloc[-1]
            resistance2 = pivot + (self.data['high'].iloc[-1] - self.data['low'].iloc[-1])
            
            return {
                'support1': float(support1),
                'support2': float(support2),
                'resistance1': float(resistance1),
                'resistance2': float(resistance2)
            }
        except Exception as e:
            print(f"Error calculating support/resistance: {str(e)}")
            return {'support1': 0, 'support2': 0, 'resistance1': 0, 'resistance2': 0}
    
    def get_all_indicators(self):
        """Get current values of all indicators"""
        try:
            return {
                'rsi': float(self.data['rsi'].iloc[-1]),
                'stoch_k': float(self.data['stoch_k'].iloc[-1]),
                'stoch_d': float(self.data['stoch_d'].iloc[-1]),
                'macd': float(self.data['macd'].iloc[-1]),
                'macd_signal': float(self.data['macd_signal'].iloc[-1]),
                'williams_r': float(self.data['williams_r'].iloc[-1]),
                'bb_position': float(self.data['bb_position'].iloc[-1]),
                'volatility': self.get_volatility(),
                'trend': self.get_trend()[0],
                'trend_strength': self.get_trend()[1]
            }
        except Exception as e:
            print(f"Error getting indicators: {str(e)}")
            raise

class PositionSizer:
    @staticmethod
    def calculate_position_size(total_capital, volatility, volume, max_risk_per_trade=0.15):
        """Calculate position size based on volatility and volume"""
        # Base position size on volatility (more balanced)
        if volatility > 0.02:
            position_pct = 0.25  # 25% of available capital for high volatility
            stop_loss_pct = 0.02
        elif volatility > 0.01:
            position_pct = 0.20  # 20% of available capital for medium volatility
            stop_loss_pct = 0.015
        else:
            position_pct = 0.15  # 15% of available capital for low volatility
            stop_loss_pct = 0.01
        
        # Adjust based on volume (more balanced)
        volume_factor = min(volume / 50000, 1)  # Scale with volume up to 50K USDC
        position_size = total_capital * position_pct * volume_factor
        
        # Risk management
        max_position = (total_capital * max_risk_per_trade) / stop_loss_pct
        position_size = min(position_size, max_position)
        
        # Ensure we never risk more than 20% of total capital per trade
        max_capital_risk = total_capital * 0.20
        position_size = min(position_size, max_capital_risk)
        
        # Ensure minimum position size for meaningful trades
        min_position = total_capital * 0.10  # Minimum 10% of capital
        position_size = max(position_size, min_position)
        
        # Ensure position size is at least $40
        position_size = max(position_size, 40)
        
        return {
            'position_size': position_size,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': stop_loss_pct * 2.0,  # 2:1 reward-to-risk ratio
            'trailing_stop_activation': stop_loss_pct * 1.5,  # Activate at 150% of stop loss
            'trailing_stop_distance': stop_loss_pct * 0.75  # Trail at 75% of stop loss distance
        }

class CryptoScanner:
    def __init__(self, client):
        """Initialize with Binance client instance"""
        self.client = client
    
    def get_usdc_pairs(self):
        """Get all USDC trading pairs"""
        exchange_info = self.client.get_exchange_info()
        return [symbol['symbol'] for symbol in exchange_info['symbols'] 
                if symbol['symbol'].endswith('USDC') and symbol['status'] == 'TRADING']
    
    def get_historical_data(self, symbol, interval='5m', limit=100):  # Changed from 1h to 5m
        """Get historical klines/candlestick data"""
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
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
            
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def calculate_volatility(self, symbol):
        """Calculate volatility and other metrics for a trading pair"""
        try:
            df = self.get_historical_data(symbol)
            if df is None or len(df) < 100:  # Need enough data for indicators
                return None
            
            analyzer = TechnicalAnalyzer(df)
            indicators = analyzer.get_all_indicators()
            
            # Calculate average volume in USDC
            avg_volume = df['volume'].mean() * df['close'].mean()
            
            # Calculate price momentum
            price_change_5m = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
            price_change_15m = ((df['close'].iloc[-1] - df['close'].iloc[-4]) / df['close'].iloc[-4]) * 100
            
            return {
                'symbol': symbol,
                'volatility': indicators['volatility'],
                'avg_volume': avg_volume,
                'rsi': indicators['rsi'],
                'stoch_k': indicators['stoch_k'],
                'macd': indicators['macd'],
                'williams_r': indicators['williams_r'],
                'trend': indicators['trend'],
                'trend_strength': indicators['trend_strength'],
                'bb_position': indicators['bb_position'],
                'current_price': float(df['close'].iloc[-1]),
                'price_change_5m': price_change_5m,
                'price_change_15m': price_change_15m
            }
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")
            return None
    
    def scan_market(self, min_volume_usdc=10000):  # Reduced from 25000
        """Scan market for volatile pairs with good volume"""
        print("\nScanning market for volatile trading pairs...")
        
        pairs = self.get_usdc_pairs()
        print(f"Found {len(pairs)} USDC trading pairs")
        
        # Calculate metrics for all pairs in parallel
        results = []
        try:
            # Create a new executor for each scan
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self.calculate_volatility, pair) for pair in pairs]
                # Wait for all futures to complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        print(f"Error processing pair: {str(e)}")
        except Exception as e:
            print(f"Error in parallel processing: {str(e)}")
        
        # Filter out None results and low volume pairs
        results = [r for r in results if r is not None and r['avg_volume'] >= min_volume_usdc]
        
        # Sort by opportunity score
        for r in results:
            r['opportunity_score'] = self.calculate_opportunity_score(r)
        
        results.sort(key=lambda x: x['opportunity_score'], reverse=True)
        
        return results
    
    def calculate_opportunity_score(self, metrics):
        """Calculate an opportunity score based on multiple factors"""
        score = 0
        
        # RSI (more balanced)
        if metrics['rsi'] < 35:  # Strong oversold
            score += 40
        elif metrics['rsi'] < 45:  # Moderately oversold
            score += 30
        elif metrics['rsi'] > 60:  # Overbought
            score -= 10
        
        # MACD
        if metrics['macd'] > 0 and metrics['macd'] > metrics['macd'] * 1.1:  # Strong momentum
            score += 35
        elif metrics['macd'] > 0:  # Positive momentum
            score += 25
        
        # Stochastic
        if metrics['stoch_k'] < 20:  # Strong oversold
            score += 35
        elif metrics['stoch_k'] < 30:  # Moderately oversold
            score += 25
        elif metrics['stoch_k'] > 70:  # Overbought
            score -= 10
        
        # Williams %R
        if metrics['williams_r'] < -80:  # Strong oversold
            score += 35
        elif metrics['williams_r'] < -65:  # Moderately oversold
            score += 25
        elif metrics['williams_r'] > -30:  # Overbought
            score -= 10
        
        # Trend
        if metrics['trend'] == 'uptrend':
            score += 30
        if metrics['trend_strength'] > 8:
            score += 30
        
        # Bollinger Bands position
        bb_pos = metrics['bb_position']
        if bb_pos < 0.2:  # Strong oversold
            score += 35
        elif bb_pos < 0.4:  # Moderately oversold
            score += 25
        elif bb_pos > 0.6:  # Overbought
            score -= 10
        
        # Volume
        if metrics['avg_volume'] > 100000:
            score += 30
        
        # Short-term price momentum
        if metrics['price_change_5m'] > 0.2:
            score += 20
        if metrics['price_change_15m'] > 0.5:
            score += 20
        
        # Volatility
        if 0.008 < metrics['volatility'] < 0.05:
            score += 30
        elif metrics['volatility'] > 0.05:
            score -= 10
        
        return max(0, score)  # Ensure score is not negative

class TradingSignal:
    def __init__(self, symbol, price, rsi, stoch_k, macd, volume, volatility,
                 williams_r=None, trend=None, trend_strength=None, bb_position=None):
        self.symbol = symbol
        self.price = price
        self.rsi = rsi
        self.stoch_k = stoch_k
        self.macd = macd
        self.volume = volume
        self.volatility = volatility
        self.williams_r = williams_r
        self.trend = trend
        self.trend_strength = trend_strength
        self.bb_position = bb_position
        
        # Calculate signal strength
        self.signal = self._calculate_signal()
        self.strength = self._calculate_strength()
    
    def _calculate_signal(self):
        """Calculate trading signal based on multiple indicators"""
        buy_signals = 0
        total_signals = 0
        
        # RSI (more balanced)
        if self.rsi < 35:  # Strong oversold
            buy_signals += 3.0
        elif self.rsi < 45:  # Moderately oversold
            buy_signals += 2.0
        total_signals += 1
        
        # Stochastic (more balanced)
        if self.stoch_k < 20:  # Strong oversold
            buy_signals += 3.0
        elif self.stoch_k < 30:  # Moderately oversold
            buy_signals += 2.0
        total_signals += 1
        
        # MACD
        if self.macd > 0 and self.macd > self.macd * 1.1:  # Strong upward momentum
            buy_signals += 3.0
        elif self.macd > 0:  # Positive momentum
            buy_signals += 2.0
        total_signals += 1
        
        # Williams %R (more balanced)
        if self.williams_r and self.williams_r < -80:  # Strong oversold
            buy_signals += 3.0
        elif self.williams_r and self.williams_r < -65:  # Moderately oversold
            buy_signals += 2.0
        total_signals += 1
        
        # Trend
        if self.trend == 'uptrend' and self.trend_strength and self.trend_strength > 10:
            buy_signals += 3.0
        elif self.trend == 'uptrend' and self.trend_strength and self.trend_strength > 5:
            buy_signals += 2.0
        total_signals += 1
        
        # Bollinger Bands (more balanced)
        if self.bb_position and self.bb_position < 0.2:  # Strong oversold
            buy_signals += 3.0
        elif self.bb_position and self.bb_position < 0.4:  # Moderately oversold
            buy_signals += 2.0
        total_signals += 1
        
        # Calculate signal threshold
        signal_ratio = buy_signals / total_signals
        
        if signal_ratio >= 0.6:  # Strong buy signal
            return 'BUY'
        elif signal_ratio <= 0.3:  # Strong sell signal
            return 'SELL'
        return 'NEUTRAL'
    
    def _calculate_strength(self):
        """Calculate signal strength (0-100) based on multiple factors"""
        if self.signal == 'NEUTRAL':
            return 0
        
        strength = 0
        
        # RSI strength (30% weight)
        if self.signal == 'BUY':
            rsi_strength = (45 - min(self.rsi, 45)) / 15
        else:
            rsi_strength = (max(self.rsi, 55) - 55) / 15
        strength += rsi_strength * 30
        
        # Stochastic strength (20% weight)
        if self.signal == 'BUY':
            stoch_strength = (30 - min(self.stoch_k, 30)) / 30
        else:
            stoch_strength = (max(self.stoch_k, 70) - 70) / 30
        strength += stoch_strength * 20
        
        # MACD strength (20% weight)
        macd_strength = min(abs(self.macd), 1)
        strength += macd_strength * 20
        
        # Volume strength (15% weight)
        volume_strength = min(self.volume / 100000, 1)
        strength += volume_strength * 15
        
        # Trend strength (15% weight)
        if self.trend_strength:
            trend_str = min(self.trend_strength / 20, 1)
            if (self.signal == 'BUY' and self.trend == 'uptrend') or \
               (self.signal == 'SELL' and self.trend == 'downtrend'):
                strength += trend_str * 15
        
        return min(max(strength, 0), 100)  # Ensure final strength is between 0-100
