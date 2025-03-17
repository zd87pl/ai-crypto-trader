import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [VolumeProfile] %(message)s',
    handlers=[
        logging.FileHandler('logs/volume_profile.log'),
        logging.StreamHandler()
    ]
)

class VolumeProfileAnalyzer:
    """
    Class for analyzing trading volume distribution across price levels (Volume Profile).
    
    Volume Profile helps identify:
    1. High volume nodes (HVN): Price ranges with high trading activity (potential support/resistance)
    2. Low volume nodes (LVN): Price ranges with low trading activity (potential breakout zones)
    3. Point of control (POC): Price level with the highest trading volume
    4. Value area: Range containing a specified percentage of total volume (typically 70%)
    """
    
    def __init__(self, num_bins: int = 20, value_area_pct: float = 0.7):
        """
        Initialize the Volume Profile Analyzer
        
        Args:
            num_bins: Number of price bins to divide the price range into
            value_area_pct: Percentage of total volume to include in the value area
        """
        self.num_bins = num_bins
        self.value_area_pct = value_area_pct
        self.logger = logging.getLogger(__name__)
    
    def analyze_volume_profile(self, data: pd.DataFrame) -> Dict:
        """
        Analyze volume profile from price and volume data
        
        Args:
            data: DataFrame with price (high, low, close) and volume data
            
        Returns:
            Dictionary with volume profile analysis
        """
        try:
            # Verify required data is present
            required_columns = ['high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in data.columns:
                    self.logger.warning(f"Missing required column: {col}")
                    return {'error': f"Missing required column: {col}"}
            
            # Calculate volume profile
            profile_data = self._calculate_volume_profile(data)
            
            # Prepare results
            result = {
                'timestamp': datetime.now().isoformat(),
                'price_levels': profile_data['price_levels'].tolist(),
                'volumes': profile_data['volumes'].tolist(),
                'normalized_volumes': profile_data['normalized_volumes'].tolist(),
                'poc': float(profile_data['poc']),
                'value_area_high': float(profile_data['value_area_high']),
                'value_area_low': float(profile_data['value_area_low']),
                'high_volume_nodes': [float(x) for x in profile_data['high_volume_nodes']],
                'low_volume_nodes': [float(x) for x in profile_data['low_volume_nodes']],
                'histogram_base64': self._generate_histogram(profile_data) if len(data) > 0 else None,
                'signals': self._generate_signals(profile_data, data),
                'summary': self._create_summary(profile_data, data)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume profile: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_volume_profile(self, data: pd.DataFrame) -> Dict:
        """
        Calculate the volume profile from price and volume data
        
        Args:
            data: DataFrame with price and volume data
            
        Returns:
            Dictionary with volume profile data
        """
        # Get price range
        price_max = data['high'].max()
        price_min = data['low'].min()
        current_price = data['close'].iloc[-1]
        
        # Create price bins
        bin_size = (price_max - price_min) / self.num_bins
        price_bins = np.linspace(price_min, price_max, self.num_bins + 1)
        price_levels = (price_bins[:-1] + price_bins[1:]) / 2  # Midpoints
        
        # Initialize volume array
        volumes = np.zeros(self.num_bins)
        
        # Distribute volume across price range
        for idx, row in data.iterrows():
            # Calculate price range and volume for this candle
            candle_min = row['low']
            candle_max = row['high']
            candle_volume = row['volume']
            
            # Find which bins this candle spans
            low_bin = max(0, int((candle_min - price_min) / bin_size))
            high_bin = min(self.num_bins - 1, int((candle_max - price_min) / bin_size))
            
            # Distribute volume equally among covered bins
            if high_bin >= low_bin:
                bins_covered = high_bin - low_bin + 1
                volume_per_bin = candle_volume / bins_covered
                volumes[low_bin:high_bin + 1] += volume_per_bin
        
        # Normalize volumes (0-1 scale)
        max_volume = volumes.max() if volumes.max() > 0 else 1
        normalized_volumes = volumes / max_volume
        
        # Find point of control (POC) - price level with highest volume
        poc_idx = np.argmax(volumes)
        poc = price_levels[poc_idx]
        
        # Calculate value area (typically 70% of volume)
        sorted_idx = np.argsort(volumes)[::-1]  # Sort by volume descending
        cumulative_volume = 0
        total_volume = volumes.sum()
        value_area_threshold = total_volume * self.value_area_pct
        value_area_indices = []
        
        for idx in sorted_idx:
            cumulative_volume += volumes[idx]
            value_area_indices.append(idx)
            if cumulative_volume >= value_area_threshold:
                break
        
        # Get value area high and low
        if value_area_indices:
            value_area_low = price_levels[min(value_area_indices)]
            value_area_high = price_levels[max(value_area_indices)]
        else:
            value_area_low = price_min
            value_area_high = price_max
        
        # Identify high volume nodes (HVN) - bins with at least 70% of max volume
        hvn_threshold = 0.7 * max_volume
        high_volume_nodes = price_levels[volumes >= hvn_threshold]
        
        # Identify low volume nodes (LVN) - bins with at most 30% of max volume
        lvn_threshold = 0.3 * max_volume
        low_volume_nodes = price_levels[volumes <= lvn_threshold]
        
        return {
            'price_levels': price_levels,
            'volumes': volumes,
            'normalized_volumes': normalized_volumes,
            'poc': poc,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'high_volume_nodes': high_volume_nodes,
            'low_volume_nodes': low_volume_nodes,
            'current_price': current_price
        }
    
    def _generate_histogram(self, profile_data: Dict) -> str:
        """
        Generate a horizontal histogram visualization of the volume profile
        
        Args:
            profile_data: Dictionary with volume profile data
            
        Returns:
            Base64 encoded string of the histogram image
        """
        try:
            # Extract data from profile
            price_levels = profile_data['price_levels']
            volumes = profile_data['volumes']
            poc = profile_data['poc']
            value_area_high = profile_data['value_area_high']
            value_area_low = profile_data['value_area_low']
            current_price = profile_data.get('current_price', price_levels[-1])
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Plot horizontal histogram
            plt.barh(price_levels, volumes, height=price_levels[1]-price_levels[0])
            
            # Add POC line
            plt.axhline(y=poc, color='r', linestyle='-', label='POC')
            
            # Add value area
            plt.axhspan(value_area_low, value_area_high, alpha=0.2, color='green', label='Value Area')
            
            # Add current price line
            plt.axhline(y=current_price, color='blue', linestyle='--', label='Current Price')
            
            # Add labels and legend
            plt.ylabel('Price')
            plt.xlabel('Volume')
            plt.title('Volume Profile Analysis')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save as PNG
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Encode as base64
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str
            
        except Exception as e:
            self.logger.error(f"Error generating histogram: {str(e)}")
            return ""
    
    def _generate_signals(self, profile_data: Dict, data: pd.DataFrame) -> Dict:
        """
        Generate trading signals based on volume profile analysis
        
        Args:
            profile_data: Dictionary with volume profile data
            data: Original price and volume data
            
        Returns:
            Dictionary with trading signals
        """
        signals = {}
        
        # Get current price (last close price)
        current_price = data['close'].iloc[-1]
        
        # Check if price is approaching POC
        poc = profile_data['poc']
        poc_distance_pct = abs(current_price - poc) / current_price * 100
        
        if poc_distance_pct < 0.5:
            if current_price < poc:
                signals['poc'] = 'approaching_poc_from_below'
            else:
                signals['poc'] = 'approaching_poc_from_above'
        elif current_price < poc:
            signals['poc'] = 'below_poc'
        else:
            signals['poc'] = 'above_poc'
        
        # Check value area
        value_area_high = profile_data['value_area_high']
        value_area_low = profile_data['value_area_low']
        
        if current_price > value_area_high:
            vah_distance_pct = (current_price - value_area_high) / current_price * 100
            if vah_distance_pct < 0.3:
                signals['value_area'] = 'at_value_area_high'
            else:
                signals['value_area'] = 'above_value_area'
        elif current_price < value_area_low:
            val_distance_pct = (value_area_low - current_price) / current_price * 100
            if val_distance_pct < 0.3:
                signals['value_area'] = 'at_value_area_low'
            else:
                signals['value_area'] = 'below_value_area'
        else:
            # Inside value area
            relative_position = (current_price - value_area_low) / (value_area_high - value_area_low)
            if relative_position < 0.3:
                signals['value_area'] = 'inside_value_area_lower'
            elif relative_position > 0.7:
                signals['value_area'] = 'inside_value_area_upper'
            else:
                signals['value_area'] = 'inside_value_area_middle'
        
        # Check for price at low volume node (potential breakout zone)
        for lvn in profile_data['low_volume_nodes']:
            lvn_distance_pct = abs(current_price - lvn) / current_price * 100
            if lvn_distance_pct < 0.3:
                signals['low_volume_node'] = 'at_low_volume_node'
                break
        
        # Check for price at high volume node (potential support/resistance)
        for hvn in profile_data['high_volume_nodes']:
            hvn_distance_pct = abs(current_price - hvn) / current_price * 100
            if hvn_distance_pct < 0.3:
                signals['high_volume_node'] = 'at_high_volume_node'
                break
        
        # Generate overall signal
        if 'at_value_area_high' in signals.values() or signals['poc'] == 'approaching_poc_from_above':
            signals['overall'] = 'potential_resistance'
        elif 'at_value_area_low' in signals.values() or signals['poc'] == 'approaching_poc_from_below':
            signals['overall'] = 'potential_support'
        elif 'at_low_volume_node' in signals.values():
            signals['overall'] = 'potential_breakout_zone'
        elif 'above_value_area' in signals.values():
            signals['overall'] = 'overbought'
        elif 'below_value_area' in signals.values():
            signals['overall'] = 'oversold'
        else:
            signals['overall'] = 'neutral'
        
        return signals
    
    def _create_summary(self, profile_data: Dict, data: pd.DataFrame) -> str:
        """
        Create a text summary of the volume profile analysis
        
        Args:
            profile_data: Dictionary with volume profile data
            data: Original price and volume data
            
        Returns:
            String with summary text
        """
        current_price = data['close'].iloc[-1]
        poc = profile_data['poc']
        value_area_high = profile_data['value_area_high']
        value_area_low = profile_data['value_area_low']
        
        # Format price values with appropriate precision
        current_price_str = f"{current_price:.8f}" if current_price < 1 else f"{current_price:.2f}"
        poc_str = f"{poc:.8f}" if poc < 1 else f"{poc:.2f}"
        vah_str = f"{value_area_high:.8f}" if value_area_high < 1 else f"{value_area_high:.2f}"
        val_str = f"{value_area_low:.8f}" if value_area_low < 1 else f"{value_area_low:.2f}"
        
        # Check if price is inside or outside value area
        if current_price >= value_area_low and current_price <= value_area_high:
            value_area_position = "inside"
        elif current_price > value_area_high:
            value_area_position = "above"
        else:
            value_area_position = "below"
        
        # Determine relation to POC
        if abs(current_price - poc) / current_price < 0.005:  # Within 0.5%
            poc_position = "at"
        elif current_price > poc:
            poc_position = "above"
        else:
            poc_position = "below"
        
        # Create summary
        summary = (
            f"Current price {current_price_str} is {value_area_position} the value area "
            f"({val_str} - {vah_str}) and {poc_position} the point of control ({poc_str}). "
        )
        
        # Add high volume nodes context if relevant
        high_volume_nodes = profile_data['high_volume_nodes']
        if len(high_volume_nodes) > 0:
            if len(high_volume_nodes) <= 3:
                hvn_prices = ', '.join([f"{p:.8f}" if p < 1 else f"{p:.2f}" for p in high_volume_nodes])
                summary += f"High volume nodes (potential support/resistance) at: {hvn_prices}. "
            else:
                summary += f"Multiple high volume nodes identified ({len(high_volume_nodes)}). "
        
        # Add low volume node context if relevant
        low_volume_nodes = profile_data['low_volume_nodes']
        if len(low_volume_nodes) > 0:
            closest_lvn = min(low_volume_nodes, key=lambda x: abs(x - current_price))
            closest_lvn_str = f"{closest_lvn:.8f}" if closest_lvn < 1 else f"{closest_lvn:.2f}"
            distance_pct = abs(closest_lvn - current_price) / current_price * 100
            
            if distance_pct < 1.0:
                summary += f"Price is near a low volume node at {closest_lvn_str} (potential breakout zone). "
        
        # Add trading implications based on signals
        signals = self._generate_signals(profile_data, data)
        if signals['overall'] == 'potential_resistance':
            summary += "Volume profile suggests potential resistance at current level."
        elif signals['overall'] == 'potential_support':
            summary += "Volume profile suggests potential support at current level."
        elif signals['overall'] == 'potential_breakout_zone':
            summary += "Current price is in a low volume area, suggesting a potential breakout zone."
        elif signals['overall'] == 'overbought':
            summary += "Price is above the value area, potentially overbought based on volume profile."
        elif signals['overall'] == 'oversold':
            summary += "Price is below the value area, potentially oversold based on volume profile."
        
        return summary
    
    def analyze_volume_by_time(self, data: pd.DataFrame, intervals: Optional[List[str]] = None) -> Dict:
        """
        Analyze volume distribution by time
        
        Args:
            data: DataFrame with volume and timestamp data
            intervals: Optional list of time intervals to analyze
                (e.g., ['hour', 'day_of_week'])
            
        Returns:
            Dictionary with volume by time analysis
        """
        try:
            # Verify required data is present
            if 'volume' not in data.columns:
                return {'error': "Missing required column: volume"}
            
            # Ensure timestamp is a datetime
            if 'timestamp' in data.columns:
                if not pd.api.types.is_datetime64_dtype(data['timestamp']):
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
            else:
                return {'error': "Missing timestamp column"}
            
            # Use default intervals if none provided
            if intervals is None:
                intervals = ['hour', 'day_of_week']
            
            result = {'timestamp': datetime.now().isoformat()}
            
            # Analyze volume by hour of day
            if 'hour' in intervals:
                data['hour'] = data['timestamp'].dt.hour
                hourly_volume = data.groupby('hour')['volume'].sum()
                result['hourly_volume'] = {
                    'hours': hourly_volume.index.tolist(),
                    'volumes': hourly_volume.values.tolist(),
                    'peak_hour': int(hourly_volume.idxmax()),
                    'lowest_hour': int(hourly_volume.idxmin())
                }
            
            # Analyze volume by day of week
            if 'day_of_week' in intervals:
                data['day_of_week'] = data['timestamp'].dt.dayofweek
                daily_volume = data.groupby('day_of_week')['volume'].sum()
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                result['daily_volume'] = {
                    'days': [days[i] for i in daily_volume.index],
                    'volumes': daily_volume.values.tolist(),
                    'peak_day': days[int(daily_volume.idxmax())],
                    'lowest_day': days[int(daily_volume.idxmin())]
                }
            
            # Analyze volume by session (Asia, Europe, US)
            if 'session' in intervals:
                # Define trading sessions based on UTC hours
                data['session'] = 'Other'
                data.loc[(data['timestamp'].dt.hour >= 0) & (data['timestamp'].dt.hour < 8), 'session'] = 'Asia'
                data.loc[(data['timestamp'].dt.hour >= 8) & (data['timestamp'].dt.hour < 16), 'session'] = 'Europe'
                data.loc[(data['timestamp'].dt.hour >= 16) & (data['timestamp'].dt.hour < 24), 'session'] = 'US'
                
                session_volume = data.groupby('session')['volume'].sum()
                result['session_volume'] = {
                    'sessions': session_volume.index.tolist(),
                    'volumes': session_volume.values.tolist(),
                    'peak_session': session_volume.idxmax()
                }
            
            # Generate insights
            insights = []
            
            if 'hourly_volume' in result:
                peak_hour = result['hourly_volume']['peak_hour']
                insights.append(f"Peak trading volume occurs at {peak_hour}:00 UTC")
            
            if 'daily_volume' in result:
                peak_day = result['daily_volume']['peak_day']
                insights.append(f"Highest trading volume is on {peak_day}")
            
            if 'session_volume' in result:
                peak_session = result['session_volume']['peak_session']
                insights.append(f"{peak_session} session shows the highest trading activity")
            
            result['insights'] = insights
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume by time: {str(e)}")
            return {'error': str(e)}
    
    def detect_volume_anomalies(self, data: pd.DataFrame, window: int = 20, std_threshold: float = 2.0) -> Dict:
        """
        Detect unusual volume patterns and anomalies
        
        Args:
            data: DataFrame with volume data
            window: Moving average window for baseline volume
            std_threshold: Standard deviation threshold for anomaly detection
            
        Returns:
            Dictionary with volume anomalies
        """
        try:
            # Verify required data is present
            if 'volume' not in data.columns:
                return {'error': "Missing required column: volume"}
            
            if len(data) < window:
                return {'error': f"Insufficient data for window size {window}"}
            
            # Calculate rolling mean and standard deviation
            data['volume_ma'] = data['volume'].rolling(window=window).mean()
            data['volume_std'] = data['volume'].rolling(window=window).std()
            
            # Define threshold for anomalies
            data['upper_threshold'] = data['volume_ma'] + (std_threshold * data['volume_std'])
            
            # Detect anomalies
            data['is_anomaly'] = data['volume'] > data['upper_threshold']
            
            # Extract anomalies
            anomalies = data[data['is_anomaly']].copy()
            
            if len(anomalies) == 0:
                return {
                    'timestamp': datetime.now().isoformat(),
                    'anomalies_detected': False,
                    'message': "No volume anomalies detected"
                }
            
            # Calculate anomaly significance
            anomalies['anomaly_significance'] = (anomalies['volume'] - anomalies['volume_ma']) / anomalies['volume_std']
            
            # Prepare result
            recent_anomalies = anomalies.tail(5).copy()  # Last 5 anomalies
            
            # Format timestamps if present
            if 'timestamp' in recent_anomalies.columns:
                if pd.api.types.is_datetime64_dtype(recent_anomalies['timestamp']):
                    recent_anomalies['timestamp'] = recent_anomalies['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'anomalies_detected': True,
                'total_anomalies': len(anomalies),
                'recent_anomalies': recent_anomalies[['timestamp', 'volume', 'volume_ma', 'anomaly_significance']].to_dict('records'),
                'strongest_anomaly': {
                    'significance': float(anomalies['anomaly_significance'].max()),
                    'volume': float(anomalies.loc[anomalies['anomaly_significance'].idxmax(), 'volume']),
                    'normal_volume': float(anomalies.loc[anomalies['anomaly_significance'].idxmax(), 'volume_ma'])
                }
            }
            
            # Calculate if recent periods have anomalies
            recent_period = min(20, len(data))
            recent_anomaly_count = data.tail(recent_period)['is_anomaly'].sum()
            result['recent_anomaly_percentage'] = float(recent_anomaly_count / recent_period * 100)
            
            if result['recent_anomaly_percentage'] > 15:
                result['message'] = "There is elevated trading volume activity recently"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting volume anomalies: {str(e)}")
            return {'error': str(e)}

    def analyze_volume_delta(self, data: pd.DataFrame) -> Dict:
        """
        Analyze buying vs selling volume pressure (volume delta)
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with volume delta analysis
        """
        try:
            # Verify required data is present
            required_columns = ['open', 'close', 'volume']
            for col in required_columns:
                if col not in data.columns:
                    return {'error': f"Missing required column: {col}"}
            
            # Calculate volume delta (buying vs selling volume)
            data['delta'] = data.apply(
                lambda row: row['volume'] if row['close'] >= row['open'] else -row['volume'], 
                axis=1
            )
            
            # Calculate cumulative volume delta
            data['cumulative_delta'] = data['delta'].cumsum()
            
            # Prepare analysis
            result = {
                'timestamp': datetime.now().isoformat(),
                'current_cumulative_delta': float(data['cumulative_delta'].iloc[-1]),
                'net_period_delta': float(data['delta'].sum()),
                'buying_volume': float(data[data['delta'] > 0]['delta'].sum()),
                'selling_volume': float(abs(data[data['delta'] < 0]['delta'].sum())),
            }
            
            # Calculate buying vs selling volume ratio
            if result['selling_volume'] > 0:
                result['buy_sell_ratio'] = result['buying_volume'] / result['selling_volume']
            else:
                result['buy_sell_ratio'] = float('inf')
            
            # Calculate recent trend (last 20 periods or fewer if less data available)
            periods = min(20, len(data))
            recent_delta = data['delta'].tail(periods).sum()
            result['recent_delta'] = float(recent_delta)
            
            # Calculate delta momentum (acceleration/deceleration)
            if len(data) >= 40:
                prev_delta = data['delta'].iloc[-40:-20].sum()
                recent_delta = data['delta'].iloc[-20:].sum()
                delta_momentum = recent_delta - prev_delta
                result['delta_momentum'] = float(delta_momentum)
                
                if delta_momentum > 0:
                    result['delta_momentum_direction'] = 'accelerating_buying'
                else:
                    result['delta_momentum_direction'] = 'accelerating_selling'
            
            # Generate signals
            signals = {}
            
            # Overall volume pressure
            if result['buy_sell_ratio'] > 1.5:
                signals['overall_pressure'] = 'strong_buying'
            elif result['buy_sell_ratio'] > 1.1:
                signals['overall_pressure'] = 'moderate_buying'
            elif result['buy_sell_ratio'] < 0.67:
                signals['overall_pressure'] = 'strong_selling'
            elif result['buy_sell_ratio'] < 0.9:
                signals['overall_pressure'] = 'moderate_selling'
            else:
                signals['overall_pressure'] = 'neutral'
            
            # Recent volume pressure
            if 'recent_delta' in result:
                if result['recent_delta'] > 0:
                    signals['recent_pressure'] = 'recent_buying'
                else:
                    signals['recent_pressure'] = 'recent_selling'
            
            # Delta divergence (price vs volume delta)
            last_price_change = data['close'].iloc[-1] - data['close'].iloc[-periods]
            price_up = last_price_change > 0
            delta_up = result['recent_delta'] > 0
            
            if price_up and not delta_up:
                signals['divergence'] = 'bearish_divergence'
            elif not price_up and delta_up:
                signals['divergence'] = 'bullish_divergence'
            else:
                signals['divergence'] = 'no_divergence'
            
            # Add signals to result
            result['signals'] = signals
            
            # Add insights
            insights = []
            
            if signals['overall_pressure'] in ['strong_buying', 'moderate_buying']:
                insights.append(f"Overall buying pressure exceeds selling ({result['buy_sell_ratio']:.2f}x)")
            elif signals['overall_pressure'] in ['strong_selling', 'moderate_selling']:
                insights.append(f"Overall selling pressure exceeds buying ({1/result['buy_sell_ratio']:.2f}x)")
            
            if 'delta_momentum' in result and abs(result['delta_momentum']) > 0:
                if result['delta_momentum'] > 0:
                    insights.append("Buying pressure is accelerating")
                else:
                    insights.append("Selling pressure is accelerating")
            
            if signals['divergence'] == 'bullish_divergence':
                insights.append("Bullish divergence: price down, but buying volume increasing")
            elif signals['divergence'] == 'bearish_divergence':
                insights.append("Bearish divergence: price up, but selling volume increasing")
            
            result['insights'] = insights
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume delta: {str(e)}")
            return {'error': str(e)}

# Standalone functions for direct use

def analyze_volume_profile(df: pd.DataFrame, num_bins: int = 20, value_area_pct: float = 0.7) -> Dict:
    """
    Analyze volume profile for the given OHLCV dataframe
    
    Args:
        df: DataFrame with price and volume data
        num_bins: Number of price bins to divide the price range into
        value_area_pct: Percentage of total volume to include in the value area
        
    Returns:
        Dictionary with volume profile analysis
    """
    analyzer = VolumeProfileAnalyzer(num_bins=num_bins, value_area_pct=value_area_pct)
    return analyzer.analyze_volume_profile(df)

def analyze_volume_by_time(df: pd.DataFrame, intervals: Optional[List[str]] = None) -> Dict:
    """
    Analyze volume distribution by time
    
    Args:
        df: DataFrame with volume and timestamp data
        intervals: Optional list of time intervals to analyze
            
    Returns:
        Dictionary with volume by time analysis
    """
    analyzer = VolumeProfileAnalyzer()
    return analyzer.analyze_volume_by_time(df, intervals)

def detect_volume_anomalies(df: pd.DataFrame, window: int = 20, std_threshold: float = 2.0) -> Dict:
    """
    Detect unusual volume patterns and anomalies
    
    Args:
        df: DataFrame with volume data
        window: Moving average window for baseline volume
        std_threshold: Standard deviation threshold for anomaly detection
        
    Returns:
        Dictionary with volume anomalies
    """
    analyzer = VolumeProfileAnalyzer()
    return analyzer.detect_volume_anomalies(df, window, std_threshold)

def analyze_volume_delta(df: pd.DataFrame) -> Dict:
    """
    Analyze buying vs selling volume pressure (volume delta)
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Dictionary with volume delta analysis
    """
    analyzer = VolumeProfileAnalyzer()
    return analyzer.analyze_volume_delta(df)