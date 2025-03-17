import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [IndicatorCombinations] %(message)s',
    handlers=[
        logging.FileHandler('logs/indicator_combinations.log'),
        logging.StreamHandler()
    ]
)

class IndicatorCombinations:
    """
    Class for creating and calculating combined technical indicators.
    
    Combined indicators can provide more reliable trading signals by:
    1. Reducing false signals through confirmation across multiple indicators
    2. Capturing different market aspects simultaneously
    3. Creating custom indicators optimized for specific market conditions
    4. Improving signal quality through weighted combinations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.combinations = {
            # Trend strength combinations
            "trend_confirmation": self._trend_confirmation,
            "momentum_trend_alignment": self._momentum_trend_alignment,
            "triple_moving_average": self._triple_moving_average,
            
            # Volatility-adjusted indicators
            "volatility_adjusted_momentum": self._volatility_adjusted_momentum,
            "volatility_trend_score": self._volatility_trend_score,
            
            # Oscillator combinations
            "oscillator_consensus": self._oscillator_consensus,
            "stoch_rsi": self._stoch_rsi,
            "double_rsi": self._double_rsi,
            
            # Volume-based combinations
            "volume_weighted_price_momentum": self._volume_weighted_price_momentum,
            "volume_price_confirmation": self._volume_price_confirmation,
            
            # Compound indicators
            "trend_strength_index": self._trend_strength_index,
            "market_regime_indicator": self._market_regime_indicator,
            "reversal_probability": self._reversal_probability,
            "breakout_confirmation": self._breakout_confirmation,
            "divergence_detector": self._divergence_detector
        }
    
    def calculate_combined_indicators(self, market_data: Dict) -> Dict:
        """
        Calculate all combined technical indicators
        
        Args:
            market_data: Dictionary containing technical indicators and price data
            
        Returns:
            Dictionary with calculated combined indicators
        """
        result = {}
        
        try:
            # Verify required data is present
            required_fields = [
                'rsi', 'macd', 'stoch_k', 'williams_r', 'bb_position',
                'price_change_1m', 'price_change_5m', 'trend', 'trend_strength'
            ]
            
            for field in required_fields:
                if field not in market_data:
                    self.logger.warning(f"Missing required field: {field}")
                    return {'error': f"Missing required field: {field}"}
            
            # Calculate each combination
            for name, func in self.combinations.items():
                try:
                    result[name] = func(market_data)
                except Exception as e:
                    self.logger.error(f"Error calculating {name}: {str(e)}")
                    result[name] = None
                    
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating combined indicators: {str(e)}")
            return {'error': str(e)}
    
    # ======= Trend strength combinations =======
    
    def _trend_confirmation(self, data: Dict) -> float:
        """
        Combines MACD and EMA direction to confirm trend
        Range: -1.0 to 1.0 (strong downtrend to strong uptrend)
        """
        macd = data['macd']
        trend_strength = data['trend_strength']
        trend_direction = 1 if data['trend'] == 'uptrend' else (-1 if data['trend'] == 'downtrend' else 0)
        
        # Normalize MACD influence
        macd_factor = np.tanh(macd * 5)  # Scale and bound between -1 and 1
        
        # Combine signals with trend direction as the dominant factor
        confirmation = (0.6 * trend_direction * trend_strength) + (0.4 * macd_factor)
        
        return round(confirmation, 4)
    
    def _momentum_trend_alignment(self, data: Dict) -> float:
        """
        Measures alignment between momentum indicators and trend direction
        Range: 0.0 to 1.0 (no alignment to perfect alignment)
        """
        # Determine if RSI agrees with trend
        rsi = data['rsi']
        rsi_bullish = rsi > 50
        trend_bullish = data['trend'] == 'uptrend'
        
        # Check if MACD agrees with trend
        macd_bullish = data['macd'] > 0
        
        # Check if Williams %R agrees with trend
        williams_bullish = data['williams_r'] > -50
        
        # Count agreements
        agreements = sum([
            1 if rsi_bullish == trend_bullish else 0,
            1 if macd_bullish == trend_bullish else 0,
            1 if williams_bullish == trend_bullish else 0
        ])
        
        # Calculate alignment score - maximum of 3 agreements
        alignment = agreements / 3.0
        
        # Weight by trend strength
        weighted_alignment = alignment * min(1.0, data['trend_strength'])
        
        return round(weighted_alignment, 4)
    
    def _triple_moving_average(self, data: Dict) -> Dict:
        """
        Composite signal based on the alignment of multiple timeframe EMAs
        Returns dict with score and state
        """
        # Use EMAs from different timeframes if available
        ema_short = data.get('ema_12', None)
        ema_medium = data.get('ema_26', None)
        
        # If EMAs are not available, use trend as proxy
        if ema_short is None or ema_medium is None:
            trend = data['trend']
            trend_strength = data['trend_strength']
            
            if trend == 'uptrend':
                score = 0.5 + (trend_strength / 2)
                state = 'bullish' if trend_strength > 0.3 else 'neutral'
            elif trend == 'downtrend':
                score = 0.5 - (trend_strength / 2)
                state = 'bearish' if trend_strength > 0.3 else 'neutral'
            else:
                score = 0.5
                state = 'neutral'
        else:
            # Real calculation based on EMAs
            # Short above medium: bullish, short below medium: bearish
            if ema_short > ema_medium:
                diff_pct = (ema_short - ema_medium) / ema_medium * 100
                score = min(1.0, 0.5 + (diff_pct * 0.1))
                state = 'bullish' if score > 0.7 else ('neutral' if score > 0.55 else 'neutral')
            else:
                diff_pct = (ema_medium - ema_short) / ema_medium * 100
                score = max(0.0, 0.5 - (diff_pct * 0.1))
                state = 'bearish' if score < 0.3 else ('neutral' if score < 0.45 else 'neutral')
        
        return {
            'score': round(score, 4),
            'state': state
        }
    
    # ======= Volatility-adjusted indicators =======
    
    def _volatility_adjusted_momentum(self, data: Dict) -> float:
        """
        Adjusts momentum indicators by recent volatility for adaptive signals
        Range: -1.0 to 1.0 (strong bearish to strong bullish)
        """
        # Estimate volatility from price changes
        price_changes = [
            abs(data['price_change_1m']),
            abs(data['price_change_3m']),
            abs(data['price_change_5m'])
        ]
        volatility = sum(price_changes) / len(price_changes)
        
        # Normalize RSI to -1 to 1 range (50 is neutral)
        rsi_normalized = (data['rsi'] - 50) / 50
        
        # Normalize Williams %R to -1 to 1 range (-50 is neutral)
        williams_normalized = (data['williams_r'] + 50) / 50
        
        # Get momentum signals
        momentum_signals = [
            rsi_normalized,
            williams_normalized,
            np.tanh(data['macd'] * 10)  # Scale MACD into bounded range
        ]
        
        # Weight based on volatility: higher volatility = more importance to momentum signals
        volatility_factor = min(3.0, max(0.5, volatility)) / 3.0
        composite_momentum = sum(momentum_signals) / len(momentum_signals)
        
        # Adjust signal strength based on volatility
        adjusted_momentum = composite_momentum * (0.5 + volatility_factor)
        
        # Bound result
        return round(max(-1.0, min(1.0, adjusted_momentum)), 4)
    
    def _volatility_trend_score(self, data: Dict) -> float:
        """
        Combines BB position with trend for volatility-aware trend signals
        Range: 0.0 to 1.0 (ranging to trending)
        """
        # Get Bollinger Band position (0 to 1)
        bb_pos = data['bb_position']
        
        # Calculate distance from middle of band (0.5)
        band_extremity = abs(bb_pos - 0.5) * 2  # 0 = middle, 1 = edge
        
        # Combine with trend strength
        trend_strength = data['trend_strength']
        
        # Higher score when strong trend with price near bands
        # Lower score when weak trend with price in middle of bands
        combined_score = (0.7 * band_extremity) + (0.3 * trend_strength)
        
        return round(combined_score, 4)
    
    # ======= Oscillator combinations =======
    
    def _oscillator_consensus(self, data: Dict) -> Dict:
        """
        Combines multiple oscillators to determine consensus signals
        Returns dict with signal and strength
        """
        # Set up oscillator buy/sell signals
        rsi = data['rsi']
        williams = data['williams_r']
        stoch = data['stoch_k']
        
        # Define overbought/oversold thresholds
        signals = {
            'rsi': {
                'overbought': rsi > 70,
                'oversold': rsi < 30,
                'strength': min(1.0, max(0.0, abs(rsi - 50) / 30))
            },
            'williams': {
                'overbought': williams > -20,
                'oversold': williams < -80,
                'strength': min(1.0, max(0.0, abs(williams + 50) / 30))
            },
            'stoch': {
                'overbought': stoch > 80,
                'oversold': stoch < 20,
                'strength': min(1.0, max(0.0, abs(stoch - 50) / 30))
            }
        }
        
        # Count signals in each direction
        overbought_count = sum(1 for s in signals.values() if s['overbought'])
        oversold_count = sum(1 for s in signals.values() if s['oversold'])
        
        # Calculate average signal strength
        if overbought_count > 0:
            overbought_strength = sum(s['strength'] for s in signals.values() if s['overbought']) / overbought_count
        else:
            overbought_strength = 0
            
        if oversold_count > 0:
            oversold_strength = sum(s['strength'] for s in signals.values() if s['oversold']) / oversold_count
        else:
            oversold_strength = 0
        
        # Determine consensus signal
        if overbought_count >= 2:
            signal = 'overbought'
            strength = overbought_strength
        elif oversold_count >= 2:
            signal = 'oversold'
            strength = oversold_strength
        else:
            signal = 'neutral'
            strength = 0.0
        
        # Add agreement level (how many oscillators agree)
        if signal == 'overbought':
            agreement = overbought_count / 3.0
        elif signal == 'oversold':
            agreement = oversold_count / 3.0
        else:
            agreement = 0.0
            
        return {
            'signal': signal,
            'strength': round(strength, 4),
            'agreement': round(agreement, 4)
        }
    
    def _stoch_rsi(self, data: Dict) -> float:
        """
        Stochastic RSI - combines features of RSI and Stochastic
        Range: 0.0 to 1.0
        """
        rsi = data['rsi']
        
        # Simple approximation without requiring historical data
        # Maps RSI to 0-1 range with higher sensitivity to mid-range changes
        if rsi <= 30:
            stoch_rsi = rsi / 30
        elif rsi >= 70:
            stoch_rsi = 0.67 + ((rsi - 70) / 30) * 0.33
        else:
            # Map 30-70 range to 0.33-0.67 range with higher slope
            stoch_rsi = 0.33 + ((rsi - 30) / 40) * 0.34
        
        return round(stoch_rsi, 4)
    
    def _double_rsi(self, data: Dict) -> Dict:
        """
        Combines RSI from multiple timeframes for better signals
        Returns dict with signal and divergence
        """
        rsi_1m = data['rsi']  # 1-minute RSI
        rsi_5m = data.get('rsi_5m', None)  # 5-minute RSI if available
        
        # If 5-minute RSI is not available, use 3-minute or approximate
        if rsi_5m is None:
            rsi_5m = data.get('rsi_3m', rsi_1m)  # Fallback to 3-minute or 1-minute
        
        # Calculate divergence between timeframes
        divergence = rsi_1m - rsi_5m
        
        # Determine signal based on both RSIs
        if rsi_1m < 30 and rsi_5m < 30:
            signal = 'strong_oversold'
        elif rsi_1m < 30:
            signal = 'oversold'
        elif rsi_1m > 70 and rsi_5m > 70:
            signal = 'strong_overbought'
        elif rsi_1m > 70:
            signal = 'overbought'
        elif rsi_1m > 50 and rsi_5m > 50:
            signal = 'bullish'
        elif rsi_1m < 50 and rsi_5m < 50:
            signal = 'bearish'
        else:
            signal = 'neutral'
        
        return {
            'signal': signal,
            'divergence': round(divergence, 4)
        }
    
    # ======= Volume-based combinations =======
    
    def _volume_weighted_price_momentum(self, data: Dict) -> float:
        """
        Combines volume and price momentum for stronger signals
        Range: -1.0 to 1.0 (bearish to bullish)
        """
        # Get price changes
        price_change_1m = data['price_change_1m']
        price_change_5m = data['price_change_5m']
        
        # Calculate price momentum (weighted average of timeframes)
        price_momentum = (0.4 * price_change_1m) + (0.6 * price_change_5m)
        
        # Get volume data if available, otherwise use 1.0 as neutral value
        volume = data.get('volume', 1.0)
        avg_volume = data.get('avg_volume', volume)
        
        # Calculate relative volume
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        # Apply volume weighting to momentum
        # Higher volume strengthens the signal
        if price_momentum > 0:
            weighted_momentum = price_momentum * min(2.0, volume_ratio)
        else:
            weighted_momentum = price_momentum * min(2.0, volume_ratio)
        
        # Normalize to -1 to 1 range
        normalized = np.tanh(weighted_momentum / 5.0)  # Scale by 5 to fit typical price changes
        
        return round(normalized, 4)
    
    def _volume_price_confirmation(self, data: Dict) -> Dict:
        """
        Checks if volume confirms price direction
        Returns dict with confirmation status and strength
        """
        # Get price change and volume
        price_change = data['price_change_1m']
        volume = data.get('volume', 0)
        avg_volume = data.get('avg_volume', volume)
        
        # Calculate volume ratio
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        # Determine if volume confirms price movement
        if abs(price_change) < 0.1:  # Small price change
            confirmation = 'neutral'
            strength = 0
        elif price_change > 0 and volume_ratio > 1.2:  # Up move with higher volume
            confirmation = 'strong_bullish'
            strength = min(1.0, (volume_ratio - 1.0))
        elif price_change > 0:  # Up move with normal/low volume
            confirmation = 'weak_bullish'
            strength = max(0.0, min(0.5, (volume_ratio - 0.8) / 0.4))
        elif price_change < 0 and volume_ratio > 1.2:  # Down move with higher volume
            confirmation = 'strong_bearish'
            strength = min(1.0, (volume_ratio - 1.0))
        elif price_change < 0:  # Down move with normal/low volume
            confirmation = 'weak_bearish'
            strength = max(0.0, min(0.5, (volume_ratio - 0.8) / 0.4))
        else:
            confirmation = 'neutral'
            strength = 0
        
        return {
            'confirmation': confirmation,
            'strength': round(strength, 4)
        }
    
    # ======= Compound indicators =======
    
    def _trend_strength_index(self, data: Dict) -> Dict:
        """
        Comprehensive trend strength measure combining multiple indicators
        Returns dict with trend direction, strength, and confidence
        """
        # Get key indicators
        trend = data['trend']
        trend_strength = data['trend_strength']
        rsi = data['rsi']
        macd = data['macd']
        bb_position = data['bb_position']
        
        # Determine trend direction factor (-1 to 1)
        if trend == 'uptrend':
            direction = 1
        elif trend == 'downtrend':
            direction = -1
        else:
            direction = 0
        
        # Calculate RSI contribution (0 to 1)
        if direction > 0:  # Uptrend
            rsi_factor = (rsi - 50) / 50 if rsi > 50 else 0
        elif direction < 0:  # Downtrend
            rsi_factor = (50 - rsi) / 50 if rsi < 50 else 0
        else:  # Sideways
            rsi_factor = 1 - (abs(rsi - 50) / 25)  # Higher near 50
        
        # Calculate MACD contribution (-1 to 1)
        macd_factor = np.tanh(macd * 20)  # Scale to get -1 to 1 range
        
        # Calculate BB contribution (0 to 1)
        if direction > 0:  # Uptrend
            bb_factor = bb_position
        elif direction < 0:  # Downtrend
            bb_factor = 1 - bb_position
        else:  # Sideways
            bb_factor = 1 - abs(bb_position - 0.5) * 2  # Higher in middle
        
        # Combine factors with weights
        combined_strength = (
            0.4 * trend_strength +
            0.25 * rsi_factor +
            0.25 * abs(macd_factor) +
            0.1 * bb_factor
        )
        
        # Calculate confidence based on agreement of indicators
        if direction != 0:
            # For trending markets, check if indicators agree with trend direction
            indicator_direction = 1 if (rsi > 50 and macd > 0) else (-1 if (rsi < 50 and macd < 0) else 0)
            agreement = (direction == np.sign(indicator_direction))
            confidence = 0.5 + (0.5 * agreement)
        else:
            # For sideways markets, higher confidence if indicators are neutral
            neutrality = (abs(rsi - 50) < 10) and (abs(macd) < 0.0005)
            confidence = 0.5 + (0.3 * neutrality)
        
        return {
            'direction': direction,
            'strength': round(combined_strength, 4),
            'confidence': round(confidence, 4)
        }
    
    def _market_regime_indicator(self, data: Dict) -> Dict:
        """
        Identifies market regime (trending, ranging, volatile)
        Returns dict with regime and confidence
        """
        # Get indicators
        trend_strength = data['trend_strength']
        bb_position = data['bb_position']
        price_changes = [
            abs(data['price_change_1m']),
            abs(data['price_change_3m']),
            abs(data['price_change_5m'])
        ]
        
        # Calculate volatility
        volatility = sum(price_changes) / len(price_changes)
        
        # Determine regime
        if trend_strength > 0.6:
            regime = 'trending'
            confidence = min(1.0, trend_strength * 1.1)
        elif volatility > 2.0:
            regime = 'volatile'
            confidence = min(1.0, volatility / 3.0)
        else:
            regime = 'ranging'
            # Higher confidence when price is in middle of bands and trend is weak
            range_evidence = (1 - trend_strength) * (1 - abs(bb_position - 0.5) * 2)
            confidence = min(1.0, 0.5 + range_evidence)
        
        return {
            'regime': regime,
            'confidence': round(confidence, 4)
        }
    
    def _reversal_probability(self, data: Dict) -> Dict:
        """
        Estimates probability of trend reversal
        Returns dict with probability and signals
        """
        # Get key indicators
        trend = data['trend']
        rsi = data['rsi']
        williams_r = data['williams_r']
        bb_position = data['bb_position']
        
        # Initialize probability and signals
        probability = 0.0
        signals = []
        
        # RSI extreme levels
        if trend == 'uptrend' and rsi > 70:
            probability += 0.25
            signals.append('rsi_overbought')
        elif trend == 'downtrend' and rsi < 30:
            probability += 0.25
            signals.append('rsi_oversold')
        
        # Williams %R extreme levels
        if trend == 'uptrend' and williams_r > -20:
            probability += 0.2
            signals.append('williams_overbought')
        elif trend == 'downtrend' and williams_r < -80:
            probability += 0.2
            signals.append('williams_oversold')
        
        # BB position extreme (price near bands)
        if trend == 'uptrend' and bb_position > 0.9:
            probability += 0.15
            signals.append('price_near_upper_band')
        elif trend == 'downtrend' and bb_position < 0.1:
            probability += 0.15
            signals.append('price_near_lower_band')
        
        # Divergences (simplified - would be more accurate with historical data)
        if trend == 'uptrend' and rsi < 60:
            probability += 0.2
            signals.append('potential_bearish_divergence')
        elif trend == 'downtrend' and rsi > 40:
            probability += 0.2
            signals.append('potential_bullish_divergence')
        
        # Cap probability
        probability = min(0.95, probability)
        
        return {
            'probability': round(probability, 4),
            'signals': signals
        }
    
    def _breakout_confirmation(self, data: Dict) -> Dict:
        """
        Confirms price breakouts using multiple indicators
        Returns dict with breakout direction and confirmation level
        """
        # Get relevant data
        price_change_5m = data['price_change_5m']
        bb_position = data['bb_position']
        rsi = data['rsi']
        
        # Determine if there's a potential breakout
        breakout_direction = 0
        confirmation = 0.0
        
        if price_change_5m > 1.0 and bb_position > 0.8:
            # Potential upward breakout
            breakout_direction = 1
            # Stronger confirmation with higher RSI
            confirmation = 0.5 + (0.5 * min(1.0, (rsi - 50) / 30))
        elif price_change_5m < -1.0 and bb_position < 0.2:
            # Potential downward breakout
            breakout_direction = -1
            # Stronger confirmation with lower RSI
            confirmation = 0.5 + (0.5 * min(1.0, (50 - rsi) / 30))
        
        # Classify the breakout
        if breakout_direction == 0:
            status = 'none'
        elif confirmation > 0.8:
            status = 'strong_' + ('bullish' if breakout_direction > 0 else 'bearish')
        elif confirmation > 0.5:
            status = 'confirmed_' + ('bullish' if breakout_direction > 0 else 'bearish')
        else:
            status = 'potential_' + ('bullish' if breakout_direction > 0 else 'bearish')
        
        return {
            'direction': breakout_direction,
            'confirmation': round(confirmation, 4),
            'status': status
        }
    
    def _divergence_detector(self, data: Dict) -> Dict:
        """
        Detects potential divergences between price and indicators
        Returns dict with divergence type and strength
        """
        # Note: This is a simplified version that approximates divergences
        # For accurate divergence detection, historical data analysis is needed
        
        # Get relevant data
        trend = data['trend']
        price_change_5m = data['price_change_5m']
        rsi = data['rsi']
        macd = data['macd']
        
        # Initialize result
        divergence = 'none'
        strength = 0.0
        
        # Check for RSI divergence
        if trend == 'uptrend' and price_change_5m > 0 and rsi < 50:
            divergence = 'bearish_rsi'
            strength = 0.5 + (0.5 * (1 - (rsi / 50)))
        elif trend == 'downtrend' and price_change_5m < 0 and rsi > 50:
            divergence = 'bullish_rsi'
            strength = 0.5 + (0.5 * ((rsi - 50) / 50))
        
        # Check for MACD divergence (prioritize if stronger)
        macd_strength = 0.0
        if trend == 'uptrend' and price_change_5m > 0 and macd < 0:
            macd_divergence = 'bearish_macd'
            macd_strength = 0.6 + (0.4 * min(1.0, abs(macd) * 1000))
        elif trend == 'downtrend' and price_change_5m < 0 and macd > 0:
            macd_divergence = 'bullish_macd'
            macd_strength = 0.6 + (0.4 * min(1.0, abs(macd) * 1000))
            
        # Use the stronger divergence
        if macd_strength > strength:
            divergence = macd_divergence
            strength = macd_strength
        
        return {
            'divergence': divergence,
            'strength': round(strength, 4)
        }


# Standalone function for calculating all indicator combinations
def calculate_indicator_combinations(market_data: Dict) -> Dict:
    """
    Calculate all technical indicator combinations from market data
    
    Args:
        market_data: Dictionary containing technical indicators and price data
        
    Returns:
        Dictionary with calculated combined indicators
    """
    indicator_combinations = IndicatorCombinations()
    return indicator_combinations.calculate_combined_indicators(market_data)