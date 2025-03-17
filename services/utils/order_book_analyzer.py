import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import linregress
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

class OrderBookAnalyzer:
    """
    Utility class for analyzing order book depth data.
    
    This analyzer provides insights from order book data including:
    - Liquidity imbalances
    - Support/resistance level detection
    - Order clustering analysis
    - Potential price impact estimation
    - Buy/sell pressure metrics
    - Market microstructure patterns
    """
    
    def __init__(self, config):
        """
        Initialize the OrderBookAnalyzer with configuration.
        
        Args:
            config (dict): Configuration dictionary with order_book_analysis settings
        """
        self.config = config.get('order_book_analysis', {})
        self.logger = logging.getLogger(__name__)
        
        # Default configurations if not provided
        self.max_levels = self.config.get('max_levels', 20)
        self.significant_level_threshold = self.config.get('significant_level_threshold', 2.0)
        self.clustering_enabled = self.config.get('clustering_enabled', True)
        self.n_clusters = self.config.get('n_clusters', 3)
        self.smoothing_window = self.config.get('smoothing_window', 3)
        self.impact_price_levels = self.config.get('impact_price_levels', 5)
        self.support_resistance_window = self.config.get('support_resistance_window', 5)
        self.min_order_size = self.config.get('min_order_size', 1000.0)  # Minimum order size to consider significant
        self.imbalance_threshold = self.config.get('imbalance_threshold', 0.2)  # 20% imbalance threshold
        self.store_historical = self.config.get('store_historical', True)
        self.historical_metrics = []
        
    def analyze_order_book(self, bids, asks, current_price):
        """
        Analyze order book data to extract trading insights.
        
        Args:
            bids (list): List of [price, quantity] pairs for buy orders
            asks (list): List of [price, quantity] pairs for sell orders
            current_price (float): Current market price
            
        Returns:
            dict: Analysis results containing various order book metrics
        """
        try:
            # Prepare data
            bids_df = pd.DataFrame(bids[:self.max_levels], columns=['price', 'quantity'])
            asks_df = pd.DataFrame(asks[:self.max_levels], columns=['price', 'quantity'])
            
            # Calculate basic metrics
            bid_volume = bids_df['quantity'].sum()
            ask_volume = asks_df['quantity'].sum()
            
            # Calculate cumulative volumes
            bids_df['cumulative'] = bids_df['quantity'].cumsum()
            asks_df['cumulative'] = asks_df['quantity'].cumsum()
            
            # Calculate value (price * quantity) at each level
            bids_df['value'] = bids_df['price'] * bids_df['quantity']
            asks_df['value'] = asks_df['price'] * asks_df['quantity']
            
            # Calculate price distances from current price
            bids_df['distance'] = (current_price - bids_df['price']) / current_price * 100
            asks_df['distance'] = (asks_df['price'] - current_price) / current_price * 100
            
            # Distance-weighted volumes
            bids_df['weighted_volume'] = bids_df['quantity'] / (bids_df['distance'] + 0.01)
            asks_df['weighted_volume'] = asks_df['quantity'] / (asks_df['distance'] + 0.01)
            
            # Calculate liquidity metrics
            results = {
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'bid_ask_ratio': bid_volume / max(ask_volume, 0.0001),  # Avoid division by zero
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'total_volume': bid_volume + ask_volume,
                'liquidity_imbalance': (bid_volume - ask_volume) / (bid_volume + ask_volume),
                'average_bid_size': bids_df['quantity'].mean(),
                'average_ask_size': asks_df['quantity'].mean(),
                'spread': asks_df['price'].iloc[0] - bids_df['price'].iloc[0],
                'spread_percentage': (asks_df['price'].iloc[0] - bids_df['price'].iloc[0]) / current_price * 100
            }
            
            # Identify price impact for standard trade sizes
            results.update(self._calculate_price_impact(bids_df, asks_df, current_price))
            
            # Identify support and resistance levels
            results.update(self._identify_support_resistance(bids_df, asks_df))
            
            # Analyze order clusters
            if self.clustering_enabled:
                results.update(self._analyze_order_clusters(bids_df, asks_df))
            
            # Calculate buy/sell pressure metrics
            results.update(self._calculate_pressure_metrics(bids_df, asks_df, current_price))
            
            # Store historical data if enabled
            if self.store_historical:
                self._update_historical_metrics(results)
            
            # Calculate market microstructure patterns
            results.update(self._analyze_microstructure(bids_df, asks_df))
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error analyzing order book: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price
            }
    
    def _calculate_price_impact(self, bids_df, asks_df, current_price):
        """
        Calculate potential price impact of various order sizes.
        
        Returns:
            dict: Price impact metrics
        """
        results = {}
        
        # Define standard order sizes to analyze in USD
        trade_sizes = self.config.get('trade_sizes', [10000, 50000, 100000, 500000, 1000000])
        
        # Calculate impact for buys (asks)
        buy_impacts = {}
        for size in trade_sizes:
            impact = self._calculate_single_impact(asks_df, size, current_price, 'buy')
            buy_impacts[f"buy_impact_{size}"] = impact
        
        # Calculate impact for sells (bids)
        sell_impacts = {}
        for size in trade_sizes:
            impact = self._calculate_single_impact(bids_df, size, current_price, 'sell')
            sell_impacts[f"sell_impact_{size}"] = impact
        
        # Calculate size needed to move price by 1%
        size_for_1pct_up = self._size_for_price_move(asks_df, current_price, 1.0)
        size_for_1pct_down = self._size_for_price_move(bids_df, current_price, -1.0)
        
        results.update({
            'buy_impacts': buy_impacts,
            'sell_impacts': sell_impacts,
            'size_for_1pct_up': size_for_1pct_up,
            'size_for_1pct_down': size_for_1pct_down,
            'buy_sell_impact_ratio': sum(buy_impacts.values()) / max(sum(sell_impacts.values()), 0.0001)
        })
        
        return results
    
    def _calculate_single_impact(self, df, size_usd, current_price, direction):
        """
        Calculate price impact for a single trade size.
        
        Args:
            df (DataFrame): Order book dataframe (bids or asks)
            size_usd (float): Trade size in USD
            current_price (float): Current market price
            direction (str): 'buy' or 'sell'
            
        Returns:
            float: Percentage price impact
        """
        remaining_size = size_usd
        impact_price = current_price
        
        if direction == 'buy':
            # For buys, we consume asks from lowest to highest
            for _, row in df.iterrows():
                order_value = row['price'] * row['quantity']
                if remaining_size <= order_value:
                    # This order can fill our remaining size
                    impact_price = row['price']
                    break
                else:
                    # Consume this order and continue
                    remaining_size -= order_value
            
            # Calculate percentage impact
            impact_pct = (impact_price - current_price) / current_price * 100
        else:
            # For sells, we consume bids from highest to lowest
            for _, row in df.iterrows():
                order_value = row['price'] * row['quantity']
                if remaining_size <= order_value:
                    # This order can fill our remaining size
                    impact_price = row['price']
                    break
                else:
                    # Consume this order and continue
                    remaining_size -= order_value
            
            # Calculate percentage impact
            impact_pct = (current_price - impact_price) / current_price * 100
        
        return impact_pct
    
    def _size_for_price_move(self, df, current_price, target_pct):
        """
        Calculate the order size needed to move price by target_pct.
        
        Args:
            df (DataFrame): Order book dataframe (bids or asks)
            current_price (float): Current market price
            target_pct (float): Target percentage move
            
        Returns:
            float: Order size in USD needed to move price by target_pct
        """
        if target_pct == 0:
            return 0
        
        target_price = current_price * (1 + target_pct / 100)
        size_needed = 0
        
        if target_pct > 0:
            # For price increase, we consume asks
            for _, row in df.iterrows():
                if row['price'] >= target_price:
                    break
                size_needed += row['price'] * row['quantity']
        else:
            # For price decrease, we consume bids
            for _, row in df.iterrows():
                if row['price'] <= target_price:
                    break
                size_needed += row['price'] * row['quantity']
        
        return size_needed
    
    def _identify_support_resistance(self, bids_df, asks_df):
        """
        Identify support and resistance levels from order book.
        
        Returns:
            dict: Support and resistance levels with their strengths
        """
        # Smooth volumes to reduce noise
        bids_smoothed = bids_df['quantity'].rolling(window=self.smoothing_window, min_periods=1).mean()
        asks_smoothed = asks_df['quantity'].rolling(window=self.smoothing_window, min_periods=1).mean()
        
        # Calculate volume derivatives to identify spikes
        bid_derivatives = np.gradient(bids_smoothed)
        ask_derivatives = np.gradient(asks_smoothed)
        
        # Identify significant levels (volume spikes)
        bid_std = np.std(bids_df['quantity'])
        ask_std = np.std(asks_df['quantity'])
        
        # Support levels (from bids)
        support_levels = []
        for i, (_, row) in enumerate(bids_df.iterrows()):
            if i < len(bid_derivatives) and row['quantity'] > self.significant_level_threshold * bid_std:
                support_levels.append({
                    'price': row['price'],
                    'strength': row['quantity'] / bid_std,
                    'volume': row['quantity']
                })
        
        # Resistance levels (from asks)
        resistance_levels = []
        for i, (_, row) in enumerate(asks_df.iterrows()):
            if i < len(ask_derivatives) and row['quantity'] > self.significant_level_threshold * ask_std:
                resistance_levels.append({
                    'price': row['price'],
                    'strength': row['quantity'] / ask_std,
                    'volume': row['quantity']
                })
        
        # Sort by strength
        support_levels = sorted(support_levels, key=lambda x: x['strength'], reverse=True)[:self.support_resistance_window]
        resistance_levels = sorted(resistance_levels, key=lambda x: x['strength'], reverse=True)[:self.support_resistance_window]
        
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels
        }
    
    def _analyze_order_clusters(self, bids_df, asks_df):
        """
        Analyze order clusters using KMeans clustering.
        
        Returns:
            dict: Cluster analysis results
        """
        results = {}
        
        if len(bids_df) > self.n_clusters and len(asks_df) > self.n_clusters:
            try:
                # Prepare bid data for clustering
                bid_features = bids_df[['price', 'quantity']].copy()
                bid_features = StandardScaler().fit_transform(bid_features)
                
                # Apply KMeans clustering
                bid_kmeans = KMeans(n_clusters=self.n_clusters, random_state=42).fit(bid_features)
                bid_clusters = []
                
                # Get cluster centers in original scale
                bid_centers = StandardScaler().fit(bids_df[['price', 'quantity']]).inverse_transform(bid_kmeans.cluster_centers_)
                
                for i in range(self.n_clusters):
                    cluster_mask = bid_kmeans.labels_ == i
                    cluster_points = bids_df[cluster_mask]
                    
                    if not cluster_points.empty:
                        bid_clusters.append({
                            'center_price': bid_centers[i][0],
                            'center_quantity': bid_centers[i][1],
                            'total_volume': cluster_points['quantity'].sum(),
                            'points': len(cluster_points),
                            'avg_price': cluster_points['price'].mean()
                        })
                
                # Same for asks
                ask_features = asks_df[['price', 'quantity']].copy()
                ask_features = StandardScaler().fit_transform(ask_features)
                
                ask_kmeans = KMeans(n_clusters=self.n_clusters, random_state=42).fit(ask_features)
                ask_clusters = []
                
                ask_centers = StandardScaler().fit(asks_df[['price', 'quantity']]).inverse_transform(ask_kmeans.cluster_centers_)
                
                for i in range(self.n_clusters):
                    cluster_mask = ask_kmeans.labels_ == i
                    cluster_points = asks_df[cluster_mask]
                    
                    if not cluster_points.empty:
                        ask_clusters.append({
                            'center_price': ask_centers[i][0],
                            'center_quantity': ask_centers[i][1],
                            'total_volume': cluster_points['quantity'].sum(),
                            'points': len(cluster_points),
                            'avg_price': cluster_points['price'].mean()
                        })
                
                # Sort clusters by volume
                bid_clusters = sorted(bid_clusters, key=lambda x: x['total_volume'], reverse=True)
                ask_clusters = sorted(ask_clusters, key=lambda x: x['total_volume'], reverse=True)
                
                results['bid_clusters'] = bid_clusters
                results['ask_clusters'] = ask_clusters
                
                # Calculate cluster metrics
                results['cluster_imbalance'] = (
                    sum(c['total_volume'] for c in bid_clusters) - 
                    sum(c['total_volume'] for c in ask_clusters)
                ) / max(sum(c['total_volume'] for c in bid_clusters + ask_clusters), 0.0001)
                
                # Identify dominant clusters
                results['dominant_bid_cluster'] = bid_clusters[0] if bid_clusters else None
                results['dominant_ask_cluster'] = ask_clusters[0] if ask_clusters else None
                
            except Exception as e:
                self.logger.warning(f"Clustering analysis failed: {str(e)}")
                results['clustering_error'] = str(e)
        
        return results
    
    def _calculate_pressure_metrics(self, bids_df, asks_df, current_price):
        """
        Calculate buy/sell pressure metrics.
        
        Returns:
            dict: Pressure metrics
        """
        # Calculate near-term buy pressure (asks close to current price)
        near_asks = asks_df[asks_df['distance'] <= self.config.get('near_threshold', 1.0)]
        near_bids = bids_df[bids_df['distance'] <= self.config.get('near_threshold', 1.0)]
        
        # Calculate buy/sell walls
        max_bid_idx = bids_df['quantity'].idxmax() if not bids_df.empty else None
        max_ask_idx = asks_df['quantity'].idxmax() if not asks_df.empty else None
        
        bid_wall = {
            'price': bids_df.loc[max_bid_idx, 'price'] if max_bid_idx is not None else None,
            'quantity': bids_df.loc[max_bid_idx, 'quantity'] if max_bid_idx is not None else None,
            'distance': bids_df.loc[max_bid_idx, 'distance'] if max_bid_idx is not None else None
        } if max_bid_idx is not None else None
        
        ask_wall = {
            'price': asks_df.loc[max_ask_idx, 'price'] if max_ask_idx is not None else None,
            'quantity': asks_df.loc[max_ask_idx, 'quantity'] if max_ask_idx is not None else None,
            'distance': asks_df.loc[max_ask_idx, 'distance'] if max_ask_idx is not None else None
        } if max_ask_idx is not None else None
        
        # Bid/ask walls with significantly large orders
        bid_wall_significant = bid_wall and bid_wall['quantity'] > self.min_order_size
        ask_wall_significant = ask_wall and ask_wall['quantity'] > self.min_order_size
        
        # Calculate pressure from walls
        wall_pressure = 0
        if bid_wall_significant and ask_wall_significant:
            # Compare walls to determine pressure direction
            if bid_wall['quantity'] > ask_wall['quantity'] * (1 + self.imbalance_threshold):
                wall_pressure = 1  # Buy pressure from bid wall
            elif ask_wall['quantity'] > bid_wall['quantity'] * (1 + self.imbalance_threshold):
                wall_pressure = -1  # Sell pressure from ask wall
        elif bid_wall_significant:
            wall_pressure = 1
        elif ask_wall_significant:
            wall_pressure = -1
        
        # Calculate regression slope for bid/ask curves
        bid_slope = 0
        if len(bids_df) > 2:
            bid_slope_result = linregress(bids_df['distance'].values, bids_df['quantity'].values)
            bid_slope = bid_slope_result.slope
        
        ask_slope = 0
        if len(asks_df) > 2:
            ask_slope_result = linregress(asks_df['distance'].values, asks_df['quantity'].values)
            ask_slope = ask_slope_result.slope
        
        # Calculate overall pressure metrics
        return {
            'near_bid_volume': near_bids['quantity'].sum() if not near_bids.empty else 0,
            'near_ask_volume': near_asks['quantity'].sum() if not near_asks.empty else 0,
            'near_pressure': (
                (near_bids['quantity'].sum() if not near_bids.empty else 0) - 
                (near_asks['quantity'].sum() if not near_asks.empty else 0)
            ) / max(
                (near_bids['quantity'].sum() if not near_bids.empty else 0) + 
                (near_asks['quantity'].sum() if not near_asks.empty else 0), 
                0.0001
            ),
            'bid_wall': bid_wall,
            'ask_wall': ask_wall,
            'wall_pressure': wall_pressure,
            'bid_slope': bid_slope,
            'ask_slope': ask_slope,
            'slope_ratio': bid_slope / max(abs(ask_slope), 0.0001) * (1 if ask_slope < 0 else -1)
        }
    
    def _update_historical_metrics(self, results):
        """
        Update historical metrics for trend analysis.
        
        Args:
            results (dict): Current analysis results
        """
        # Store only the key metrics for historical tracking
        historical_entry = {
            'timestamp': results['timestamp'],
            'current_price': results['current_price'],
            'bid_ask_ratio': results['bid_ask_ratio'],
            'liquidity_imbalance': results['liquidity_imbalance'],
            'spread_percentage': results['spread_percentage'],
            'near_pressure': results.get('near_pressure', 0),
            'wall_pressure': results.get('wall_pressure', 0)
        }
        
        self.historical_metrics.append(historical_entry)
        
        # Limit the size of the historical data
        max_history = self.config.get('max_history_size', 100)
        if len(self.historical_metrics) > max_history:
            self.historical_metrics = self.historical_metrics[-max_history:]
    
    def _analyze_microstructure(self, bids_df, asks_df):
        """
        Analyze market microstructure patterns.
        
        Returns:
            dict: Microstructure analysis
        """
        # Identify spoofing patterns (large orders far from midpoint)
        far_threshold = self.config.get('far_threshold', 5.0)  # 5% from current price
        spoof_candidate_bids = bids_df[bids_df['distance'] > far_threshold]
        spoof_candidate_asks = asks_df[asks_df['distance'] > far_threshold]
        
        # Check for large orders
        large_order_threshold = self.config.get('large_order_multiplier', 5.0) * bids_df['quantity'].mean()
        
        potential_spoof_bids = spoof_candidate_bids[spoof_candidate_bids['quantity'] > large_order_threshold]
        potential_spoof_asks = spoof_candidate_asks[spoof_candidate_asks['quantity'] > large_order_threshold]
        
        # Iceberg detection (check for repeated same-sized orders)
        bid_value_counts = bids_df['quantity'].value_counts()
        ask_value_counts = asks_df['quantity'].value_counts()
        
        potential_iceberg_bids = [
            (val, count) for val, count in bid_value_counts.items() 
            if count >= self.config.get('iceberg_repeat_threshold', 3)
        ]
        potential_iceberg_asks = [
            (val, count) for val, count in ask_value_counts.items() 
            if count >= self.config.get('iceberg_repeat_threshold', 3)
        ]
        
        # Calculate depth evenness (how evenly distributed is liquidity)
        bid_gini = self._calculate_gini(bids_df['quantity'])
        ask_gini = self._calculate_gini(asks_df['quantity'])
        
        return {
            'potential_spoofing': {
                'bids': len(potential_spoof_bids),
                'asks': len(potential_spoof_asks)
            },
            'potential_icebergs': {
                'bids': potential_iceberg_bids[:3],  # Top 3 suspected iceberg orders
                'asks': potential_iceberg_asks[:3]
            },
            'liquidity_distribution': {
                'bid_gini': bid_gini,  # Higher = more concentrated, lower = more distributed
                'ask_gini': ask_gini
            },
            'microstructure_signals': self._interpret_microstructure(
                bid_gini, ask_gini, 
                len(potential_spoof_bids), len(potential_spoof_asks),
                potential_iceberg_bids, potential_iceberg_asks
            )
        }
    
    def _calculate_gini(self, series):
        """
        Calculate Gini coefficient to measure concentration.
        
        Args:
            series (Series): Data series
            
        Returns:
            float: Gini coefficient (0 = perfectly even, 1 = perfectly concentrated)
        """
        if series.empty or series.sum() == 0:
            return 0
        
        array = series.values
        array = array.flatten()
        if np.amin(array) < 0:
            array -= np.amin(array)
        array += 0.0000001
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))
    
    def _interpret_microstructure(self, bid_gini, ask_gini, spoof_bids, spoof_asks, 
                                iceberg_bids, iceberg_asks):
        """
        Interpret microstructure patterns into usable signals.
        
        Returns:
            dict: Dict of signals with their strengths
        """
        signals = {}
        
        # Interpret gini coefficients
        bid_concentration = "high" if bid_gini > 0.6 else "medium" if bid_gini > 0.4 else "low"
        ask_concentration = "high" if ask_gini > 0.6 else "medium" if ask_gini > 0.4 else "low"
        
        # Asymmetry in concentration might indicate sentiment
        gini_asymmetry = bid_gini - ask_gini
        if abs(gini_asymmetry) > 0.2:
            signals['liquidity_concentration'] = {
                'signal': 'bullish' if gini_asymmetry < 0 else 'bearish',
                'strength': abs(gini_asymmetry),
                'interpretation': (
                    "Liquidity more evenly distributed on bid side than ask side, possibly bullish" 
                    if gini_asymmetry < 0 else 
                    "Liquidity more evenly distributed on ask side than bid side, possibly bearish"
                )
            }
        
        # Spoofing patterns
        if spoof_bids > 0 or spoof_asks > 0:
            signals['potential_spoofing'] = {
                'signal': 'bullish' if spoof_asks > spoof_bids else 'bearish',
                'strength': min(1.0, (spoof_asks - spoof_bids) / 10),
                'interpretation': (
                    f"Potential spoofing detected with {spoof_asks} ask-side and {spoof_bids} bid-side candidates"
                )
            }
        
        # Iceberg signals
        has_iceberg_bids = len(iceberg_bids) > 0
        has_iceberg_asks = len(iceberg_asks) > 0
        
        if has_iceberg_bids or has_iceberg_asks:
            # Calculate total volume in potential icebergs
            bid_iceberg_volume = sum(qty * count for qty, count in iceberg_bids)
            ask_iceberg_volume = sum(qty * count for qty, count in iceberg_asks)
            
            signals['potential_icebergs'] = {
                'signal': 'bullish' if bid_iceberg_volume > ask_iceberg_volume else 'bearish',
                'strength': min(1.0, abs(bid_iceberg_volume - ask_iceberg_volume) / max(bid_iceberg_volume + ask_iceberg_volume, 1)),
                'interpretation': (
                    f"Potential iceberg orders detected with volume {bid_iceberg_volume:.2f} (bids) vs {ask_iceberg_volume:.2f} (asks)"
                )
            }
        
        return signals
    
    def get_historical_trends(self):
        """
        Calculate trends from historical metrics.
        
        Returns:
            dict: Trend analysis of key metrics
        """
        if not self.historical_metrics or len(self.historical_metrics) < 2:
            return {'error': 'Not enough historical data for trend analysis'}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.historical_metrics)
        
        # Calculate trends
        trends = {}
        for metric in ['bid_ask_ratio', 'liquidity_imbalance', 'spread_percentage', 'near_pressure']:
            if metric in df.columns:
                # Calculate trend direction and strength
                first_half = df[metric].iloc[:len(df)//2].mean()
                second_half = df[metric].iloc[len(df)//2:].mean()
                
                change = second_half - first_half
                change_pct = change / max(abs(first_half), 0.0001) * 100
                
                # Calculate linear trend
                if len(df) >= 3:
                    x = np.arange(len(df))
                    y = df[metric].values
                    slope, _, r_value, _, _ = linregress(x, y)
                    trend_strength = abs(r_value)
                    trend_direction = "increasing" if slope > 0 else "decreasing"
                else:
                    trend_strength = abs(change_pct) / 100
                    trend_direction = "increasing" if change > 0 else "decreasing"
                
                trends[metric] = {
                    'direction': trend_direction,
                    'strength': trend_strength,
                    'change_pct': change_pct,
                    'current': df[metric].iloc[-1],
                    'average': df[metric].mean()
                }
        
        # Overall trend score (-1 to 1, negative = bearish, positive = bullish)
        trend_score = 0
        if 'liquidity_imbalance' in trends:
            trend_score += trends['liquidity_imbalance']['change_pct'] / 100
        if 'near_pressure' in trends:
            trend_score += trends['near_pressure']['change_pct'] / 100
        
        trend_score = max(-1, min(1, trend_score / 2))  # Average and clamp
        
        trends['overall'] = {
            'score': trend_score,
            'interpretation': "bullish" if trend_score > 0.2 else "bearish" if trend_score < -0.2 else "neutral",
            'strength': abs(trend_score)
        }
        
        return trends
    
    def generate_trading_signals(self, analysis_results):
        """
        Generate trading signals from order book analysis.
        
        Args:
            analysis_results (dict): Results from analyze_order_book
            
        Returns:
            dict: Trading signals with confidence levels
        """
        signals = {}
        
        # Extract key metrics
        liquidity_imbalance = analysis_results.get('liquidity_imbalance', 0)
        near_pressure = analysis_results.get('near_pressure', 0)
        wall_pressure = analysis_results.get('wall_pressure', 0)
        
        # Get the strongest support and resistance levels
        support_levels = analysis_results.get('support_levels', [])
        resistance_levels = analysis_results.get('resistance_levels', [])
        
        strongest_support = support_levels[0] if support_levels else None
        strongest_resistance = resistance_levels[0] if resistance_levels else None
        
        current_price = analysis_results.get('current_price', 0)
        
        # 1. Liquidity imbalance signal
        liquidity_threshold = self.config.get('liquidity_signal_threshold', 0.2)
        if abs(liquidity_imbalance) > liquidity_threshold:
            signals['liquidity_imbalance'] = {
                'signal': 'buy' if liquidity_imbalance > 0 else 'sell',
                'confidence': min(1.0, abs(liquidity_imbalance) / 0.5),  # Scale to max 1.0
                'reasoning': (
                    f"Strong {'buy' if liquidity_imbalance > 0 else 'sell'} signal from "
                    f"liquidity imbalance of {liquidity_imbalance:.2f}"
                )
            }
        
        # 2. Near pressure signal
        pressure_threshold = self.config.get('pressure_signal_threshold', 0.15)
        if abs(near_pressure) > pressure_threshold:
            signals['near_pressure'] = {
                'signal': 'buy' if near_pressure > 0 else 'sell',
                'confidence': min(1.0, abs(near_pressure) / 0.3),
                'reasoning': (
                    f"{'Buy' if near_pressure > 0 else 'Sell'} pressure near current price "
                    f"with imbalance of {near_pressure:.2f}"
                )
            }
        
        # 3. Wall signal
        if wall_pressure != 0:
            signals['wall_pressure'] = {
                'signal': 'buy' if wall_pressure > 0 else 'sell',
                'confidence': 0.7,  # Walls are significant signals
                'reasoning': f"{'Buy' if wall_pressure > 0 else 'Sell'} signal from {'bid' if wall_pressure > 0 else 'ask'} wall"
            }
        
        # 4. Support/Resistance proximity
        if strongest_support and strongest_resistance:
            support_distance = (current_price - strongest_support['price']) / current_price * 100
            resistance_distance = (strongest_resistance['price'] - current_price) / current_price * 100
            
            # Close to support or resistance
            proximity_threshold = self.config.get('level_proximity_threshold', 1.0)  # 1%
            
            if support_distance < proximity_threshold:
                signals['support_proximity'] = {
                    'signal': 'buy',
                    'confidence': 0.5 + (0.5 * (1 - support_distance / proximity_threshold)),
                    'reasoning': f"Price near strong support at {strongest_support['price']:.4f}"
                }
            elif resistance_distance < proximity_threshold:
                signals['resistance_proximity'] = {
                    'signal': 'sell',
                    'confidence': 0.5 + (0.5 * (1 - resistance_distance / proximity_threshold)),
                    'reasoning': f"Price near strong resistance at {strongest_resistance['price']:.4f}"
                }
        
        # 5. Price impact asymmetry signal
        buy_impacts = analysis_results.get('buy_impacts', {})
        sell_impacts = analysis_results.get('sell_impacts', {})
        
        if buy_impacts and sell_impacts:
            # Compare impacts of equal-sized trades
            keys = [k for k in buy_impacts.keys() if k.replace('buy_impact_', 'sell_impact_') in sell_impacts]
            
            buy_avg_impact = sum(buy_impacts[k] for k in keys) / len(keys) if keys else 0
            sell_avg_impact = sum(sell_impacts[k.replace('buy_impact_', 'sell_impact_')] for k in keys) / len(keys) if keys else 0
            
            impact_ratio = buy_avg_impact / max(sell_avg_impact, 0.0001)
            
            impact_threshold = self.config.get('impact_signal_threshold', 1.5)
            if impact_ratio > impact_threshold:
                signals['price_impact'] = {
                    'signal': 'sell',
                    'confidence': min(1.0, (impact_ratio - impact_threshold) / 2 + 0.5),
                    'reasoning': f"Selling has less market impact than buying (ratio: {impact_ratio:.2f})"
                }
            elif impact_ratio < 1/impact_threshold:
                signals['price_impact'] = {
                    'signal': 'buy',
                    'confidence': min(1.0, (1/impact_ratio - impact_threshold) / 2 + 0.5),
                    'reasoning': f"Buying has less market impact than selling (ratio: {1/impact_ratio:.2f})"
                }
        
        # 6. Microstructure signals
        micro_signals = analysis_results.get('microstructure_signals', {})
        for key, data in micro_signals.items():
            signals[f"microstructure_{key}"] = {
                'signal': 'buy' if data['signal'] == 'bullish' else 'sell',
                'confidence': data['strength'],
                'reasoning': data['interpretation']
            }
        
        # Calculate overall signal
        if signals:
            buy_signals = [s for s in signals.values() if s['signal'] == 'buy']
            sell_signals = [s for s in signals.values() if s['signal'] == 'sell']
            
            buy_confidence = sum(s['confidence'] for s in buy_signals) / len(buy_signals) if buy_signals else 0
            sell_confidence = sum(s['confidence'] for s in sell_signals) / len(sell_signals) if sell_signals else 0
            
            # Calculate net signal (-1 to 1)
            if buy_confidence > 0 or sell_confidence > 0:
                net_signal = (buy_confidence * len(buy_signals) - sell_confidence * len(sell_signals)) / (len(buy_signals) + len(sell_signals))
            else:
                net_signal = 0
            
            signal_threshold = self.config.get('overall_signal_threshold', 0.2)
            if net_signal > signal_threshold:
                overall_signal = 'buy'
                confidence = min(1.0, net_signal)
            elif net_signal < -signal_threshold:
                overall_signal = 'sell'
                confidence = min(1.0, -net_signal)
            else:
                overall_signal = 'neutral'
                confidence = 1.0 - abs(net_signal) / signal_threshold
            
            signals['overall'] = {
                'signal': overall_signal,
                'confidence': confidence,
                'net_score': net_signal,
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'reasoning': f"Overall {'bullish' if net_signal > 0 else 'bearish' if net_signal < 0 else 'neutral'} bias from order book analysis"
            }
        else:
            signals['overall'] = {
                'signal': 'neutral',
                'confidence': 0.5,
                'net_score': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'reasoning': "No significant signals from order book analysis"
            }
        
        return signals