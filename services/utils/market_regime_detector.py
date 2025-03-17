import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from hmmlearn import hmm
import pickle
import logging as logger

class MarketRegimeDetector:
    """
    Machine learning-based market regime detection.
    
    This class implements multiple methods for detecting market regimes:
    1. Clustering based (KMeans, GMM)
    2. Hidden Markov Models (HMM)
    3. Supervised learning with labels
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the market regime detector with configuration.
        
        Args:
            config: Dictionary with detector configuration
        """
        self.config = config
        self.method = config.get('method', 'kmeans')
        self.window_size = config.get('window_size', 20)
        self.n_regimes = config.get('n_regimes', 4)
        self.features = config.get('features', [
            'return', 'volatility', 'rsi', 'macd', 'bb_width'
        ])
        
        # Common regime names across detection methods
        self.regime_names = {
            0: 'bull',
            1: 'bear',
            2: 'ranging',
            3: 'volatile'
        }
        
        # Model instances
        self.scaler = StandardScaler()
        self.model = None
        self.pca = None
        self.is_trained = False
        
        # Feature transformers
        self.feature_transformers = {
            'return': self._calculate_returns,
            'volatility': self._calculate_volatility,
            'trend_strength': self._calculate_trend_strength,
            'rsi': self._calculate_rsi,
            'macd': self._calculate_macd,
            'bb_width': self._calculate_bollinger_width
        }
        
        logger.debug(f"MarketRegimeDetector initialized with method: {self.method}")
    
    def _calculate_returns(self, df: pd.DataFrame) -> pd.Series:
        """Calculate returns from price series."""
        return df['close'].pct_change().fillna(0)
    
    def _calculate_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate rolling volatility."""
        return df['close'].pct_change().rolling(self.window_size).std().fillna(0)
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength using slope of linear regression."""
        returns = df['close'].pct_change().fillna(0)
        return returns.rolling(self.window_size).apply(
            lambda x: np.abs(np.polyfit(np.arange(len(x)), x, 1)[0]) * 100,
            raw=True
        ).fillna(0)
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """Calculate RSI indicator."""
        delta = df['close'].diff().fillna(0)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean().fillna(0)
        avg_loss = loss.rolling(window=14).mean().fillna(0)
        
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.Series:
        """Calculate MACD indicator."""
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        return macd
    
    def _calculate_bollinger_width(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Bollinger Band width."""
        rolling_mean = df['close'].rolling(window=20).mean()
        rolling_std = df['close'].rolling(window=20).std()
        
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        
        bandwidth = (upper_band - lower_band) / rolling_mean
        return bandwidth.fillna(0)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for regime detection.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with calculated features
        """
        features_df = pd.DataFrame(index=df.index)
        
        # Calculate each requested feature
        for feature in self.features:
            if feature in self.feature_transformers:
                features_df[feature] = self.feature_transformers[feature](df)
            elif feature in df.columns:
                features_df[feature] = df[feature]
            else:
                logger.warning(f"Feature {feature} not found or cannot be calculated")
        
        # Drop rows with NaN values
        features_df = features_df.dropna()
        
        return features_df
    
    def _train_kmeans(self, X: np.ndarray) -> KMeans:
        """Train KMeans clustering model."""
        model = KMeans(n_clusters=self.n_regimes, random_state=42)
        model.fit(X)
        return model
    
    def _train_gmm(self, X: np.ndarray) -> GaussianMixture:
        """Train Gaussian Mixture Model."""
        model = GaussianMixture(n_components=self.n_regimes, random_state=42)
        model.fit(X)
        return model
    
    def _train_hmm(self, X: np.ndarray) -> hmm.GaussianHMM:
        """Train Hidden Markov Model."""
        model = hmm.GaussianHMM(n_components=self.n_regimes, random_state=42)
        model.fit(X)
        return model
    
    def _train_random_forest(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        """Train Random Forest classifier."""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    
    def train(self, df: pd.DataFrame, labels: Optional[np.ndarray] = None) -> bool:
        """
        Train the market regime detection model.
        
        Args:
            df: DataFrame with OHLCV data
            labels: Optional array of known regime labels for supervised methods
            
        Returns:
            True if training was successful
        """
        try:
            # Prepare features
            features_df = self.prepare_features(df)
            
            if len(features_df) < self.window_size * 2:
                logger.warning(f"Not enough data for training: {len(features_df)} samples")
                return False
            
            # Scale features
            X = self.scaler.fit_transform(features_df)
            
            # Apply PCA if more than 5 features to avoid curse of dimensionality
            if X.shape[1] > 5:
                self.pca = PCA(n_components=5)
                X = self.pca.fit_transform(X)
                logger.debug(f"Applied PCA, explained variance: {np.sum(self.pca.explained_variance_ratio_):.2f}")
            
            # Train the selected model
            if self.method == 'kmeans':
                self.model = self._train_kmeans(X)
                logger.info(f"KMeans model trained with {len(features_df)} samples")
            
            elif self.method == 'gmm':
                self.model = self._train_gmm(X)
                logger.info(f"GMM model trained with {len(features_df)} samples")
            
            elif self.method == 'hmm':
                self.model = self._train_hmm(X)
                logger.info(f"HMM model trained with {len(features_df)} samples")
            
            elif self.method == 'supervised' and labels is not None:
                if len(labels) != len(X):
                    logger.error(f"Labels length ({len(labels)}) does not match data length ({len(X)})")
                    return False
                    
                self.model = self._train_random_forest(X, labels)
                logger.info(f"Random Forest model trained with {len(features_df)} samples")
            
            elif self.method == 'supervised' and labels is None:
                logger.error("Labels required for supervised learning but none provided")
                return False
            
            else:
                logger.error(f"Unsupported model type: {self.method}")
                return False
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Error training regime detection model: {str(e)}")
            return False
    
    def _map_regime_labels(self, labels: np.ndarray, X: np.ndarray) -> Dict[int, str]:
        """
        Map numerical labels to regime names (bull, bear, ranging, volatile).
        
        For clustering methods, we use a heuristic approach:
        - Regimes with highest return -> bull
        - Regimes with lowest return -> bear
        - Regimes with lowest volatility -> ranging
        - Regimes with highest volatility -> volatile
        """
        # Dictionary to store characteristics of each regime
        regime_stats = {}
        
        # Check which features we have available
        feature_indices = {}
        for i, feature in enumerate(self.features):
            feature_indices[feature] = i
        
        # Calculate statistics for each regime
        for label in range(self.n_regimes):
            regime_points = X[labels == label]
            
            # Calculate statistics for this regime
            regime_stats[label] = {
                'count': len(regime_points),
                'return_mean': np.mean(regime_points[:, feature_indices.get('return', 0)]) if 'return' in feature_indices else 0,
                'volatility_mean': np.mean(regime_points[:, feature_indices.get('volatility', 1)]) if 'volatility' in feature_indices else 0,
                'rsi_mean': np.mean(regime_points[:, feature_indices.get('rsi', 2)]) if 'rsi' in feature_indices else 50,
            }
        
        # Map regimes to meaningful names
        # First, set default mapping
        regime_mapping = {i: f'regime_{i}' for i in range(self.n_regimes)}
        
        # If we have return and volatility, use them to identify regimes
        if 'return' in feature_indices and 'volatility' in feature_indices:
            # Sort regimes by return (ascending)
            return_sorted = sorted(regime_stats.keys(), key=lambda x: regime_stats[x]['return_mean'])
            
            # Sort regimes by volatility (ascending)
            volatility_sorted = sorted(regime_stats.keys(), key=lambda x: regime_stats[x]['volatility_mean'])
            
            # Map labels
            if len(return_sorted) >= 4 and len(volatility_sorted) >= 4:
                # Bull regime: Highest return
                regime_mapping[return_sorted[-1]] = 'bull'
                
                # Bear regime: Lowest return
                regime_mapping[return_sorted[0]] = 'bear'
                
                # Ranging regime: Lowest volatility
                regime_mapping[volatility_sorted[0]] = 'ranging'
                
                # Volatile regime: Highest volatility
                regime_mapping[volatility_sorted[-1]] = 'volatile'
            
            # If we have 2 or 3 regimes, use a simpler mapping
            elif len(return_sorted) >= 2:
                # Bull regime: Highest return
                regime_mapping[return_sorted[-1]] = 'bull'
                
                # Bear regime: Lowest return
                regime_mapping[return_sorted[0]] = 'bear'
                
                # If we have 3 regimes, add ranging
                if len(return_sorted) >= 3:
                    # Ranging regime: Middle return or lowest volatility
                    regime_mapping[return_sorted[1]] = 'ranging'
        
        logger.debug(f"Regime mapping: {regime_mapping}")
        return regime_mapping
    
    def detect_regime(self, df: pd.DataFrame, window_size: Optional[int] = None) -> Dict:
        """
        Detect the market regime from recent data.
        
        Args:
            df: DataFrame with OHLCV data
            window_size: Optional window size for recent data
            
        Returns:
            Dictionary with detected regime and confidence
        """
        try:
            if not self.is_trained:
                logger.warning("Model not trained yet, cannot detect regime")
                return {'regime': 'unknown', 'confidence': 0, 'probs': {}}
            
            # Use default window size if not specified
            if window_size is None:
                window_size = self.window_size
            
            # Get the most recent data
            if len(df) > window_size:
                recent_df = df.iloc[-window_size:]
            else:
                recent_df = df
            
            # Prepare features
            features_df = self.prepare_features(recent_df)
            
            if len(features_df) == 0:
                logger.warning("No valid features for regime detection")
                return {'regime': 'unknown', 'confidence': 0, 'probs': {}}
            
            # Scale features
            X = self.scaler.transform(features_df)
            
            # Apply PCA if needed
            if self.pca is not None:
                X = self.pca.transform(X)
            
            # Detect regime using the trained model
            result = {'probs': {}}
            
            if self.method == 'kmeans':
                labels = self.model.predict(X)
                distances = self.model.transform(X)
                
                # Use most frequent label as the regime
                unique_labels, counts = np.unique(labels, return_counts=True)
                most_common_label = unique_labels[np.argmax(counts)]
                
                # Get minimum distance to centroid as confidence
                confidence_scores = 1.0 / (1.0 + np.min(distances, axis=1))
                avg_confidence = float(np.mean(confidence_scores))
                
                # Map regime label
                regime_mapping = self._map_regime_labels(labels, X)
                regime_name = regime_mapping.get(most_common_label, f'regime_{most_common_label}')
                
                # Count occurrences of each regime
                for label in unique_labels:
                    regime = regime_mapping.get(label, f'regime_{label}')
                    probability = float(counts[np.where(unique_labels == label)[0][0]] / len(labels))
                    result['probs'][regime] = probability
                
                result['regime'] = regime_name
                result['confidence'] = avg_confidence
            
            elif self.method == 'gmm':
                probs = self.model.predict_proba(X)
                labels = self.model.predict(X)
                
                # Use most frequent label as the regime
                unique_labels, counts = np.unique(labels, return_counts=True)
                most_common_label = unique_labels[np.argmax(counts)]
                
                # Use probability as confidence
                avg_confidence = float(np.max(np.mean(probs, axis=0)))
                
                # Map regime label
                regime_mapping = self._map_regime_labels(labels, X)
                regime_name = regime_mapping.get(most_common_label, f'regime_{most_common_label}')
                
                # Count occurrences of each regime
                for i in range(self.n_regimes):
                    regime = regime_mapping.get(i, f'regime_{i}')
                    probability = float(np.mean(probs[:, i]))
                    result['probs'][regime] = probability
                
                result['regime'] = regime_name
                result['confidence'] = avg_confidence
            
            elif self.method == 'hmm':
                # Convert to correct shape for HMM
                X_reshaped = X.reshape(-1, X.shape[1])
                
                # Get regime sequence
                hidden_states = self.model.predict(X_reshaped)
                
                # Use most frequent state as the regime
                unique_states, counts = np.unique(hidden_states, return_counts=True)
                most_common_state = unique_states[np.argmax(counts)]
                
                # Calculate log-likelihood as a confidence proxy
                log_likelihood = self.model.score(X_reshaped)
                confidence = 1.0 / (1.0 + np.abs(log_likelihood))
                
                # Map regime label
                regime_mapping = self._map_regime_labels(hidden_states, X)
                regime_name = regime_mapping.get(most_common_state, f'regime_{most_common_state}')
                
                # Count occurrences of each regime
                for state in unique_states:
                    regime = regime_mapping.get(state, f'regime_{state}')
                    probability = float(counts[np.where(unique_states == state)[0][0]] / len(hidden_states))
                    result['probs'][regime] = probability
                
                result['regime'] = regime_name
                result['confidence'] = float(confidence)
            
            elif self.method == 'supervised':
                # Make predictions
                labels = self.model.predict(X)
                probs = self.model.predict_proba(X)
                
                # Use most frequent label as the regime
                unique_labels, counts = np.unique(labels, return_counts=True)
                most_common_label = unique_labels[np.argmax(counts)]
                
                # Get confidence from probability
                avg_confidence = float(np.max(np.mean(probs, axis=0)))
                
                # Get regime name from mapping
                regime_name = self.regime_names.get(most_common_label, f'regime_{most_common_label}')
                
                # Store probabilities
                for i, label in enumerate(self.model.classes_):
                    regime = self.regime_names.get(label, f'regime_{label}')
                    probability = float(np.mean(probs[:, i]))
                    result['probs'][regime] = probability
                
                result['regime'] = regime_name
                result['confidence'] = avg_confidence
            
            else:
                logger.error(f"Unsupported model type: {self.method}")
                return {'regime': 'unknown', 'confidence': 0, 'probs': {}}
            
            # Add timestamps
            result['timestamp'] = pd.Timestamp.now().isoformat()
            result['data_end'] = df.index[-1].isoformat() if hasattr(df.index[-1], 'isoformat') else str(df.index[-1])
            
            logger.info(f"Detected regime: {result['regime']} with confidence: {result['confidence']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return {'regime': 'unknown', 'confidence': 0, 'probs': {}}
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful
        """
        try:
            if not self.is_trained:
                logger.warning("Model not trained yet, cannot save")
                return False
                
            model_data = {
                'method': self.method,
                'window_size': self.window_size,
                'n_regimes': self.n_regimes,
                'features': self.features,
                'scaler': self.scaler,
                'model': self.model,
                'pca': self.pca,
                'is_trained': self.is_trained
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            self.method = model_data['method']
            self.window_size = model_data['window_size']
            self.n_regimes = model_data['n_regimes']
            self.features = model_data['features']
            self.scaler = model_data['scaler']
            self.model = model_data['model']
            self.pca = model_data['pca']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False