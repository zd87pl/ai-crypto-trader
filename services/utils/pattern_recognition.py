import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from typing import Dict, List, Union, Optional, Tuple
import matplotlib.pyplot as plt
import io
import base64
import logging
import os
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [PatternRecognition] %(message)s',
    handlers=[
        logging.FileHandler('logs/pattern_recognition.log'),
        logging.StreamHandler()
    ]
)

class PatternRecognitionModel:
    """
    Deep learning model for identifying chart patterns in price data.
    
    This class implements CNN and LSTM-based models to recognize common chart patterns:
    1. Head and Shoulders
    2. Double Top/Bottom
    3. Triangle (Ascending, Descending, Symmetric)
    4. Rectangle
    5. Flag/Pennant
    6. Cup and Handle
    7. Wedge
    """
    
    def __init__(
        self, 
        config: Optional[Dict] = None, 
        sequence_length: int = 60,
        model_type: str = "cnn_lstm"
    ):
        """
        Initialize the Pattern Recognition Model
        
        Args:
            config: Configuration dictionary
            sequence_length: Number of time steps to analyze for patterns
            model_type: Type of model to use ('cnn', 'lstm', 'cnn_lstm')
        """
        self.logger = logging.getLogger(__name__)
        self.config = config if config else {}
        self.sequence_length = sequence_length
        self.model_type = model_type
        self.model = None
        self.pattern_classes = [
            'head_and_shoulders', 'inverse_head_and_shoulders',
            'double_top', 'double_bottom',
            'ascending_triangle', 'descending_triangle', 'symmetric_triangle',
            'rectangle', 'flag_bull', 'flag_bear',
            'pennant', 'cup_and_handle', 'rising_wedge', 'falling_wedge',
            'no_pattern'
        ]
        self.checkpoint_dir = "models/pattern_recognition"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Create visualization directory
        self.viz_dir = "visualizations/patterns"
        os.makedirs(self.viz_dir, exist_ok=True)
    
    def build_model(self, input_shape: Tuple[int, int] = None) -> None:
        """
        Build the neural network model architecture
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
                Default is (sequence_length, 5) for OHLCV data
        """
        if input_shape is None:
            input_shape = (self.sequence_length, 5)  # OHLCV data by default
            
        if self.model_type == "cnn":
            self._build_cnn_model(input_shape)
        elif self.model_type == "lstm":
            self._build_lstm_model(input_shape)
        elif self.model_type == "cnn_lstm":
            self._build_cnn_lstm_model(input_shape)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _build_cnn_model(self, input_shape: Tuple[int, int]) -> None:
        """Build a CNN model for pattern recognition"""
        model = models.Sequential([
            layers.Input(shape=input_shape),
            # Reshape for 1D convolution (batch, sequence, features) -> (batch, sequence, features, 1)
            layers.Reshape((*input_shape, 1)),
            
            # First convolutional block
            layers.Conv2D(32, kernel_size=(3, 1), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 1)),
            
            # Second convolutional block
            layers.Conv2D(64, kernel_size=(3, 1), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 1)),
            
            # Third convolutional block
            layers.Conv2D(128, kernel_size=(3, 1), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 1)),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.pattern_classes), activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        self.logger.info(f"Built CNN model with input shape {input_shape}")
    
    def _build_lstm_model(self, input_shape: Tuple[int, int]) -> None:
        """Build an LSTM model for pattern recognition"""
        model = models.Sequential([
            layers.Input(shape=input_shape),
            
            # LSTM layers
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(64),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(self.pattern_classes), activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        self.logger.info(f"Built LSTM model with input shape {input_shape}")
    
    def _build_cnn_lstm_model(self, input_shape: Tuple[int, int]) -> None:
        """Build a hybrid CNN-LSTM model for pattern recognition"""
        model = models.Sequential([
            layers.Input(shape=input_shape),
            
            # 1D Convolutional layers for feature extraction
            layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            
            layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            
            # LSTM layers
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(64),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(self.pattern_classes), activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        self.logger.info(f"Built CNN-LSTM model with input shape {input_shape}")
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10
    ) -> Dict:
        """
        Train the pattern recognition model
        
        Args:
            X_train: Training data features
            y_train: Training data labels
            X_val: Validation data features (optional)
            y_val: Validation data labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of training data to use for validation
            early_stopping_patience: Number of epochs to wait for improvement
            
        Returns:
            Dictionary with training history
        """
        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=early_stopping_patience,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.checkpoint_dir, f"pattern_model_{self.model_type}.h5"),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train the model
        if X_val is not None and y_val is not None:
            # Use provided validation data
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Use validation split
            history = self.model.fit(
                X_train, y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        # Save training history
        history_dict = {key: [float(x) for x in val] for key, val in history.history.items()}
        with open(os.path.join(self.checkpoint_dir, f"pattern_model_{self.model_type}_history.json"), 'w') as f:
            json.dump(history_dict, f)
        
        self.logger.info(f"Model trained for {len(history.history['loss'])} epochs")
        
        # Generate training visualization
        self._generate_training_plot(history.history)
        
        return history.history
    
    def _generate_training_plot(self, history: Dict) -> str:
        """Generate and save training history plot"""
        try:
            plt.figure(figsize=(12, 5))
            
            # Plot training & validation accuracy
            plt.subplot(1, 2, 1)
            plt.plot(history['accuracy'])
            plt.plot(history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            
            # Plot training & validation loss
            plt.subplot(1, 2, 2)
            plt.plot(history['loss'])
            plt.plot(history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, f"pattern_model_{self.model_type}_training.png"))
            plt.close()
            
            self.logger.info(f"Training visualization saved to {self.viz_dir}")
            return os.path.join(self.viz_dir, f"pattern_model_{self.model_type}_training.png")
            
        except Exception as e:
            self.logger.error(f"Error generating training plot: {str(e)}")
            return ""
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load a trained model from file
        
        Args:
            model_path: Path to the model file, if None use default path
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if model_path is None:
                model_path = os.path.join(self.checkpoint_dir, f"pattern_model_{self.model_type}.h5")
            
            if not os.path.exists(model_path):
                self.logger.warning(f"Model file not found: {model_path}")
                return False
            
            self.model = models.load_model(model_path)
            self.logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess data for pattern recognition
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Normalized and formatted data array
        """
        try:
            # Ensure we have required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"Missing required column: {col}")
                    raise ValueError(f"Missing required column: {col}")
            
            # Convert to numpy array
            data = df[required_columns].values
            
            # Normalize data (price by close and volume separately)
            close_values = data[:, 3].reshape(-1, 1)  # Close prices
            data_norm = np.zeros_like(data, dtype=np.float32)
            
            # Normalize prices by the last close price
            for i in range(4):  # OHLC
                data_norm[:, i] = data[:, i] / close_values[-1]
            
            # Normalize volume by max volume
            max_volume = np.max(data[:, 4])
            if max_volume > 0:
                data_norm[:, 4] = data[:, 4] / max_volume
            
            return data_norm
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def generate_sequences(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Generate sequences from data for pattern detection
        
        Args:
            data: Normalized price and volume data
            
        Returns:
            List of sequences for pattern detection
        """
        sequences = []
        seq_length = self.sequence_length
        
        # If data is too short, pad with zeros
        if len(data) < seq_length:
            pad_length = seq_length - len(data)
            padded_data = np.vstack([np.zeros((pad_length, data.shape[1])), data])
            sequences.append(padded_data)
        else:
            # Generate overlapping sequences with stride of 5
            stride = 5
            for i in range(0, len(data) - seq_length + 1, stride):
                seq = data[i:i+seq_length]
                sequences.append(seq)
        
        return sequences
    
    def detect_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Detect chart patterns in the given data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with detected patterns and confidence scores
        """
        try:
            if self.model is None:
                model_loaded = self.load_model()
                if not model_loaded:
                    self.logger.error("No model available for pattern detection")
                    return {"error": "No model available for pattern detection"}
            
            # Preprocess data
            normalized_data = self.preprocess_data(df)
            sequences = self.generate_sequences(normalized_data)
            
            # Convert to numpy array
            X = np.array(sequences)
            
            # Get predictions
            predictions = self.model.predict(X)
            
            # Calculate average probability for each pattern across all sequences
            avg_predictions = np.mean(predictions, axis=0)
            
            # Get top 3 patterns
            top_indices = np.argsort(avg_predictions)[-3:][::-1]
            top_patterns = [
                {
                    "pattern": self.pattern_classes[idx],
                    "confidence": float(avg_predictions[idx])
                }
                for idx in top_indices if avg_predictions[idx] > 0.2  # Only include if confidence > 20%
            ]
            
            # Get primary pattern (highest confidence)
            if top_patterns and top_patterns[0]["confidence"] > 0.5:
                primary_pattern = top_patterns[0]["pattern"]
                confidence = top_patterns[0]["confidence"]
            else:
                primary_pattern = "no_pattern"
                confidence = 1.0 - (top_patterns[0]["confidence"] if top_patterns else 0.0)
            
            # Check for pattern completion percentage
            completion_pct = self._estimate_pattern_completion(df, primary_pattern)
            
            # Generate pattern visualization
            visualization = None
            if primary_pattern != "no_pattern":
                visualization = self._generate_pattern_visualization(df, primary_pattern)
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "primary_pattern": primary_pattern,
                "confidence": float(confidence),
                "completion_percentage": completion_pct,
                "all_patterns": top_patterns,
                "visualization": visualization,
                "pattern_description": self._get_pattern_description(primary_pattern),
                "trading_implications": self._get_trading_implications(primary_pattern, completion_pct)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {str(e)}")
            return {"error": str(e)}
    
    def _estimate_pattern_completion(
        self, 
        df: pd.DataFrame, 
        pattern: str
    ) -> float:
        """
        Estimate the completion percentage of the detected pattern
        This is a heuristic approach based on typical pattern lengths
        
        Args:
            df: DataFrame with price data
            pattern: Detected pattern
            
        Returns:
            Estimated pattern completion percentage (0-100)
        """
        # Typical pattern lengths in candles
        pattern_lengths = {
            "head_and_shoulders": 30,
            "inverse_head_and_shoulders": 30,
            "double_top": 25,
            "double_bottom": 25,
            "ascending_triangle": 20,
            "descending_triangle": 20,
            "symmetric_triangle": 20,
            "rectangle": 20,
            "flag_bull": 10,
            "flag_bear": 10,
            "pennant": 15,
            "cup_and_handle": 40,
            "rising_wedge": 25,
            "falling_wedge": 25,
            "no_pattern": 1
        }
        
        # If pattern is 'no_pattern', return 0
        if pattern == "no_pattern":
            return 0.0
        
        # Get typical length for the pattern
        typical_length = pattern_lengths.get(pattern, 20)
        
        # Calculate completion percentage based on available data
        # More data = potentially more complete pattern
        completion = min(100.0, (len(df) / typical_length) * 100)
        
        # Adjust completion based on recent price action for certain patterns
        if pattern in ["double_top", "double_bottom", "head_and_shoulders", "inverse_head_and_shoulders"]:
            # These patterns need a confirmation move after formation
            recent_moves = df["close"].pct_change().tail(5).abs().mean() * 100
            completion = min(100.0, completion * (1.0 + recent_moves / 10.0))
        
        # Round to nearest 5%
        return round(completion / 5) * 5
    
    def _generate_pattern_visualization(
        self, 
        df: pd.DataFrame, 
        pattern: str
    ) -> Optional[str]:
        """
        Generate visualization for the detected pattern
        
        Args:
            df: DataFrame with price data
            pattern: Detected pattern
            
        Returns:
            Base64 encoded image or None if error
        """
        try:
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Get price data
            dates = df.index[-min(100, len(df)):] if hasattr(df.index, 'date') else range(len(df[-min(100, len(df)):]))
            closes = df['close'][-min(100, len(df)):].values
            highs = df['high'][-min(100, len(df)):].values
            lows = df['low'][-min(100, len(df)):].values
            
            # Plot price
            plt.plot(dates, closes, label='Close Price', color='black', linewidth=1.5)
            
            # Highlight pattern region (simplified visualization)
            if pattern != "no_pattern":
                # Different colors for different pattern types
                pattern_colors = {
                    "head_and_shoulders": "red",
                    "inverse_head_and_shoulders": "green",
                    "double_top": "red",
                    "double_bottom": "green",
                    "ascending_triangle": "blue",
                    "descending_triangle": "orange",
                    "symmetric_triangle": "purple",
                    "rectangle": "gray",
                    "flag_bull": "green",
                    "flag_bear": "red",
                    "pennant": "blue",
                    "cup_and_handle": "green",
                    "rising_wedge": "red",
                    "falling_wedge": "green"
                }
                
                color = pattern_colors.get(pattern, "blue")
                
                # Highlight the pattern area
                plt.fill_between(
                    dates, 
                    lows, 
                    highs, 
                    alpha=0.2, 
                    color=color,
                    label=f"{pattern.replace('_', ' ').title()} Pattern"
                )
            
            # Add labels and legend
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.title(f'Detected Pattern: {pattern.replace("_", " ").title()}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save to buffer
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert to base64
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            # Also save to file for reference
            file_path = os.path.join(self.viz_dir, f"pattern_{pattern}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            with open(file_path, 'wb') as f:
                f.write(base64.b64decode(img_str))
            
            return img_str
            
        except Exception as e:
            self.logger.error(f"Error generating pattern visualization: {str(e)}")
            return None
    
    def _get_pattern_description(self, pattern: str) -> str:
        """
        Get description for the detected pattern
        
        Args:
            pattern: Detected pattern
            
        Returns:
            Pattern description
        """
        descriptions = {
            "head_and_shoulders": (
                "The Head and Shoulders pattern is a reversal pattern that signals a trend change "
                "from bullish to bearish. It consists of three peaks, with the middle peak (head) "
                "being the highest and the two outer peaks (shoulders) being lower and roughly equal."
            ),
            "inverse_head_and_shoulders": (
                "The Inverse Head and Shoulders pattern is a reversal pattern that signals a trend change "
                "from bearish to bullish. It consists of three troughs, with the middle trough (head) "
                "being the lowest and the two outer troughs (shoulders) being higher and roughly equal."
            ),
            "double_top": (
                "The Double Top pattern is a reversal pattern that signals a trend change from bullish "
                "to bearish. It consists of two peaks at approximately the same price level, with a "
                "trough between them."
            ),
            "double_bottom": (
                "The Double Bottom pattern is a reversal pattern that signals a trend change from bearish "
                "to bullish. It consists of two troughs at approximately the same price level, with a "
                "peak between them."
            ),
            "ascending_triangle": (
                "The Ascending Triangle is a bullish continuation pattern characterized by a horizontal "
                "upper resistance line and an upward-sloping lower support line. It indicates accumulation "
                "and typically resolves with an upward breakout."
            ),
            "descending_triangle": (
                "The Descending Triangle is a bearish continuation pattern characterized by a horizontal "
                "lower support line and a downward-sloping upper resistance line. It indicates distribution "
                "and typically resolves with a downward breakout."
            ),
            "symmetric_triangle": (
                "The Symmetric Triangle is a continuation pattern characterized by converging trendlines of "
                "support and resistance. It can break in either direction, and the breakout direction is "
                "typically followed by a significant price move."
            ),
            "rectangle": (
                "The Rectangle pattern is a continuation pattern characterized by horizontal support and "
                "resistance lines. It represents a period of consolidation before continuation of the trend."
            ),
            "flag_bull": (
                "The Bullish Flag is a continuation pattern that forms as a brief pause in an uptrend. "
                "It consists of a strong price move up (the pole) followed by a downward-sloping "
                "consolidation period (the flag)."
            ),
            "flag_bear": (
                "The Bearish Flag is a continuation pattern that forms as a brief pause in a downtrend. "
                "It consists of a strong price move down (the pole) followed by an upward-sloping "
                "consolidation period (the flag)."
            ),
            "pennant": (
                "The Pennant is a continuation pattern similar to a flag but with converging trendlines "
                "forming a small symmetric triangle. It typically forms after a strong price move and "
                "signals a brief consolidation before continuation."
            ),
            "cup_and_handle": (
                "The Cup and Handle is a bullish continuation pattern resembling a cup (a rounded bottom) "
                "followed by a handle (a slight downward drift). It signals a period of consolidation "
                "before continuing the uptrend."
            ),
            "rising_wedge": (
                "The Rising Wedge is typically a bearish reversal pattern (but can be a continuation pattern "
                "in a downtrend). It is characterized by converging trendlines both sloping upward, with "
                "the lower trendline having a steeper slope."
            ),
            "falling_wedge": (
                "The Falling Wedge is typically a bullish reversal pattern (but can be a continuation pattern "
                "in an uptrend). It is characterized by converging trendlines both sloping downward, with "
                "the upper trendline having a steeper slope."
            ),
            "no_pattern": (
                "No significant chart pattern detected. The price action does not currently conform to "
                "any of the recognized technical chart patterns."
            )
        }
        
        return descriptions.get(pattern, "Pattern description not available")
    
    def _get_trading_implications(self, pattern: str, completion_pct: float) -> Dict:
        """
        Get trading implications for the detected pattern
        
        Args:
            pattern: Detected pattern
            completion_pct: Pattern completion percentage
            
        Returns:
            Dictionary with trading implications
        """
        # Default implications
        implications = {
            "bias": "neutral",
            "signal_strength": "weak",
            "potential_targets": [],
            "confirmation_criteria": [],
            "invalidation_criteria": []
        }
        
        # Classify patterns by expected direction
        bullish_patterns = [
            "inverse_head_and_shoulders", "double_bottom", "ascending_triangle",
            "flag_bull", "cup_and_handle", "falling_wedge"
        ]
        
        bearish_patterns = [
            "head_and_shoulders", "double_top", "descending_triangle",
            "flag_bear", "rising_wedge"
        ]
        
        neutral_patterns = ["symmetric_triangle", "rectangle", "pennant", "no_pattern"]
        
        # Determine bias based on pattern type
        if pattern in bullish_patterns:
            implications["bias"] = "bullish"
        elif pattern in bearish_patterns:
            implications["bias"] = "bearish"
        else:
            implications["bias"] = "neutral"
        
        # Determine signal strength based on completion percentage
        if completion_pct >= 90:
            implications["signal_strength"] = "very_strong"
        elif completion_pct >= 75:
            implications["signal_strength"] = "strong"
        elif completion_pct >= 50:
            implications["signal_strength"] = "moderate"
        else:
            implications["signal_strength"] = "weak"
        
        # Add pattern-specific implications
        if pattern == "head_and_shoulders":
            implications["confirmation_criteria"] = ["Break below neckline", "Increased volume on break"]
            implications["invalidation_criteria"] = ["Price moves above right shoulder", "Failure to break neckline"]
            
        elif pattern == "inverse_head_and_shoulders":
            implications["confirmation_criteria"] = ["Break above neckline", "Increased volume on break"]
            implications["invalidation_criteria"] = ["Price moves below right shoulder", "Failure to break neckline"]
            
        elif pattern == "double_top":
            implications["confirmation_criteria"] = ["Break below support level between tops", "Volume expansion on break"]
            implications["invalidation_criteria"] = ["Price exceeds the second top", "Low volume on downward move"]
            
        elif pattern == "double_bottom":
            implications["confirmation_criteria"] = ["Break above resistance level between bottoms", "Volume expansion on break"]
            implications["invalidation_criteria"] = ["Price falls below the second bottom", "Low volume on upward move"]
            
        elif pattern == "ascending_triangle":
            implications["confirmation_criteria"] = ["Break above upper resistance line", "Increased volume on breakout"]
            implications["invalidation_criteria"] = ["Break below upward trendline", "Failure of support levels"]
            
        elif pattern == "descending_triangle":
            implications["confirmation_criteria"] = ["Break below lower support line", "Increased volume on breakdown"]
            implications["invalidation_criteria"] = ["Break above downward trendline", "Failure of resistance levels"]
            
        elif pattern == "symmetric_triangle":
            implications["confirmation_criteria"] = ["Break of either trendline", "Increased volume on breakout"]
            implications["invalidation_criteria"] = ["False breakout followed by retreat", "Low volume on breakout"]
            
        elif pattern == "rectangle":
            implications["confirmation_criteria"] = ["Break of support or resistance", "Volume confirmation"]
            implications["invalidation_criteria"] = ["Price remains within rectangle", "False breakout"]
            
        elif pattern in ["flag_bull", "flag_bear"]:
            implications["confirmation_criteria"] = ["Break in direction of prior trend", "Increased volume"]
            implications["invalidation_criteria"] = ["Break against prior trend", "Pattern extends too long (>20 bars)"]
            
        elif pattern == "pennant":
            implications["confirmation_criteria"] = ["Break in direction of prior trend", "Occurs within 1-3 weeks of flagpole"]
            implications["invalidation_criteria"] = ["Break against prior trend", "Low volume on breakout"]
            
        elif pattern == "cup_and_handle":
            implications["confirmation_criteria"] = ["Break above resistance after handle forms", "Increased volume"]
            implications["invalidation_criteria"] = ["Drop below cup low", "Handle drops more than 50% of cup depth"]
            
        elif pattern == "rising_wedge":
            implications["confirmation_criteria"] = ["Break below lower trendline", "Declining volume within pattern"]
            implications["invalidation_criteria"] = ["Break above upper trendline", "Strong upward momentum"]
            
        elif pattern == "falling_wedge":
            implications["confirmation_criteria"] = ["Break above upper trendline", "Declining volume within pattern"]
            implications["invalidation_criteria"] = ["Break below lower trendline", "Strong downward momentum"]
        
        return implications
    
    def generate_pattern_examples(self, save_dir: Optional[str] = None) -> List[str]:
        """
        Generate example visualizations for different chart patterns
        Useful for documentation and training purposes
        
        Args:
            save_dir: Directory to save pattern examples
            
        Returns:
            List of file paths to generated pattern examples
        """
        if save_dir is None:
            save_dir = self.viz_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate synthetic pattern data
        patterns = {
            "head_and_shoulders": self._generate_synthetic_head_and_shoulders(bearish=True),
            "inverse_head_and_shoulders": self._generate_synthetic_head_and_shoulders(bearish=False),
            "double_top": self._generate_synthetic_double_top_bottom(bearish=True),
            "double_bottom": self._generate_synthetic_double_top_bottom(bearish=False),
            "ascending_triangle": self._generate_synthetic_triangle(type="ascending"),
            "descending_triangle": self._generate_synthetic_triangle(type="descending"),
            "symmetric_triangle": self._generate_synthetic_triangle(type="symmetric"),
            "rectangle": self._generate_synthetic_rectangle(),
            "cup_and_handle": self._generate_synthetic_cup_and_handle()
        }
        
        # Create and save visualizations
        file_paths = []
        for pattern_name, pattern_data in patterns.items():
            plt.figure(figsize=(10, 6))
            plt.plot(pattern_data, label=pattern_name.replace('_', ' ').title())
            plt.title(f"{pattern_name.replace('_', ' ').title()} Chart Pattern")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            file_path = os.path.join(save_dir, f"example_{pattern_name}.png")
            plt.savefig(file_path)
            plt.close()
            
            file_paths.append(file_path)
        
        return file_paths
    
    # Methods to generate synthetic pattern data for training and testing
    
    def _generate_synthetic_head_and_shoulders(self, bearish=True) -> np.ndarray:
        """Generate synthetic Head and Shoulders pattern data"""
        x = np.linspace(0, 100, 100)
        y = np.zeros_like(x, dtype=float)
        
        # Base trend
        base = 100
        noise_level = 2
        
        # Left shoulder
        y[10:25] = base + 10 * np.sin(np.linspace(0, np.pi, 15)) + np.random.normal(0, noise_level, 15)
        
        # Head
        y[35:55] = base + 20 * np.sin(np.linspace(0, np.pi, 20)) + np.random.normal(0, noise_level, 20)
        
        # Right shoulder
        y[65:80] = base + 10 * np.sin(np.linspace(0, np.pi, 15)) + np.random.normal(0, noise_level, 15)
        
        # Neckline and connecting segments
        y[0:10] = np.linspace(base-5, y[10], 10) + np.random.normal(0, noise_level, 10)
        y[25:35] = np.linspace(y[24], y[35], 10) + np.random.normal(0, noise_level, 10)
        y[55:65] = np.linspace(y[54], y[65], 10) + np.random.normal(0, noise_level, 10)
        y[80:90] = np.linspace(y[79], base-5, 10) + np.random.normal(0, noise_level, 10)
        y[90:100] = np.linspace(base-5, base-15, 10) + np.random.normal(0, noise_level, 10)
        
        # Invert for inverse pattern
        if not bearish:
            y = 2*base - y
            
        return y
    
    def _generate_synthetic_double_top_bottom(self, bearish=True) -> np.ndarray:
        """Generate synthetic Double Top/Bottom pattern data"""
        x = np.linspace(0, 100, 100)
        y = np.zeros_like(x, dtype=float)
        
        # Base trend
        base = 100
        noise_level = 2
        
        # First peak/trough
        y[15:30] = base + 15 * np.sin(np.linspace(0, np.pi, 15)) + np.random.normal(0, noise_level, 15)
        
        # Middle section
        y[30:45] = np.linspace(y[29], base, 15) + np.random.normal(0, noise_level, 15)
        y[45:60] = np.linspace(base, y[60], 15) + np.random.normal(0, noise_level, 15)
        
        # Second peak/trough (slightly lower/higher for realism)
        peak_adjust = -2 if bearish else 2
        y[60:75] = base + (15 + peak_adjust) * np.sin(np.linspace(0, np.pi, 15)) + np.random.normal(0, noise_level, 15)
        
        # Before and after
        y[0:15] = np.linspace(base-10, y[15], 15) + np.random.normal(0, noise_level, 15)
        
        # Breakdown/breakout
        if bearish:
            y[75:100] = np.linspace(y[74], base-20, 25) + np.random.normal(0, noise_level, 25)
        else:
            y[75:100] = np.linspace(y[74], base+20, 25) + np.random.normal(0, noise_level, 25)
        
        # Invert for double bottom
        if not bearish:
            y = 2*base - y
            
        return y
    
    def _generate_synthetic_triangle(self, type="symmetric") -> np.ndarray:
        """Generate synthetic Triangle pattern data"""
        x = np.linspace(0, 100, 100)
        y = np.zeros_like(x, dtype=float)
        
        # Base trend
        base = 100
        noise_level = 1.5
        
        # Pre-pattern trend
        if type == "ascending" or type == "symmetric":
            y[0:20] = np.linspace(base-15, base, 20) + np.random.normal(0, noise_level, 20)
        else:  # descending
            y[0:20] = np.linspace(base+15, base, 20) + np.random.normal(0, noise_level, 20)
        
        # Triangle pattern
        if type == "ascending":
            # Higher lows, constant highs
            for i in range(4):
                start_idx = 20 + i*15
                end_idx = start_idx + 15
                if i % 2 == 0:  # Highs
                    y[start_idx:end_idx] = base + 10 + np.random.normal(0, noise_level, 15)
                else:  # Lows (ascending)
                    low_level = base - 10 + (i*5)
                    y[start_idx:end_idx] = low_level + np.random.normal(0, noise_level, 15)
        
        elif type == "descending":
            # Lower highs, constant lows
            for i in range(4):
                start_idx = 20 + i*15
                end_idx = start_idx + 15
                if i % 2 == 0:  # Highs (descending)
                    high_level = base + 10 - (i*5)
                    y[start_idx:end_idx] = high_level + np.random.normal(0, noise_level, 15)
                else:  # Lows
                    y[start_idx:end_idx] = base - 10 + np.random.normal(0, noise_level, 15)
        
        else:  # symmetric
            # Both converging
            for i in range(4):
                start_idx = 20 + i*15
                end_idx = start_idx + 15
                if i % 2 == 0:  # Highs (descending)
                    high_level = base + 10 - (i*2.5)
                    y[start_idx:end_idx] = high_level + np.random.normal(0, noise_level, 15)
                else:  # Lows (ascending)
                    low_level = base - 10 + (i*2.5)
                    y[start_idx:end_idx] = low_level + np.random.normal(0, noise_level, 15)
        
        # Breakout
        y[80:100] = np.linspace(y[79], y[79] + 15, 20) + np.random.normal(0, noise_level, 20)
        
        return y
    
    def _generate_synthetic_rectangle(self) -> np.ndarray:
        """Generate synthetic Rectangle pattern data"""
        x = np.linspace(0, 100, 100)
        y = np.zeros_like(x, dtype=float)
        
        # Base trend
        base = 100
        noise_level = 1.5
        
        # Pre-pattern trend
        y[0:20] = np.linspace(base-15, base, 20) + np.random.normal(0, noise_level, 20)
        
        # Rectangle pattern
        rectangle_high = base + 10
        rectangle_low = base - 5
        
        for i in range(4):
            start_idx = 20 + i*15
            end_idx = start_idx + 15
            
            if i % 2 == 0:  # Highs
                y[start_idx:end_idx] = rectangle_high + np.random.normal(0, noise_level, 15)
            else:  # Lows
                y[start_idx:end_idx] = rectangle_low + np.random.normal(0, noise_level, 15)
        
        # Breakout
        y[80:100] = np.linspace(y[79], y[79] + 20, 20) + np.random.normal(0, noise_level, 20)
        
        return y
    
    def _generate_synthetic_cup_and_handle(self) -> np.ndarray:
        """Generate synthetic Cup and Handle pattern data"""
        x = np.linspace(0, 100, 100)
        y = np.zeros_like(x, dtype=float)
        
        # Base trend
        base = 100
        noise_level = 1.5
        
        # Initial level
        y[0:10] = base + 10 + np.random.normal(0, noise_level, 10)
        
        # Cup
        cup_x = np.linspace(0, np.pi, 50)
        cup_y = base + 10 - 15 * np.sin(cup_x) + np.random.normal(0, noise_level, 50)
        y[10:60] = cup_y
        
        # Handle (smaller dip)
        handle_x = np.linspace(0, np.pi, 20)
        handle_y = base + 10 - 5 * np.sin(handle_x) + np.random.normal(0, noise_level, 20)
        y[60:80] = handle_y
        
        # Breakout
        y[80:100] = np.linspace(y[79], y[79] + 15, 20) + np.random.normal(0, noise_level, 20)
        
        return y


class ChartPatternRecognitionService:
    """
    Service class for chart pattern recognition in crypto trading.
    
    This service detects chart patterns in price data, which can be used for:
    1. Identifying potential trading opportunities based on classical chart patterns
    2. Providing visual confirmation of technical setups
    3. Calculating pattern-based price targets
    4. Enhancing trading signals with pattern context
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Chart Pattern Recognition Service
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.pattern_enabled = config.get('pattern_recognition', {}).get('enabled', True)
        
        # Get model configurations
        pattern_config = config.get('pattern_recognition', {})
        self.model_type = pattern_config.get('model_type', 'cnn_lstm')
        self.sequence_length = pattern_config.get('sequence_length', 60)
        self.confidence_threshold = pattern_config.get('confidence_threshold', 0.6)
        
        # Storage for detected patterns
        self.patterns_by_symbol = {}
        self.last_analysis_time = {}
        
        # Analysis interval in seconds
        self.analysis_interval = pattern_config.get('analysis_interval', 300)  # 5 minutes
        
        # Whether to create visualizations
        self.create_visualizations = pattern_config.get('create_visualizations', True)
        
        # Initialize pattern recognition model
        self.model = PatternRecognitionModel(
            config=pattern_config,
            sequence_length=self.sequence_length,
            model_type=self.model_type
        )
        
        # Try to load the pre-trained model
        model_loaded = self.model.load_model()
        if model_loaded:
            self.logger.info(f"Successfully loaded pattern recognition model")
        else:
            self.logger.warning("No pre-trained pattern recognition model found")
            
    def analyze_chart_patterns(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Analyze chart patterns for a symbol
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            
        Returns:
            Dictionary with pattern analysis results
        """
        try:
            if not self.pattern_enabled:
                self.logger.debug(f"Pattern recognition is disabled, skipping for {symbol}")
                return {"enabled": False}
            
            current_time = datetime.now()
            last_time = self.last_analysis_time.get(symbol, datetime.min)
            
            # Check if we need to analyze again based on interval
            if (current_time - last_time).total_seconds() < self.analysis_interval:
                self.logger.debug(f"Skipping pattern analysis for {symbol}, last analysis was {(current_time - last_time).seconds}s ago")
                return self.patterns_by_symbol.get(symbol, {"enabled": True, "pattern": "no_pattern"})
            
            # Ensure we have enough data for pattern detection
            if len(df) < self.sequence_length:
                self.logger.warning(f"Insufficient data for pattern detection for {symbol}: {len(df)} points available, {self.sequence_length} required")
                return {"enabled": True, "pattern": "insufficient_data"}
            
            # Detect patterns
            self.logger.info(f"Analyzing chart patterns for {symbol}")
            pattern_results = self.model.detect_patterns(df)
            
            if "error" in pattern_results:
                self.logger.error(f"Error detecting patterns for {symbol}: {pattern_results['error']}")
                return {"enabled": True, "error": pattern_results['error']}
            
            # Store results and update analysis time
            pattern_results["symbol"] = symbol
            self.patterns_by_symbol[symbol] = pattern_results
            self.last_analysis_time[symbol] = current_time
            
            # Log the detected pattern
            pattern = pattern_results.get("primary_pattern", "unknown")
            confidence = pattern_results.get("confidence", 0.0)
            self.logger.info(f"Detected pattern for {symbol}: {pattern} (confidence: {confidence:.2f})")
            
            return pattern_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing chart patterns for {symbol}: {str(e)}")
            return {"enabled": True, "error": str(e)}
    
    def get_pattern_trading_signals(self, pattern_analysis: Dict) -> Dict:
        """
        Extract trading signals from pattern analysis
        
        Args:
            pattern_analysis: Pattern analysis dictionary
            
        Returns:
            Dictionary with trading signals derived from patterns
        """
        if not pattern_analysis or "enabled" in pattern_analysis and not pattern_analysis["enabled"]:
            return {"signal": "neutral", "strength": 0.0}
        
        pattern = pattern_analysis.get("primary_pattern", "no_pattern")
        confidence = pattern_analysis.get("confidence", 0.0)
        completion_pct = pattern_analysis.get("completion_percentage", 0.0)
        
        # Skip if pattern confidence is below threshold
        if confidence < self.confidence_threshold:
            return {"signal": "neutral", "strength": 0.0}
        
        # Get trading implications
        implications = pattern_analysis.get("trading_implications", {})
        bias = implications.get("bias", "neutral")
        signal_strength = implications.get("signal_strength", "weak")
        
        # Convert signal strength to numeric value
        strength_values = {
            "very_strong": 0.9,
            "strong": 0.7,
            "moderate": 0.5,
            "weak": 0.3
        }
        
        numeric_strength = strength_values.get(signal_strength, 0.3)
        
        # Adjust by confidence
        adjusted_strength = numeric_strength * confidence
        
        # Adjust by completion percentage
        adjusted_strength *= (completion_pct / 100)
        
        # Determine signal
        if bias == "bullish" and adjusted_strength > 0.3:
            signal = "buy"
        elif bias == "bearish" and adjusted_strength > 0.3:
            signal = "sell"
        else:
            signal = "neutral"
        
        # Add pattern characteristics
        result = {
            "signal": signal,
            "strength": round(adjusted_strength, 2),
            "pattern": pattern,
            "bias": bias,
            "completion": completion_pct
        }
        
        # Add confirmation criteria
        if "confirmation_criteria" in implications:
            result["confirmation"] = implications["confirmation_criteria"]
        
        # Add invalidation criteria
        if "invalidation_criteria" in implications:
            result["invalidation"] = implications["invalidation_criteria"]
        
        return result
    
    def get_all_pattern_signals(self) -> Dict:
        """
        Get pattern-based trading signals for all analyzed symbols
        
        Returns:
            Dictionary with symbols and their pattern signals
        """
        result = {}
        
        for symbol, pattern_analysis in self.patterns_by_symbol.items():
            signals = self.get_pattern_trading_signals(pattern_analysis)
            if signals["signal"] != "neutral":
                result[symbol] = signals
        
        return result

# Standalone functions for direct use

def detect_chart_patterns(df: pd.DataFrame, model_type: str = "cnn_lstm") -> Dict:
    """
    Detect chart patterns in the given OHLCV data
    
    Args:
        df: DataFrame with OHLCV data
        model_type: Type of model to use
        
    Returns:
        Dictionary with detected patterns
    """
    model = PatternRecognitionModel(model_type=model_type)
    model_loaded = model.load_model()
    
    if not model_loaded:
        return {"error": "No pre-trained model available"}
    
    return model.detect_patterns(df)

def generate_pattern_examples(save_dir: str = "visualizations/patterns") -> List[str]:
    """
    Generate example visualizations for different chart patterns
    
    Args:
        save_dir: Directory to save pattern examples
        
    Returns:
        List of file paths to generated pattern examples
    """
    model = PatternRecognitionModel()
    return model.generate_pattern_examples(save_dir)