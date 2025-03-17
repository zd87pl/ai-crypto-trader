import os
import json
import redis
import logging
import numpy as np
import pandas as pd
import asyncio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from redis.asyncio import Redis

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [NeuralNetwork] %(message)s',
    handlers=[
        logging.FileHandler('logs/neural_network.log'),
        logging.StreamHandler()
    ]
)

class NeuralNetworkService:
    """
    Neural Network Service for price prediction
    
    This service implements various neural network architectures for price prediction:
    1. LSTM (Long Short-Term Memory) networks for time series forecasting
    2. GRU (Gated Recurrent Unit) networks as a lighter alternative to LSTM
    3. Bidirectional LSTM for capturing future and past context
    4. CNN-LSTM hybrid model for extracting hierarchical patterns
    5. Attention-based models for focusing on relevant time steps
    """
    
    def __init__(self):
        """Initialize the Neural Network Service"""
        logging.info("Initializing Neural Network Service...")
        
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Get configuration settings with defaults
        self.nn_config = self.config.get('neural_network', {})
        self.enabled = self.nn_config.get('enabled', True)
        self.prediction_intervals = self.nn_config.get('prediction_intervals', ['1h', '4h', '24h'])
        self.service_port = self.nn_config.get('service_port', 8010)
        self.model_checkpoint_interval = self.nn_config.get('model_checkpoint_interval', 86400)  # 24 hours
        self.training_lookback_days = self.nn_config.get('training_lookback_days', 60)
        self.prediction_features = self.nn_config.get('features', [])
        self.model_type = self.nn_config.get('model_type', 'lstm')
        self.ensemble_enabled = self.nn_config.get('ensemble_enabled', False)
        self.models_dir = 'models'
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Redis connection settings
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis = None
        
        # Tracking state
        self.last_training_time = None
        self.running = True
        self.models = {}
        self.training_history = {}
        self.latest_predictions = {}
        
        # If no feature list is provided, use a default set
        if not self.prediction_features:
            self.prediction_features = [
                'close', 'volume', 'rsi', 'macd', 'bb_position', 
                'stoch_k', 'williams_r', 'ema_12', 'ema_26'
            ]
            
        logging.info(f"Neural Network Service initialized with model type: {self.model_type}")
        logging.info(f"Prediction intervals: {self.prediction_intervals}")
        logging.info(f"Features: {self.prediction_features}")
        logging.info(f"Ensemble enabled: {self.ensemble_enabled}")
    
    async def connect_redis(self, max_retries=10, retry_delay=5):
        """Connect to Redis with retries"""
        retries = 0
        while retries < max_retries and self.running:
            try:
                if self.redis is None:
                    logging.debug(f"Attempting Redis connection (attempt {retries + 1}/{max_retries})")
                    self.redis = Redis(
                        host=self.redis_host,
                        port=self.redis_port,
                        decode_responses=True,
                        socket_connect_timeout=5.0,
                        socket_keepalive=True,
                        health_check_interval=15
                    )
                await self.redis.ping()
                logging.info(f"Successfully connected to Redis at {self.redis_host}:{self.redis_port}")
                return True
            except Exception as e:
                retries += 1
                logging.error(f"Failed to connect to Redis (attempt {retries}/{max_retries}): {str(e)}")
                if self.redis:
                    await self.redis.close()
                    self.redis = None
                if retries < max_retries:
                    await asyncio.sleep(retry_delay)
                else:
                    logging.error("Max retries reached. Could not connect to Redis.")
                    return False
    
    async def initialize_models(self):
        """Initialize neural network models"""
        try:
            # Check for TensorFlow
            try:
                import tensorflow as tf
                from tensorflow.keras.models import Sequential, Model, load_model
                from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional
                from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, TimeDistributed
                from tensorflow.keras.layers import Input, Concatenate, Attention, MultiHeadAttention
                from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
                from tensorflow.keras.optimizers import Adam
                
                tf.get_logger().setLevel('ERROR')
                logging.info("TensorFlow successfully imported")
                self.tf_available = True
            except ImportError:
                logging.warning("TensorFlow not available. Neural network predictions will be disabled.")
                self.tf_available = False
                return False
            
            # Initialize empty models dictionary
            self.models = {}
            
            # Load existing models if available
            for interval in self.prediction_intervals:
                model_path = os.path.join(self.models_dir, f'nn_model_{self.model_type}_{interval}.h5')
                if os.path.exists(model_path):
                    try:
                        self.models[interval] = load_model(model_path)
                        logging.info(f"Loaded existing model for {interval} interval from {model_path}")
                    except Exception as e:
                        logging.error(f"Error loading model for {interval}: {str(e)}")
                        # Will create a new model for this interval during training
            
            logging.info("Neural network models initialized successfully")
            return True
        
        except Exception as e:
            logging.error(f"Error initializing neural network models: {str(e)}")
            return False
    
    def create_model(self, input_shape: Tuple[int, int], output_dim: int = 1):
        """
        Create a neural network model based on the configured type
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            output_dim: Number of output predictions (typically 1 for price prediction)
            
        Returns:
            A TensorFlow/Keras model
        """
        if not self.tf_available:
            logging.error("TensorFlow not available, cannot create model")
            return None
        
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential, Model
            from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional
            from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, TimeDistributed
            from tensorflow.keras.layers import Input, Concatenate, Attention, MultiHeadAttention
            from tensorflow.keras.optimizers import Adam
            
            model_type = self.model_type.lower()
            
            if model_type == 'lstm':
                # Standard LSTM model
                model = Sequential([
                    LSTM(64, input_shape=input_shape, return_sequences=True),
                    Dropout(0.2),
                    LSTM(32),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dense(output_dim)
                ])
            
            elif model_type == 'gru':
                # GRU model (lighter and faster than LSTM)
                model = Sequential([
                    GRU(64, input_shape=input_shape, return_sequences=True),
                    Dropout(0.2),
                    GRU(32),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dense(output_dim)
                ])
            
            elif model_type == 'bidirectional':
                # Bidirectional LSTM
                model = Sequential([
                    Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
                    Dropout(0.2),
                    Bidirectional(LSTM(32)),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dense(output_dim)
                ])
            
            elif model_type == 'cnn_lstm':
                # CNN-LSTM hybrid
                model = Sequential([
                    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
                    MaxPooling1D(pool_size=2),
                    Conv1D(filters=32, kernel_size=3, activation='relu'),
                    LSTM(32, return_sequences=False),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dense(output_dim)
                ])
            
            elif model_type == 'attention':
                # Attention-based model
                inputs = Input(shape=input_shape)
                lstm = LSTM(64, return_sequences=True)(inputs)
                attention = MultiHeadAttention(num_heads=2, key_dim=32)(lstm, lstm)
                x = tf.keras.layers.Flatten()(attention)
                x = Dense(32, activation='relu')(x)
                x = Dropout(0.2)(x)
                outputs = Dense(output_dim)(x)
                model = Model(inputs=inputs, outputs=outputs)
            
            else:
                # Default to LSTM if model type not recognized
                logging.warning(f"Unknown model type '{model_type}', defaulting to LSTM")
                model = Sequential([
                    LSTM(64, input_shape=input_shape, return_sequences=True),
                    Dropout(0.2),
                    LSTM(32),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dense(output_dim)
                ])
            
            # Compile the model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae']
            )
            
            return model
        
        except Exception as e:
            logging.error(f"Error creating neural network model: {str(e)}")
            return None
    
    def create_ensemble_model(self, input_shape: Tuple[int, int], output_dim: int = 1):
        """
        Create an ensemble model combining multiple architectures
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            output_dim: Number of output predictions
            
        Returns:
            A TensorFlow/Keras ensemble model
        """
        if not self.tf_available:
            logging.error("TensorFlow not available, cannot create ensemble model")
            return None
        
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional
            from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Input, Concatenate
            from tensorflow.keras.optimizers import Adam
            
            # Shared input layer
            inputs = Input(shape=input_shape)
            
            # LSTM branch
            lstm = LSTM(64, return_sequences=False)(inputs)
            lstm = Dropout(0.2)(lstm)
            lstm_output = Dense(16)(lstm)
            
            # GRU branch
            gru = GRU(64, return_sequences=False)(inputs)
            gru = Dropout(0.2)(gru)
            gru_output = Dense(16)(gru)
            
            # CNN branch
            cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
            cnn = MaxPooling1D(pool_size=2)(cnn)
            cnn = Flatten()(cnn)
            cnn = Dense(32, activation='relu')(cnn)
            cnn_output = Dense(16)(cnn)
            
            # Combine branches
            combined = Concatenate()([lstm_output, gru_output, cnn_output])
            combined = Dense(32, activation='relu')(combined)
            combined = Dropout(0.2)(combined)
            output = Dense(output_dim)(combined)
            
            # Create model
            model = Model(inputs=inputs, outputs=output)
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae']
            )
            
            return model
        
        except Exception as e:
            logging.error(f"Error creating ensemble model: {str(e)}")
            return None
    
    async def fetch_historical_data(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """
        Fetch historical market data for model training
        
        Args:
            symbol: The trading symbol (e.g., 'BTCUSDC')
            interval: Timeframe interval (e.g., '1h')
            days: Number of days of historical data to fetch
            
        Returns:
            DataFrame with historical data and indicators
        """
        try:
            # Check if data is available in Redis
            key = f'historical_data_{symbol}_{interval}'
            data_json = await self.redis.get(key)
            
            if data_json:
                # Data is available in Redis
                data = json.loads(data_json)
                df = pd.DataFrame(data)
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Filter for requested time period
                if 'timestamp' in df.columns:
                    cutoff_time = datetime.now() - timedelta(days=days)
                    df = df[df['timestamp'] >= cutoff_time]
                
                logging.info(f"Loaded historical data for {symbol} ({interval}) from Redis: {len(df)} rows")
                
                return df
            else:
                # No data in Redis
                logging.warning(f"No historical data found in Redis for {symbol} ({interval})")
                return pd.DataFrame()
        
        except Exception as e:
            logging.error(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()
    
    def prepare_training_data(self, df: pd.DataFrame, sequence_length: int = 60) -> Tuple:
        """
        Prepare data for neural network training
        
        Args:
            df: DataFrame with historical market data
            sequence_length: Number of time steps to use for sequence prediction
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, scaler)
        """
        if df.empty:
            logging.error("Cannot prepare training data: DataFrame is empty")
            return None, None, None, None, None
        
        try:
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.model_selection import train_test_split
            
            # Select features for prediction
            features = [f for f in self.prediction_features if f in df.columns]
            
            if len(features) < 2:
                logging.error(f"Not enough features available for training. Found: {features}")
                return None, None, None, None, None
            
            # Use 'close' as the target feature if available, otherwise use the first feature
            target_feature = 'close' if 'close' in features else features[0]
            logging.debug(f"Using {target_feature} as the target feature for prediction")
            
            # Create feature dataset
            data = df[features].astype(float).values
            
            # Normalize data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            
            # Create sequences for training
            X, y = [], []
            for i in range(len(scaled_data) - sequence_length):
                X.append(scaled_data[i:i+sequence_length])
                # Use the target feature's index for prediction
                target_idx = features.index(target_feature)
                y.append(scaled_data[i+sequence_length, target_idx])
            
            X, y = np.array(X), np.array(y)
            
            # Split into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            logging.info(f"Prepared training data: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
            
            return X_train, y_train, X_val, y_val, scaler
        
        except Exception as e:
            logging.error(f"Error preparing training data: {str(e)}")
            return None, None, None, None, None
    
    async def train_model(self, symbol: str, interval: str) -> bool:
        """
        Train a neural network model for the specified symbol and interval
        
        Args:
            symbol: Trading symbol
            interval: Time interval for prediction
            
        Returns:
            Boolean indicating training success
        """
        if not self.tf_available:
            logging.error("TensorFlow not available, cannot train model")
            return False
        
        try:
            import tensorflow as tf
            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
            
            logging.info(f"Starting model training for {symbol} ({interval})")
            
            # Fetch historical data
            df = await self.fetch_historical_data(symbol, interval, self.training_lookback_days)
            
            if df.empty:
                logging.error(f"No historical data available for {symbol} ({interval})")
                return False
            
            # Prepare training data
            sequence_length = self.nn_config.get('sequence_length', 60)
            X_train, y_train, X_val, y_val, scaler = self.prepare_training_data(df, sequence_length)
            
            if X_train is None:
                logging.error("Failed to prepare training data")
                return False
            
            # Create or get existing model
            if interval in self.models:
                model = self.models[interval]
                logging.info(f"Using existing model for {interval}")
            else:
                # Create new model
                input_shape = (X_train.shape[1], X_train.shape[2])
                if self.ensemble_enabled:
                    model = self.create_ensemble_model(input_shape)
                else:
                    model = self.create_model(input_shape)
                
                if model is None:
                    logging.error("Failed to create model")
                    return False
                
                self.models[interval] = model
                logging.info(f"Created new {self.model_type} model for {interval}")
            
            # Set up callbacks
            checkpoint_path = os.path.join(self.models_dir, f'nn_model_{self.model_type}_{interval}.h5')
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
            ]
            
            # Train the model
            batch_size = self.nn_config.get('batch_size', 32)
            epochs = self.nn_config.get('epochs', 100)
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Save training history
            self.training_history[interval] = {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'mae': history.history['mae'],
                'val_mae': history.history['val_mae'],
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'interval': interval
            }
            
            # Save training history to Redis
            await self.redis.set(
                f'nn_training_history_{symbol}_{interval}',
                json.dumps(self.training_history[interval])
            )
            
            # Create and save performance visualization
            self._create_training_visualization(history, symbol, interval)
            
            logging.info(f"Model training completed for {symbol} ({interval})")
            self.last_training_time = datetime.now()
            
            return True
        
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            return False
    
    def _create_training_visualization(self, history, symbol: str, interval: str):
        """Create and save visualizations of model training performance"""
        try:
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot training & validation loss
            ax1.plot(history.history['loss'], label='Training Loss')
            ax1.plot(history.history['val_loss'], label='Validation Loss')
            ax1.set_title(f'Model Loss: {symbol} ({interval})')
            ax1.set_ylabel('Loss')
            ax1.set_xlabel('Epoch')
            ax1.legend()
            ax1.grid(True)
            
            # Plot training & validation MAE
            ax2.plot(history.history['mae'], label='Training MAE')
            ax2.plot(history.history['val_mae'], label='Validation MAE')
            ax2.set_title(f'Model MAE: {symbol} ({interval})')
            ax2.set_ylabel('Mean Absolute Error')
            ax2.set_xlabel('Epoch')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Save visualization to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Encode as base64 and store in Redis
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            data_uri = f"data:image/png;base64,{image_base64}"
            
            # Store visualization in Redis asynchronously
            asyncio.create_task(self.redis.set(f'nn_training_viz_{symbol}_{interval}', data_uri))
            
            plt.close(fig)
            
        except Exception as e:
            logging.error(f"Error creating training visualization: {str(e)}")
    
    async def predict_prices(self, symbol: str, interval: str) -> Dict:
        """
        Make price predictions using the trained neural network model
        
        Args:
            symbol: Trading symbol
            interval: Time interval for prediction
            
        Returns:
            Dictionary with prediction results
        """
        if not self.tf_available:
            logging.error("TensorFlow not available, cannot make predictions")
            return {"status": "error", "message": "TensorFlow not available"}
        
        try:
            # Check if model exists for the interval
            if interval not in self.models:
                logging.warning(f"No model available for {interval}, training new model")
                success = await self.train_model(symbol, interval)
                if not success:
                    return {"status": "error", "message": f"Failed to train model for {interval}"}
            
            # Fetch recent data for prediction
            sequence_length = self.nn_config.get('sequence_length', 60)
            df = await self.fetch_historical_data(symbol, interval, 30)  # 30 days should be enough
            
            if df.empty:
                return {"status": "error", "message": "No historical data available for prediction"}
            
            # Prepare prediction data
            features = [f for f in self.prediction_features if f in df.columns]
            
            if len(features) < 2:
                return {"status": "error", "message": f"Not enough features available. Found: {features}"}
            
            # Use 'close' as the target feature if available, otherwise use the first feature
            target_feature = 'close' if 'close' in features else features[0]
            target_idx = features.index(target_feature)
            
            # Get the last price for denormalization
            last_price = df[target_feature].iloc[-1]
            
            # Get the model for this interval
            model = self.models[interval]
            
            # Create scaler and normalize data
            from sklearn.preprocessing import MinMaxScaler
            data = df[features].astype(float).values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            
            # Create prediction sequence from most recent data
            pred_sequence = scaled_data[-sequence_length:]
            pred_sequence = np.reshape(pred_sequence, (1, pred_sequence.shape[0], pred_sequence.shape[1]))
            
            # Make prediction
            prediction = model.predict(pred_sequence)
            
            # Denormalize prediction for the target feature
            # Create a dummy array to denormalize
            dummy = np.zeros((1, len(features)))
            dummy[0, target_idx] = prediction[0][0]
            denormalized = scaler.inverse_transform(dummy)[0, target_idx]
            
            # Calculate prediction timeframe in hours
            hours_map = {
                '1m': 1/60,
                '5m': 5/60,
                '15m': 15/60,
                '30m': 30/60,
                '1h': 1,
                '4h': 4,
                '12h': 12,
                '1d': 24,
                '3d': 72,
                '1w': 168
            }
            prediction_hours = hours_map.get(interval, 1)
            
            # Calculate change percentage
            change_pct = ((denormalized - last_price) / last_price) * 100
            
            # Create prediction timepoint
            current_time = datetime.now()
            prediction_time = current_time + timedelta(hours=prediction_hours)
            
            # Calculate confidence
            # Use prediction variance or uncertainty from model if available
            # For now, use a simple confidence estimate based on training history
            confidence = 0.7
            if interval in self.training_history:
                # Lower validation loss = higher confidence
                val_loss = self.training_history[interval]['val_loss'][-1]
                # Normalize to 0-1 range with a reasonable scale
                confidence = max(0.4, min(0.9, 1.0 - (val_loss * 10)))
            
            # Store prediction result
            result = {
                "symbol": symbol,
                "interval": interval,
                "current_price": float(last_price),
                "predicted_price": float(denormalized),
                "change_pct": float(change_pct),
                "prediction_time": prediction_time.isoformat(),
                "reference_time": current_time.isoformat(),
                "confidence": float(confidence),
                "model_type": self.model_type,
                "status": "success"
            }
            
            # Store prediction in Redis
            await self.redis.set(
                f'nn_prediction_{symbol}_{interval}',
                json.dumps(result)
            )
            
            # Update latest predictions
            self.latest_predictions[(symbol, interval)] = result
            
            # Create and save prediction visualization
            self._create_prediction_visualization(df, symbol, interval, result)
            
            logging.info(f"Made prediction for {symbol} ({interval}): {denormalized:.4f} (change: {change_pct:.2f}%)")
            
            return result
        
        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _create_prediction_visualization(self, df: pd.DataFrame, symbol: str, interval: str, prediction: Dict):
        """Create and save visualization of price prediction"""
        try:
            # Get the target feature (usually 'close')
            target_feature = 'close' if 'close' in df.columns else df.columns[0]
            
            # Get timestamps
            timestamps = df['timestamp'] if 'timestamp' in df.columns else pd.date_range(
                end=datetime.now(), periods=len(df), freq=interval
            )
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot historical prices
            plt.plot(timestamps, df[target_feature], label='Historical Price', color='blue')
            
            # Plot prediction point
            prediction_time = datetime.fromisoformat(prediction['prediction_time'])
            predicted_price = prediction['predicted_price']
            
            # Add prediction point
            plt.scatter([prediction_time], [predicted_price], color='red', marker='o', s=100, label='Prediction')
            
            # Connect the last actual point to the prediction with a dashed line
            plt.plot([timestamps.iloc[-1], prediction_time], 
                     [df[target_feature].iloc[-1], predicted_price], 
                     'r--', label='Predicted Trajectory')
            
            # Add confidence interval if available
            conf = prediction['confidence']
            if conf > 0:
                error_margin = predicted_price * (1 - conf)
                plt.fill_between(
                    [timestamps.iloc[-1], prediction_time],
                    [df[target_feature].iloc[-1], predicted_price - error_margin],
                    [df[target_feature].iloc[-1], predicted_price + error_margin],
                    color='red', alpha=0.2, label='Confidence Interval'
                )
            
            # Add prediction details
            change_pct = prediction['change_pct']
            direction = "↗️" if change_pct > 0 else "↘️"
            plt.title(f"{symbol} Price Prediction ({interval}): {direction} {abs(change_pct):.2f}%")
            plt.xlabel('Time')
            plt.ylabel(f'Price (USDC)')
            plt.grid(True)
            plt.legend()
            
            # Add annotation for the prediction
            plt.annotate(
                f"{predicted_price:.4f} USDC\n{change_pct:+.2f}%",
                xy=(prediction_time, predicted_price),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5)
            )
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            
            # Save visualization to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Encode as base64 and store in Redis
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            data_uri = f"data:image/png;base64,{image_base64}"
            
            # Store visualization in Redis asynchronously
            asyncio.create_task(self.redis.set(f'nn_prediction_viz_{symbol}_{interval}', data_uri))
            
            plt.close()
            
        except Exception as e:
            logging.error(f"Error creating prediction visualization: {str(e)}")
    
    async def maintain_redis(self):
        """Maintain Redis connection"""
        logging.debug("Starting Redis connection maintenance...")
        while self.running:
            try:
                if self.redis:
                    await self.redis.ping()
                else:
                    await self.connect_redis()
                await asyncio.sleep(5)
            except Exception as e:
                logging.error(f"Redis connection error: {str(e)}")
                self.redis = None
                await asyncio.sleep(5)
    
    async def prediction_loop(self):
        """Main loop for making predictions"""
        logging.info("Starting prediction loop...")
        
        # Default symbols to monitor if not specified in config
        symbols = self.nn_config.get('symbols', ['BTCUSDC', 'ETHUSDC', 'BNBUSDC'])
        
        while self.running:
            try:
                for symbol in symbols:
                    for interval in self.prediction_intervals:
                        # Check if prediction is needed (not done recently)
                        key = f'nn_prediction_{symbol}_{interval}'
                        prediction_json = await self.redis.get(key)
                        
                        needs_prediction = True
                        if prediction_json:
                            prediction = json.loads(prediction_json)
                            # If prediction is less than interval old, skip
                            try:
                                pred_time = datetime.fromisoformat(prediction['reference_time'])
                                hours_since_pred = (datetime.now() - pred_time).total_seconds() / 3600
                                
                                # Map interval to hours
                                interval_hours = {
                                    '1m': 1/60, '5m': 5/60, '15m': 15/60, '30m': 30/60,
                                    '1h': 1, '4h': 4, '12h': 12, '1d': 24, '3d': 72, '1w': 168
                                }.get(interval, 1)
                                
                                # Only predict if more than half the interval has passed
                                if hours_since_pred < (interval_hours / 2):
                                    needs_prediction = False
                            except Exception:
                                pass
                        
                        if needs_prediction:
                            logging.info(f"Making prediction for {symbol} ({interval})")
                            prediction = await self.predict_prices(symbol, interval)
                            
                            # Publish prediction to Redis
                            if prediction['status'] == 'success':
                                await self.redis.publish(
                                    'neural_network_predictions',
                                    json.dumps({
                                        'type': 'prediction',
                                        'symbol': symbol,
                                        'interval': interval,
                                        'prediction': prediction
                                    })
                                )
                
                # Check if model training is needed
                if self.last_training_time is None or \
                   (datetime.now() - self.last_training_time).total_seconds() > self.model_checkpoint_interval:
                    logging.info("Periodic model training triggered")
                    for symbol in symbols:
                        for interval in self.prediction_intervals:
                            await self.train_model(symbol, interval)
                
                # Sleep before next prediction cycle
                await asyncio.sleep(60)  # Check every minute
            
            except Exception as e:
                logging.error(f"Error in prediction loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def get_model_summary(self, interval: str) -> Dict:
        """
        Get summary information about a model
        
        Args:
            interval: The interval for the model
            
        Returns:
            Dictionary with model summary information
        """
        if interval not in self.models:
            return {"status": "error", "message": f"No model found for interval {interval}"}
        
        try:
            model = self.models[interval]
            
            # Get model architecture summary
            stringlist = []
            model.summary(print_fn=lambda x: stringlist.append(x))
            summary = "\n".join(stringlist)
            
            # Get training history if available
            history = None
            if interval in self.training_history:
                history = {
                    'loss': self.training_history[interval]['loss'][-1],
                    'val_loss': self.training_history[interval]['val_loss'][-1],
                    'mae': self.training_history[interval]['mae'][-1],
                    'val_mae': self.training_history[interval]['val_mae'][-1],
                    'epochs': len(self.training_history[interval]['loss'])
                }
            
            # Get last prediction if available
            last_predictions = [pred for (sym, intv), pred in self.latest_predictions.items() 
                               if intv == interval]
            
            return {
                "interval": interval,
                "model_type": self.model_type,
                "summary": summary,
                "training_history": history,
                "last_predictions": last_predictions,
                "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
                "status": "success"
            }
        
        except Exception as e:
            logging.error(f"Error getting model summary: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def run(self):
        """Main service runner"""
        try:
            logging.info("Starting Neural Network Service...")
            
            # First establish Redis connection
            if not await self.connect_redis(max_retries=15, retry_delay=2):
                raise Exception("Failed to establish initial Redis connection")
            
            # Initialize neural network models
            await self.initialize_models()
            
            # Create tasks for all service components
            tasks = [
                asyncio.create_task(self.maintain_redis()),
                asyncio.create_task(self.prediction_loop())
            ]
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks)
        
        except Exception as e:
            logging.error(f"Critical error in Neural Network Service: {str(e)}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the service"""
        logging.info("Stopping Neural Network Service...")
        self.running = False
        
        # Save models
        for interval, model in self.models.items():
            try:
                model_path = os.path.join(self.models_dir, f'nn_model_{self.model_type}_{interval}.h5')
                model.save(model_path)
                logging.info(f"Saved model for {interval} to {model_path}")
            except Exception as e:
                logging.error(f"Error saving model for {interval}: {str(e)}")
        
        # Close Redis connection
        if self.redis:
            await self.redis.close()
            logging.info("Closed Redis connection")

if __name__ == "__main__":
    service = NeuralNetworkService()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        logging.info("Service interrupted by user")
        asyncio.run(service.stop())
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        asyncio.run(service.stop())