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
    6. Transformer models for capturing long-range dependencies
    7. Multi-task learning models for predicting multiple timeframes
    8. Probabilistic models for uncertainty estimation
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
            from tensorflow.keras.layers import LayerNormalization, Reshape, Lambda, GlobalAveragePooling1D
            from tensorflow.keras.optimizers import Adam
            import tensorflow_probability as tfp
            
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
            
            elif model_type == 'transformer':
                # Transformer model for time series
                inputs = Input(shape=input_shape)
                
                # Positional encoding to preserve temporal information
                def positional_encoding(position, d_model):
                    def get_angles(pos, i, d_model):
                        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
                        return pos * angle_rates
                    
                    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                                          np.arange(d_model)[np.newaxis, :],
                                          d_model)
                    
                    # Apply sin to even indices and cos to odd indices
                    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
                    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
                    
                    pos_encoding = angle_rads[np.newaxis, ...]
                    return tf.cast(pos_encoding, dtype=tf.float32)
                
                # Create the positional encoding layer
                pos_encoding = positional_encoding(input_shape[0], input_shape[1])
                pos_encoding_layer = Lambda(lambda x: x + pos_encoding)(inputs)
                
                # Multi-head attention layers
                x = LayerNormalization(epsilon=1e-6)(pos_encoding_layer)
                
                # First transformer block
                attn_output1 = MultiHeadAttention(
                    num_heads=4, key_dim=input_shape[1]//4)(x, x)
                attn_output1 = Dropout(0.1)(attn_output1)
                out1 = LayerNormalization(epsilon=1e-6)(x + attn_output1)
                
                # Feed forward network
                ffn_output = Dense(64, activation='relu')(out1)
                ffn_output = Dense(input_shape[1])(ffn_output)
                ffn_output = Dropout(0.1)(ffn_output)
                
                # Second transformer block
                out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
                attn_output2 = MultiHeadAttention(
                    num_heads=4, key_dim=input_shape[1]//4)(out2, out2)
                attn_output2 = Dropout(0.1)(attn_output2)
                out3 = LayerNormalization(epsilon=1e-6)(out2 + attn_output2)
                
                # Final feed forward network
                ffn_output2 = Dense(64, activation='relu')(out3)
                ffn_output2 = Dense(input_shape[1])(ffn_output2)
                ffn_output2 = Dropout(0.1)(ffn_output2)
                out4 = LayerNormalization(epsilon=1e-6)(out3 + ffn_output2)
                
                # Global average pooling to get a single vector per sequence
                pooled = GlobalAveragePooling1D()(out4)
                
                # Final dense layers
                x = Dense(32, activation='relu')(pooled)
                outputs = Dense(output_dim)(x)
                
                model = Model(inputs=inputs, outputs=outputs)
            
            elif model_type == 'multitask':
                # Multi-task model predicting multiple timeframes
                inputs = Input(shape=input_shape)
                
                # Shared layers
                shared = LSTM(64, return_sequences=True)(inputs)
                shared = Dropout(0.2)(shared)
                shared = LSTM(32)(shared)
                shared = Dropout(0.2)(shared)
                
                # Task-specific layers (predicting different timeframes)
                # Short-term prediction (e.g., 1h)
                short_term = Dense(16, activation='relu')(shared)
                short_term_output = Dense(1, name='short_term')(short_term)
                
                # Medium-term prediction (e.g., 4h)
                medium_term = Dense(16, activation='relu')(shared)
                medium_term_output = Dense(1, name='medium_term')(medium_term)
                
                # Long-term prediction (e.g., 24h)
                long_term = Dense(16, activation='relu')(shared)
                long_term_output = Dense(1, name='long_term')(long_term)
                
                # Create model with multiple outputs
                model = Model(
                    inputs=inputs, 
                    outputs=[short_term_output, medium_term_output, long_term_output]
                )
                
                # Compile with different loss weights for each task
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss={
                        'short_term': 'mean_squared_error',
                        'medium_term': 'mean_squared_error',
                        'long_term': 'mean_squared_error'
                    },
                    loss_weights={
                        'short_term': 1.0,
                        'medium_term': 0.7,
                        'long_term': 0.5
                    },
                    metrics=['mae']
                )
                
                return model  # Return early as compilation is different
            
            elif model_type == 'probabilistic':
                # Probabilistic model for uncertainty estimation
                try:
                    # Use TensorFlow Probability for distribution outputs
                    import tensorflow_probability as tfp
                    tfd = tfp.distributions
                    
                    inputs = Input(shape=input_shape)
                    x = LSTM(64, return_sequences=True)(inputs)
                    x = Dropout(0.2)(x)
                    x = LSTM(32)(x)
                    x = Dropout(0.2)(x)
                    x = Dense(16, activation='relu')(x)
                    
                    # Output parameters for normal distribution (mean and standard deviation)
                    mu = Dense(output_dim)(x)  # Mean prediction
                    sigma = Dense(output_dim, activation='softplus')(x)  # Standard deviation
                    
                    # Create distribution output
                    dist = tfp.layers.DistributionLambda(
                        lambda params: tfd.Normal(loc=params[0], scale=params[1]),
                        name='normal_distribution'
                    )([mu, sigma])
                    
                    model = Model(inputs=inputs, outputs=dist)
                    
                    # Use negative log likelihood loss
                    def neg_log_likelihood(y, dist):
                        return -dist.log_prob(y)
                    
                    model.compile(
                        optimizer=Adam(learning_rate=0.001),
                        loss=neg_log_likelihood,
                        metrics=['mae']
                    )
                    
                    return model  # Return early as compilation is different
                    
                except ImportError:
                    logging.warning("TensorFlow Probability not available, falling back to LSTM")
                    model_type = 'lstm'  # Fall back to LSTM
            
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
            
            # Compile the model (skip for models that need custom compilation)
            if model_type not in ['multitask', 'probabilistic']:
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
    
    async def optimize_hyperparameters(self, symbol: str, interval: str) -> Dict:
        """
        Optimize hyperparameters for model using Bayesian optimization
        
        Args:
            symbol: Trading symbol
            interval: Time interval for prediction
            
        Returns:
            Dictionary with best hyperparameters and optimization results
        """
        if not self.tf_available:
            logging.error("TensorFlow not available, cannot optimize hyperparameters")
            return {"status": "error", "message": "TensorFlow not available"}
        
        try:
            import optuna
            from tensorflow.keras.callbacks import EarlyStopping
            
            logging.info(f"Starting hyperparameter optimization for {symbol} ({interval})")
            
            # Fetch historical data
            df = await self.fetch_historical_data(symbol, interval, self.training_lookback_days)
            
            if df.empty:
                logging.error(f"No historical data available for {symbol} ({interval})")
                return {"status": "error", "message": "No historical data available"}
            
            # Prepare training data
            sequence_length = self.nn_config.get('sequence_length', 60)
            X_train, y_train, X_val, y_val, scaler = self.prepare_training_data(df, sequence_length)
            
            if X_train is None:
                logging.error("Failed to prepare training data")
                return {"status": "error", "message": "Failed to prepare training data"}
            
            # Get input shape for model creation
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            # Define the objective function for optimization
            def objective(trial):
                # Define hyperparameters to search
                model_type = trial.suggest_categorical('model_type', 
                                                     ['lstm', 'gru', 'bidirectional', 'attention'])
                
                # Layer sizes
                first_layer_units = trial.suggest_int('first_layer_units', 32, 128, 32)
                second_layer_units = trial.suggest_int('second_layer_units', 16, 64, 16)
                
                # Dropout rates
                dropout1 = trial.suggest_float('dropout1', 0.1, 0.5, step=0.1)
                dropout2 = trial.suggest_float('dropout2', 0.1, 0.5, step=0.1)
                
                # Learning rate
                learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                
                # Temporary override model type
                original_model_type = self.model_type
                self.model_type = model_type
                
                # Create a custom model
                model = None
                if model_type == 'lstm':
                    model = Sequential([
                        LSTM(first_layer_units, input_shape=input_shape, return_sequences=True),
                        Dropout(dropout1),
                        LSTM(second_layer_units),
                        Dropout(dropout2),
                        Dense(16, activation='relu'),
                        Dense(1)
                    ])
                elif model_type == 'gru':
                    model = Sequential([
                        GRU(first_layer_units, input_shape=input_shape, return_sequences=True),
                        Dropout(dropout1),
                        GRU(second_layer_units),
                        Dropout(dropout2),
                        Dense(16, activation='relu'),
                        Dense(1)
                    ])
                elif model_type == 'bidirectional':
                    model = Sequential([
                        Bidirectional(LSTM(first_layer_units, return_sequences=True), input_shape=input_shape),
                        Dropout(dropout1),
                        Bidirectional(LSTM(second_layer_units)),
                        Dropout(dropout2),
                        Dense(16, activation='relu'),
                        Dense(1)
                    ])
                elif model_type == 'attention':
                    inputs = Input(shape=input_shape)
                    lstm = LSTM(first_layer_units, return_sequences=True)(inputs)
                    lstm = Dropout(dropout1)(lstm)
                    attention = MultiHeadAttention(
                        num_heads=trial.suggest_int('num_heads', 1, 4), 
                        key_dim=16
                    )(lstm, lstm)
                    x = tf.keras.layers.Flatten()(attention)
                    x = Dense(second_layer_units, activation='relu')(x)
                    x = Dropout(dropout2)(x)
                    outputs = Dense(1)(x)
                    model = Model(inputs=inputs, outputs=outputs)
                
                # Restore original model type
                self.model_type = original_model_type
                
                if model is None:
                    return float('inf')  # Return large error value if model creation fails
                
                # Compile model
                optimizer = Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
                
                # Set up callbacks
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                ]
                
                # Train model
                batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
                
                # Try/catch block for training
                try:
                    history = model.fit(
                        X_train, y_train,
                        epochs=30,  # Use fewer epochs for optimization
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks,
                        verbose=0  # Silent
                    )
                    
                    # Get the best validation loss
                    val_loss = min(history.history['val_loss'])
                    return val_loss
                    
                except Exception as e:
                    logging.error(f"Error training model in optimization: {str(e)}")
                    return float('inf')  # Return large error value if training fails
            
            # Create study and optimize
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=self.nn_config.get('optuna_trials', 20))
            
            # Get best parameters
            best_params = study.best_params
            best_val_loss = study.best_value
            
            # Log results
            logging.info(f"Hyperparameter optimization completed for {symbol} ({interval})")
            logging.info(f"Best parameters: {best_params}")
            logging.info(f"Best validation loss: {best_val_loss:.6f}")
            
            # Store optimization results in Redis
            optimization_results = {
                "symbol": symbol,
                "interval": interval,
                "best_params": best_params,
                "best_val_loss": best_val_loss,
                "timestamp": datetime.now().isoformat(),
                "model_type": best_params.get('model_type', self.model_type),
                "status": "success"
            }
            
            await self.redis.set(
                f'nn_hyperparameter_optimization_{symbol}_{interval}',
                json.dumps(optimization_results)
            )
            
            # Create a visualization of the optimization process
            self._create_optimization_visualization(study, symbol, interval)
            
            return optimization_results
            
        except ImportError as e:
            logging.error(f"Required package for hyperparameter optimization not available: {str(e)}")
            return {"status": "error", "message": f"Required package not available: {str(e)}"}
        except Exception as e:
            logging.error(f"Error during hyperparameter optimization: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    def _create_optimization_visualization(self, study, symbol: str, interval: str):
        """Create and save visualization of hyperparameter optimization results"""
        try:
            import optuna
            
            # Create figure with optimization history
            fig = optuna.visualization.plot_optimization_history(study)
            fig.update_layout(title=f"Optimization History: {symbol} ({interval})")
            
            # Convert to PNG
            img_bytes = fig.to_image(format="png")
            
            # Encode as base64 and store in Redis
            image_base64 = base64.b64encode(img_bytes).decode('utf-8')
            data_uri = f"data:image/png;base64,{image_base64}"
            
            # Store visualization in Redis asynchronously
            asyncio.create_task(self.redis.set(f'nn_optim_viz_{symbol}_{interval}', data_uri))
            
            # Create a second visualization of parameter importances
            fig2 = optuna.visualization.plot_param_importances(study)
            fig2.update_layout(title=f"Parameter Importances: {symbol} ({interval})")
            
            # Convert to PNG
            img_bytes2 = fig2.to_image(format="png")
            
            # Encode as base64 and store in Redis
            image_base64_2 = base64.b64encode(img_bytes2).decode('utf-8')
            data_uri_2 = f"data:image/png;base64,{image_base64_2}"
            
            # Store visualization in Redis asynchronously
            asyncio.create_task(self.redis.set(f'nn_optim_param_viz_{symbol}_{interval}', data_uri_2))
            
        except Exception as e:
            logging.error(f"Error creating optimization visualization: {str(e)}")
    
    async def train_model(self, symbol: str, interval: str, use_optimized_params: bool = True) -> bool:
        """
        Train a neural network model for the specified symbol and interval
        
        Args:
            symbol: Trading symbol
            interval: Time interval for prediction
            use_optimized_params: Whether to use hyperparameter optimization results
            
        Returns:
            Boolean indicating training success
        """
        if not self.tf_available:
            logging.error("TensorFlow not available, cannot train model")
            return False
        
        try:
            import tensorflow as tf
            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
            
            logging.info(f"Starting model training for {symbol} ({interval})")
            
            # Check if we should use optimized hyperparameters
            optimized_params = None
            if use_optimized_params:
                optim_results = await self.redis.get(f'nn_hyperparameter_optimization_{symbol}_{interval}')
                if optim_results:
                    try:
                        optimized_params = json.loads(optim_results)
                        logging.info(f"Using optimized hyperparameters for {symbol} ({interval})")
                    except Exception as e:
                        logging.error(f"Error parsing optimized params: {str(e)}")
            
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
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            if optimized_params and 'best_params' in optimized_params:
                # Create a model with optimized parameters
                best_params = optimized_params['best_params']
                model_type = best_params.get('model_type', self.model_type)
                
                # Save current model type
                original_model_type = self.model_type
                
                # Temporarily set model type to optimized one for model creation
                self.model_type = model_type
                
                # Create model
                model = self.create_model(input_shape)
                
                # Restore original model type
                self.model_type = original_model_type
                
                # Update model in dictionary if created successfully
                if model is not None:
                    self.models[interval] = model
                    logging.info(f"Created optimized {model_type} model for {interval}")
                else:
                    # Fall back to regular model creation
                    if interval in self.models:
                        model = self.models[interval]
                        logging.info(f"Using existing model for {interval}")
                    else:
                        model = self.create_model(input_shape) if not self.ensemble_enabled else self.create_ensemble_model(input_shape)
                        if model is not None:
                            self.models[interval] = model
                            logging.info(f"Created new {self.model_type} model for {interval}")
            else:
                # Regular model creation (without optimized params)
                if interval in self.models:
                    model = self.models[interval]
                    logging.info(f"Using existing model for {interval}")
                else:
                    # Create new model
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
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss'),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)
            ]
            
            # Train the model
            batch_size = self.nn_config.get('batch_size', 32)
            epochs = self.nn_config.get('epochs', 100)
            
            # Use SHAP for feature importance if requested
            enable_shap = self.nn_config.get('enable_shap', False)
            
            # If using a probabilistic model, handle the different output type
            if hasattr(model, 'output') and 'normal_distribution' in str(model.output):
                # Special handling for probabilistic model training
                logging.info("Training probabilistic model...")
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Save training history
            if 'loss' in history.history and 'val_loss' in history.history:
                self.training_history[interval] = {
                    'loss': history.history['loss'],
                    'val_loss': history.history['val_loss'],
                    'mae': history.history.get('mae', []),
                    'val_mae': history.history.get('val_mae', []),
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
            
            # Generate SHAP values for feature importance if enabled
            if enable_shap:
                try:
                    import shap
                    # Get a sample batch of data for SHAP analysis
                    sample_data = X_train[:100]  # Use a small subset for speed
                    
                    # Create a background dataset
                    background = X_train[:10]
                    
                    # Create an explainer
                    explainer = shap.DeepExplainer(model, background)
                    
                    # Calculate SHAP values
                    shap_values = explainer.shap_values(sample_data)
                    
                    # Store feature importance based on mean absolute SHAP values
                    feature_importance = {}
                    for i, feature in enumerate(self.prediction_features):
                        # Average across all timesteps for this feature
                        if isinstance(shap_values, list):
                            # For classification models
                            feature_importance[feature] = float(np.abs(shap_values[0][:, :, i]).mean())
                        else:
                            # For regression models
                            feature_importance[feature] = float(np.abs(shap_values[:, :, i]).mean())
                    
                    # Sort by importance
                    sorted_importance = {k: v for k, v in sorted(
                        feature_importance.items(), key=lambda item: item[1], reverse=True)}
                    
                    # Save feature importance to Redis
                    await self.redis.set(
                        f'nn_feature_importance_{symbol}_{interval}',
                        json.dumps({
                            'feature_importance': sorted_importance,
                            'timestamp': datetime.now().isoformat(),
                            'symbol': symbol,
                            'interval': interval
                        })
                    )
                    
                    # Create feature importance visualization
                    self._create_feature_importance_visualization(sorted_importance, symbol, interval)
                    
                except Exception as e:
                    logging.error(f"Error generating SHAP feature importance: {str(e)}")
            
            logging.info(f"Model training completed for {symbol} ({interval})")
            self.last_training_time = datetime.now()
            
            return True
        
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            return False
            
    def _create_feature_importance_visualization(self, feature_importance: Dict, symbol: str, interval: str):
        """Create and save visualization of feature importance"""
        try:
            # Sort features by importance
            sorted_items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            features = [item[0] for item in sorted_items]
            importance = [item[1] for item in sorted_items]
            
            # Create bar chart
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(features)), importance, align='center')
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title(f'Feature Importance: {symbol} ({interval})')
            plt.tight_layout()
            
            # Save visualization to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Encode as base64 and store in Redis
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            data_uri = f"data:image/png;base64,{image_base64}"
            
            # Store visualization in Redis asynchronously
            asyncio.create_task(self.redis.set(f'nn_feature_importance_viz_{symbol}_{interval}', data_uri))
            
            plt.close()
            
        except Exception as e:
            logging.error(f"Error creating feature importance visualization: {str(e)}")
    
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
        
        # Track hyperparameter optimization intervals
        optimization_intervals = {}
        
        while self.running:
            try:
                # First check for hyperparameter optimization requests
                optimization_request = await self.redis.get('nn_optimization_request')
                if optimization_request:
                    try:
                        request = json.loads(optimization_request)
                        if 'symbol' in request and 'interval' in request:
                            symbol = request['symbol']
                            interval = request['interval']
                            
                            # Check if we're already optimizing this interval
                            if (symbol, interval) not in optimization_intervals:
                                logging.info(f"Starting hyperparameter optimization for {symbol} ({interval})")
                                
                                # Mark as in progress
                                optimization_intervals[(symbol, interval)] = datetime.now()
                                
                                # Run optimization in background task
                                asyncio.create_task(self._run_optimization(symbol, interval))
                                
                                # Clear the request
                                await self.redis.delete('nn_optimization_request')
                    except Exception as e:
                        logging.error(f"Error processing optimization request: {str(e)}")
                        await self.redis.delete('nn_optimization_request')
                
                # Clean up completed optimizations
                current_time = datetime.now()
                for key in list(optimization_intervals.keys()):
                    if (current_time - optimization_intervals[key]).total_seconds() > 3600:  # 1 hour timeout
                        del optimization_intervals[key]
                
                # Make predictions for each symbol and interval
                for symbol in symbols:
                    for interval in self.prediction_intervals:
                        # Skip if currently optimizing
                        if (symbol, interval) in optimization_intervals:
                            continue
                            
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
                            # Skip if currently optimizing
                            if (symbol, interval) in optimization_intervals:
                                continue
                                
                            # Check if we should optimize hyperparameters instead of regular training
                            auto_optimize = self.nn_config.get('auto_optimize', False)
                            optimization_interval = self.nn_config.get('optimization_interval', 604800)  # 7 days default
                            
                            if auto_optimize:
                                # Check when the last optimization was performed
                                last_optimization = await self.redis.get(f'nn_last_optimization_{symbol}_{interval}')
                                
                                if last_optimization:
                                    try:
                                        last_optim_time = datetime.fromisoformat(last_optimization)
                                        time_since_optim = (datetime.now() - last_optim_time).total_seconds()
                                        
                                        if time_since_optim > optimization_interval:
                                            # Time to re-optimize
                                            logging.info(f"Auto-optimization triggered for {symbol} ({interval})")
                                            optimization_intervals[(symbol, interval)] = datetime.now()
                                            asyncio.create_task(self._run_optimization(symbol, interval))
                                            continue
                                    except Exception as e:
                                        logging.error(f"Error checking optimization time: {str(e)}")
                                else:
                                    # No record of previous optimization, run it
                                    logging.info(f"First auto-optimization for {symbol} ({interval})")
                                    optimization_intervals[(symbol, interval)] = datetime.now()
                                    asyncio.create_task(self._run_optimization(symbol, interval))
                                    continue
                            
                            # Regular training if not optimizing
                            await self.train_model(symbol, interval)
                
                # Check if we should integrate with market regime detector
                integrate_with_regime = self.nn_config.get('integrate_with_regime', False)
                if integrate_with_regime:
                    # Get current market regime
                    market_regime_data = await self.redis.get('market_regime_history')
                    if market_regime_data:
                        try:
                            regime_history = json.loads(market_regime_data)
                            if regime_history and len(regime_history) > 0:
                                # Get the most recent regime
                                current_regime = regime_history[-1]
                                regime_name = current_regime.get('regime', 'unknown')
                                
                                # Store regime-specific model if needed
                                for symbol in symbols:
                                    for interval in self.prediction_intervals:
                                        if regime_name != 'unknown' and interval in self.models:
                                            model_path = os.path.join(
                                                self.models_dir, 
                                                f'nn_model_{self.model_type}_{interval}_{regime_name}.h5'
                                            )
                                            # Save a copy of the model trained for this specific regime
                                            try:
                                                self.models[interval].save(model_path)
                                                logging.info(f"Saved regime-specific model for {regime_name}")
                                            except Exception as e:
                                                logging.error(f"Error saving regime model: {str(e)}")
                        except Exception as e:
                            logging.error(f"Error processing market regime data: {str(e)}")
                
                # Sleep before next prediction cycle
                await asyncio.sleep(60)  # Check every minute
            
            except Exception as e:
                logging.error(f"Error in prediction loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _run_optimization(self, symbol: str, interval: str):
        """Run hyperparameter optimization and then train with best params"""
        try:
            # Run optimization
            optimization_results = await self.optimize_hyperparameters(symbol, interval)
            
            if optimization_results.get('status') == 'success':
                # Store the timestamp of this optimization
                await self.redis.set(
                    f'nn_last_optimization_{symbol}_{interval}',
                    datetime.now().isoformat()
                )
                
                # Train with the optimized parameters
                await self.train_model(symbol, interval, use_optimized_params=True)
                
                # Publish completion notification
                await self.redis.publish(
                    'neural_network_events',
                    json.dumps({
                        'type': 'optimization_complete',
                        'symbol': symbol,
                        'interval': interval,
                        'timestamp': datetime.now().isoformat(),
                        'best_params': optimization_results.get('best_params', {})
                    })
                )
            else:
                logging.error(f"Optimization failed for {symbol} ({interval})")
        except Exception as e:
            logging.error(f"Error in optimization task: {str(e)}")
    
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