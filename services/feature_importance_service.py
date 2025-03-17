import os
import json
import socket
import asyncio
import logging as logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from logging.handlers import RotatingFileHandler
from redis.asyncio import Redis
from redis.exceptions import ConnectionError
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.inspection import permutation_importance
import shap
import pickle
import io
import base64

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure rotating file handler
rotating_handler = RotatingFileHandler(
    'logs/feature_importance.log',
    maxBytes=10*1024*1024,  # 10MB per file
    backupCount=5,  # Keep 5 backup files
    encoding='utf-8'
)

# Configure logging with rotation
logger.basicConfig(
    level=logger.DEBUG,
    format='%(asctime)s - %(levelname)s - [FeatureImportance] %(message)s',
    handlers=[
        rotating_handler,
        logger.StreamHandler()
    ]
)

class FeatureImportanceService:
    """Service for analyzing and reporting the importance of different features in trading decisions."""
    
    def __init__(self):
        """Initialize the Feature Importance Service."""
        logger.debug("Initializing Feature Importance Service...")
        
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        logger.debug(f"Loaded configuration")
        
        # Redis configuration
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        logger.debug(f"Redis configuration - Host: {self.redis_host}, Port: {self.redis_port}")
        
        # Redis will be initialized in connect_redis
        self.redis = None
        self.running = True
        
        # Analysis parameters
        self.analysis_interval = 3600  # 1 hour
        self.minimum_samples = 100  # Minimum data points needed for analysis
        self.lookback_days = 7  # Amount of history to consider
        self.feature_cache = {}  # Cache for feature importance results
        self.update_frequency = 3600  # How often to update models (1 hour)
        self.last_update = datetime.min
        
        # Service port
        self.service_port = int(os.getenv('FEATURE_IMPORTANCE_PORT', 8010))
        logger.debug(f"Service port configured as: {self.service_port}")
        
        # Initialize models
        self.classifier = None
        self.regressor = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Feature groups
        self.feature_groups = {
            "price_action": ["current_price", "price_change_1m", "price_change_3m", 
                           "price_change_5m", "price_change_15m"],
            "momentum": ["rsi", "rsi_3m", "rsi_5m", "stoch_k", "williams_r"],
            "trend": ["trend_strength", "macd", "macd_3m", "macd_5m"],
            "volatility": ["bb_position"],
            "social": ["social_volume", "social_engagement", "social_sentiment", "social_contributors"]
        }
        
        logger.info("Feature Importance Service initialized")
    
    async def connect_redis(self, max_retries=10, retry_delay=5):
        """Establish Redis connection with retries."""
        retries = 0
        while retries < max_retries and self.running:
            try:
                if self.redis is None:
                    logger.debug(f"Attempting Redis connection (attempt {retries + 1}/{max_retries})")
                    self.redis = Redis(
                        host=self.redis_host,
                        port=self.redis_port,
                        decode_responses=True,
                        socket_connect_timeout=5.0,
                        socket_keepalive=True,
                        health_check_interval=15
                    )
                await self.redis.ping()
                logger.info(f"Successfully connected to Redis at {self.redis_host}:{self.redis_port}")
                return True
            except (ConnectionError, OSError) as e:
                retries += 1
                logger.error(f"Failed to connect to Redis (attempt {retries}/{max_retries}): {str(e)}")
                if self.redis:
                    await self.redis.close()
                    self.redis = None
                if retries < max_retries:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("Max retries reached. Could not connect to Redis.")
                    return False
            except Exception as e:
                logger.error(f"Unexpected error connecting to Redis: {str(e)}")
                if self.redis:
                    await self.redis.close()
                    self.redis = None
                await asyncio.sleep(retry_delay)
                retries += 1
    
    async def get_historical_trading_data(self) -> pd.DataFrame:
        """Fetch historical trading signals and their outcomes for analysis."""
        try:
            # Get raw data from Redis
            signals_data = await self.redis.get('historical_trading_signals')
            outcomes_data = await self.redis.get('historical_trading_outcomes')
            
            if not signals_data or not outcomes_data:
                logger.warning("No historical trading data found in Redis")
                return None
                
            # Parse JSON data
            signals = json.loads(signals_data)
            outcomes = json.loads(outcomes_data)
            
            # Combine signals with outcomes
            combined_data = []
            for signal_id, signal in signals.items():
                if signal_id in outcomes:
                    # Combine signal features with outcome
                    combined_signal = {
                        'signal_id': signal_id,
                        'timestamp': signal.get('timestamp'),
                        'symbol': signal.get('symbol'),
                        'decision': signal.get('decision'),  # BUY, SELL, HOLD
                        'confidence': signal.get('confidence'),
                        # Extract features from market_data
                        **signal.get('market_data', {}),
                        # Add outcome
                        'outcome': outcomes[signal_id].get('outcome'),  # success, failure
                        'profit_pct': outcomes[signal_id].get('profit_pct', 0),
                        'holding_time': outcomes[signal_id].get('holding_time', 0),
                    }
                    combined_data.append(combined_signal)
            
            # Convert to DataFrame
            df = pd.DataFrame(combined_data)
            
            # Filter by date if needed
            if self.lookback_days > 0:
                cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
                df['datetime'] = pd.to_datetime(df['timestamp'])
                df = df[df['datetime'] >= cutoff_date]
            
            # Drop non-feature columns
            cols_to_drop = ['signal_id', 'timestamp', 'symbol', 'datetime', 'decision', 
                           'market_context', 'recent_news']
            feature_cols = [col for col in df.columns if col not in cols_to_drop]
            
            # Store feature names for later use
            self.feature_names = feature_cols
            
            logger.info(f"Retrieved {len(df)} historical trading data points")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical trading data: {str(e)}")
            return None
    
    def train_models(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Train machine learning models on historical data for feature importance."""
        try:
            if df is None or len(df) < self.minimum_samples:
                logger.warning(f"Insufficient data for model training: {len(df) if df is not None else 0} samples")
                return False, {}
            
            # Prepare classification data (predicting trading decision success)
            X = df[self.feature_names].copy()
            y_class = (df['outcome'] == 'success').astype(int)  # Binary classification target
            y_reg = df['profit_pct'].copy()  # Regression target
            
            # Handle missing values
            X = X.fillna(0)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
                X_scaled, y_class, y_reg, test_size=0.2, random_state=42
            )
            
            # Train classification model
            logger.info("Training classification model...")
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.classifier.fit(X_train, y_class_train)
            
            # Train regression model
            logger.info("Training regression model...")
            self.regressor = RandomForestRegressor(n_estimators=100, random_state=42)
            self.regressor.fit(X_train, y_reg_train)
            
            # Evaluate models
            y_class_pred = self.classifier.predict(X_test)
            y_reg_pred = self.regressor.predict(X_test)
            
            class_accuracy = accuracy_score(y_class_test, y_class_pred)
            class_precision = precision_score(y_class_test, y_class_pred, zero_division=0)
            class_recall = recall_score(y_class_test, y_class_pred, zero_division=0)
            class_f1 = f1_score(y_class_test, y_class_pred, zero_division=0)
            
            reg_mse = mean_squared_error(y_reg_test, y_reg_pred)
            reg_rmse = np.sqrt(reg_mse)
            
            logger.info(f"Classification metrics - Accuracy: {class_accuracy:.4f}, Precision: {class_precision:.4f}, Recall: {class_recall:.4f}, F1: {class_f1:.4f}")
            logger.info(f"Regression metrics - RMSE: {reg_rmse:.4f}")
            
            # Return metrics for reporting
            metrics = {
                "classification": {
                    "accuracy": class_accuracy,
                    "precision": class_precision,
                    "recall": class_recall,
                    "f1": class_f1,
                    "samples": len(df)
                },
                "regression": {
                    "rmse": reg_rmse,
                    "mse": reg_mse,
                    "samples": len(df)
                }
            }
            
            return True, metrics
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return False, {}
    
    def calculate_feature_importance(self) -> Dict:
        """Calculate feature importance using multiple methods."""
        try:
            if self.classifier is None or self.regressor is None:
                logger.warning("Models not trained yet, cannot calculate feature importance")
                return {}
            
            importance_results = {
                "classification": {},
                "regression": {},
                "permutation": {},
                "feature_groups": {},
                "shap": {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Built-in feature importance from Random Forest
            clf_importance = self.classifier.feature_importances_
            reg_importance = self.regressor.feature_importances_
            
            # Store in dictionary with feature names
            importance_results["classification"] = {
                self.feature_names[i]: float(clf_importance[i]) 
                for i in range(len(self.feature_names))
            }
            
            importance_results["regression"] = {
                self.feature_names[i]: float(reg_importance[i]) 
                for i in range(len(self.feature_names))
            }
            
            # Generate feature group importance
            group_importance = {}
            for group_name, features in self.feature_groups.items():
                # Only include features that exist in our data
                valid_features = [f for f in features if f in self.feature_names]
                if not valid_features:
                    continue
                    
                # Sum importance for all features in group
                group_clf_importance = sum(importance_results["classification"].get(f, 0) for f in valid_features)
                group_reg_importance = sum(importance_results["regression"].get(f, 0) for f in valid_features)
                
                group_importance[group_name] = {
                    "classification": float(group_clf_importance),
                    "regression": float(group_reg_importance),
                    "features": valid_features
                }
            
            importance_results["feature_groups"] = group_importance
            
            # Generate plots and attach them for visualization
            self.generate_importance_plots(importance_results)
            
            # Cache the results
            self.feature_cache = importance_results
            
            logger.info(f"Feature importance calculated for {len(self.feature_names)} features")
            return importance_results
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {}
    
    def generate_importance_plots(self, importance_results: Dict) -> None:
        """Generate visualizations for feature importance and attach to results."""
        try:
            # Convert dictionaries to sorted lists for plotting
            clf_features = list(importance_results["classification"].keys())
            clf_importance = list(importance_results["classification"].values())
            
            reg_features = list(importance_results["regression"].keys())
            reg_importance = list(importance_results["regression"].values())
            
            # Sort by importance
            clf_sorted_idx = np.argsort(clf_importance)[::-1]
            reg_sorted_idx = np.argsort(reg_importance)[::-1]
            
            clf_sorted_features = [clf_features[i] for i in clf_sorted_idx]
            clf_sorted_importance = [clf_importance[i] for i in clf_sorted_idx]
            
            reg_sorted_features = [reg_features[i] for i in reg_sorted_idx]
            reg_sorted_importance = [reg_importance[i] for i in reg_sorted_idx]
            
            # Limit to top 15 features for readability
            clf_sorted_features = clf_sorted_features[:15]
            clf_sorted_importance = clf_sorted_importance[:15]
            reg_sorted_features = reg_sorted_features[:15]
            reg_sorted_importance = reg_sorted_importance[:15]
            
            # Create classification importance plot
            plt.figure(figsize=(10, 8))
            plt.barh(clf_sorted_features, clf_sorted_importance)
            plt.xlabel('Feature Importance')
            plt.ylabel('Features')
            plt.title('Classification Feature Importance (Trade Success Prediction)')
            plt.tight_layout()
            
            # Save plot to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Encode as base64 string
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            importance_results["plots"] = {
                "classification": f"data:image/png;base64,{image_base64}"
            }
            plt.close()
            
            # Create regression importance plot
            plt.figure(figsize=(10, 8))
            plt.barh(reg_sorted_features, reg_sorted_importance)
            plt.xlabel('Feature Importance')
            plt.ylabel('Features')
            plt.title('Regression Feature Importance (Profit Prediction)')
            plt.tight_layout()
            
            # Save plot to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Encode as base64 string
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            importance_results["plots"]["regression"] = f"data:image/png;base64,{image_base64}"
            plt.close()
            
            # Create feature group importance plot
            group_names = list(importance_results["feature_groups"].keys())
            group_clf_importance = [g["classification"] for g in importance_results["feature_groups"].values()]
            
            # Sort groups by importance
            group_sorted_idx = np.argsort(group_clf_importance)[::-1]
            group_sorted_names = [group_names[i] for i in group_sorted_idx]
            group_sorted_importance = [group_clf_importance[i] for i in group_sorted_idx]
            
            plt.figure(figsize=(10, 6))
            plt.barh(group_sorted_names, group_sorted_importance)
            plt.xlabel('Feature Group Importance')
            plt.ylabel('Feature Groups')
            plt.title('Feature Group Importance for Trading Success')
            plt.tight_layout()
            
            # Save plot to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Encode as base64 string
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            importance_results["plots"]["feature_groups"] = f"data:image/png;base64,{image_base64}"
            plt.close()
            
            logger.info("Generated feature importance visualizations")
            
        except Exception as e:
            logger.error(f"Error generating importance plots: {str(e)}")
    
    async def analyze_feature_importance(self) -> Dict:
        """Run a complete feature importance analysis cycle."""
        try:
            # Check if we need to update the models
            current_time = datetime.now()
            time_since_update = (current_time - self.last_update).total_seconds()
            
            if time_since_update < self.update_frequency and self.feature_cache:
                logger.info(f"Using cached feature importance (updated {time_since_update:.0f}s ago)")
                return self.feature_cache
                
            # Get historical data
            df = await self.get_historical_trading_data()
            
            if df is None or len(df) < self.minimum_samples:
                logger.warning(f"Insufficient data for feature importance analysis: {len(df) if df is not None else 0} samples")
                return {}
                
            # Train models
            success, metrics = self.train_models(df)
            
            if not success:
                logger.warning("Model training failed, cannot calculate feature importance")
                return {}
                
            # Calculate feature importance
            importance_results = self.calculate_feature_importance()
            
            # Add model metrics
            importance_results["model_metrics"] = metrics
            
            # Update cache and timestamp
            self.feature_cache = importance_results
            self.last_update = current_time
            
            # Store results in Redis for other services
            await self.redis.set('feature_importance', json.dumps(importance_results))
            
            logger.info("Feature importance analysis completed and cached")
            return importance_results
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {str(e)}")
            return {}
    
    async def maintain_redis(self):
        """Maintain Redis connection."""
        logger.debug("Starting Redis connection maintenance...")
        while self.running:
            try:
                if self.redis:
                    await self.redis.ping()
                else:
                    await self.connect_redis()
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Redis connection error: {str(e)}")
                self.redis = None
                await asyncio.sleep(5)
    
    async def health_check_server(self):
        """Run a simple TCP server for health checks."""
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('0.0.0.0', self.service_port))
            server.listen(1)
            server.setblocking(False)
            
            logger.info(f"Health check server listening on port {self.service_port}")
            
            while self.running:
                try:
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Health check server loop error: {str(e)}")
                    break
                    
        except Exception as e:
            logger.error(f"Failed to start health check server: {str(e)}")
            raise  # Re-raise the exception to trigger service restart
        finally:
            try:
                server.close()
            except Exception:
                pass
    
    async def periodic_analysis(self):
        """Periodically run feature importance analysis."""
        logger.debug("Starting periodic feature importance analysis...")
        while self.running:
            try:
                logger.info("Running scheduled feature importance analysis...")
                await self.analyze_feature_importance()
                
                # Wait for next analysis interval
                await asyncio.sleep(self.analysis_interval)
                
            except Exception as e:
                logger.error(f"Error in periodic analysis: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute and try again
    
    async def run(self):
        """Run the feature importance service."""
        try:
            logger.info("Starting Feature Importance Service...")
            
            # First establish Redis connection
            if not await self.connect_redis(max_retries=15, retry_delay=2):
                raise Exception("Failed to establish initial Redis connection")
            
            # Create tasks for periodic analysis, Redis maintenance, and health check
            tasks = [
                asyncio.create_task(self.periodic_analysis()),
                asyncio.create_task(self.maintain_redis()),
                asyncio.create_task(self.health_check_server())
            ]
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Error in Feature Importance Service: {str(e)}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the feature importance service."""
        logger.info("Stopping Feature Importance Service...")
        self.running = False
        if self.redis:
            await self.redis.close()

if __name__ == "__main__":
    service = FeatureImportanceService()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        asyncio.run(service.stop())
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        asyncio.run(service.stop())