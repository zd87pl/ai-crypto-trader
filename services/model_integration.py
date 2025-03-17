import os
import json
import redis
import logging as logger
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import joblib

# Configure logging
logger.basicConfig(
    level=logger.INFO,
    format='%(asctime)s - %(levelname)s - [ModelIntegration] %(message)s',
    handlers=[
        logger.FileHandler('logs/model_integration.log'),
        logger.StreamHandler()
    ]
)

class FeatureImportanceIntegrator:
    """
    Integrates feature importance analysis into the trading system
    by interpreting feature importance reports and applying insights
    to enhance trading decisions and strategy selection.
    """
    
    def __init__(self):
        """Initialize the Feature Importance Integrator"""
        logger.debug("Initializing Feature Importance Integrator...")
        
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Redis setup
        redis_host = os.getenv('REDIS_HOST', 'redis')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        
        # Cached feature importance data
        self.feature_importance_data = None
        self.last_update_time = None
        self.update_interval = self.config.get('feature_importance', {}).get('integration_update_interval', 3600)
        
        # Model paths
        self.models_dir = 'models'
        self.latest_model = None
        self.loaded_model = None
        
        # Feature weights for strategy adjustment
        self.feature_weights = {}
        self.category_weights = {}
        
        logger.info("Feature Importance Integrator initialized successfully")
    
    def update_feature_importance_data(self, force: bool = False) -> bool:
        """
        Update feature importance data from Redis
        
        Args:
            force: Force update even if update interval hasn't elapsed
            
        Returns:
            bool: True if data was updated, False otherwise
        """
        current_time = datetime.now()
        
        # Check if update is needed
        if (not force and self.last_update_time and 
                (current_time - self.last_update_time).total_seconds() < self.update_interval):
            return False
        
        try:
            # Get latest feature importance data
            importance_data = self.redis.get('latest_feature_importance')
            if not importance_data:
                logger.warning("No feature importance data found in Redis")
                return False
                
            importance_data = json.loads(importance_data)
            
            # Validate data
            if not self._validate_feature_importance_data(importance_data):
                logger.warning("Invalid feature importance data")
                return False
                
            # Update cached data
            self.feature_importance_data = importance_data
            self.last_update_time = current_time
            
            # Update feature weights
            self._update_feature_weights()
            
            # Check for new optimized model
            self._check_for_new_model()
            
            logger.info("Feature importance data updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating feature importance data: {str(e)}")
            return False
    
    def _validate_feature_importance_data(self, data: Dict) -> bool:
        """
        Validate feature importance data
        
        Args:
            data: Feature importance data dictionary
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        required_keys = ['timestamp', 'top_features_permutation', 'top_categories']
        if not all(key in data for key in required_keys):
            logger.warning(f"Missing required keys in feature importance data: {required_keys}")
            return False
            
        # Check data freshness (3 days max)
        try:
            timestamp = datetime.fromisoformat(data['timestamp'])
            if (datetime.now() - timestamp) > timedelta(days=3):
                logger.warning("Feature importance data is more than 3 days old")
                return False
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid timestamp in feature importance data: {str(e)}")
            return False
            
        return True
    
    def _update_feature_weights(self) -> None:
        """Update feature and category weights based on importance data"""
        if not self.feature_importance_data:
            return
            
        # Get permutation importance as feature weights
        if 'top_features_permutation' in self.feature_importance_data:
            self.feature_weights = self.feature_importance_data['top_features_permutation']
            
        # Get category weights
        if 'top_categories' in self.feature_importance_data:
            self.category_weights = self.feature_importance_data['top_categories']
            
        logger.debug(f"Updated feature weights: {self.feature_weights}")
        logger.debug(f"Updated category weights: {self.category_weights}")
    
    def _check_for_new_model(self) -> None:
        """Check for new optimized model files"""
        try:
            if not os.path.exists(self.models_dir):
                logger.warning(f"Models directory {self.models_dir} does not exist")
                return
                
            # Get all model files
            model_files = [f for f in os.listdir(self.models_dir) 
                          if f.startswith('optimized_rf_model_') and f.endswith('.joblib')]
            
            if not model_files:
                logger.debug("No optimized model files found")
                return
                
            # Get the most recent model file
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join(self.models_dir, latest_model)
            
            # Check if model is already loaded
            if self.latest_model == model_path:
                return
                
            logger.info(f"Found new optimized model: {latest_model}")
            
            # Load the model
            try:
                self.loaded_model = joblib.load(model_path)
                self.latest_model = model_path
                logger.info(f"Loaded optimized model from {model_path}")
                
                # Store model name in Redis for other services
                self.redis.set('current_optimized_model', latest_model)
                
                # Store feature names used by the model if available
                if hasattr(self.loaded_model, 'feature_names'):
                    self.redis.set('current_model_features', json.dumps(self.loaded_model.feature_names))
                    
            except Exception as e:
                logger.error(f"Error loading model {model_path}: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error checking for new models: {str(e)}")
    
    def get_feature_weight(self, feature_name: str) -> float:
        """
        Get weight for a specific feature
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            float: Weight between 0 and 1, or 0.5 if feature not found
        """
        return self.feature_weights.get(feature_name, 0.5)
    
    def get_category_weight(self, category_name: str) -> float:
        """
        Get weight for a feature category
        
        Args:
            category_name: Name of the category
            
        Returns:
            float: Weight between 0 and 1, or 0.5 if category not found
        """
        return self.category_weights.get(category_name, 0.5)
    
    def predict_trade_outcome(self, features: Dict) -> Dict:
        """
        Use the optimized model to predict trade outcome
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Dict: Prediction results with confidence
        """
        if not self.loaded_model:
            logger.warning("No model loaded for prediction")
            return {
                'success_probability': 0.5,
                'confidence': 0.0,
                'status': 'no_model'
            }
            
        try:
            # Check if model has feature_names attribute
            if not hasattr(self.loaded_model, 'feature_names'):
                logger.warning("Model does not have feature_names attribute")
                return {
                    'success_probability': 0.5,
                    'confidence': 0.0,
                    'status': 'missing_feature_names'
                }
                
            # Create feature vector
            feature_values = []
            for feature in self.loaded_model.feature_names:
                if feature in features:
                    feature_values.append(features[feature])
                else:
                    logger.warning(f"Feature {feature} not found in input, using 0")
                    feature_values.append(0)
                    
            # Create numpy array
            X = np.array([feature_values])
            
            # Make prediction
            if hasattr(self.loaded_model, 'predict_proba'):
                probabilities = self.loaded_model.predict_proba(X)[0]
                success_probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            else:
                prediction = self.loaded_model.predict(X)[0]
                success_probability = float(prediction)
            
            # Calculate confidence based on distance from 0.5
            confidence = abs(success_probability - 0.5) * 2  # Scale to 0-1
            
            return {
                'success_probability': float(success_probability),
                'confidence': float(confidence),
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'model_file': os.path.basename(self.latest_model) if self.latest_model else None
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                'success_probability': 0.5,
                'confidence': 0.0,
                'status': 'error',
                'error': str(e)
            }
    
    def adjust_strategy_weights(self, strategy_weights: Dict) -> Dict:
        """
        Adjust strategy weights based on feature importance
        
        Args:
            strategy_weights: Original strategy weights
            
        Returns:
            Dict: Adjusted strategy weights
        """
        if not self.feature_importance_data:
            return strategy_weights
            
        try:
            # Create a copy of the weights to adjust
            adjusted_weights = strategy_weights.copy()
            
            # Get recommendations
            if 'recommendations' in self.feature_importance_data:
                recommendations = self.feature_importance_data['recommendations']
                
                # Adjust weights based on feature categories to prioritize
                if 'categories_to_prioritize' in recommendations:
                    for category in recommendations['categories_to_prioritize']:
                        category_lower = category.lower()
                        
                        # Boost social-based strategies if social features are important
                        if category_lower == 'social' and 'social_sentiment' in adjusted_weights:
                            adjusted_weights['social_sentiment'] *= 1.2
                            
                        # Boost momentum strategies if momentum features are important
                        if category_lower == 'momentum' and 'price_momentum' in adjusted_weights:
                            adjusted_weights['price_momentum'] *= 1.2
                            
                        # Boost volatility strategies if volatility features are important
                        if category_lower == 'volatility' and 'market_volatility' in adjusted_weights:
                            adjusted_weights['market_volatility'] *= 1.2
                        
                # Adjust weights based on feature categories to reconsider
                if 'categories_to_reconsider' in recommendations:
                    for category in recommendations['categories_to_reconsider']:
                        category_lower = category.lower()
                        
                        # Reduce weights for low-importance categories
                        if category_lower == 'volume' and 'volume_analysis' in adjusted_weights:
                            adjusted_weights['volume_analysis'] *= 0.8
                            
                        if category_lower == 'trend' and 'trend_following' in adjusted_weights:
                            adjusted_weights['trend_following'] *= 0.8
            
            # Normalize weights to sum to 1.0
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                for key in adjusted_weights:
                    adjusted_weights[key] /= total_weight
                    
            logger.debug(f"Adjusted strategy weights: {adjusted_weights}")
            return adjusted_weights
            
        except Exception as e:
            logger.error(f"Error adjusting strategy weights: {str(e)}")
            return strategy_weights
    
    def get_feature_importance_summary(self) -> Dict:
        """
        Get a summary of feature importance for display
        
        Returns:
            Dict: Summary of feature importance data
        """
        if not self.feature_importance_data:
            return {
                'status': 'no_data',
                'message': 'No feature importance data available'
            }
            
        try:
            # Create summary with the most relevant information
            summary = {
                'timestamp': self.feature_importance_data.get('timestamp'),
                'top_features': dict(list(self.feature_importance_data.get('top_features_permutation', {}).items())[:5]),
                'top_categories': self.feature_importance_data.get('top_categories', {}),
                'model_file': os.path.basename(self.latest_model) if self.latest_model else None,
                'status': 'success'
            }
            
            # Add recommendations if available
            if 'recommendations' in self.feature_importance_data:
                summary['recommendations'] = {
                    'features_to_prioritize': self.feature_importance_data['recommendations'].get('features_to_prioritize', [])[:3],
                    'categories_to_prioritize': self.feature_importance_data['recommendations'].get('categories_to_prioritize', [])
                }
                
            return summary
            
        except Exception as e:
            logger.error(f"Error creating feature importance summary: {str(e)}")
            return {
                'status': 'error',
                'message': f'Error creating summary: {str(e)}'
            }