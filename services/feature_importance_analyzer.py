import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging as logger
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from eli5 import explain_weights
import joblib
import redis
import asyncio
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Configure rotating file handler
rotating_handler = RotatingFileHandler(
    'logs/feature_importance.log',
    maxBytes=10*1024*1024,  # 10MB per file
    backupCount=5,  # Keep 5 backup files
    encoding='utf-8'
)

# Configure logging
logger.basicConfig(
    level=logger.DEBUG,
    format='%(asctime)s - %(levelname)s - [FeatureImportance] %(message)s',
    handlers=[
        rotating_handler,
        logger.StreamHandler()
    ]
)

class FeatureImportanceAnalyzer:
    """
    Analyzes the importance of various features used in the ML trading models
    by using permutation importance, SHAP values, and correlation analysis.
    """
    
    def __init__(self):
        """Initialize the Feature Importance Analyzer"""
        logger.debug("Initializing Feature Importance Analyzer...")
        
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Feature importance configuration
        self.feature_importance_config = self.config.get('feature_importance', {})
        self.analysis_interval = self.feature_importance_config.get('analysis_interval', 86400)  # Default: daily
        self.min_data_points = self.feature_importance_config.get('min_data_points', 1000)
        self.n_permutations = self.feature_importance_config.get('n_permutations', 30)
        self.test_size = self.feature_importance_config.get('test_size', 0.3)
        self.random_state = self.feature_importance_config.get('random_state', 42)
        
        # Initialize Redis connection
        redis_host = os.getenv('REDIS_HOST', 'redis')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        
        # Get service port from environment variable or config
        self.service_port = int(os.getenv('FEATURE_SERVICE_PORT', 
                                        self.feature_importance_config.get('service_port', 8007)))
        
        # Track analyzed models
        self.analyzed_models = set()
        self.running = True
        
        # Define feature categories
        self.feature_categories = {
            'price_action': ['price_change_1m', 'price_change_3m', 'price_change_5m', 
                             'price_change_15m', 'price_change_30m', 'price_change_1h',
                             'price_change_4h', 'current_price'],
            'momentum': ['rsi', 'stoch_k', 'stoch_d', 'williams_r', 'macd', 'macd_signal'],
            'volatility': ['atr', 'bollinger_width', 'volatility'],
            'trend': ['trend', 'trend_strength', 'sma_20', 'sma_50', 'sma_200', 
                     'ema_12', 'ema_26', 'ichimoku_a', 'ichimoku_b'],
            'volume': ['volume', 'vwap', 'avg_volume', 'quote_volume'],
            'social': ['social_volume', 'social_engagement', 'social_contributors', 'social_sentiment']
        }
        
        # Feature normalization ranges
        self.normalization_ranges = {
            'rsi': (0, 100),
            'stoch_k': (0, 100),
            'stoch_d': (0, 100),
            'williams_r': (-100, 0),
            'macd': (-10, 10),  # Typical range, adjust based on your data
            'bollinger_width': (0, 5),  # Typical range, adjust based on your data
            'social_sentiment': (0, 1)
        }
        
        logger.info("Feature Importance Analyzer initialized successfully")
    
    async def start(self):
        """Start the feature importance analyzer service"""
        logger.info("Starting Feature Importance Analyzer Service...")
        
        try:
            # Start the main analysis loop
            while self.running:
                try:
                    # Check if it's time to run analysis
                    await self.run_analysis()
                    
                    # Wait for the next analysis interval
                    await asyncio.sleep(self.analysis_interval)
                    
                except Exception as e:
                    logger.error(f"Error in analysis loop: {str(e)}", exc_info=True)
                    await asyncio.sleep(60)  # Wait before retrying
                    
        except asyncio.CancelledError:
            logger.info("Feature Importance Analyzer Service task was cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in Feature Importance Analyzer Service: {str(e)}", exc_info=True)
        finally:
            logger.info("Feature Importance Analyzer Service stopped")
    
    async def run_analysis(self):
        """Run a full feature importance analysis"""
        logger.info("Starting feature importance analysis...")
        
        try:
            # Collect training data from Redis
            training_data = await self.collect_training_data()
            
            if not training_data or len(training_data) < self.min_data_points:
                logger.warning(f"Insufficient data for analysis: {len(training_data) if training_data else 0} data points")
                return
            
            # Prepare the dataset
            X, y, feature_names = self.prepare_dataset(training_data)
            
            if X is None or y is None:
                logger.warning("Failed to prepare dataset")
                return
            
            # Train a Random Forest model on the data
            model = self.train_model(X, y)
            
            # Calculate various types of feature importance
            importance_results = self.calculate_feature_importance(model, X, y, feature_names)
            
            # Generate feature importance report
            report = self.generate_report(importance_results, feature_names)
            
            # Save report to Redis
            report_key = f'feature_importance_report:{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
            self.redis.set(report_key, json.dumps(report))
            self.redis.expire(report_key, 60 * 60 * 24 * 7)  # Keep report for 7 days
            
            # Generate and save visualizations
            self.generate_visualizations(importance_results, feature_names)
            
            # Optimize model based on feature importance
            pruned_model = self.create_optimized_model(model, importance_results, X, y, feature_names)
            
            # Save optimized model
            model_path = os.path.join('models', f'optimized_rf_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib')
            joblib.dump(pruned_model, model_path)
            
            logger.info(f"Feature importance analysis completed. Model saved to {model_path}")
            
            # Publish results to interested services
            await self.publish_results(report)
            
        except Exception as e:
            logger.error(f"Error running feature importance analysis: {str(e)}", exc_info=True)
    
    async def collect_training_data(self) -> List[Dict]:
        """Collect training data from Redis and historical sources"""
        logger.debug("Collecting training data...")
        
        training_data = []
        
        try:
            # Get keys for stored market data
            market_data_keys = self.redis.keys('market_data:*')
            logger.debug(f"Found {len(market_data_keys)} market data records in Redis")
            
            # Get keys for trading signals
            trading_signal_keys = self.redis.keys('trading_signal:*')
            logger.debug(f"Found {len(trading_signal_keys)} trading signal records in Redis")
            
            # Get keys for trade results
            trade_result_keys = self.redis.keys('trade_result:*')
            logger.debug(f"Found {len(trade_result_keys)} trade result records in Redis")
            
            # Process data: combine market data with signals and results
            for signal_key in trading_signal_keys:
                try:
                    signal_data = json.loads(self.redis.get(signal_key))
                    symbol = signal_data.get('symbol')
                    timestamp = signal_data.get('timestamp')
                    
                    if not symbol or not timestamp:
                        continue
                    
                    # Look for corresponding market data
                    market_key = f"market_data:{symbol}:{timestamp}"
                    market_data = self.redis.get(market_key)
                    
                    if not market_data:
                        continue
                        
                    market_data = json.loads(market_data)
                    
                    # Look for corresponding trade result
                    result_key = f"trade_result:{symbol}:{timestamp}"
                    trade_result = self.redis.get(result_key)
                    
                    if not trade_result:
                        continue
                        
                    trade_result = json.loads(trade_result)
                    
                    # Combine data
                    combined_data = {**market_data, **signal_data, 'outcome': trade_result.get('outcome', 'neutral')}
                    training_data.append(combined_data)
                    
                except Exception as e:
                    logger.error(f"Error processing data record: {str(e)}")
                    continue
            
            logger.info(f"Collected {len(training_data)} complete training records")
            
            return training_data
            
        except Exception as e:
            logger.error(f"Error collecting training data: {str(e)}", exc_info=True)
            return []
    
    def prepare_dataset(self, training_data: List[Dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """Prepare dataset for feature importance analysis"""
        logger.debug("Preparing dataset...")
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(training_data)
            
            # Define features to include
            feature_columns = []
            for category in self.feature_categories.values():
                for feature in category:
                    if feature in df.columns:
                        feature_columns.append(feature)
            
            logger.debug(f"Using features: {feature_columns}")
            
            # Handle categorical features
            if 'trend' in df.columns:
                df['trend'] = df['trend'].map({'uptrend': 1, 'downtrend': -1, 'sideways': 0})
            
            # Define target variable (trading signal outcome)
            df['target'] = df['outcome'].map({'profit': 1, 'loss': 0, 'neutral': 0.5})
            
            # Drop rows with NaN values
            df = df.dropna(subset=feature_columns + ['target'])
            
            if len(df) < self.min_data_points:
                logger.warning(f"Insufficient clean data after preprocessing: {len(df)} records")
                return None, None, []
            
            # Normalize features
            df_norm = df.copy()
            for col in feature_columns:
                if col in self.normalization_ranges:
                    min_val, max_val = self.normalization_ranges[col]
                    df_norm[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
            
            # Prepare X and y
            X = df_norm[feature_columns].values
            y = (df['target'] > 0.5).astype(int).values  # Binary classification: profit vs non-profit
            
            logger.info(f"Dataset prepared with {len(df)} records and {len(feature_columns)} features")
            
            return X, y, feature_columns
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}", exc_info=True)
            return None, None, []
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        """Train a Random Forest model for feature importance analysis"""
        logger.debug("Training Random Forest model...")
        
        try:
            # Configure model parameters
            n_estimators = self.feature_importance_config.get('n_estimators', 100)
            max_depth = self.feature_importance_config.get('max_depth', None)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth,
                random_state=self.random_state,
                n_jobs=-1  # Use all available cores
            )
            
            model.fit(X, y)
            
            # Check model accuracy
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
            
            logger.info(f"Model trained with metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                       f"Recall: {recall:.4f}, F1: {f1:.4f}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}", exc_info=True)
            # Return a dummy model in case of error
            return RandomForestClassifier()
    
    def calculate_feature_importance(self, model: RandomForestClassifier, X: np.ndarray, 
                                  y: np.ndarray, feature_names: List[str]) -> Dict:
        """Calculate various feature importance metrics"""
        logger.debug("Calculating feature importance metrics...")
        
        results = {}
        
        try:
            # 1. Built-in feature importance
            builtin_importance = model.feature_importances_
            results['builtin_importance'] = {
                feature_names[i]: float(builtin_importance[i]) 
                for i in range(len(feature_names))
            }
            
            # 2. Permutation importance
            perm_importance = permutation_importance(
                model, X, y, 
                n_repeats=self.n_permutations,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            results['permutation_importance'] = {
                feature_names[i]: float(perm_importance.importances_mean[i])
                for i in range(len(feature_names))
            }
            
            # 3. Feature importance by category
            cat_importance = {}
            for category, features in self.feature_categories.items():
                # Only include features that are in our dataset
                valid_features = [f for f in features if f in feature_names]
                if not valid_features:
                    continue
                
                # Get the indices of these features
                indices = [feature_names.index(f) for f in valid_features]
                
                # Calculate the average importance for this category
                if indices:
                    avg_importance = np.mean([builtin_importance[i] for i in indices])
                    cat_importance[category] = float(avg_importance)
            
            results['category_importance'] = cat_importance
            
            # Log top features
            top_features = sorted(
                results['permutation_importance'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            logger.info("Top 10 features by permutation importance:")
            for feature, importance in top_features:
                logger.info(f"  {feature}: {importance:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}", exc_info=True)
            return {'error': str(e)}
    
    def generate_report(self, importance_results: Dict, feature_names: List[str]) -> Dict:
        """Generate a comprehensive feature importance report"""
        logger.debug("Generating feature importance report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'feature_importance',
            'model_type': 'RandomForest',
            'feature_count': len(feature_names),
            'importance_metrics': importance_results
        }
        
        # Add top features by each importance metric
        try:
            # Top features by built-in importance
            if 'builtin_importance' in importance_results:
                builtin_top = sorted(
                    importance_results['builtin_importance'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                report['top_features_builtin'] = dict(builtin_top[:10])
            
            # Top features by permutation importance
            if 'permutation_importance' in importance_results:
                perm_top = sorted(
                    importance_results['permutation_importance'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                report['top_features_permutation'] = dict(perm_top[:10])
            
            # Top feature categories
            if 'category_importance' in importance_results:
                category_top = sorted(
                    importance_results['category_importance'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                report['top_categories'] = dict(category_top)
            
            # Add recommendations based on feature importance
            report['recommendations'] = self.generate_recommendations(importance_results, feature_names)
            
            logger.info("Feature importance report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}", exc_info=True)
            return {'error': str(e)}
    
    def generate_recommendations(self, importance_results: Dict, feature_names: List[str]) -> Dict:
        """Generate recommendations based on feature importance"""
        recommendations = {
            'features_to_prioritize': [],
            'features_to_reconsider': [],
            'categories_to_prioritize': [],
            'categories_to_reconsider': []
        }
        
        try:
            # Identify key features to prioritize/reconsider
            if 'permutation_importance' in importance_results:
                perm_imp = importance_results['permutation_importance']
                
                # Sort features by importance
                sorted_features = sorted(perm_imp.items(), key=lambda x: x[1], reverse=True)
                
                # Top 20% features to prioritize
                top_count = max(1, int(len(sorted_features) * 0.2))
                recommendations['features_to_prioritize'] = [f[0] for f in sorted_features[:top_count]]
                
                # Bottom 20% features to reconsider
                bottom_count = max(1, int(len(sorted_features) * 0.2))
                recommendations['features_to_reconsider'] = [f[0] for f in sorted_features[-bottom_count:]]
            
            # Identify key categories to prioritize/reconsider
            if 'category_importance' in importance_results:
                cat_imp = importance_results['category_importance']
                
                # Sort categories by importance
                sorted_cats = sorted(cat_imp.items(), key=lambda x: x[1], reverse=True)
                
                # Top 33% categories to prioritize
                top_count = max(1, int(len(sorted_cats) * 0.33))
                recommendations['categories_to_prioritize'] = [c[0] for c in sorted_cats[:top_count]]
                
                # Bottom 33% categories to reconsider
                bottom_count = max(1, int(len(sorted_cats) * 0.33))
                recommendations['categories_to_reconsider'] = [c[0] for c in sorted_cats[-bottom_count:]]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
            return {'error': str(e)}
    
    def generate_visualizations(self, importance_results: Dict, feature_names: List[str]) -> None:
        """Generate visualizations of feature importance"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Bar chart of top features by permutation importance
            if 'permutation_importance' in importance_results:
                plt.figure(figsize=(12, 8))
                
                # Sort features by importance
                perm_imp = importance_results['permutation_importance']
                sorted_features = sorted(perm_imp.items(), key=lambda x: x[1], reverse=True)[:15]  # Top 15
                
                # Create bar chart
                features = [f[0] for f in sorted_features]
                importances = [f[1] for f in sorted_features]
                
                plt.barh(features, importances)
                plt.xlabel('Permutation Importance')
                plt.ylabel('Feature')
                plt.title('Top 15 Features by Permutation Importance')
                plt.tight_layout()
                
                # Save figure
                plt.savefig(f'reports/feature_importance_top15_{timestamp}.png', dpi=300)
                plt.close()
            
            # 2. Bar chart of feature categories
            if 'category_importance' in importance_results:
                plt.figure(figsize=(10, 6))
                
                # Get category importances
                cat_imp = importance_results['category_importance']
                categories = list(cat_imp.keys())
                cat_importances = list(cat_imp.values())
                
                # Sort categories by importance
                sorted_indices = np.argsort(cat_importances)[::-1]
                sorted_categories = [categories[i] for i in sorted_indices]
                sorted_importances = [cat_importances[i] for i in sorted_indices]
                
                # Create bar chart
                plt.barh(sorted_categories, sorted_importances)
                plt.xlabel('Average Feature Importance')
                plt.ylabel('Feature Category')
                plt.title('Feature Importance by Category')
                plt.tight_layout()
                
                # Save figure
                plt.savefig(f'reports/feature_category_importance_{timestamp}.png', dpi=300)
                plt.close()
            
            logger.info(f"Visualizations generated and saved to reports directory")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}", exc_info=True)
    
    def create_optimized_model(self, model: RandomForestClassifier, importance_results: Dict,
                            X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> RandomForestClassifier:
        """Create an optimized model based on feature importance"""
        logger.debug("Creating optimized model...")
        
        try:
            # Identify low-importance features to prune
            if 'permutation_importance' not in importance_results:
                logger.warning("No permutation importance results available for model optimization")
                return model
            
            perm_imp = importance_results['permutation_importance']
            
            # Calculate importance threshold (keep features with importance > 25% of max importance)
            max_importance = max(perm_imp.values())
            threshold = max_importance * 0.25
            
            # Identify features to keep
            features_to_keep = [
                feature for feature, importance in perm_imp.items() 
                if importance > threshold
            ]
            
            logger.info(f"Keeping {len(features_to_keep)} out of {len(feature_names)} features for optimized model")
            
            # Get indices of features to keep
            keep_indices = [feature_names.index(f) for f in features_to_keep if f in feature_names]
            
            # Create new dataset with only important features
            X_optimized = X[:, keep_indices]
            
            # Train new model
            optimized_model = RandomForestClassifier(
                n_estimators=model.n_estimators,
                max_depth=model.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            optimized_model.fit(X_optimized, y)
            
            # Check optimized model performance
            y_pred = optimized_model.predict(X_optimized)
            accuracy = accuracy_score(y, y_pred)
            
            logger.info(f"Optimized model trained with accuracy: {accuracy:.4f}")
            
            # Save feature names for optimized model
            optimized_model.feature_names = features_to_keep
            
            return optimized_model
            
        except Exception as e:
            logger.error(f"Error creating optimized model: {str(e)}", exc_info=True)
            return model
    
    async def publish_results(self, report: Dict) -> None:
        """Publish feature importance results to Redis for other services"""
        try:
            # Publish to feature_importance channel
            self.redis.publish('feature_importance', json.dumps({
                'timestamp': datetime.now().isoformat(),
                'report': report
            }))
            
            # Store latest report in a known key for easy access
            self.redis.set('latest_feature_importance', json.dumps(report))
            
            logger.info("Published feature importance results to Redis")
            
        except Exception as e:
            logger.error(f"Error publishing results: {str(e)}", exc_info=True)
    
    async def stop(self) -> None:
        """Stop the feature importance analyzer service"""
        logger.info("Stopping Feature Importance Analyzer Service...")
        self.running = False

if __name__ == "__main__":
    analyzer = FeatureImportanceAnalyzer()
    try:
        asyncio.run(analyzer.start())
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service stopped due to error: {str(e)}", exc_info=True)