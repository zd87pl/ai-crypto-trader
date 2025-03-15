import os
import json
import uuid
import time
import asyncio
import logging as logger
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import redis.asyncio as redis_async
from redis.exceptions import ConnectionError

# Configure logging
logger.basicConfig(
    level=logger.INFO,
    format='%(asctime)s - %(levelname)s - [ModelRegistry] %(message)s',
    handlers=[
        logger.FileHandler('logs/model_registry.log'),
        logger.StreamHandler()
    ]
)

class ModelRegistryService:
    """
    Service for managing AI model versions, tracking their performance, and providing version control.
    Implements AI-05 (model performance monitoring) and AI-06 (model registry for version control).
    """
    
    def __init__(self):
        """Initialize the model registry service"""
        # Ensure directories exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Redis configuration
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis = None
        self.pubsub = None
        
        # Service state
        self.running = True
        self.registered_models = {}
        self.active_models = {}
        
        # Model storage paths
        self.model_registry_path = Path('models/registry')
        self.model_registry_path.mkdir(exist_ok=True)
        self.model_registry_file = self.model_registry_path / 'registry.json'
        
        # Load existing registry if available
        self._load_registry()
        
        logger.info(f"Model Registry Service initialized with {len(self.registered_models)} registered models")

    def _load_registry(self) -> None:
        """Load the model registry from file"""
        try:
            if self.model_registry_file.exists():
                with open(self.model_registry_file, 'r') as f:
                    registry_data = json.load(f)
                    self.registered_models = registry_data.get('models', {})
                    logger.info(f"Loaded model registry with {len(self.registered_models)} models")
            else:
                # Initialize empty registry
                self._save_registry()
        except Exception as e:
            logger.error(f"Error loading model registry: {str(e)}")
            self.registered_models = {}
            
    def _save_registry(self) -> None:
        """Save the model registry to file"""
        try:
            registry_data = {
                'models': self.registered_models,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.model_registry_file, 'w') as f:
                json.dump(registry_data, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving model registry: {str(e)}")
            
    async def connect_redis(self, max_retries=10, retry_delay=5):
        """Connect to Redis with retries"""
        retries = 0
        while retries < max_retries and self.running:
            try:
                if self.redis is None:
                    logger.debug(f"Attempting Redis connection (attempt {retries + 1}/{max_retries})")
                    self.redis = redis_async.Redis(
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
                
    async def setup_pubsub(self):
        """Set up Redis pubsub for model-related events"""
        try:
            # Verify Redis connection
            if not self.redis or not await self.redis.ping():
                logger.error("Redis connection not available for pubsub setup")
                return False
                
            # Close existing pubsub if any
            if self.pubsub:
                logger.debug("Closing existing pubsub connection")
                await self.pubsub.close()
                self.pubsub = None
                
            # Create new pubsub instance
            logger.debug("Creating new pubsub instance")
            self.pubsub = self.redis.pubsub()
            
            # Subscribe to channels
            logger.debug("Subscribing to model-related channels")
            await self.pubsub.subscribe('model_registry_events', 'model_performance_updates')
            
            # Get first messages to confirm subscriptions
            logger.debug("Waiting for subscription confirmation messages")
            subscribed_count = 0
            while subscribed_count < 2:  # Wait for both subscriptions
                message = await self.pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'subscribe':
                    subscribed_count += 1
                    
            if subscribed_count == 2:
                logger.info("Successfully subscribed to all channels")
                return True
            else:
                logger.error("Failed to subscribe to all channels")
                return False
                
        except Exception as e:
            logger.error(f"Error in pubsub setup: {str(e)}", exc_info=True)
            if self.pubsub:
                await self.pubsub.close()
                self.pubsub = None
            return False
            
    async def register_model(self, model_info: Dict) -> str:
        """Register a new model version in the registry"""
        try:
            model_id = model_info.get('version_id', str(uuid.uuid4())[:8])
            model_name = model_info.get('version_name', f"model_{model_id}")
            
            # Create a model registry entry
            model_entry = {
                'version_id': model_id,
                'version_name': model_name,
                'model_type': model_info.get('model', 'unknown'),
                'creation_date': model_info.get('creation_date', datetime.now().isoformat()),
                'registration_date': datetime.now().isoformat(),
                'config': model_info.get('config', {}),
                'performance_metrics': model_info.get('performance_metrics', {
                    'total_trades': 0,
                    'successful_trades': 0,
                    'failed_trades': 0,
                    'win_rate': 0.0,
                    'avg_confidence': 0.0
                }),
                'status': 'registered',
                'last_updated': datetime.now().isoformat()
            }
            
            # Add to registry
            self.registered_models[model_id] = model_entry
            self._save_registry()
            
            # Publish event to Redis
            if self.redis and await self.redis.ping():
                event_data = {
                    'event_type': 'model_registered',
                    'timestamp': datetime.now().isoformat(),
                    'model_id': model_id,
                    'model_name': model_name
                }
                await self.redis.publish('model_registry_events', json.dumps(event_data))
                
                # Store model metadata in Redis for quick access
                await self.redis.hset(
                    'model_registry', 
                    model_id, 
                    json.dumps(model_entry)
                )
                
            logger.info(f"Registered new model: {model_name} (ID: {model_id})")
            return model_id
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            return None
            
    async def update_model_performance(self, model_id: str, metrics: Dict) -> bool:
        """Update performance metrics for a model"""
        try:
            if model_id not in self.registered_models:
                logger.error(f"Model ID {model_id} not found in registry")
                return False
                
            # Get current model entry
            model_entry = self.registered_models[model_id]
            
            # Update performance metrics
            for key, value in metrics.items():
                if key in model_entry['performance_metrics']:
                    model_entry['performance_metrics'][key] = value
                    
            # Calculate win rate
            total = model_entry['performance_metrics'].get('total_trades', 0)
            successful = model_entry['performance_metrics'].get('successful_trades', 0)
            if total > 0:
                model_entry['performance_metrics']['win_rate'] = successful / total
                
            # Update timestamps
            model_entry['last_updated'] = datetime.now().isoformat()
            
            # Save to registry
            self.registered_models[model_id] = model_entry
            self._save_registry()
            
            # Update Redis
            if self.redis and await self.redis.ping():
                # Publish update event
                event_data = {
                    'event_type': 'model_performance_updated',
                    'timestamp': datetime.now().isoformat(),
                    'model_id': model_id,
                    'model_name': model_entry['version_name'],
                    'metrics': model_entry['performance_metrics']
                }
                await self.redis.publish('model_performance_updates', json.dumps(event_data))
                
                # Update stored model data
                await self.redis.hset(
                    'model_registry', 
                    model_id, 
                    json.dumps(model_entry)
                )
                
            logger.info(f"Updated performance metrics for model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating model performance: {str(e)}")
            return False
            
    async def get_model_by_id(self, model_id: str) -> Optional[Dict]:
        """Get a model by its ID"""
        try:
            # Try to get from Redis first for speed
            if self.redis and await self.redis.ping():
                model_data = await self.redis.hget('model_registry', model_id)
                if model_data:
                    return json.loads(model_data)
                    
            # Fall back to local registry
            if model_id in self.registered_models:
                return self.registered_models[model_id]
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting model {model_id}: {str(e)}")
            return None
            
    async def get_best_model(self, metric: str = 'win_rate', min_trades: int = 10) -> Optional[Dict]:
        """Get the best performing model based on a specific metric"""
        try:
            best_model = None
            best_value = -1
            
            for model_id, model in self.registered_models.items():
                # Only consider models with minimum number of trades
                if model['performance_metrics'].get('total_trades', 0) < min_trades:
                    continue
                    
                # Compare metric values
                metric_value = model['performance_metrics'].get(metric, 0)
                if metric_value > best_value:
                    best_value = metric_value
                    best_model = model
                    
            return best_model
            
        except Exception as e:
            logger.error(f"Error finding best model: {str(e)}")
            return None
            
    async def set_model_status(self, model_id: str, status: str) -> bool:
        """Set the status of a model (e.g., 'active', 'archived', 'deprecated')"""
        try:
            if model_id not in self.registered_models:
                logger.error(f"Model ID {model_id} not found in registry")
                return False
                
            # Update status
            self.registered_models[model_id]['status'] = status
            self.registered_models[model_id]['last_updated'] = datetime.now().isoformat()
            self._save_registry()
            
            # Update Redis
            if self.redis and await self.redis.ping():
                # Publish status change event
                event_data = {
                    'event_type': 'model_status_changed',
                    'timestamp': datetime.now().isoformat(),
                    'model_id': model_id,
                    'model_name': self.registered_models[model_id]['version_name'],
                    'status': status
                }
                await self.redis.publish('model_registry_events', json.dumps(event_data))
                
                # Update stored model data
                await self.redis.hset(
                    'model_registry', 
                    model_id, 
                    json.dumps(self.registered_models[model_id])
                )
                
            logger.info(f"Set model {model_id} status to '{status}'")
            return True
            
        except Exception as e:
            logger.error(f"Error setting model status: {str(e)}")
            return False
            
    async def compare_models(self, model_ids: List[str]) -> Dict:
        """Compare multiple models based on their performance metrics"""
        try:
            comparison = {
                'timestamp': datetime.now().isoformat(),
                'models': {},
                'metrics_comparison': {}
            }
            
            # Collect model data
            for model_id in model_ids:
                model = await self.get_model_by_id(model_id)
                if model:
                    comparison['models'][model_id] = {
                        'name': model['version_name'],
                        'type': model['model_type'],
                        'creation_date': model['creation_date'],
                        'metrics': model['performance_metrics']
                    }
            
            # Skip comparison if less than 2 models are found
            if len(comparison['models']) < 2:
                return comparison
                
            # Compare key metrics across models
            key_metrics = ['win_rate', 'avg_confidence', 'total_trades', 'successful_trades']
            for metric in key_metrics:
                comparison['metrics_comparison'][metric] = {}
                for model_id, model_data in comparison['models'].items():
                    comparison['metrics_comparison'][metric][model_id] = model_data['metrics'].get(metric, 0)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            return {'error': str(e)}
            
    async def process_registry_events(self):
        """Process events from the model registry"""
        try:
            if not self.pubsub:
                if not await self.setup_pubsub():
                    logger.error("Failed to setup pubsub for registry events")
                    return
                    
            # Process incoming messages
            message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if not message:
                return
                
            # Handle message
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    
                    if message['channel'] == 'model_performance_updates':
                        logger.debug(f"Received model performance update: {data}")
                        
                        # Update local registry with latest metrics
                        if 'model_id' in data and 'metrics' in data:
                            model_id = data['model_id']
                            if model_id in self.registered_models:
                                self.registered_models[model_id]['performance_metrics'] = data['metrics']
                                self.registered_models[model_id]['last_updated'] = data.get('timestamp', datetime.now().isoformat())
                                self._save_registry()
                    
                    elif message['channel'] == 'model_registry_events':
                        logger.debug(f"Received model registry event: {data}")
                        
                        # Handle different event types
                        event_type = data.get('event_type')
                        
                        if event_type == 'model_registered':
                            # Already handled by the register_model method
                            pass
                            
                        elif event_type == 'model_status_changed':
                            # Update model status locally if needed
                            model_id = data.get('model_id')
                            if model_id in self.registered_models:
                                self.registered_models[model_id]['status'] = data.get('status', 'unknown')
                                self.registered_models[model_id]['last_updated'] = data.get('timestamp', datetime.now().isoformat())
                                self._save_registry()
                                
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in message: {message['data']}")
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error in process_registry_events: {str(e)}")
            
    async def maintain_redis(self):
        """Maintain Redis connection"""
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
                
    async def run(self):
        """Run the model registry service"""
        try:
            logger.info("Starting Model Registry Service...")
            
            # First establish Redis connection
            if not await self.connect_redis(max_retries=15, retry_delay=2):
                raise Exception("Failed to establish initial Redis connection")
                
            # Setup pubsub
            if not await self.setup_pubsub():
                logger.warning("Failed to set up pubsub, will retry later")
                
            # Start Redis maintenance task
            redis_task = asyncio.create_task(self.maintain_redis())
            
            # Main service loop
            while self.running:
                try:
                    # Process registry events
                    await self.process_registry_events()
                    
                    # Periodically save registry (every 5 minutes)
                    if int(time.time()) % 300 == 0:
                        self._save_registry()
                        
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error in main service loop: {str(e)}")
                    await asyncio.sleep(5)
                    
        except Exception as e:
            logger.error(f"Error in Model Registry Service: {str(e)}")
            
        finally:
            # Clean up
            self._save_registry()
            
            if self.pubsub:
                await self.pubsub.close()
                
            if self.redis:
                await self.redis.close()
                
            # Cancel maintenance task
            if 'redis_task' in locals():
                redis_task.cancel()
                try:
                    await redis_task
                except asyncio.CancelledError:
                    pass
                    
            logger.info("Model Registry Service stopped")
            
    async def stop(self):
        """Stop the model registry service"""
        logger.info("Stopping Model Registry Service...")
        self.running = False
        
        # Save registry before stopping
        self._save_registry()
        
        # Close Redis connections
        if self.pubsub:
            await self.pubsub.close()
            
        if self.redis:
            await self.redis.close()

# Run the service if executed directly
if __name__ == "__main__":
    service = ModelRegistryService()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        asyncio.run(service.stop())
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        asyncio.run(service.stop())