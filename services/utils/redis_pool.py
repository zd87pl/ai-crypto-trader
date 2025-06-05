"""
Redis Connection Pool Manager
Optimized Redis connections with pooling, clustering support, and health monitoring
"""
import os
import asyncio
import logging
from typing import Optional, Dict, Any
from redis.asyncio import Redis, ConnectionPool, RedisCluster
from redis.exceptions import ConnectionError, TimeoutError, RedisError
from services.utils.circuit_breaker import circuit_breaker, CircuitBreakerConfig
from services.utils.metrics import get_metrics, is_metrics_enabled
import json
import time

logger = logging.getLogger(__name__)

class RedisPoolManager:
    """
    Manages Redis connection pools with health monitoring, retries, and metrics
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._load_default_config()
        self.pools: Dict[str, ConnectionPool] = {}
        self.clients: Dict[str, Redis] = {}
        self.cluster_client: Optional[RedisCluster] = None
        self.is_cluster_mode = self.config.get('cluster_mode', False)
        
        # Health monitoring
        self.health_stats = {}
        self.last_health_check = {}
        
        # Initialize metrics if enabled
        self.metrics = None
        if is_metrics_enabled():
            self.metrics = get_metrics('redis_pool_manager')
        
        # Circuit breaker for Redis operations
        self.redis_cb = circuit_breaker(
            name='redis_pool_operations',
            failure_threshold=self.config.get('circuit_breaker', {}).get('failure_threshold', 5),
            recovery_timeout=self.config.get('circuit_breaker', {}).get('recovery_timeout', 30),
            timeout=self.config.get('circuit_breaker', {}).get('timeout', 10.0)
        )
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default Redis configuration"""
        return {
            'host': os.getenv('REDIS_HOST', 'redis'),
            'port': int(os.getenv('REDIS_PORT', 6379)),
            'password': os.getenv('REDIS_PASSWORD'),
            'db': int(os.getenv('REDIS_DB', 0)),
            'cluster_mode': os.getenv('REDIS_CLUSTER_MODE', 'false').lower() == 'true',
            'cluster_nodes': os.getenv('REDIS_CLUSTER_NODES', '').split(',') if os.getenv('REDIS_CLUSTER_NODES') else [],
            'pool_config': {
                'max_connections': int(os.getenv('REDIS_MAX_CONNECTIONS', 20)),
                'max_connections_per_node': int(os.getenv('REDIS_MAX_CONNECTIONS_PER_NODE', 10)),
                'retry_on_timeout': True,
                'retry_on_error': [ConnectionError, TimeoutError],
                'health_check_interval': int(os.getenv('REDIS_HEALTH_CHECK_INTERVAL', 30)),
                'socket_keepalive': True,
                'socket_keepalive_options': {},
                'connection_timeout': float(os.getenv('REDIS_CONNECTION_TIMEOUT', 5.0)),
                'socket_timeout': float(os.getenv('REDIS_SOCKET_TIMEOUT', 5.0)),
            },
            'circuit_breaker': {
                'failure_threshold': 5,
                'recovery_timeout': 30,
                'timeout': 10.0
            }
        }
    
    async def initialize(self):
        """Initialize Redis pools and connections"""
        try:
            if self.is_cluster_mode:
                await self._initialize_cluster()
            else:
                await self._initialize_standalone()
            
            logger.info("Redis Pool Manager initialized successfully")
            if self.metrics:
                self.metrics.set_service_health(True)
                
        except Exception as e:
            logger.error(f"Failed to initialize Redis Pool Manager: {e}")
            if self.metrics:
                self.metrics.set_service_health(False)
            raise
    
    async def _initialize_standalone(self):
        """Initialize standalone Redis with connection pooling"""
        pool_config = self.config['pool_config']
        
        # Create connection pool
        pool = ConnectionPool(
            host=self.config['host'],
            port=self.config['port'],
            password=self.config['password'],
            db=self.config['db'],
            max_connections=pool_config['max_connections'],
            retry_on_timeout=pool_config['retry_on_timeout'],
            retry_on_error=pool_config['retry_on_error'],
            socket_keepalive=pool_config['socket_keepalive'],
            socket_keepalive_options=pool_config['socket_keepalive_options'],
            socket_connect_timeout=pool_config['connection_timeout'],
            socket_timeout=pool_config['socket_timeout'],
            decode_responses=True
        )
        
        self.pools['default'] = pool
        self.clients['default'] = Redis(connection_pool=pool)
        
        # Test connection
        await self._test_connection('default')
        logger.info(f"Standalone Redis pool initialized with {pool_config['max_connections']} max connections")
    
    async def _initialize_cluster(self):
        """Initialize Redis Cluster"""
        if not self.config['cluster_nodes']:
            raise ValueError("Cluster nodes must be specified for cluster mode")
        
        pool_config = self.config['pool_config']
        
        # Parse cluster nodes
        nodes = []
        for node in self.config['cluster_nodes']:
            if ':' in node:
                host, port = node.split(':')
                nodes.append({'host': host.strip(), 'port': int(port.strip())})
            else:
                nodes.append({'host': node.strip(), 'port': 6379})
        
        # Create cluster client
        self.cluster_client = RedisCluster(
            startup_nodes=nodes,
            password=self.config['password'],
            max_connections_per_node=pool_config['max_connections_per_node'],
            socket_keepalive=pool_config['socket_keepalive'],
            socket_connect_timeout=pool_config['connection_timeout'],
            socket_timeout=pool_config['socket_timeout'],
            decode_responses=True
        )
        
        # Test cluster connection
        await self.cluster_client.ping()
        logger.info(f"Redis Cluster initialized with {len(nodes)} nodes")
    
    async def _test_connection(self, pool_name: str):
        """Test Redis connection"""
        client = self.clients[pool_name]
        await client.ping()
        self.health_stats[pool_name] = {
            'status': 'healthy',
            'last_ping': time.time(),
            'connection_count': self.pools[pool_name].created_connections
        }
    
    def get_client(self, pool_name: str = 'default') -> Redis:
        """Get Redis client from pool"""
        if self.is_cluster_mode:
            return self.cluster_client
        
        if pool_name not in self.clients:
            raise ValueError(f"Pool {pool_name} not found")
        
        return self.clients[pool_name]
    
    @circuit_breaker(name='redis_execute', failure_threshold=3, recovery_timeout=15)
    async def execute(self, command: str, *args, pool_name: str = 'default', **kwargs) -> Any:
        """Execute Redis command with circuit breaker protection"""
        client = self.get_client(pool_name)
        
        start_time = time.time()
        try:
            # Record request metrics
            if self.metrics:
                self.metrics.request_counter.labels(
                    service='redis_pool',
                    endpoint=command,
                    method='REDIS'
                ).inc()
            
            # Execute command
            result = await getattr(client, command)(*args, **kwargs)
            
            # Record successful operation
            if self.metrics:
                duration = time.time() - start_time
                self.metrics.request_duration.labels(
                    service='redis_pool',
                    endpoint=command,
                    method='REDIS'
                ).observe(duration)
            
            return result
            
        except Exception as e:
            # Record error metrics
            if self.metrics:
                self.metrics.error_counter.labels(
                    service='redis_pool',
                    error_type=type(e).__name__,
                    endpoint=command
                ).inc()
            
            logger.error(f"Redis command {command} failed: {e}")
            raise
    
    async def publish(self, channel: str, message: str, pool_name: str = 'default') -> int:
        """Publish message to Redis channel"""
        return await self.execute('publish', channel, message, pool_name=pool_name)
    
    async def subscribe(self, *channels, pool_name: str = 'default'):
        """Subscribe to Redis channels"""
        client = self.get_client(pool_name)
        pubsub = client.pubsub()
        await pubsub.subscribe(*channels)
        return pubsub
    
    async def set(self, key: str, value: str, ex: Optional[int] = None, pool_name: str = 'default'):
        """Set key-value pair in Redis"""
        return await self.execute('set', key, value, ex=ex, pool_name=pool_name)
    
    async def get(self, key: str, pool_name: str = 'default') -> Optional[str]:
        """Get value from Redis"""
        return await self.execute('get', key, pool_name=pool_name)
    
    async def hset(self, name: str, key: str, value: str, pool_name: str = 'default'):
        """Set hash field"""
        return await self.execute('hset', name, key, value, pool_name=pool_name)
    
    async def hget(self, name: str, key: str, pool_name: str = 'default') -> Optional[str]:
        """Get hash field"""
        return await self.execute('hget', name, key, pool_name=pool_name)
    
    async def hgetall(self, name: str, pool_name: str = 'default') -> Dict[str, str]:
        """Get all hash fields"""
        return await self.execute('hgetall', name, pool_name=pool_name)
    
    async def delete(self, *keys, pool_name: str = 'default') -> int:
        """Delete keys"""
        return await self.execute('delete', *keys, pool_name=pool_name)
    
    async def ping(self, pool_name: str = 'default') -> bool:
        """Ping Redis"""
        try:
            await self.execute('ping', pool_name=pool_name)
            return True
        except Exception:
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_report = {
            'overall_status': 'healthy',
            'pools': {},
            'cluster_status': None,
            'metrics': {
                'total_connections': 0,
                'active_connections': 0,
                'failed_connections': 0
            }
        }
        
        try:
            if self.is_cluster_mode:
                # Check cluster health
                cluster_info = await self.cluster_client.cluster_info()
                health_report['cluster_status'] = {
                    'state': cluster_info.get('cluster_state', 'unknown'),
                    'slots_assigned': cluster_info.get('cluster_slots_assigned', 0),
                    'nodes_count': cluster_info.get('cluster_known_nodes', 0)
                }
                
                # Test cluster ping
                ping_success = await self.ping()
                if not ping_success:
                    health_report['overall_status'] = 'unhealthy'
                    
            else:
                # Check standalone pools
                for pool_name, pool in self.pools.items():
                    try:
                        ping_success = await self.ping(pool_name)
                        
                        pool_stats = {
                            'status': 'healthy' if ping_success else 'unhealthy',
                            'created_connections': pool.created_connections,
                            'available_connections': len(pool._available_connections),
                            'in_use_connections': len(pool._in_use_connections),
                            'max_connections': pool.max_connections
                        }
                        
                        health_report['pools'][pool_name] = pool_stats
                        
                        # Update overall metrics
                        health_report['metrics']['total_connections'] += pool_stats['created_connections']
                        health_report['metrics']['active_connections'] += pool_stats['in_use_connections']
                        
                        if not ping_success:
                            health_report['overall_status'] = 'degraded'
                            health_report['metrics']['failed_connections'] += 1
                            
                    except Exception as e:
                        logger.error(f"Health check failed for pool {pool_name}: {e}")
                        health_report['pools'][pool_name] = {
                            'status': 'unhealthy',
                            'error': str(e)
                        }
                        health_report['overall_status'] = 'unhealthy'
            
            # Update health metrics
            if self.metrics:
                if health_report['overall_status'] == 'healthy':
                    self.metrics.set_service_health(True)
                else:
                    self.metrics.set_service_health(False)
            
            return health_report
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_report['overall_status'] = 'unhealthy'
            health_report['error'] = str(e)
            
            if self.metrics:
                self.metrics.set_service_health(False)
            
            return health_report
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis pool statistics"""
        stats = {
            'mode': 'cluster' if self.is_cluster_mode else 'standalone',
            'config': self.config,
            'health': await self.health_check(),
            'circuit_breaker_stats': {}
        }
        
        # Add circuit breaker stats
        from services.utils.circuit_breaker import get_all_circuit_breakers
        circuit_breakers = get_all_circuit_breakers()
        for name, cb in circuit_breakers.items():
            if 'redis' in name.lower():
                stats['circuit_breaker_stats'][name] = cb.get_stats()
        
        return stats
    
    async def close(self):
        """Close all Redis connections"""
        try:
            if self.is_cluster_mode and self.cluster_client:
                await self.cluster_client.close()
                logger.info("Redis cluster connections closed")
            else:
                for pool_name, client in self.clients.items():
                    await client.close()
                    logger.info(f"Redis pool {pool_name} closed")
            
            self.pools.clear()
            self.clients.clear()
            self.cluster_client = None
            
        except Exception as e:
            logger.error(f"Error closing Redis connections: {e}")


# Global Redis pool manager instance
_redis_pool_manager: Optional[RedisPoolManager] = None

async def get_redis_pool_manager(config: Dict[str, Any] = None) -> RedisPoolManager:
    """Get or create Redis pool manager"""
    global _redis_pool_manager
    
    if _redis_pool_manager is None:
        _redis_pool_manager = RedisPoolManager(config)
        await _redis_pool_manager.initialize()
    
    return _redis_pool_manager

async def get_redis_client(pool_name: str = 'default') -> Redis:
    """Get Redis client from global pool manager"""
    pool_manager = await get_redis_pool_manager()
    return pool_manager.get_client(pool_name)

# Convenience functions for common operations
async def redis_execute(command: str, *args, pool_name: str = 'default', **kwargs) -> Any:
    """Execute Redis command using pool manager"""
    pool_manager = await get_redis_pool_manager()
    return await pool_manager.execute(command, *args, pool_name=pool_name, **kwargs)

async def redis_publish(channel: str, message: str, pool_name: str = 'default') -> int:
    """Publish message to Redis channel"""
    pool_manager = await get_redis_pool_manager()
    return await pool_manager.publish(channel, message, pool_name)

async def redis_set(key: str, value: str, ex: Optional[int] = None, pool_name: str = 'default'):
    """Set key-value pair in Redis"""
    pool_manager = await get_redis_pool_manager()
    return await pool_manager.set(key, value, ex=ex, pool_name=pool_name)

async def redis_get(key: str, pool_name: str = 'default') -> Optional[str]:
    """Get value from Redis"""
    pool_manager = await get_redis_pool_manager()
    return await pool_manager.get(key, pool_name)
