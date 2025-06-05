"""
Rate Limiting System for AI Crypto Trader
Implements multiple rate limiting algorithms with Redis-backed storage
"""
import time
import asyncio
import logging
from typing import Dict, Any, Optional, Union, List
from enum import Enum
from dataclasses import dataclass
from services.utils.redis_pool import get_redis_pool_manager
from services.utils.metrics import get_metrics, is_metrics_enabled
import json
import hashlib

logger = logging.getLogger(__name__)

class RateLimitAlgorithm(Enum):
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"

@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests: int              # Number of requests allowed
    window: int               # Time window in seconds
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    burst_limit: Optional[int] = None  # Max burst size for token bucket
    leak_rate: Optional[float] = None  # Leak rate for leaky bucket

@dataclass
class RateLimitResult:
    """Result of rate limit check"""
    allowed: bool
    remaining: int
    reset_time: float
    retry_after: Optional[int] = None
    algorithm_used: str = ""
    metadata: Dict[str, Any] = None

class RateLimiter:
    """
    Advanced rate limiter with multiple algorithms and Redis backend
    """
    
    def __init__(self, redis_pool_manager=None):
        self.redis_pool_manager = redis_pool_manager
        self.metrics = None
        if is_metrics_enabled():
            self.metrics = get_metrics('rate_limiter')
        
        # Default rate limits for different endpoints
        self.default_limits = {
            'trading_signals': RateLimit(100, 60),  # 100 signals per minute
            'api_requests': RateLimit(1000, 60),    # 1000 API requests per minute
            'ai_analysis': RateLimit(50, 60),       # 50 AI analyses per minute
            'market_data': RateLimit(10000, 60),    # 10K market data requests per minute
            'user_actions': RateLimit(20, 60),      # 20 user actions per minute
            'binance_api': RateLimit(1200, 60),     # Binance API limit
            'openai_api': RateLimit(60, 60),        # OpenAI API limit
            'redis_operations': RateLimit(10000, 60), # Redis operations
        }
    
    async def initialize(self):
        """Initialize rate limiter"""
        if not self.redis_pool_manager:
            self.redis_pool_manager = await get_redis_pool_manager()
        logger.info("Rate Limiter initialized")
    
    def _get_key(self, identifier: str, endpoint: str) -> str:
        """Generate Redis key for rate limit tracking"""
        return f"rate_limit:{endpoint}:{identifier}"
    
    async def check_limit(
        self,
        identifier: str,
        endpoint: str,
        rate_limit: Optional[RateLimit] = None,
        cost: int = 1
    ) -> RateLimitResult:
        """
        Check if request is within rate limit
        
        Args:
            identifier: Unique identifier (IP, user ID, service name)
            endpoint: Endpoint or operation being rate limited
            rate_limit: Custom rate limit (uses default if None)
            cost: Cost of this request (default 1)
        """
        if not self.redis_pool_manager:
            await self.initialize()
        
        rate_limit = rate_limit or self.default_limits.get(endpoint, RateLimit(100, 60))
        
        try:
            if rate_limit.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                result = await self._check_token_bucket(identifier, endpoint, rate_limit, cost)
            elif rate_limit.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                result = await self._check_sliding_window(identifier, endpoint, rate_limit, cost)
            elif rate_limit.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                result = await self._check_fixed_window(identifier, endpoint, rate_limit, cost)
            elif rate_limit.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
                result = await self._check_leaky_bucket(identifier, endpoint, rate_limit, cost)
            else:
                # Default to sliding window
                result = await self._check_sliding_window(identifier, endpoint, rate_limit, cost)
            
            # Record metrics
            if self.metrics:
                self.metrics.request_counter.labels(
                    service='rate_limiter',
                    endpoint=endpoint,
                    method='check'
                ).inc()
                
                if not result.allowed:
                    self.metrics.error_counter.labels(
                        service='rate_limiter',
                        error_type='rate_limit_exceeded',
                        endpoint=endpoint
                    ).inc()
            
            return result
            
        except Exception as e:
            logger.error(f"Rate limit check failed for {identifier}:{endpoint}: {e}")
            # Fail open - allow the request if rate limiting fails
            return RateLimitResult(
                allowed=True,
                remaining=rate_limit.requests,
                reset_time=time.time() + rate_limit.window,
                algorithm_used="fail_open",
                metadata={"error": str(e)}
            )
    
    async def _check_sliding_window(
        self, 
        identifier: str, 
        endpoint: str, 
        rate_limit: RateLimit, 
        cost: int
    ) -> RateLimitResult:
        """Sliding window rate limiting algorithm"""
        key = self._get_key(identifier, endpoint)
        now = time.time()
        window_start = now - rate_limit.window
        
        # Use Redis pipeline for atomic operations
        redis_client = self.redis_pool_manager.get_client()
        pipe = redis_client.pipeline()
        
        # Remove expired entries and count current requests
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.expire(key, rate_limit.window * 2)  # Set expiration
        
        results = await pipe.execute()
        current_requests = results[1]
        
        if current_requests + cost <= rate_limit.requests:
            # Add current request(s)
            for _ in range(cost):
                await redis_client.zadd(key, {f"{now}_{id({})}": now})
            
            return RateLimitResult(
                allowed=True,
                remaining=rate_limit.requests - current_requests - cost,
                reset_time=now + rate_limit.window,
                algorithm_used="sliding_window"
            )
        else:
            # Rate limit exceeded
            oldest_request = await redis_client.zrange(key, 0, 0, withscores=True)
            if oldest_request:
                reset_time = oldest_request[0][1] + rate_limit.window
            else:
                reset_time = now + rate_limit.window
            
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=reset_time,
                retry_after=int(reset_time - now),
                algorithm_used="sliding_window"
            )
    
    async def _check_fixed_window(
        self, 
        identifier: str, 
        endpoint: str, 
        rate_limit: RateLimit, 
        cost: int
    ) -> RateLimitResult:
        """Fixed window rate limiting algorithm"""
        key = self._get_key(identifier, endpoint)
        now = time.time()
        window_start = int(now // rate_limit.window) * rate_limit.window
        window_key = f"{key}:{window_start}"
        
        redis_client = self.redis_pool_manager.get_client()
        
        # Get current count and increment atomically
        pipe = redis_client.pipeline()
        pipe.get(window_key)
        pipe.expire(window_key, rate_limit.window)
        
        results = await pipe.execute()
        current_count = int(results[0] or 0)
        
        if current_count + cost <= rate_limit.requests:
            # Increment counter
            await redis_client.incrby(window_key, cost)
            await redis_client.expire(window_key, rate_limit.window)
            
            return RateLimitResult(
                allowed=True,
                remaining=rate_limit.requests - current_count - cost,
                reset_time=window_start + rate_limit.window,
                algorithm_used="fixed_window"
            )
        else:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=window_start + rate_limit.window,
                retry_after=int(window_start + rate_limit.window - now),
                algorithm_used="fixed_window"
            )
    
    async def _check_token_bucket(
        self, 
        identifier: str, 
        endpoint: str, 
        rate_limit: RateLimit, 
        cost: int
    ) -> RateLimitResult:
        """Token bucket rate limiting algorithm"""
        key = self._get_key(identifier, endpoint)
        now = time.time()
        
        burst_limit = rate_limit.burst_limit or rate_limit.requests
        refill_rate = rate_limit.requests / rate_limit.window  # tokens per second
        
        redis_client = self.redis_pool_manager.get_client()
        
        # Get current bucket state
        bucket_data = await redis_client.get(key)
        if bucket_data:
            bucket = json.loads(bucket_data)
            last_refill = bucket['last_refill']
            tokens = bucket['tokens']
        else:
            last_refill = now
            tokens = burst_limit
        
        # Calculate tokens to add
        time_passed = now - last_refill
        tokens_to_add = time_passed * refill_rate
        tokens = min(burst_limit, tokens + tokens_to_add)
        
        if tokens >= cost:
            # Consume tokens
            tokens -= cost
            
            # Update bucket state
            bucket_data = {
                'tokens': tokens,
                'last_refill': now
            }
            await redis_client.setex(key, rate_limit.window * 2, json.dumps(bucket_data))
            
            return RateLimitResult(
                allowed=True,
                remaining=int(tokens),
                reset_time=now + (burst_limit - tokens) / refill_rate,
                algorithm_used="token_bucket"
            )
        else:
            # Not enough tokens
            time_until_token = (cost - tokens) / refill_rate
            
            return RateLimitResult(
                allowed=False,
                remaining=int(tokens),
                reset_time=now + time_until_token,
                retry_after=int(time_until_token),
                algorithm_used="token_bucket"
            )
    
    async def _check_leaky_bucket(
        self, 
        identifier: str, 
        endpoint: str, 
        rate_limit: RateLimit, 
        cost: int
    ) -> RateLimitResult:
        """Leaky bucket rate limiting algorithm"""
        key = self._get_key(identifier, endpoint)
        now = time.time()
        
        leak_rate = rate_limit.leak_rate or (rate_limit.requests / rate_limit.window)
        bucket_size = rate_limit.requests
        
        redis_client = self.redis_pool_manager.get_client()
        
        # Get current bucket state
        bucket_data = await redis_client.get(key)
        if bucket_data:
            bucket = json.loads(bucket_data)
            last_leak = bucket['last_leak']
            level = bucket['level']
        else:
            last_leak = now
            level = 0
        
        # Calculate leaked amount
        time_passed = now - last_leak
        leaked = time_passed * leak_rate
        level = max(0, level - leaked)
        
        if level + cost <= bucket_size:
            # Add to bucket
            level += cost
            
            bucket_data = {
                'level': level,
                'last_leak': now
            }
            await redis_client.setex(key, rate_limit.window * 2, json.dumps(bucket_data))
            
            return RateLimitResult(
                allowed=True,
                remaining=int(bucket_size - level),
                reset_time=now + level / leak_rate,
                algorithm_used="leaky_bucket"
            )
        else:
            # Bucket overflow
            time_until_space = (level + cost - bucket_size) / leak_rate
            
            return RateLimitResult(
                allowed=False,
                remaining=int(bucket_size - level),
                reset_time=now + time_until_space,
                retry_after=int(time_until_space),
                algorithm_used="leaky_bucket"
            )
    
    async def reset_limit(self, identifier: str, endpoint: str) -> bool:
        """Reset rate limit for specific identifier/endpoint"""
        try:
            key = self._get_key(identifier, endpoint)
            redis_client = self.redis_pool_manager.get_client()
            await redis_client.delete(key)
            
            logger.info(f"Rate limit reset for {identifier}:{endpoint}")
            return True
        except Exception as e:
            logger.error(f"Failed to reset rate limit for {identifier}:{endpoint}: {e}")
            return False
    
    async def get_status(self, identifier: str, endpoint: str) -> Dict[str, Any]:
        """Get current rate limit status"""
        try:
            rate_limit = self.default_limits.get(endpoint, RateLimit(100, 60))
            
            # Check current status without consuming quota
            result = await self.check_limit(identifier, endpoint, rate_limit, cost=0)
            
            return {
                'identifier': identifier,
                'endpoint': endpoint,
                'rate_limit': {
                    'requests': rate_limit.requests,
                    'window': rate_limit.window,
                    'algorithm': rate_limit.algorithm.value
                },
                'current_status': {
                    'remaining': result.remaining,
                    'reset_time': result.reset_time,
                    'allowed': result.allowed
                }
            }
        except Exception as e:
            logger.error(f"Failed to get rate limit status for {identifier}:{endpoint}: {e}")
            return {'error': str(e)}
    
    async def get_all_limits(self, identifier: str) -> Dict[str, Any]:
        """Get rate limit status for all endpoints for an identifier"""
        status = {}
        for endpoint in self.default_limits.keys():
            status[endpoint] = await self.get_status(identifier, endpoint)
        return status
    
    def add_custom_limit(self, endpoint: str, rate_limit: RateLimit):
        """Add custom rate limit for endpoint"""
        self.default_limits[endpoint] = rate_limit
        logger.info(f"Added custom rate limit for {endpoint}: {rate_limit}")
    
    async def cleanup_expired(self, batch_size: int = 1000) -> int:
        """Clean up expired rate limit keys"""
        try:
            redis_client = self.redis_pool_manager.get_client()
            
            # Find rate limit keys
            keys = await redis_client.keys("rate_limit:*")
            cleaned = 0
            
            for i in range(0, len(keys), batch_size):
                batch = keys[i:i + batch_size]
                
                # Check which keys are expired or empty
                pipe = redis_client.pipeline()
                for key in batch:
                    pipe.ttl(key)
                    pipe.exists(key)
                
                results = await pipe.execute()
                
                # Delete expired keys
                to_delete = []
                for j, key in enumerate(batch):
                    ttl = results[j * 2]
                    exists = results[j * 2 + 1]
                    
                    if ttl == -1 or not exists:  # No TTL set or key doesn't exist
                        to_delete.append(key)
                
                if to_delete:
                    await redis_client.delete(*to_delete)
                    cleaned += len(to_delete)
            
            logger.info(f"Cleaned up {cleaned} expired rate limit keys")
            return cleaned
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired rate limits: {e}")
            return 0


# Decorator for automatic rate limiting
def rate_limit(
    endpoint: str,
    identifier_func: callable = None,
    rate_limit_config: RateLimit = None,
    cost: int = 1
):
    """
    Decorator to automatically apply rate limiting to functions
    
    Args:
        endpoint: Endpoint name for rate limiting
        identifier_func: Function to extract identifier from args/kwargs
        rate_limit_config: Custom rate limit configuration
        cost: Cost of this operation
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Get identifier
            if identifier_func:
                identifier = identifier_func(*args, **kwargs)
            else:
                identifier = 'default'
            
            # Get rate limiter
            limiter = RateLimiter()
            await limiter.initialize()
            
            # Check rate limit
            result = await limiter.check_limit(identifier, endpoint, rate_limit_config, cost)
            
            if not result.allowed:
                from aiohttp import web
                raise web.HTTPTooManyRequests(
                    text=f"Rate limit exceeded. Try again in {result.retry_after} seconds.",
                    headers={
                        'X-RateLimit-Limit': str(rate_limit_config.requests if rate_limit_config else 100),
                        'X-RateLimit-Remaining': str(result.remaining),
                        'X-RateLimit-Reset': str(int(result.reset_time)),
                        'Retry-After': str(result.retry_after or 60)
                    }
                )
            
            return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we'd need to handle differently
            # For now, just call the function
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None

async def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance"""
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
        await _rate_limiter.initialize()
    
    return _rate_limiter

# Convenience functions
async def check_rate_limit(
    identifier: str,
    endpoint: str,
    rate_limit: Optional[RateLimit] = None,
    cost: int = 1
) -> RateLimitResult:
    """Check rate limit using global limiter"""
    limiter = await get_rate_limiter()
    return await limiter.check_limit(identifier, endpoint, rate_limit, cost)

async def reset_rate_limit(identifier: str, endpoint: str) -> bool:
    """Reset rate limit using global limiter"""
    limiter = await get_rate_limiter()
    return await limiter.reset_limit(identifier, endpoint)
