"""
Circuit Breaker implementation for AI Crypto Trader services
"""
import asyncio
import time
import logging
from enum import Enum
from typing import Callable, Any, Optional, Dict
from functools import wraps
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "CLOSED"       # Normal operation
    OPEN = "OPEN"          # Circuit is open, requests fail fast
    HALF_OPEN = "HALF_OPEN" # Testing if service is back

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5          # Number of failures to open circuit
    recovery_timeout: int = 60          # Seconds before trying half-open
    expected_exception: type = Exception # Exception type that counts as failure
    success_threshold: int = 3          # Successes needed to close from half-open
    timeout: float = 30.0               # Request timeout in seconds

class CircuitBreakerOpenException(Exception):
    """Raised when circuit breaker is open"""
    pass

class CircuitBreaker:
    """
    Circuit Breaker implementation for protecting external service calls
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        
        # Metrics tracking
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.total_timeouts = 0
        self.total_circuit_opens = 0
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker"""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self._call_async(func, *args, **kwargs)
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return self._call_sync(func, *args, **kwargs)
            
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    async def _call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection"""
        self.total_requests += 1
        
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._move_to_half_open()
            else:
                logger.warning(f"Circuit breaker {self.name} is OPEN, failing fast")
                raise CircuitBreakerOpenException(f"Circuit breaker {self.name} is open")
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            self._on_success()
            return result
            
        except asyncio.TimeoutError:
            self.total_timeouts += 1
            self._on_failure()
            logger.error(f"Circuit breaker {self.name}: Request timed out after {self.config.timeout}s")
            raise
            
        except self.config.expected_exception as e:
            self._on_failure()
            logger.error(f"Circuit breaker {self.name}: Expected failure: {str(e)}")
            raise
            
        except Exception as e:
            # Unexpected exceptions don't count as failures
            logger.error(f"Circuit breaker {self.name}: Unexpected error: {str(e)}")
            raise
    
    def _call_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute sync function with circuit breaker protection"""
        self.total_requests += 1
        
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._move_to_half_open()
            else:
                logger.warning(f"Circuit breaker {self.name} is OPEN, failing fast")
                raise CircuitBreakerOpenException(f"Circuit breaker {self.name} is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.config.expected_exception as e:
            self._on_failure()
            logger.error(f"Circuit breaker {self.name}: Expected failure: {str(e)}")
            raise
            
        except Exception as e:
            # Unexpected exceptions don't count as failures
            logger.error(f"Circuit breaker {self.name}: Unexpected error: {str(e)}")
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def _move_to_half_open(self):
        """Move circuit to half-open state"""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        logger.info(f"Circuit breaker {self.name} moved to HALF_OPEN")
    
    def _on_success(self):
        """Handle successful request"""
        self.total_successes += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._close_circuit()
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed request"""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open moves back to open
            self._open_circuit()
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._open_circuit()
    
    def _open_circuit(self):
        """Open the circuit breaker"""
        self.state = CircuitState.OPEN
        self.total_circuit_opens += 1
        logger.warning(f"Circuit breaker {self.name} OPENED after {self.failure_count} failures")
    
    def _close_circuit(self):
        """Close the circuit breaker"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker {self.name} CLOSED - service recovered")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'total_requests': self.total_requests,
            'total_failures': self.total_failures,
            'total_successes': self.total_successes,
            'total_timeouts': self.total_timeouts,
            'total_circuit_opens': self.total_circuit_opens,
            'failure_rate': self.total_failures / max(self.total_requests, 1),
            'last_failure_time': self.last_failure_time,
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout': self.config.recovery_timeout,
                'success_threshold': self.config.success_threshold,
                'timeout': self.config.timeout
            }
        }
    
    def reset(self):
        """Manually reset the circuit breaker"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info(f"Circuit breaker {self.name} manually reset")


class RetryConfig:
    """Configuration for retry mechanism"""
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter


async def retry_with_backoff(
    func: Callable,
    config: RetryConfig = None,
    exceptions: tuple = (Exception,),
    *args,
    **kwargs
) -> Any:
    """
    Retry function with exponential backoff
    """
    import random
    
    config = config or RetryConfig()
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
                
        except exceptions as e:
            last_exception = e
            
            if attempt == config.max_attempts - 1:
                # Last attempt, re-raise the exception
                raise e
            
            # Calculate delay with exponential backoff
            delay = min(
                config.base_delay * (config.backoff_factor ** attempt),
                config.max_delay
            )
            
            # Add jitter to prevent thundering herd
            if config.jitter:
                delay *= (0.5 + random.random() * 0.5)
            
            logger.warning(
                f"Attempt {attempt + 1} failed: {str(e)}. "
                f"Retrying in {delay:.2f}s..."
            )
            
            await asyncio.sleep(delay)
    
    # Should never reach here, but just in case
    if last_exception:
        raise last_exception


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}

def get_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Get or create a circuit breaker by name"""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]

def get_all_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """Get all registered circuit breakers"""
    return _circuit_breakers.copy()

def reset_all_circuit_breakers():
    """Reset all circuit breakers - useful for testing"""
    for cb in _circuit_breakers.values():
        cb.reset()

# Convenience decorators
def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    timeout: float = 30.0
):
    """Decorator for easy circuit breaker application"""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        timeout=timeout
    )
    cb = get_circuit_breaker(name, config)
    return cb

def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for adding retry functionality"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            config = RetryConfig(max_attempts=max_attempts, base_delay=base_delay)
            return await retry_with_backoff(func, config, exceptions, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we'll need to handle this differently
            # For now, just call the function directly
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
