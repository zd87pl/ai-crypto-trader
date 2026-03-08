"""
API Security and Key Rotation System
Comprehensive security management for API keys, authentication, and access control
"""
import os
import asyncio
import logging
import secrets
import hashlib
import hmac
import time
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from services.utils.redis_pool import get_redis_pool_manager
from services.utils.metrics import get_metrics, is_metrics_enabled
from services.utils.rate_limiter import get_rate_limiter, RateLimit
import jwt
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

class KeyStatus(Enum):
    ACTIVE = "active"
    ROTATING = "rotating"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"

class AccessLevel(Enum):
    READ_ONLY = "read_only"
    TRADING = "trading"
    ADMIN = "admin"
    SYSTEM = "system"

@dataclass
class APIKey:
    """API Key information"""
    key_id: str
    key_hash: str
    name: str
    access_level: AccessLevel
    status: KeyStatus
    created_at: datetime
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    rate_limits: Optional[Dict[str, RateLimit]] = field(default=None)
    allowed_ips: Optional[List[str]] = field(default=None)
    metadata: Optional[Dict[str, Any]] = field(default=None)

@dataclass
class AuthResult:
    """Authentication result"""
    success: bool
    key_info: Optional[APIKey] = None
    error: Optional[str] = None
    rate_limit_remaining: Optional[int] = None

class APISecurityManager:
    """
    Comprehensive API security management system
    """
    
    def __init__(self):
        self.redis_pool_manager = None
        self.rate_limiter = None
        self.metrics = None
        
        if is_metrics_enabled():
            self.metrics = get_metrics('api_security')
        
        # Security configuration
        jwt_secret = os.getenv('JWT_SECRET')
        if not jwt_secret:
            jwt_secret = self._generate_secret()
            logger.warning("JWT_SECRET not set - generating ephemeral secret. Tokens will be invalid after restart.")

        encryption_key = os.getenv('ENCRYPTION_KEY')
        if not encryption_key:
            encryption_key = Fernet.generate_key().decode()
            logger.warning("ENCRYPTION_KEY not set - generating ephemeral key. Encrypted data will be unreadable after restart.")

        self.config = {
            'key_rotation_interval': int(os.getenv('API_KEY_ROTATION_INTERVAL', 86400)),  # 24 hours
            'key_expiry_days': int(os.getenv('API_KEY_EXPIRY_DAYS', 90)),
            'max_keys_per_user': int(os.getenv('MAX_KEYS_PER_USER', 5)),
            'require_ip_whitelist': os.getenv('REQUIRE_IP_WHITELIST', 'false').lower() == 'true',
            'jwt_secret': jwt_secret,
            'encryption_key': encryption_key,
            'audit_log_retention_days': int(os.getenv('AUDIT_LOG_RETENTION_DAYS', 30))
        }
        
        # Initialize encryption
        self.fernet = Fernet(self.config['encryption_key'].encode())
        
        # Default rate limits by access level
        self.default_rate_limits = {
            AccessLevel.READ_ONLY: {
                'api_requests': RateLimit(100, 60),
                'market_data': RateLimit(1000, 60)
            },
            AccessLevel.TRADING: {
                'api_requests': RateLimit(500, 60),
                'trading_signals': RateLimit(50, 60),
                'market_data': RateLimit(2000, 60)
            },
            AccessLevel.ADMIN: {
                'api_requests': RateLimit(1000, 60),
                'admin_actions': RateLimit(100, 60)
            },
            AccessLevel.SYSTEM: {
                'api_requests': RateLimit(10000, 60),
                'system_operations': RateLimit(1000, 60)
            }
        }
    
    def _generate_secret(self) -> str:
        """Generate a secure random secret"""
        return secrets.token_urlsafe(32)
    
    async def initialize(self):
        """Initialize security manager"""
        self.redis_pool_manager = await get_redis_pool_manager()
        self.rate_limiter = await get_rate_limiter()
        
        # Schedule key rotation check
        asyncio.create_task(self._rotation_scheduler())
        
        logger.info("API Security Manager initialized")
    
    def _hash_key(self, api_key: str) -> str:
        """Hash API key for secure storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def _generate_api_key(self) -> Tuple[str, str]:
        """Generate API key and its hash"""
        # Format: prefix_random_checksum
        prefix = "ak"  # api key
        random_part = secrets.token_urlsafe(24)
        checksum = hashlib.sha256(f"{prefix}_{random_part}".encode()).hexdigest()[:8]
        api_key = f"{prefix}_{random_part}_{checksum}"
        
        return api_key, self._hash_key(api_key)
    
    async def create_api_key(
        self,
        name: str,
        access_level: AccessLevel,
        user_id: str = "system",
        expires_in_days: Optional[int] = None,
        allowed_ips: Optional[List[str]] = None,
        custom_rate_limits: Optional[Dict[str, RateLimit]] = None
    ) -> Tuple[str, str]:  # Returns (api_key, key_id)
        """Create a new API key"""
        try:
            # Check if user has too many keys
            existing_keys = await self._get_user_keys(user_id)
            if len(existing_keys) >= self.config['max_keys_per_user']:
                raise ValueError(f"User has reached maximum number of API keys ({self.config['max_keys_per_user']})")
            
            # Generate key
            api_key, key_hash = self._generate_api_key()
            key_id = secrets.token_urlsafe(16)
            
            # Set expiration
            expires_at = None
            if expires_in_days:
                expires_at = datetime.now() + timedelta(days=expires_in_days)
            elif self.config['key_expiry_days'] > 0:
                expires_at = datetime.now() + timedelta(days=self.config['key_expiry_days'])
            
            # Create API key object
            api_key_obj = APIKey(
                key_id=key_id,
                key_hash=key_hash,
                name=name,
                access_level=access_level,
                status=KeyStatus.ACTIVE,
                created_at=datetime.now(),
                expires_at=expires_at,
                rate_limits=custom_rate_limits or self.default_rate_limits.get(access_level, {}),
                allowed_ips=allowed_ips,
                metadata={'user_id': user_id, 'created_by': 'system'}
            )
            
            # Store in Redis
            await self._store_api_key(api_key_obj)
            
            # Add to user's key list
            await self._add_user_key(user_id, key_id)
            
            # Log creation
            await self._audit_log('key_created', {
                'key_id': key_id,
                'name': name,
                'access_level': access_level.value,
                'user_id': user_id
            })
            
            # Update metrics
            if self.metrics:
                self.metrics.request_counter.labels(
                    service='api_security',
                    endpoint='create_key',
                    method='POST'
                ).inc()
            
            logger.info(f"Created API key {key_id} for user {user_id} with access level {access_level.value}")
            return api_key, key_id
            
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            if self.metrics:
                self.metrics.error_counter.labels(
                    service='api_security',
                    error_type=type(e).__name__,
                    endpoint='create_key'
                ).inc()
            raise
    
    async def authenticate(
        self,
        api_key: str,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None
    ) -> AuthResult:
        """Authenticate API key and check permissions"""
        try:
            # Hash the provided key
            key_hash = self._hash_key(api_key)
            
            # Get key info from Redis
            key_info = await self._get_api_key_by_hash(key_hash)
            if not key_info:
                await self._audit_log('auth_failed', {
                    'reason': 'invalid_key',
                    'ip_address': ip_address
                })
                return AuthResult(success=False, error="Invalid API key")
            
            # Check key status
            if key_info.status != KeyStatus.ACTIVE:
                await self._audit_log('auth_failed', {
                    'key_id': key_info.key_id,
                    'reason': 'key_not_active',
                    'status': key_info.status.value,
                    'ip_address': ip_address
                })
                return AuthResult(success=False, error=f"API key is {key_info.status.value}")
            
            # Check expiration
            if key_info.expires_at and datetime.now() > key_info.expires_at:
                await self._revoke_key(key_info.key_id, "expired")
                return AuthResult(success=False, error="API key has expired")
            
            # Check IP whitelist
            if self.config['require_ip_whitelist'] and key_info.allowed_ips:
                if ip_address not in key_info.allowed_ips:
                    await self._audit_log('auth_failed', {
                        'key_id': key_info.key_id,
                        'reason': 'ip_not_allowed',
                        'ip_address': ip_address,
                        'allowed_ips': key_info.allowed_ips
                    })
                    return AuthResult(success=False, error="IP address not allowed")
            
            # Check rate limits
            if endpoint and key_info.rate_limits and endpoint in key_info.rate_limits:
                rate_limit = key_info.rate_limits[endpoint]
                result = await self.rate_limiter.check_limit(
                    key_info.key_id, endpoint, rate_limit
                )
                if not result.allowed:
                    await self._audit_log('auth_failed', {
                        'key_id': key_info.key_id,
                        'reason': 'rate_limit_exceeded',
                        'endpoint': endpoint,
                        'ip_address': ip_address
                    })
                    return AuthResult(
                        success=False,
                        error="Rate limit exceeded",
                        rate_limit_remaining=result.remaining
                    )
            
            # Update last used timestamp
            await self._update_last_used(key_info.key_id)
            
            # Log successful authentication
            await self._audit_log('auth_success', {
                'key_id': key_info.key_id,
                'access_level': key_info.access_level.value,
                'ip_address': ip_address,
                'endpoint': endpoint
            })
            
            # Update metrics
            if self.metrics:
                self.metrics.request_counter.labels(
                    service='api_security',
                    endpoint='authenticate',
                    method='POST'
                ).inc()
            
            return AuthResult(success=True, key_info=key_info)
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            if self.metrics:
                self.metrics.error_counter.labels(
                    service='api_security',
                    error_type=type(e).__name__,
                    endpoint='authenticate'
                ).inc()
            return AuthResult(success=False, error="Authentication failed")
    
    async def rotate_key(self, key_id: str) -> Tuple[str, str]:
        """Rotate an existing API key"""
        try:
            # Get existing key
            key_info = await self._get_api_key(key_id)
            if not key_info:
                raise ValueError("API key not found")
            
            # Mark old key as rotating
            key_info.status = KeyStatus.ROTATING
            await self._store_api_key(key_info)
            
            # Generate new key
            new_api_key, new_key_hash = self._generate_api_key()
            new_key_id = secrets.token_urlsafe(16)
            
            # Create new key object with same permissions
            new_key_info = APIKey(
                key_id=new_key_id,
                key_hash=new_key_hash,
                name=f"{key_info.name} (rotated)",
                access_level=key_info.access_level,
                status=KeyStatus.ACTIVE,
                created_at=datetime.now(),
                expires_at=key_info.expires_at,
                rate_limits=key_info.rate_limits,
                allowed_ips=key_info.allowed_ips,
                metadata={**key_info.metadata, 'rotated_from': key_id}
            )
            
            # Store new key
            await self._store_api_key(new_key_info)
            
            # Update user's key list
            user_id = key_info.metadata.get('user_id', 'system')
            await self._add_user_key(user_id, new_key_id)
            
            # Schedule old key deprecation (grace period)
            await self._schedule_key_deprecation(key_id, 3600)  # 1 hour grace period
            
            # Log rotation
            await self._audit_log('key_rotated', {
                'old_key_id': key_id,
                'new_key_id': new_key_id,
                'user_id': user_id
            })
            
            logger.info(f"Rotated API key {key_id} to {new_key_id}")
            return new_api_key, new_key_id
            
        except Exception as e:
            logger.error(f"Failed to rotate key {key_id}: {e}")
            raise
    
    async def revoke_key(self, key_id: str, reason: str = "manual") -> bool:
        """Revoke an API key"""
        return await self._revoke_key(key_id, reason)
    
    async def _revoke_key(self, key_id: str, reason: str) -> bool:
        """Internal key revocation"""
        try:
            key_info = await self._get_api_key(key_id)
            if not key_info:
                return False
            
            # Update status
            key_info.status = KeyStatus.REVOKED
            key_info.metadata['revoked_at'] = datetime.now().isoformat()
            key_info.metadata['revocation_reason'] = reason
            
            await self._store_api_key(key_info)
            
            # Remove from user's active keys
            user_id = key_info.metadata.get('user_id', 'system')
            await self._remove_user_key(user_id, key_id)
            
            # Log revocation
            await self._audit_log('key_revoked', {
                'key_id': key_id,
                'reason': reason,
                'user_id': user_id
            })
            
            logger.info(f"Revoked API key {key_id} (reason: {reason})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke key {key_id}: {e}")
            return False
    
    async def get_key_info(self, key_id: str) -> Optional[APIKey]:
        """Get API key information"""
        return await self._get_api_key(key_id)
    
    async def list_user_keys(self, user_id: str) -> List[APIKey]:
        """List all keys for a user"""
        try:
            key_ids = await self._get_user_keys(user_id)
            keys = []
            
            for key_id in key_ids:
                key_info = await self._get_api_key(key_id)
                if key_info:
                    keys.append(key_info)
            
            return keys
            
        except Exception as e:
            logger.error(f"Failed to list keys for user {user_id}: {e}")
            return []
    
    async def cleanup_expired_keys(self) -> int:
        """Clean up expired and deprecated keys"""
        try:
            cleaned = 0
            
            # Get all key IDs using SCAN (non-blocking unlike KEYS)
            redis_client = self.redis_pool_manager.get_client()
            keys = []
            async for redis_key in redis_client.scan_iter(match="api_key:*", count=100):
                keys.append(redis_key)

            for redis_key in keys:
                try:
                    key_data = await redis_client.get(redis_key)
                    if not key_data:
                        continue
                    
                    key_info_dict = json.loads(key_data)
                    
                    # Check expiration
                    if key_info_dict.get('expires_at'):
                        expires_at = datetime.fromisoformat(key_info_dict['expires_at'])
                        if datetime.now() > expires_at:
                            key_id = key_info_dict['key_id']
                            await self._revoke_key(key_id, "expired")
                            cleaned += 1
                    
                    # Check deprecated keys older than grace period
                    if key_info_dict.get('status') == 'deprecated':
                        created_at = datetime.fromisoformat(key_info_dict['created_at'])
                        if datetime.now() > created_at + timedelta(hours=24):
                            key_id = key_info_dict['key_id']
                            await self._revoke_key(key_id, "cleanup")
                            cleaned += 1
                            
                except Exception as e:
                    logger.error(f"Error processing key {redis_key}: {e}")
                    continue
            
            logger.info(f"Cleaned up {cleaned} expired/deprecated API keys")
            return cleaned
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired keys: {e}")
            return 0
    
    # Internal methods for Redis operations
    async def _store_api_key(self, key_info: APIKey):
        """Store API key in Redis"""
        redis_client = self.redis_pool_manager.get_client()
        
        # Store by key ID
        key_data = {
            'key_id': key_info.key_id,
            'key_hash': key_info.key_hash,
            'name': key_info.name,
            'access_level': key_info.access_level.value,
            'status': key_info.status.value,
            'created_at': key_info.created_at.isoformat(),
            'last_used': key_info.last_used.isoformat() if key_info.last_used else None,
            'expires_at': key_info.expires_at.isoformat() if key_info.expires_at else None,
            'rate_limits': {k: {'requests': v.requests, 'window': v.window, 'algorithm': v.algorithm.value} 
                           for k, v in (key_info.rate_limits or {}).items()},
            'allowed_ips': key_info.allowed_ips or [],
            'metadata': key_info.metadata or {}
        }
        
        await redis_client.setex(f"api_key:{key_info.key_id}", 86400 * 30, json.dumps(key_data))
        
        # Store hash -> key_id mapping for fast lookup
        await redis_client.setex(f"api_key_hash:{key_info.key_hash}", 86400 * 30, key_info.key_id)
    
    async def _get_api_key(self, key_id: str) -> Optional[APIKey]:
        """Get API key by ID"""
        try:
            redis_client = self.redis_pool_manager.get_client()
            key_data = await redis_client.get(f"api_key:{key_id}")
            
            if not key_data:
                return None
            
            data = json.loads(key_data)
            return self._deserialize_api_key(data)
            
        except Exception as e:
            logger.error(f"Failed to get API key {key_id}: {e}")
            return None
    
    async def _get_api_key_by_hash(self, key_hash: str) -> Optional[APIKey]:
        """Get API key by hash"""
        try:
            redis_client = self.redis_pool_manager.get_client()
            key_id = await redis_client.get(f"api_key_hash:{key_hash}")
            
            if not key_id:
                return None
            
            return await self._get_api_key(key_id)
            
        except Exception as e:
            logger.error(f"Failed to get API key by hash: {e}")
            return None
    
    def _deserialize_api_key(self, data: Dict) -> APIKey:
        """Convert dict to APIKey object"""
        from services.utils.rate_limiter import RateLimit, RateLimitAlgorithm
        
        # Deserialize rate limits
        rate_limits = {}
        if data.get('rate_limits'):
            for endpoint, limit_data in data['rate_limits'].items():
                rate_limits[endpoint] = RateLimit(
                    requests=limit_data['requests'],
                    window=limit_data['window'],
                    algorithm=RateLimitAlgorithm(limit_data['algorithm'])
                )
        
        return APIKey(
            key_id=data['key_id'],
            key_hash=data['key_hash'],
            name=data['name'],
            access_level=AccessLevel(data['access_level']),
            status=KeyStatus(data['status']),
            created_at=datetime.fromisoformat(data['created_at']),
            last_used=datetime.fromisoformat(data['last_used']) if data.get('last_used') else None,
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            rate_limits=rate_limits if rate_limits else None,
            allowed_ips=data.get('allowed_ips'),
            metadata=data.get('metadata', {})
        )
    
    async def _update_last_used(self, key_id: str):
        """Update last used timestamp"""
        try:
            key_info = await self._get_api_key(key_id)
            if key_info:
                key_info.last_used = datetime.now()
                await self._store_api_key(key_info)
        except Exception as e:
            logger.error(f"Failed to update last used for key {key_id}: {e}")
    
    async def _get_user_keys(self, user_id: str) -> List[str]:
        """Get list of key IDs for user"""
        try:
            redis_client = self.redis_pool_manager.get_client()
            keys = await redis_client.smembers(f"user_keys:{user_id}")
            return list(keys) if keys else []
        except Exception as e:
            logger.error(f"Failed to get user keys for {user_id}: {e}")
            return []
    
    async def _add_user_key(self, user_id: str, key_id: str):
        """Add key to user's key set"""
        redis_client = self.redis_pool_manager.get_client()
        await redis_client.sadd(f"user_keys:{user_id}", key_id)
    
    async def _remove_user_key(self, user_id: str, key_id: str):
        """Remove key from user's key set"""
        redis_client = self.redis_pool_manager.get_client()
        await redis_client.srem(f"user_keys:{user_id}", key_id)
    
    async def _audit_log(self, action: str, details: Dict[str, Any]):
        """Log security events for audit"""
        try:
            redis_client = self.redis_pool_manager.get_client()
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'details': details
            }
            
            # Store in audit log
            await redis_client.lpush('security_audit_log', json.dumps(log_entry))
            
            # Trim log to retention period
            max_entries = self.config['audit_log_retention_days'] * 1000  # Rough estimate
            await redis_client.ltrim('security_audit_log', 0, max_entries - 1)
            
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    async def _schedule_key_deprecation(self, key_id: str, delay_seconds: int):
        """Schedule key deprecation after delay"""
        async def deprecate_later():
            await asyncio.sleep(delay_seconds)
            try:
                key_info = await self._get_api_key(key_id)
                if key_info and key_info.status == KeyStatus.ROTATING:
                    key_info.status = KeyStatus.DEPRECATED
                    await self._store_api_key(key_info)
                    logger.info(f"Deprecated API key {key_id} after rotation grace period")
            except Exception as e:
                logger.error(f"Failed to deprecate key {key_id}: {e}")
        
        asyncio.create_task(deprecate_later())
    
    async def _rotation_scheduler(self):
        """Background task to check for keys needing rotation"""
        while True:
            try:
                await asyncio.sleep(self.config['key_rotation_interval'])
                
                # Get all active keys using SCAN (non-blocking unlike KEYS)
                redis_client = self.redis_pool_manager.get_client()
                keys = []
                async for redis_key in redis_client.scan_iter(match="api_key:*", count=100):
                    keys.append(redis_key)

                rotation_threshold = datetime.now() - timedelta(days=30)  # Rotate monthly

                for redis_key in keys:
                    try:
                        key_data = await redis_client.get(redis_key)
                        if not key_data:
                            continue
                        
                        key_info_dict = json.loads(key_data)
                        
                        # Check if key needs rotation
                        if (key_info_dict.get('status') == 'active' and 
                            key_info_dict.get('created_at')):
                            
                            created_at = datetime.fromisoformat(key_info_dict['created_at'])
                            if created_at < rotation_threshold:
                                # Auto-rotate system keys
                                if key_info_dict.get('metadata', {}).get('user_id') == 'system':
                                    key_id = key_info_dict['key_id']
                                    logger.info(f"Auto-rotating system key {key_id}")
                                    await self.rotate_key(key_id)
                        
                    except Exception as e:
                        logger.error(f"Error checking key rotation for {redis_key}: {e}")
                        continue
                
            except Exception as e:
                logger.error(f"Error in rotation scheduler: {e}")
                await asyncio.sleep(60)  # Wait before retrying


# Global security manager instance
_security_manager: Optional[APISecurityManager] = None

async def get_security_manager() -> APISecurityManager:
    """Get global security manager instance"""
    global _security_manager
    
    if _security_manager is None:
        _security_manager = APISecurityManager()
        await _security_manager.initialize()
    
    return _security_manager

# Convenience functions
async def authenticate_api_key(
    api_key: str,
    ip_address: Optional[str] = None,
    endpoint: Optional[str] = None
) -> AuthResult:
    """Authenticate API key using global security manager"""
    manager = await get_security_manager()
    return await manager.authenticate(api_key, ip_address, endpoint)

async def create_system_api_key(name: str, access_level: AccessLevel) -> Tuple[str, str]:
    """Create system API key"""
    manager = await get_security_manager()
    return await manager.create_api_key(name, access_level, "system")
