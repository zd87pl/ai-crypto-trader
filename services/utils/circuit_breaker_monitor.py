"""
Circuit Breaker Monitoring Service
Provides real-time visibility into circuit breaker states and statistics
"""
import json
import asyncio
from aiohttp import web
from services.utils.circuit_breaker import get_all_circuit_breakers
from services.utils.metrics import get_metrics, is_metrics_enabled
import logging

logger = logging.getLogger(__name__)

class CircuitBreakerMonitor:
    """Monitor and expose circuit breaker statistics"""
    
    def __init__(self, port: int = 9091):
        self.port = port
        self.app = web.Application()
        self.setup_routes()
        
        # Initialize metrics if enabled
        self.metrics = None
        if is_metrics_enabled():
            self.metrics = get_metrics('circuit_breaker_monitor', port)
    
    def setup_routes(self):
        """Setup HTTP routes for monitoring"""
        self.app.router.add_get('/circuit-breakers', self.get_all_circuit_breakers)
        self.app.router.add_get('/circuit-breakers/{name}', self.get_circuit_breaker)
        self.app.router.add_post('/circuit-breakers/{name}/reset', self.reset_circuit_breaker)
        self.app.router.add_get('/health', self.health_check)
    
    async def get_all_circuit_breakers(self, request):
        """Get statistics for all circuit breakers"""
        try:
            circuit_breakers = get_all_circuit_breakers()
            stats = {}
            
            for name, cb in circuit_breakers.items():
                stats[name] = cb.get_stats()
                
                # Update metrics if enabled
                if self.metrics:
                    # Circuit breaker state (0=closed, 1=half-open, 2=open)
                    state_value = {'CLOSED': 0, 'HALF_OPEN': 1, 'OPEN': 2}[cb.state.value]
                    self.metrics.request_counter.labels(
                        service='circuit_breaker_monitor',
                        endpoint='get_all_circuit_breakers',
                        method='GET'
                    ).inc()
            
            return web.json_response({
                'circuit_breakers': stats,
                'total_count': len(stats),
                'timestamp': asyncio.get_event_loop().time()
            })
        except Exception as e:
            logger.error(f"Error getting circuit breaker stats: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def get_circuit_breaker(self, request):
        """Get statistics for a specific circuit breaker"""
        try:
            name = request.match_info['name']
            circuit_breakers = get_all_circuit_breakers()
            
            if name not in circuit_breakers:
                return web.json_response({'error': f'Circuit breaker {name} not found'}, status=404)
            
            cb = circuit_breakers[name]
            stats = cb.get_stats()
            
            if self.metrics:
                self.metrics.request_counter.labels(
                    service='circuit_breaker_monitor',
                    endpoint='get_circuit_breaker',
                    method='GET'
                ).inc()
            
            return web.json_response(stats)
        except Exception as e:
            logger.error(f"Error getting circuit breaker {name}: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def reset_circuit_breaker(self, request):
        """Reset a specific circuit breaker"""
        try:
            name = request.match_info['name']
            circuit_breakers = get_all_circuit_breakers()
            
            if name not in circuit_breakers:
                return web.json_response({'error': f'Circuit breaker {name} not found'}, status=404)
            
            cb = circuit_breakers[name]
            cb.reset()
            
            if self.metrics:
                self.metrics.request_counter.labels(
                    service='circuit_breaker_monitor',
                    endpoint='reset_circuit_breaker',
                    method='POST'
                ).inc()
            
            logger.info(f"Circuit breaker {name} has been reset")
            return web.json_response({'message': f'Circuit breaker {name} reset successfully'})
        except Exception as e:
            logger.error(f"Error resetting circuit breaker {name}: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def health_check(self, request):
        """Health check endpoint"""
        return web.json_response({'status': 'healthy'})
    
    async def start(self):
        """Start the monitoring server"""
        try:
            # Start metrics server if enabled
            if self.metrics:
                await self.metrics.start_server()
            
            runner = web.AppRunner(self.app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', self.port)
            await site.start()
            
            logger.info(f"Circuit Breaker Monitor started on port {self.port}")
            logger.info(f"Available endpoints:")
            logger.info(f"  GET  /circuit-breakers - Get all circuit breaker stats")
            logger.info(f"  GET  /circuit-breakers/{{name}} - Get specific circuit breaker stats")
            logger.info(f"  POST /circuit-breakers/{{name}}/reset - Reset circuit breaker")
            logger.info(f"  GET  /health - Health check")
            
        except Exception as e:
            logger.error(f"Failed to start Circuit Breaker Monitor: {e}")
            raise

if __name__ == "__main__":
    async def main():
        monitor = CircuitBreakerMonitor()
        await monitor.start()
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Circuit Breaker Monitor stopped")
    
    asyncio.run(main())
