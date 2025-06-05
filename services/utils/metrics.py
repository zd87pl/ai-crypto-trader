"""
Prometheus metrics utility for AI Crypto Trader services
"""
import time
import logging
from functools import wraps
from typing import Dict, Optional
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, start_http_server, generate_latest
import asyncio
from aiohttp import web
import os

logger = logging.getLogger(__name__)

class PrometheusMetrics:
    """Centralized Prometheus metrics for crypto trading services"""
    
    def __init__(self, service_name: str, port: Optional[int] = None):
        self.service_name = service_name
        self.port = port
        self.registry = CollectorRegistry()
        self.app = None
        self.server = None
        
        # Common metrics for all services
        self.request_counter = Counter(
            'crypto_trader_request_total',
            'Total number of requests processed by service',
            ['service', 'endpoint', 'method'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'crypto_trader_request_latency_seconds',
            'Request latency in seconds',
            ['service', 'endpoint', 'method'],
            registry=self.registry
        )
        
        self.error_counter = Counter(
            'crypto_trader_errors_total',
            'Total number of errors by service',
            ['service', 'error_type', 'endpoint'],
            registry=self.registry
        )
        
        self.service_health = Gauge(
            'crypto_trader_service_health',
            'Service health status (1=healthy, 0=unhealthy)',
            ['service'],
            registry=self.registry
        )
        
        # Trading-specific metrics
        self.trading_signals = Counter(
            'crypto_trader_trading_signals_total',
            'Total trading signals generated',
            ['symbol', 'action', 'strategy'],
            registry=self.registry
        )
        
        self.trades_executed = Counter(
            'crypto_trader_trades_executed_total',
            'Total trades executed',
            ['symbol', 'action', 'strategy'],
            registry=self.registry
        )
        
        self.portfolio_value = Gauge(
            'crypto_trader_portfolio_value_usd',
            'Current portfolio value in USD',
            registry=self.registry
        )
        
        self.asset_holdings = Gauge(
            'crypto_trader_asset_holdings',
            'Current asset holdings',
            ['symbol'],
            registry=self.registry
        )
        
        # AI-specific metrics
        self.ai_requests = Counter(
            'crypto_trader_ai_requests_total',
            'Total AI requests',
            ['model', 'service'],
            registry=self.registry
        )
        
        self.ai_request_duration = Histogram(
            'crypto_trader_ai_request_duration_seconds',
            'AI request duration',
            ['model', 'service'],
            registry=self.registry
        )
        
        self.ai_model_confidence = Gauge(
            'crypto_trader_ai_model_confidence',
            'AI model confidence scores',
            ['symbol', 'action', 'model'],
            registry=self.registry
        )
        
        # Social metrics
        self.social_sentiment = Gauge(
            'crypto_trader_social_sentiment',
            'Social sentiment scores',
            ['symbol'],
            registry=self.registry
        )
        
        self.social_volume = Gauge(
            'crypto_trader_social_volume',
            'Social media volume',
            ['symbol'],
            registry=self.registry
        )
        
        # Risk metrics
        self.portfolio_var = Gauge(
            'crypto_trader_portfolio_var',
            'Portfolio Value at Risk',
            registry=self.registry
        )
        
        self.position_risk = Gauge(
            'crypto_trader_position_risk',
            'Individual position risk scores',
            ['symbol'],
            registry=self.registry
        )
        
        # Execution metrics
        self.execution_errors = Counter(
            'crypto_trader_execution_errors_total',
            'Trading execution errors',
            ['symbol', 'error_type'],
            registry=self.registry
        )
        
        self.order_latency = Histogram(
            'crypto_trader_order_latency_seconds',
            'Order execution latency',
            ['symbol', 'order_type'],
            registry=self.registry
        )
        
        # Market data metrics
        self.market_data_updates = Counter(
            'crypto_trader_market_data_updates_total',
            'Market data updates received',
            ['symbol', 'source'],
            registry=self.registry
        )
        
        self.price_changes = Gauge(
            'crypto_trader_price_change_percent',
            'Price change percentages',
            ['symbol', 'timeframe'],
            registry=self.registry
        )
        
        # Performance metrics
        self.win_rate = Gauge(
            'crypto_trader_win_rate',
            'Trading win rate',
            ['symbol', 'strategy'],
            registry=self.registry
        )
        
        self.profit_loss = Gauge(
            'crypto_trader_profit_loss_usd',
            'Profit/Loss in USD',
            ['symbol', 'strategy'],
            registry=self.registry
        )
        
        # Strategy metrics
        self.strategy_performance = Gauge(
            'crypto_trader_strategy_performance',
            'Strategy performance scores',
            ['strategy_id', 'metric'],
            registry=self.registry
        )
        
        # Initialize service health
        self.service_health.labels(service=self.service_name).set(1)
        
    async def start_server(self, port: Optional[int] = None):
        """Start the metrics HTTP server"""
        if not port:
            port = self.port or 9090
            
        try:
            self.app = web.Application()
            self.app.router.add_get('/metrics', self._metrics_handler)
            self.app.router.add_get('/health', self._health_handler)
            
            runner = web.AppRunner(self.app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', port)
            await site.start()
            
            logger.info(f"Metrics server started on port {port} for service {self.service_name}")
            
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            
    async def _metrics_handler(self, request):
        """Handle metrics endpoint requests"""
        try:
            metrics_data = generate_latest(self.registry)
            return web.Response(text=metrics_data.decode('utf-8'), content_type='text/plain')
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            return web.Response(text="Error generating metrics", status=500)
            
    async def _health_handler(self, request):
        """Handle health check requests"""
        return web.Response(text="OK", status=200)
        
    def measure_time(self, endpoint: str, method: str = "GET"):
        """Decorator to measure request duration"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    self.request_counter.labels(
                        service=self.service_name,
                        endpoint=endpoint,
                        method=method
                    ).inc()
                    
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    self.error_counter.labels(
                        service=self.service_name,
                        error_type=type(e).__name__,
                        endpoint=endpoint
                    ).inc()
                    raise
                finally:
                    duration = time.time() - start_time
                    self.request_duration.labels(
                        service=self.service_name,
                        endpoint=endpoint,
                        method=method
                    ).observe(duration)
                    
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    self.request_counter.labels(
                        service=self.service_name,
                        endpoint=endpoint,
                        method=method
                    ).inc()
                    
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    self.error_counter.labels(
                        service=self.service_name,
                        error_type=type(e).__name__,
                        endpoint=endpoint
                    ).inc()
                    raise
                finally:
                    duration = time.time() - start_time
                    self.request_duration.labels(
                        service=self.service_name,
                        endpoint=endpoint,
                        method=method
                    ).observe(duration)
                    
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
        
    def record_trading_signal(self, symbol: str, action: str, strategy: str = "default"):
        """Record a trading signal"""
        self.trading_signals.labels(
            symbol=symbol,
            action=action,
            strategy=strategy
        ).inc()
        
    def record_trade_execution(self, symbol: str, action: str, strategy: str = "default"):
        """Record a trade execution"""
        self.trades_executed.labels(
            symbol=symbol,
            action=action,
            strategy=strategy
        ).inc()
        
    def update_portfolio_value(self, value_usd: float):
        """Update portfolio value"""
        self.portfolio_value.set(value_usd)
        
    def update_asset_holding(self, symbol: str, quantity: float):
        """Update asset holding"""
        self.asset_holdings.labels(symbol=symbol).set(quantity)
        
    def record_ai_request(self, model: str, duration: float):
        """Record AI request"""
        self.ai_requests.labels(model=model, service=self.service_name).inc()
        self.ai_request_duration.labels(model=model, service=self.service_name).observe(duration)
        
    def update_ai_confidence(self, symbol: str, action: str, model: str, confidence: float):
        """Update AI model confidence"""
        self.ai_model_confidence.labels(
            symbol=symbol,
            action=action,
            model=model
        ).set(confidence)
        
    def update_social_metrics(self, symbol: str, sentiment: float, volume: float):
        """Update social metrics"""
        self.social_sentiment.labels(symbol=symbol).set(sentiment)
        self.social_volume.labels(symbol=symbol).set(volume)
        
    def update_portfolio_var(self, var_value: float):
        """Update portfolio VaR"""
        self.portfolio_var.set(var_value)
        
    def update_position_risk(self, symbol: str, risk_score: float):
        """Update position risk"""
        self.position_risk.labels(symbol=symbol).set(risk_score)
        
    def record_execution_error(self, symbol: str, error_type: str):
        """Record execution error"""
        self.execution_errors.labels(symbol=symbol, error_type=error_type).inc()
        
    def record_order_latency(self, symbol: str, order_type: str, latency: float):
        """Record order execution latency"""
        self.order_latency.labels(symbol=symbol, order_type=order_type).observe(latency)
        
    def record_market_data_update(self, symbol: str, source: str = "binance"):
        """Record market data update"""
        self.market_data_updates.labels(symbol=symbol, source=source).inc()
        
    def update_price_change(self, symbol: str, timeframe: str, change_pct: float):
        """Update price change percentage"""
        self.price_changes.labels(symbol=symbol, timeframe=timeframe).set(change_pct)
        
    def update_win_rate(self, symbol: str, strategy: str, win_rate: float):
        """Update win rate"""
        self.win_rate.labels(symbol=symbol, strategy=strategy).set(win_rate)
        
    def update_profit_loss(self, symbol: str, strategy: str, pnl_usd: float):
        """Update profit/loss"""
        self.profit_loss.labels(symbol=symbol, strategy=strategy).set(pnl_usd)
        
    def update_strategy_performance(self, strategy_id: str, metric: str, value: float):
        """Update strategy performance metrics"""
        self.strategy_performance.labels(strategy_id=strategy_id, metric=metric).set(value)
        
    def set_service_health(self, healthy: bool):
        """Set service health status"""
        self.service_health.labels(service=self.service_name).set(1 if healthy else 0)


# Global metrics instances for easy access
_metrics_instances: Dict[str, PrometheusMetrics] = {}

def get_metrics(service_name: str, port: Optional[int] = None) -> PrometheusMetrics:
    """Get or create metrics instance for a service"""
    if service_name not in _metrics_instances:
        _metrics_instances[service_name] = PrometheusMetrics(service_name, port)
    return _metrics_instances[service_name]

def is_metrics_enabled() -> bool:
    """Check if metrics are enabled via environment variable"""
    return os.getenv('ENABLE_METRICS', 'false').lower() in ('true', '1', 'yes', 'on')
