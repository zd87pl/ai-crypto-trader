import os
import time
import json
import logging
import threading
from datetime import datetime
from functools import wraps
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional, List, Callable, Union

# Prometheus metrics
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server

# Structured logging
import structlog
from pythonjsonlogger import jsonlogger

# Constants
DEFAULT_METRICS_PORT = 8000
METRICS_NAMESPACE = "crypto_trader"

# Globals for metrics
enabled_metrics = os.environ.get("ENABLE_METRICS", "false").lower() == "true"
metrics = {}
metrics_port = None
metrics_started = False

# Setup structured logging
def setup_logging(service_name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Configure structured JSON logging for the service
    
    Args:
        service_name: Name of the service for log context
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    log_level_num = getattr(logging, log_level.upper(), logging.INFO)
    
    # Ensure logs directory exists
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create standard file handler with rotation
    log_file = os.path.join(logs_dir, f"{service_name.lower()}.log")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
    )
    
    # Create JSON formatter
    class CustomJsonFormatter(jsonlogger.JsonFormatter):
        def add_fields(self, log_record, record, message_dict):
            super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
            log_record["service"] = service_name
            log_record["timestamp"] = datetime.utcnow().isoformat() + "Z"
            log_record["level"] = record.levelname
            log_record["thread"] = threading.current_thread().name

    formatter = CustomJsonFormatter("%(timestamp)s %(level)s %(service)s %(name)s %(message)s")
    file_handler.setFormatter(formatter)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_num)
    
    # Add console handler for local development
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Create service-specific logger
    logger = structlog.get_logger(service_name)
    
    return logger

# Metrics functions
def start_metrics_server(port: Optional[int] = None) -> None:
    """
    Start the Prometheus metrics HTTP server on the specified port
    
    Args:
        port: Port to run the metrics server on (default based on service port)
    """
    global metrics_started, metrics_port
    
    if not enabled_metrics:
        return
        
    if metrics_started:
        return
    
    if port is None:
        # Default to service port + 1000 if not specified
        service_port = int(os.environ.get("SERVICE_PORT", DEFAULT_METRICS_PORT))
        port = service_port
    
    metrics_port = port
    start_http_server(port)
    metrics_started = True

def create_counter(name: str, description: str, labels: Optional[List[str]] = None) -> Counter:
    """
    Create a Prometheus counter metric
    
    Args:
        name: Metric name
        description: Metric description 
        labels: List of label names for this metric
        
    Returns:
        Prometheus Counter instance
    """
    if not enabled_metrics:
        return DummyMetric()
        
    key = f"counter_{name}"
    if key not in metrics:
        metrics[key] = Counter(
            f"{METRICS_NAMESPACE}_{name}",
            description,
            labels or []
        )
    return metrics[key]

def create_gauge(name: str, description: str, labels: Optional[List[str]] = None) -> Gauge:
    """
    Create a Prometheus gauge metric
    
    Args:
        name: Metric name
        description: Metric description 
        labels: List of label names for this metric
        
    Returns:
        Prometheus Gauge instance
    """
    if not enabled_metrics:
        return DummyMetric()
        
    key = f"gauge_{name}"
    if key not in metrics:
        metrics[key] = Gauge(
            f"{METRICS_NAMESPACE}_{name}",
            description,
            labels or []
        )
    return metrics[key]

def create_histogram(name: str, description: str, labels: Optional[List[str]] = None, 
                  buckets: Optional[List[float]] = None) -> Histogram:
    """
    Create a Prometheus histogram metric
    
    Args:
        name: Metric name
        description: Metric description 
        labels: List of label names for this metric
        buckets: Custom buckets for the histogram
        
    Returns:
        Prometheus Histogram instance
    """
    if not enabled_metrics:
        return DummyMetric()
        
    key = f"histogram_{name}"
    if key not in metrics:
        metrics[key] = Histogram(
            f"{METRICS_NAMESPACE}_{name}",
            description,
            labels or [],
            buckets=buckets
        )
    return metrics[key]

def create_summary(name: str, description: str, labels: Optional[List[str]] = None) -> Summary:
    """
    Create a Prometheus summary metric
    
    Args:
        name: Metric name
        description: Metric description 
        labels: List of label names for this metric
        
    Returns:
        Prometheus Summary instance
    """
    if not enabled_metrics:
        return DummyMetric()
        
    key = f"summary_{name}"
    if key not in metrics:
        metrics[key] = Summary(
            f"{METRICS_NAMESPACE}_{name}",
            description,
            labels or []
        )
    return metrics[key]

# Dummy metric class for when metrics are disabled
class DummyMetric:
    """Dummy metric class that implements no-op methods for all metric types"""
    def inc(self, amount=1):
        pass
        
    def dec(self, amount=1):
        pass
        
    def set(self, value):
        pass
        
    def observe(self, value):
        pass
        
    def time(self):
        return DummyTimer()
        
    def labels(self, *args, **kwargs):
        return self

class DummyTimer:
    """Dummy timer context manager"""
    def __enter__(self):
        pass
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Decorator for timing functions and capturing errors
def timed(metric_name=None, description=None, labels=None):
    """
    Decorator that times function execution and records it in a histogram
    Also tracks successes and failures as counters
    
    Args:
        metric_name: Name for the metric (defaults to function name)
        description: Description for the metric
        labels: Labels to apply to the metric
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = metric_name or func.__name__
            desc = description or f"Time spent in {name}"
            
            if enabled_metrics:
                timer = create_histogram(
                    f"{name}_duration_seconds",
                    desc,
                    labels
                )
                error_counter = create_counter(
                    f"{name}_errors_total",
                    f"Errors in {name}",
                    labels
                )
                success_counter = create_counter(
                    f"{name}_success_total",
                    f"Successful calls to {name}",
                    labels
                )
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                if enabled_metrics:
                    success_counter.inc()
                return result
            except Exception as e:
                if enabled_metrics:
                    error_counter.inc()
                raise
            finally:
                if enabled_metrics:
                    duration = time.time() - start_time
                    timer.observe(duration)
                
        return wrapper
    return decorator

# Common metrics for all services
request_latency = create_histogram(
    "request_latency_seconds", 
    "Request latency in seconds",
    ["service", "endpoint"]
)

request_count = create_counter(
    "request_total",
    "Total number of requests",
    ["service", "endpoint", "status"]
)

active_connections = create_gauge(
    "active_connections",
    "Number of active connections",
    ["service"]
)

error_count = create_counter(
    "errors_total",
    "Total number of errors",
    ["service", "type"]
)

# Initialize global metrics for each service type
def init_service_metrics(service_name):
    """Initialize service-specific metrics"""
    if not enabled_metrics:
        return
        
    if service_name == "market_monitor":
        # Market monitor specific metrics
        create_gauge(
            "market_data_age_seconds",
            "Age of latest market data in seconds",
            ["symbol", "timeframe"]
        )
        create_counter(
            "market_updates_total",
            "Total number of market updates processed",
            ["symbol", "timeframe"]
        )
        create_gauge(
            "technical_indicator_value",
            "Current value of technical indicators",
            ["symbol", "indicator", "timeframe"]
        )
        
    elif service_name == "social_monitor":
        # Social monitor specific metrics
        create_gauge(
            "social_sentiment",
            "Social sentiment score",
            ["symbol"]
        )
        create_gauge(
            "social_volume",
            "Social volume",
            ["symbol"]
        )
        create_gauge(
            "social_contributors",
            "Number of social contributors",
            ["symbol"]
        )
        create_counter(
            "social_updates_total",
            "Total number of social updates processed",
            ["symbol"]
        )
        
    elif service_name == "ai_analyzer":
        # AI analyzer specific metrics
        create_counter(
            "ai_requests_total",
            "Total number of AI API requests",
            ["model"]
        )
        create_histogram(
            "ai_request_duration_seconds",
            "Duration of AI API requests",
            ["model"]
        )
        create_counter(
            "trading_signals_total",
            "Total number of trading signals generated",
            ["symbol", "action", "confidence_level"]
        )
        create_gauge(
            "ai_model_confidence",
            "Confidence level of AI model predictions",
            ["symbol", "action"]
        )
        
    elif service_name == "trade_executor":
        # Trade executor specific metrics
        create_counter(
            "trades_executed_total",
            "Total number of trades executed",
            ["symbol", "action"]
        )
        create_gauge(
            "portfolio_value_usd",
            "Total portfolio value in USD",
        )
        create_gauge(
            "asset_holdings",
            "Current holdings of assets",
            ["symbol"]
        )
        create_counter(
            "execution_errors_total",
            "Total number of execution errors",
            ["symbol", "error_type"]
        )
        create_histogram(
            "execution_latency_seconds",
            "Trade execution latency",
            ["symbol", "action"]
        )
        
    elif service_name == "strategy_evolution":
        # Strategy evolution specific metrics
        create_counter(
            "strategies_generated_total",
            "Total number of strategies generated",
            ["status"]
        )
        create_gauge(
            "strategy_performance",
            "Performance metrics for strategies",
            ["strategy_id", "metric"]
        )
        create_counter(
            "strategy_mutations_total",
            "Total number of strategy mutations",
            ["result"]
        )
        create_histogram(
            "strategy_backtest_duration_seconds",
            "Duration of strategy backtests",
            ["strategy_id"]
        )
    
    elif service_name == "dashboard":
        # Dashboard specific metrics
        create_counter(
            "dashboard_views_total",
            "Total number of dashboard views",
            ["endpoint"]
        )
        create_gauge(
            "active_users",
            "Number of active dashboard users",
        )
        create_histogram(
            "dashboard_render_time_seconds",
            "Time to render dashboard components",
            ["component"]
        )