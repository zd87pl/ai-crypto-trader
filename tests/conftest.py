"""
Pytest configuration and fixtures for AI Crypto Trader tests.

This module provides shared fixtures, mocks, and test utilities used across all tests.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from datetime import datetime
from typing import Dict, List, Any


# ============================================================================
# Async Event Loop Fixture
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Mock Binance Client Fixtures
# ============================================================================

@pytest.fixture
def mock_binance_client():
    """Create a mock Binance client with common methods."""
    client = MagicMock()

    # Mock get_symbol_ticker
    client.get_symbol_ticker.return_value = {'price': '50000.00'}

    # Mock get_exchange_info
    client.get_exchange_info.return_value = {
        'symbols': [
            {
                'symbol': 'BTCUSDC',
                'status': 'TRADING',
                'baseAsset': 'BTC',
                'quoteAsset': 'USDC',
                'filters': [
                    {'filterType': 'PRICE_FILTER', 'tickSize': '0.01', 'minPrice': '0.01', 'maxPrice': '1000000.00'},
                    {'filterType': 'LOT_SIZE', 'stepSize': '0.00001', 'minQty': '0.00001', 'maxQty': '9000.00'},
                    {'filterType': 'MIN_NOTIONAL', 'minNotional': '10.00'}
                ]
            },
            {
                'symbol': 'ETHUSDC',
                'status': 'TRADING',
                'baseAsset': 'ETH',
                'quoteAsset': 'USDC',
                'filters': [
                    {'filterType': 'PRICE_FILTER', 'tickSize': '0.01', 'minPrice': '0.01', 'maxPrice': '100000.00'},
                    {'filterType': 'LOT_SIZE', 'stepSize': '0.0001', 'minQty': '0.0001', 'maxQty': '9000.00'},
                    {'filterType': 'MIN_NOTIONAL', 'minNotional': '10.00'}
                ]
            }
        ]
    }

    # Mock get_historical_klines
    client.get_historical_klines.return_value = [
        # [open_time, open, high, low, close, volume, ...]
        [1704067200000, '48000.00', '49000.00', '47500.00', '48500.00', '1000.0', ...],
        [1704153600000, '48500.00', '50000.00', '48000.00', '49500.00', '1200.0', ...],
        [1704240000000, '49500.00', '51000.00', '49000.00', '50000.00', '1100.0', ...],
    ]

    # Mock create_order - successful fill
    client.create_order.return_value = {
        'symbol': 'BTCUSDC',
        'orderId': 12345,
        'status': 'FILLED',
        'executedQty': '0.001',
        'fills': [
            {'price': '50000.00', 'qty': '0.001', 'commission': '0.05'}
        ]
    }

    # Mock get_order
    client.get_order.return_value = {
        'symbol': 'BTCUSDC',
        'orderId': 12345,
        'status': 'NEW',
        'executedQty': '0.0'
    }

    # Mock get_account
    client.get_account.return_value = {
        'balances': [
            {'asset': 'USDC', 'free': '10000.00', 'locked': '0.00'},
            {'asset': 'BTC', 'free': '0.5', 'locked': '0.0'}
        ]
    }

    return client


@pytest.fixture
def mock_binance_client_order_rejected():
    """Create a mock Binance client that returns rejected orders."""
    client = MagicMock()
    client.get_symbol_ticker.return_value = {'price': '50000.00'}
    client.create_order.return_value = {
        'symbol': 'BTCUSDC',
        'orderId': 12345,
        'status': 'REJECTED',
        'executedQty': '0.0',
        'fills': []
    }
    return client


# ============================================================================
# Mock Redis Fixtures
# ============================================================================

@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis_mock = MagicMock()

    # Storage for simulating Redis
    storage: Dict[str, str] = {}

    def mock_set(key, value):
        storage[key] = value
        return True

    def mock_get(key):
        return storage.get(key)

    def mock_hget(name, key):
        return storage.get(f"{name}:{key}")

    def mock_hset(name, key, value):
        storage[f"{name}:{key}"] = value
        return True

    redis_mock.set = mock_set
    redis_mock.get = mock_get
    redis_mock.hget = mock_hget
    redis_mock.hset = mock_hset
    redis_mock.publish = MagicMock(return_value=1)
    redis_mock.lpush = MagicMock(return_value=1)
    redis_mock.ltrim = MagicMock(return_value=True)

    return redis_mock


# ============================================================================
# Trading Data Fixtures
# ============================================================================

@pytest.fixture
def sample_trade_setup():
    """Create a valid sample trade setup."""
    return {
        'symbol': 'BTCUSDC',
        'price': 50000.0,
        'position_size': 100.0,
        'stop_loss_pct': 5.0,
        'take_profit_pct': 10.0
    }


@pytest.fixture
def sample_grid_config():
    """Create a sample grid trading configuration."""
    return {
        'symbol': 'BTCUSDC',
        'current_price': 50000.0,
        'lower_boundary': 47500.0,
        'upper_boundary': 52500.0,
        'grid_levels': [47500.0, 48000.0, 48500.0, 49000.0, 49500.0,
                       50000.0, 50500.0, 51000.0, 51500.0, 52000.0, 52500.0],
        'quantity': 0.001,
        'price_precision': 2,
        'quantity_precision': 5,
        'last_price': 50000.0
    }


@pytest.fixture
def sample_dca_position():
    """Create a sample DCA position."""
    return {
        'symbol': 'BTCUSDC',
        'total_invested': 500.0,
        'total_quantity': 0.01,
        'average_price': 50000.0,
        'last_purchase_time': datetime.now().isoformat(),
        'last_purchase_price': 50000.0,
        'current_price': 52000.0,
        'current_value': 520.0,
        'profit_loss': 20.0,
        'profit_loss_pct': 4.0,
        'purchases': []
    }


@pytest.fixture
def sample_order_response_filled():
    """Create a sample filled order response."""
    return {
        'symbol': 'BTCUSDC',
        'orderId': 12345,
        'clientOrderId': 'test_order_1',
        'transactTime': 1704067200000,
        'price': '0.00',
        'origQty': '0.001',
        'executedQty': '0.001',
        'status': 'FILLED',
        'type': 'MARKET',
        'side': 'BUY',
        'fills': [
            {
                'price': '50000.00',
                'qty': '0.001',
                'commission': '0.00005',
                'commissionAsset': 'BTC'
            }
        ]
    }


@pytest.fixture
def sample_order_response_rejected():
    """Create a sample rejected order response."""
    return {
        'symbol': 'BTCUSDC',
        'orderId': 12346,
        'status': 'REJECTED',
        'executedQty': '0.0',
        'fills': []
    }


@pytest.fixture
def sample_order_response_empty_fills():
    """Create a sample order with empty fills."""
    return {
        'symbol': 'BTCUSDC',
        'orderId': 12347,
        'status': 'FILLED',
        'executedQty': '0.001',
        'fills': []
    }


# ============================================================================
# Config Fixtures
# ============================================================================

@pytest.fixture
def sample_config():
    """Create a sample trading configuration."""
    return {
        'trading_params': {
            'min_volume_usdc': 50000,
            'position_size': 0.4,
            'max_positions': 5,
            'ai_confidence_threshold': 0.7,
            'min_trade_amount': 10,
            'ai_analysis_interval': 300
        },
        'risk_management': {
            'max_portfolio_var': 0.05,
            'max_drawdown_limit': 0.15,
            'adaptive_stop_loss_enabled': True,
            'trailing_stop_loss_enabled': True
        },
        'grid_trading': {
            'enabled': True,
            'symbols': ['BTCUSDC', 'ETHUSDC'],
            'num_grids': 10,
            'grid_type': 'arithmetic',
            'simulation_mode': True
        },
        'dca_strategy': {
            'enabled': True,
            'symbols': ['BTCUSDC', 'ETHUSDC'],
            'base_order_size_usdc': 100,
            'simulation_mode': True
        }
    }


# ============================================================================
# Symbol Info Fixtures
# ============================================================================

@pytest.fixture
def sample_symbol_info():
    """Create sample symbol trading rules."""
    return {
        'BTCUSDC': {
            'baseAsset': 'BTC',
            'quoteAsset': 'USDC',
            'status': 'TRADING',
            'min_price': 0.01,
            'max_price': 1000000.0,
            'tick_size': 0.01,
            'min_qty': 0.00001,
            'max_qty': 9000.0,
            'step_size': 0.00001,
            'min_notional': 10.0
        },
        'ETHUSDC': {
            'baseAsset': 'ETH',
            'quoteAsset': 'USDC',
            'status': 'TRADING',
            'min_price': 0.01,
            'max_price': 100000.0,
            'tick_size': 0.01,
            'min_qty': 0.0001,
            'max_qty': 9000.0,
            'step_size': 0.0001,
            'min_notional': 10.0
        }
    }


# ============================================================================
# Market Data Fixtures
# ============================================================================

@pytest.fixture
def sample_market_data():
    """Create sample market data."""
    return {
        'BTCUSDC': {
            'price': 50000.0,
            'volume': 1000000.0,
            'price_change': 2.5,
            'timestamp': datetime.now()
        },
        'ETHUSDC': {
            'price': 3000.0,
            'volume': 500000.0,
            'price_change': 1.8,
            'timestamp': datetime.now()
        }
    }


@pytest.fixture
def sample_klines():
    """Create sample historical klines data."""
    return [
        [1704067200000, '48000.00', '49000.00', '47500.00', '48500.00', '1000.0', 1704153599999, '48500000.00', 5000, '500.0', '24250000.00', '0'],
        [1704153600000, '48500.00', '50000.00', '48000.00', '49500.00', '1200.0', 1704239999999, '59400000.00', 6000, '600.0', '29700000.00', '0'],
        [1704240000000, '49500.00', '51000.00', '49000.00', '50000.00', '1100.0', 1704326399999, '55000000.00', 5500, '550.0', '27500000.00', '0'],
        [1704326400000, '50000.00', '52000.00', '49500.00', '51500.00', '1300.0', 1704412799999, '66950000.00', 6500, '650.0', '33475000.00', '0'],
        [1704412800000, '51500.00', '53000.00', '51000.00', '52500.00', '1400.0', 1704499199999, '73500000.00', 7000, '700.0', '36750000.00', '0'],
    ]


# ============================================================================
# Utility Functions
# ============================================================================

def create_mock_ticker(symbol: str, price: float) -> Dict[str, str]:
    """Create a mock ticker response."""
    return {'symbol': symbol, 'price': str(price)}


def create_mock_order(
    symbol: str,
    order_id: int,
    status: str = 'FILLED',
    qty: str = '0.001',
    price: str = '50000.00'
) -> Dict[str, Any]:
    """Create a mock order response."""
    fills = [{'price': price, 'qty': qty, 'commission': '0.00005'}] if status == 'FILLED' else []
    return {
        'symbol': symbol,
        'orderId': order_id,
        'status': status,
        'executedQty': qty if status == 'FILLED' else '0.0',
        'fills': fills
    }
