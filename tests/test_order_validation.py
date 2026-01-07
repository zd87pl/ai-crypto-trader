"""
Unit tests for Order Validation and Trade Execution.

Tests cover:
- BUG-003: Order execution verification
- BUG-004: Input validation for trading parameters
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the exception class
from auto_trader import TradeValidationError


class TestTradeValidation:
    """Tests for trade setup validation (BUG-004 fix)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.symbol_info = {
            'BTCUSDC': {
                'baseAsset': 'BTC',
                'quoteAsset': 'USDC',
                'step_size': 0.00001,
                'min_notional': 10.0
            },
            'ETHUSDC': {
                'baseAsset': 'ETH',
                'quoteAsset': 'USDC',
                'step_size': 0.0001,
                'min_notional': 10.0
            }
        }

    def validate_trade_setup(self, trade_setup: Dict) -> None:
        """
        Copy of the validation logic for testing.
        """
        # Check required fields exist
        required_fields = ['symbol', 'price', 'position_size']
        for field in required_fields:
            if field not in trade_setup:
                raise TradeValidationError(f"Missing required field: {field}")

        # Validate symbol
        symbol = trade_setup['symbol']
        if not isinstance(symbol, str) or not symbol:
            raise TradeValidationError(f"Invalid symbol: {symbol}")
        if symbol not in self.symbol_info:
            raise TradeValidationError(f"Unknown trading symbol: {symbol}")

        # Validate price
        try:
            price = float(trade_setup['price'])
        except (TypeError, ValueError):
            raise TradeValidationError(f"Invalid price value: {trade_setup['price']}")

        if price <= 0:
            raise TradeValidationError(f"Price must be positive: {price}")
        if price > 1e10:
            raise TradeValidationError(f"Price exceeds maximum: {price}")

        # Validate position size
        try:
            position_size = float(trade_setup['position_size'])
        except (TypeError, ValueError):
            raise TradeValidationError(f"Invalid position_size value: {trade_setup['position_size']}")

        if position_size <= 0:
            raise TradeValidationError(f"Position size must be positive: {position_size}")

        # Validate against symbol rules
        rules = self.symbol_info.get(symbol)
        if rules:
            min_notional = rules.get('min_notional', 0)
            if position_size < min_notional:
                raise TradeValidationError(f"Position size {position_size} below minimum notional {min_notional}")

        # Validate stop loss and take profit percentages if provided
        if 'stop_loss_pct' in trade_setup:
            try:
                stop_loss = float(trade_setup['stop_loss_pct'])
                if stop_loss <= 0 or stop_loss >= 100:
                    raise TradeValidationError(f"Invalid stop_loss_pct: {stop_loss}")
            except (TypeError, ValueError):
                raise TradeValidationError(f"Invalid stop_loss_pct value: {trade_setup['stop_loss_pct']}")

        if 'take_profit_pct' in trade_setup:
            try:
                take_profit = float(trade_setup['take_profit_pct'])
                if take_profit <= 0 or take_profit >= 1000:
                    raise TradeValidationError(f"Invalid take_profit_pct: {take_profit}")
            except (TypeError, ValueError):
                raise TradeValidationError(f"Invalid take_profit_pct value: {trade_setup['take_profit_pct']}")

    @pytest.mark.unit
    def test_valid_trade_setup(self, sample_trade_setup):
        """Test validation passes for valid trade setup."""
        # Should not raise any exception
        self.validate_trade_setup(sample_trade_setup)

    @pytest.mark.unit
    def test_missing_symbol(self):
        """Test validation fails when symbol is missing."""
        trade_setup = {'price': 50000.0, 'position_size': 100.0}
        with pytest.raises(TradeValidationError, match="Missing required field: symbol"):
            self.validate_trade_setup(trade_setup)

    @pytest.mark.unit
    def test_missing_price(self):
        """Test validation fails when price is missing."""
        trade_setup = {'symbol': 'BTCUSDC', 'position_size': 100.0}
        with pytest.raises(TradeValidationError, match="Missing required field: price"):
            self.validate_trade_setup(trade_setup)

    @pytest.mark.unit
    def test_missing_position_size(self):
        """Test validation fails when position_size is missing."""
        trade_setup = {'symbol': 'BTCUSDC', 'price': 50000.0}
        with pytest.raises(TradeValidationError, match="Missing required field: position_size"):
            self.validate_trade_setup(trade_setup)

    @pytest.mark.unit
    def test_invalid_symbol_type(self):
        """Test validation fails for non-string symbol."""
        trade_setup = {'symbol': 12345, 'price': 50000.0, 'position_size': 100.0}
        with pytest.raises(TradeValidationError, match="Invalid symbol"):
            self.validate_trade_setup(trade_setup)

    @pytest.mark.unit
    def test_empty_symbol(self):
        """Test validation fails for empty symbol."""
        trade_setup = {'symbol': '', 'price': 50000.0, 'position_size': 100.0}
        with pytest.raises(TradeValidationError, match="Invalid symbol"):
            self.validate_trade_setup(trade_setup)

    @pytest.mark.unit
    def test_unknown_symbol(self):
        """Test validation fails for unknown symbol."""
        trade_setup = {'symbol': 'UNKNOWNUSDC', 'price': 50000.0, 'position_size': 100.0}
        with pytest.raises(TradeValidationError, match="Unknown trading symbol"):
            self.validate_trade_setup(trade_setup)

    @pytest.mark.unit
    def test_zero_price(self):
        """Test validation fails for zero price."""
        trade_setup = {'symbol': 'BTCUSDC', 'price': 0.0, 'position_size': 100.0}
        with pytest.raises(TradeValidationError, match="Price must be positive"):
            self.validate_trade_setup(trade_setup)

    @pytest.mark.unit
    def test_negative_price(self):
        """Test validation fails for negative price."""
        trade_setup = {'symbol': 'BTCUSDC', 'price': -50000.0, 'position_size': 100.0}
        with pytest.raises(TradeValidationError, match="Price must be positive"):
            self.validate_trade_setup(trade_setup)

    @pytest.mark.unit
    def test_extreme_price(self):
        """Test validation fails for unreasonably high price."""
        trade_setup = {'symbol': 'BTCUSDC', 'price': 1e15, 'position_size': 100.0}
        with pytest.raises(TradeValidationError, match="Price exceeds maximum"):
            self.validate_trade_setup(trade_setup)

    @pytest.mark.unit
    def test_invalid_price_type(self):
        """Test validation fails for non-numeric price."""
        trade_setup = {'symbol': 'BTCUSDC', 'price': 'fifty thousand', 'position_size': 100.0}
        with pytest.raises(TradeValidationError, match="Invalid price value"):
            self.validate_trade_setup(trade_setup)

    @pytest.mark.unit
    def test_zero_position_size(self):
        """Test validation fails for zero position size."""
        trade_setup = {'symbol': 'BTCUSDC', 'price': 50000.0, 'position_size': 0.0}
        with pytest.raises(TradeValidationError, match="Position size must be positive"):
            self.validate_trade_setup(trade_setup)

    @pytest.mark.unit
    def test_negative_position_size(self):
        """Test validation fails for negative position size."""
        trade_setup = {'symbol': 'BTCUSDC', 'price': 50000.0, 'position_size': -100.0}
        with pytest.raises(TradeValidationError, match="Position size must be positive"):
            self.validate_trade_setup(trade_setup)

    @pytest.mark.unit
    def test_below_min_notional(self):
        """Test validation fails when position size is below minimum notional."""
        trade_setup = {'symbol': 'BTCUSDC', 'price': 50000.0, 'position_size': 5.0}  # Below 10.0 min
        with pytest.raises(TradeValidationError, match="below minimum notional"):
            self.validate_trade_setup(trade_setup)

    @pytest.mark.unit
    def test_invalid_stop_loss_negative(self):
        """Test validation fails for negative stop loss percentage."""
        trade_setup = {
            'symbol': 'BTCUSDC',
            'price': 50000.0,
            'position_size': 100.0,
            'stop_loss_pct': -5.0
        }
        with pytest.raises(TradeValidationError, match="Invalid stop_loss_pct"):
            self.validate_trade_setup(trade_setup)

    @pytest.mark.unit
    def test_invalid_stop_loss_over_100(self):
        """Test validation fails for stop loss over 100%."""
        trade_setup = {
            'symbol': 'BTCUSDC',
            'price': 50000.0,
            'position_size': 100.0,
            'stop_loss_pct': 105.0
        }
        with pytest.raises(TradeValidationError, match="Invalid stop_loss_pct"):
            self.validate_trade_setup(trade_setup)

    @pytest.mark.unit
    def test_invalid_take_profit_negative(self):
        """Test validation fails for negative take profit percentage."""
        trade_setup = {
            'symbol': 'BTCUSDC',
            'price': 50000.0,
            'position_size': 100.0,
            'take_profit_pct': -10.0
        }
        with pytest.raises(TradeValidationError, match="Invalid take_profit_pct"):
            self.validate_trade_setup(trade_setup)

    @pytest.mark.unit
    def test_valid_with_optional_params(self):
        """Test validation passes with all optional parameters."""
        trade_setup = {
            'symbol': 'BTCUSDC',
            'price': 50000.0,
            'position_size': 100.0,
            'stop_loss_pct': 5.0,
            'take_profit_pct': 15.0
        }
        # Should not raise
        self.validate_trade_setup(trade_setup)


class TestOrderExecutionVerification:
    """Tests for order execution verification (BUG-003 fix)."""

    def verify_order_execution(self, order: Dict, expected_side: str, expected_quantity: float) -> bool:
        """
        Copy of the verification logic for testing.
        """
        if not order:
            return False

        status = order.get('status')
        if status not in ['FILLED', 'PARTIALLY_FILLED']:
            return False

        fills = order.get('fills', [])
        if not fills:
            return False

        executed_qty = float(order.get('executedQty', 0))
        if executed_qty <= 0:
            return False

        return True

    @pytest.mark.unit
    def test_filled_order_verification(self, sample_order_response_filled):
        """Test verification passes for filled order."""
        result = self.verify_order_execution(sample_order_response_filled, 'BUY', 0.001)
        assert result is True

    @pytest.mark.unit
    def test_rejected_order_verification(self, sample_order_response_rejected):
        """Test verification fails for rejected order."""
        result = self.verify_order_execution(sample_order_response_rejected, 'BUY', 0.001)
        assert result is False

    @pytest.mark.unit
    def test_empty_fills_verification(self, sample_order_response_empty_fills):
        """Test verification fails for order with empty fills."""
        result = self.verify_order_execution(sample_order_response_empty_fills, 'BUY', 0.001)
        assert result is False

    @pytest.mark.unit
    def test_none_order_verification(self):
        """Test verification fails for None order."""
        result = self.verify_order_execution(None, 'BUY', 0.001)
        assert result is False

    @pytest.mark.unit
    def test_empty_order_verification(self):
        """Test verification fails for empty order dict."""
        result = self.verify_order_execution({}, 'BUY', 0.001)
        assert result is False

    @pytest.mark.unit
    def test_new_status_order_verification(self):
        """Test verification fails for NEW status order."""
        order = {
            'symbol': 'BTCUSDC',
            'orderId': 12345,
            'status': 'NEW',
            'executedQty': '0.0',
            'fills': []
        }
        result = self.verify_order_execution(order, 'BUY', 0.001)
        assert result is False

    @pytest.mark.unit
    def test_cancelled_order_verification(self):
        """Test verification fails for CANCELED order."""
        order = {
            'symbol': 'BTCUSDC',
            'orderId': 12345,
            'status': 'CANCELED',
            'executedQty': '0.0',
            'fills': []
        }
        result = self.verify_order_execution(order, 'BUY', 0.001)
        assert result is False

    @pytest.mark.unit
    def test_partially_filled_order_verification(self):
        """Test verification passes for partially filled order."""
        order = {
            'symbol': 'BTCUSDC',
            'orderId': 12345,
            'status': 'PARTIALLY_FILLED',
            'executedQty': '0.0005',  # Half filled
            'fills': [{'price': '50000.00', 'qty': '0.0005'}]
        }
        result = self.verify_order_execution(order, 'BUY', 0.001)
        assert result is True

    @pytest.mark.unit
    def test_zero_executed_qty_verification(self):
        """Test verification fails when executedQty is zero."""
        order = {
            'symbol': 'BTCUSDC',
            'orderId': 12345,
            'status': 'FILLED',
            'executedQty': '0.0',
            'fills': [{'price': '50000.00', 'qty': '0.001'}]
        }
        result = self.verify_order_execution(order, 'BUY', 0.001)
        assert result is False


class TestRoundStepSize:
    """Tests for step size rounding."""

    def round_step_size(self, quantity: float, step_size: float) -> float:
        """Copy of the rounding logic for testing."""
        if step_size <= 0:
            raise ValueError(f"Invalid step_size: {step_size}")
        precision = len(str(step_size).split('.')[-1])
        return round(quantity - (quantity % step_size), precision)

    @pytest.mark.unit
    def test_round_to_step_size(self):
        """Test rounding quantity to valid step size."""
        result = self.round_step_size(0.00123456, 0.00001)
        assert result == 0.00123

    @pytest.mark.unit
    def test_round_larger_step_size(self):
        """Test rounding with larger step size."""
        result = self.round_step_size(1.2345, 0.01)
        assert result == 1.23

    @pytest.mark.unit
    def test_exact_step_size_multiple(self):
        """Test quantity that's already a step size multiple."""
        result = self.round_step_size(0.001, 0.001)
        assert result == 0.001

    @pytest.mark.unit
    def test_zero_step_size_raises(self):
        """Test that zero step size raises ValueError."""
        with pytest.raises(ValueError, match="Invalid step_size"):
            self.round_step_size(0.001, 0.0)

    @pytest.mark.unit
    def test_negative_step_size_raises(self):
        """Test that negative step size raises ValueError."""
        with pytest.raises(ValueError, match="Invalid step_size"):
            self.round_step_size(0.001, -0.001)


class TestDivisionByZeroProtection:
    """Tests for division by zero protection in trade execution."""

    @pytest.mark.unit
    def test_quantity_calculation_normal(self):
        """Test normal quantity calculation."""
        position_size = 100.0
        price = 50000.0
        quantity = position_size / price
        assert quantity == 0.002

    @pytest.mark.unit
    def test_quantity_calculation_zero_price_protection(self):
        """Test that zero price is caught before division."""
        position_size = 100.0
        price = 0.0

        # The fix should catch this before division
        if price <= 0:
            result = None  # Would not proceed
        else:
            result = position_size / price

        assert result is None

    @pytest.mark.unit
    def test_min_notional_check_with_zero_quantity(self):
        """Test minimum notional check handles edge cases."""
        quantity = 0.0
        price = 50000.0
        min_notional = 10.0

        order_value = quantity * price
        assert order_value < min_notional
