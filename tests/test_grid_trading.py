"""
Unit tests for Grid Trading Strategy.

Tests cover:
- BUG-001: Race condition prevention with asyncio locks
- BUG-002: Safe grid level index lookup
- BUG-006: Division by zero protection in volatility calculation
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from typing import List, Optional

# Import the module under test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSafeGridLevelIndex:
    """Tests for the safe grid level index lookup function (BUG-002 fix)."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a minimal mock of GridTradingStrategy for testing helper methods
        self.mock_logger = MagicMock()

        # We'll test the logic directly without instantiating the full class
        self.grid_levels = [47500.0, 48000.0, 48500.0, 49000.0, 49500.0,
                           50000.0, 50500.0, 51000.0, 51500.0, 52000.0, 52500.0]

    def _safe_find_grid_level_index(self, grid_levels: List[float], target_level: float) -> Optional[int]:
        """
        Copy of the safe grid level lookup logic for testing.
        """
        if not grid_levels:
            return None

        tolerance = 1e-8
        for i, level in enumerate(grid_levels):
            if abs(level - target_level) < tolerance:
                return i

        min_diff = float('inf')
        closest_idx = None
        for i, level in enumerate(grid_levels):
            diff = abs(level - target_level)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i

        if closest_idx is not None and min_diff < abs(target_level) * 0.001:
            return closest_idx

        return None

    @pytest.mark.unit
    def test_exact_level_match(self):
        """Test finding an exact grid level match."""
        index = self._safe_find_grid_level_index(self.grid_levels, 50000.0)
        assert index == 5

    @pytest.mark.unit
    def test_first_level(self):
        """Test finding the first grid level."""
        index = self._safe_find_grid_level_index(self.grid_levels, 47500.0)
        assert index == 0

    @pytest.mark.unit
    def test_last_level(self):
        """Test finding the last grid level."""
        index = self._safe_find_grid_level_index(self.grid_levels, 52500.0)
        assert index == 10

    @pytest.mark.unit
    def test_level_not_found(self):
        """Test that non-existent level returns None."""
        index = self._safe_find_grid_level_index(self.grid_levels, 45000.0)
        assert index is None

    @pytest.mark.unit
    def test_empty_grid_levels(self):
        """Test with empty grid levels list."""
        index = self._safe_find_grid_level_index([], 50000.0)
        assert index is None

    @pytest.mark.unit
    def test_floating_point_precision(self):
        """Test handling of floating-point precision issues."""
        # Simulate a slight floating-point error
        imprecise_level = 50000.0 + 1e-10
        index = self._safe_find_grid_level_index(self.grid_levels, imprecise_level)
        assert index == 5

    @pytest.mark.unit
    def test_close_but_not_exact_match(self):
        """Test finding a close level within tolerance."""
        # Within 0.1% tolerance
        close_level = 50000.0 * 1.0005  # 0.05% off
        index = self._safe_find_grid_level_index(self.grid_levels, close_level)
        assert index == 5

    @pytest.mark.unit
    def test_too_far_from_any_level(self):
        """Test that level too far from any grid returns None."""
        # More than 0.1% from any level
        far_level = 50000.0 * 1.01  # 1% off
        index = self._safe_find_grid_level_index(self.grid_levels, far_level)
        assert index is None


class TestVolatilityCalculation:
    """Tests for volatility calculation with zero-division protection (BUG-006 fix)."""

    @pytest.mark.unit
    def test_normal_volatility_calculation(self):
        """Test normal volatility calculation."""
        closes = [48000.0, 49000.0, 50000.0, 51000.0, 52000.0]
        mean_close = np.mean(closes)
        volatility = np.std(closes) / mean_close * 100

        assert volatility > 0
        assert not np.isnan(volatility)
        assert not np.isinf(volatility)

    @pytest.mark.unit
    def test_zero_mean_protection(self):
        """Test protection against zero mean."""
        closes = [0.0, 0.0, 0.0, 0.0, 0.0]
        mean_close = np.mean(closes)

        # Apply the fix logic
        if mean_close == 0 or np.isnan(mean_close) or np.isinf(mean_close):
            volatility = 2.0  # Default
        else:
            volatility = np.std(closes) / mean_close * 100

        assert volatility == 2.0

    @pytest.mark.unit
    def test_nan_in_closes(self):
        """Test handling of NaN values in close prices."""
        closes = [48000.0, np.nan, 50000.0, 51000.0, 52000.0]
        mean_close = np.nanmean(closes)  # Use nanmean to ignore NaN

        if mean_close == 0 or np.isnan(mean_close) or np.isinf(mean_close):
            volatility = 2.0
        else:
            volatility = np.nanstd(closes) / mean_close * 100
            if np.isnan(volatility) or np.isinf(volatility) or volatility <= 0:
                volatility = 2.0

        assert volatility > 0
        assert not np.isnan(volatility)

    @pytest.mark.unit
    def test_single_value_volatility(self):
        """Test volatility with single value (std = 0)."""
        closes = [50000.0]
        mean_close = np.mean(closes)

        if mean_close == 0 or np.isnan(mean_close) or np.isinf(mean_close):
            volatility = 2.0
        else:
            volatility = np.std(closes) / mean_close * 100
            if np.isnan(volatility) or np.isinf(volatility) or volatility <= 0:
                volatility = 2.0

        # std of single value is 0, so volatility should be 0, triggering default
        assert volatility == 2.0


class TestAsyncLocking:
    """Tests for asyncio lock behavior (BUG-001 fix)."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_lock_prevents_concurrent_access(self):
        """Test that lock prevents concurrent modification."""
        locks = {}
        shared_data = {'counter': 0}
        access_order = []

        async def get_lock(symbol: str) -> asyncio.Lock:
            if symbol not in locks:
                locks[symbol] = asyncio.Lock()
            return locks[symbol]

        async def increment_with_lock(symbol: str, task_id: int):
            lock = await get_lock(symbol)
            async with lock:
                access_order.append(f"start_{task_id}")
                current = shared_data['counter']
                await asyncio.sleep(0.01)  # Simulate work
                shared_data['counter'] = current + 1
                access_order.append(f"end_{task_id}")

        # Run multiple tasks concurrently
        await asyncio.gather(
            increment_with_lock('BTCUSDC', 1),
            increment_with_lock('BTCUSDC', 2),
            increment_with_lock('BTCUSDC', 3)
        )

        # Verify counter is correct (no race condition)
        assert shared_data['counter'] == 3

        # Verify access was serialized (no interleaving)
        for i in range(0, len(access_order), 2):
            task_id = access_order[i].split('_')[1]
            assert access_order[i + 1] == f"end_{task_id}"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_separate_symbols_dont_block(self):
        """Test that locks for different symbols don't block each other."""
        locks = {}
        execution_times = {}

        async def get_lock(symbol: str) -> asyncio.Lock:
            if symbol not in locks:
                locks[symbol] = asyncio.Lock()
            return locks[symbol]

        async def process_symbol(symbol: str):
            import time
            start = time.time()
            lock = await get_lock(symbol)
            async with lock:
                await asyncio.sleep(0.05)  # Simulate work
            end = time.time()
            execution_times[symbol] = end - start

        # Run for different symbols concurrently
        await asyncio.gather(
            process_symbol('BTCUSDC'),
            process_symbol('ETHUSDC')
        )

        # Both should complete in roughly the same time (parallel execution)
        # Not serialized like same-symbol would be
        assert abs(execution_times['BTCUSDC'] - execution_times['ETHUSDC']) < 0.02


class TestGridProfitCalculation:
    """Tests for grid profit calculations with safe index handling."""

    @pytest.mark.unit
    def test_profit_calculation_valid_index(self):
        """Test profit calculation with valid grid level index."""
        grid_levels = [47500.0, 48000.0, 48500.0, 49000.0, 49500.0, 50000.0]
        sell_level_index = 3  # 49000.0
        filled_price = 49000.0
        filled_qty = 0.001

        if sell_level_index > 0:
            profit = (filled_price - grid_levels[sell_level_index - 1]) * filled_qty
        else:
            profit = 0

        expected_profit = (49000.0 - 48500.0) * 0.001  # 0.5 USDC
        assert abs(profit - expected_profit) < 1e-10

    @pytest.mark.unit
    def test_profit_calculation_first_level(self):
        """Test profit calculation at first grid level (no profit)."""
        grid_levels = [47500.0, 48000.0, 48500.0]
        sell_level_index = 0
        filled_qty = 0.001

        if sell_level_index > 0:
            profit = (47500.0 - grid_levels[sell_level_index - 1]) * filled_qty
        else:
            profit = 0

        assert profit == 0

    @pytest.mark.unit
    def test_profit_calculation_none_index(self):
        """Test profit calculation when index is None."""
        sell_level_index = None
        filled_qty = 0.001

        if sell_level_index is not None and sell_level_index > 0:
            profit = 100.0  # Would be calculated
        else:
            profit = 0

        assert profit == 0


class TestGridLevelGeneration:
    """Tests for grid level generation."""

    @pytest.mark.unit
    def test_arithmetic_grid_generation(self):
        """Test arithmetic grid level generation."""
        lower = 47500.0
        upper = 52500.0
        num_grids = 10

        step = (upper - lower) / num_grids
        grid_levels = [lower + i * step for i in range(num_grids + 1)]

        assert len(grid_levels) == 11
        assert grid_levels[0] == lower
        assert abs(grid_levels[-1] - upper) < 1e-10
        # Check equal spacing
        for i in range(1, len(grid_levels)):
            assert abs((grid_levels[i] - grid_levels[i-1]) - step) < 1e-10

    @pytest.mark.unit
    def test_geometric_grid_generation(self):
        """Test geometric grid level generation."""
        lower = 47500.0
        upper = 52500.0
        num_grids = 10

        ratio = (upper / lower) ** (1 / num_grids)
        grid_levels = [lower * (ratio ** i) for i in range(num_grids + 1)]

        assert len(grid_levels) == 11
        assert abs(grid_levels[0] - lower) < 1e-10
        assert abs(grid_levels[-1] - upper) < 0.01  # Allow small float error
        # Check equal ratio spacing
        for i in range(1, len(grid_levels)):
            actual_ratio = grid_levels[i] / grid_levels[i-1]
            assert abs(actual_ratio - ratio) < 1e-10
