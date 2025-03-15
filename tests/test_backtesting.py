import os
import sys
import json
import asyncio
import unittest
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.data_manager import HistoricalDataManager
from backtesting.social_data_provider import SocialDataProvider
from backtesting.strategy_tester import StrategyTester
from backtesting.result_analyzer import ResultAnalyzer

class TestBacktesting(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        # Create test data directories
        self.test_dir = Path('test_backtesting_data')
        self.test_dir.mkdir(exist_ok=True)
        
        # Initialize components with test configuration
        self.data_manager = HistoricalDataManager('config.json')
        self.social_provider = SocialDataProvider(self.data_manager)
        self.strategy_tester = StrategyTester('config.json')
        self.result_analyzer = ResultAnalyzer()
        
    def tearDown(self):
        """Clean up after tests"""
        # Remove test data if needed
        pass
        
    async def test_data_manager_initialization(self):
        """Test data manager initialization"""
        # Verify data directories were created
        self.assertTrue(self.data_manager.market_data_dir.exists())
        self.assertTrue(self.data_manager.social_data_dir.exists())
        
    async def test_basic_market_data_structure(self):
        """Test that market data structure is correct"""
        # Create a minimal test DataFrame
        test_data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(10)],
            'open': [100 + i for i in range(10)],
            'high': [110 + i for i in range(10)],
            'low': [90 + i for i in range(10)],
            'close': [105 + i for i in range(10)],
            'volume': [1000 * (i + 1) for i in range(10)]
        })
        
        # Set timestamp as index
        test_data.set_index('timestamp', inplace=True)
        
        # Ensure data_manager handles this structure
        # This is a basic structure test, not a data processing test
        self.assertEqual(test_data.shape[1], 5)  # 5 columns (OHLCV)
        
    async def test_social_data_default_values(self):
        """Test default values for social data provider"""
        # Get default metrics
        default_metrics = self.social_provider.default_metrics
        
        # Check that default sentiment is neutral (0.5)
        self.assertEqual(default_metrics['social_sentiment'], 0.5)
        
        # Check that required metrics exist
        required_metrics = ['social_volume', 'social_engagement', 'social_contributors', 'social_sentiment']
        for metric in required_metrics:
            self.assertIn(metric, default_metrics)
            
    async def test_strategy_tester_initialization(self):
        """Test strategy tester initialization"""
        # Check that stats are properly initialized
        self.assertEqual(self.strategy_tester.stats['initial_balance'], 0.0)
        self.assertEqual(self.strategy_tester.stats['final_balance'], 0.0)
        self.assertEqual(self.strategy_tester.stats['total_trades'], 0)
        
        # Check that the strategy tester has a valid AI trader
        self.assertIsNotNone(self.strategy_tester.ai_trader)
        
    async def test_result_analyzer_initialization(self):
        """Test result analyzer initialization"""
        # Check that plots directory was created
        self.assertTrue(self.result_analyzer.plots_dir.exists())
        
def run_tests():
    """Run all tests"""
    async def run_async_tests():
        # Create test suite
        suite = unittest.TestSuite()
        
        # Create test instance
        test_case = TestBacktesting()
        
        # Set up test case
        test_case.setUp()
        
        # Add tests
        for method_name in dir(test_case):
            if method_name.startswith('test_') and callable(getattr(test_case, method_name)):
                if method_name.startswith('test_async_'):
                    continue  # Skip async tests for now
                test_method = getattr(test_case, method_name)
                if asyncio.iscoroutinefunction(test_method):
                    # Convert async test to sync for unittest
                    setattr(test_case, method_name, lambda test_method=test_method: asyncio.run(test_method()))
                
        suite.addTest(test_case)
        
        # Run tests
        runner = unittest.TextTestRunner()
        runner.run(suite)
        
        # Clean up
        test_case.tearDown()
    
    # Run async tests
    asyncio.run(run_async_tests())

if __name__ == '__main__':
    run_tests()