import os
import sys
import json
import asyncio
import unittest
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.strategy_evolution_service import StrategyEvolutionService

class TestStrategyEvolution(unittest.TestCase):
    async def asyncSetUp(self):
        """Set up test environment"""
        # Load test configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)

        # Initialize service
        self.service = StrategyEvolutionService()

        # Test parameters
        self.test_params = {
            'type': 'mean_reversion',
            'timeframe': '5m',
            'risk_limit': 1.5,
            'target_profit': 1.0,
            'max_position_size': 5
        }

    async def test_strategy_generation(self):
        """Test strategy generation"""
        # Generate strategy
        strategy_code = await self.service.generate_strategy(self.test_params)
        
        # Verify strategy code
        self.assertIsNotNone(strategy_code)
        self.assertIn('async fetch(request, env)', strategy_code)
        self.assertIn('export default', strategy_code)

    async def test_strategy_deployment(self):
        """Test strategy deployment to Cloudflare"""
        # Generate and deploy strategy
        strategy_code = await self.service.generate_strategy(self.test_params)
        worker_id = await self.service.deploy_strategy(strategy_code)
        
        # Verify deployment
        self.assertIsNotNone(worker_id)
        self.assertTrue(worker_id in self.service.active_strategies)

    async def test_strategy_monitoring(self):
        """Test strategy performance monitoring"""
        # Deploy strategy
        strategy_code = await self.service.generate_strategy(self.test_params)
        worker_id = await self.service.deploy_strategy(strategy_code)
        
        # Monitor performance
        performance = await self.service.monitor_strategy(worker_id)
        
        # Verify monitoring data
        self.assertIsNotNone(performance)
        self.assertIn('sharpe_ratio', performance)
        self.assertIn('drawdown', performance)
        self.assertIn('win_rate', performance)

    async def test_strategy_evolution(self):
        """Test strategy evolution process"""
        # Deploy initial strategy
        strategy_code = await self.service.generate_strategy(self.test_params)
        worker_id = await self.service.deploy_strategy(strategy_code)
        
        # Create test performance data
        test_performance = {
            'sharpe_ratio': 0.5,  # Below threshold to trigger evolution
            'drawdown': 20,
            'win_rate': 0.45,
            'profit_factor': 1.1,
            'total_trades': 100,
            'period_start': (datetime.now() - timedelta(days=7)).isoformat(),
            'period_end': datetime.now().isoformat()
        }
        
        # Evolve strategy
        improved_strategy = await self.service.evolve_strategy(worker_id, test_performance)
        
        # Verify evolution
        self.assertIsNotNone(improved_strategy)
        self.assertNotEqual(improved_strategy, strategy_code)

    async def test_complete_cycle(self):
        """Test complete strategy lifecycle"""
        # 1. Generate initial strategy
        strategy_code = await self.service.generate_strategy(self.test_params)
        self.assertIsNotNone(strategy_code)
        
        # 2. Deploy strategy
        worker_id = await self.service.deploy_strategy(strategy_code)
        self.assertIsNotNone(worker_id)
        
        # 3. Monitor performance
        performance = await self.service.monitor_strategy(worker_id)
        self.assertIsNotNone(performance)
        
        # 4. Trigger evolution
        test_performance = {
            'sharpe_ratio': 0.5,
            'drawdown': 20,
            'win_rate': 0.45
        }
        improved_strategy = await self.service.evolve_strategy(worker_id, test_performance)
        self.assertIsNotNone(improved_strategy)
        
        # 5. Deploy improved version
        new_worker_id = await self.service.deploy_strategy(improved_strategy)
        self.assertIsNotNone(new_worker_id)
        self.assertNotEqual(worker_id, new_worker_id)

def run_tests():
    """Run all tests"""
    async def run_async_tests():
        # Create test suite
        suite = unittest.TestSuite()
        
        # Create test instance
        test_case = TestStrategyEvolution()
        
        # Set up test case
        await test_case.asyncSetUp()
        
        # Add tests
        suite.addTest(test_case)
        
        # Run tests
        runner = unittest.TextTestRunner()
        runner.run(suite)
    
    # Run async tests
    asyncio.run(run_async_tests())

if __name__ == '__main__':
    run_tests()
