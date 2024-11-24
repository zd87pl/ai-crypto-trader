import os
import sys
import json
import asyncio
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestEnvironment:
    """Test environment setup and teardown"""
    
    @staticmethod
    def setup():
        """Set up test environment"""
        try:
            # Load environment variables
            load_dotenv('.env')
            
            # Verify required environment variables
            required_vars = [
                'OPENAI_API_KEY',
                'CLOUDFLARE_API_TOKEN',
                'CLOUDFLARE_ACCOUNT_ID'
            ]
            
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
            # Create test config
            test_config = {
                "trading_params": {
                    "min_volume_usdc": 10000,  # Lower for testing
                    "position_size": 0.05,     # Smaller for testing
                    "max_positions": 2,        # Fewer for testing
                    "stop_loss_pct": 1,
                    "take_profit_pct": 2,
                    "min_trade_amount": 10
                },
                "evolution": {
                    "min_sharpe_ratio": 1.0,
                    "max_drawdown": 20.0,
                    "min_win_rate": 0.5,
                    "min_profit_factor": 1.2,
                    "improvement_threshold": 0.1,
                    "max_iterations": 3,        # Fewer for testing
                    "convergence_criteria": 0.05
                },
                "worker_defaults": {
                    "memory_limit": "128MB",
                    "cpu_limit": "10ms",
                    "timeout": 5000            # Shorter for testing
                }
            }
            
            # Save test config
            with open('tests/test_config.json', 'w') as f:
                json.dump(test_config, f, indent=2)
            
            logger.info("Test environment setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up test environment: {str(e)}")
            return False

    @staticmethod
    def teardown():
        """Clean up test environment"""
        try:
            # Remove test config
            if os.path.exists('tests/test_config.json'):
                os.remove('tests/test_config.json')
            
            # Clean up any test workers
            # This would typically involve calling Cloudflare API to remove test workers
            
            logger.info("Test environment cleanup completed")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up test environment: {str(e)}")
            return False

async def run_tests():
    """Run all tests"""
    try:
        # Set up test environment
        if not TestEnvironment.setup():
            logger.error("Failed to set up test environment")
            return False
        
        # Import and run tests
        from test_strategy_evolution import TestStrategyEvolution
        
        # Create test suite
        import unittest
        suite = unittest.TestSuite()
        
        # Create test instance
        test_case = TestStrategyEvolution()
        
        # Set up test case
        await test_case.asyncSetUp()
        
        # Add tests
        test_methods = [
            'test_strategy_generation',
            'test_strategy_deployment',
            'test_strategy_monitoring',
            'test_strategy_evolution',
            'test_complete_cycle'
        ]
        
        for method in test_methods:
            suite.addTest(TestStrategyEvolution(method))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Clean up test environment
        TestEnvironment.teardown()
        
        return result.wasSuccessful()
        
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        TestEnvironment.teardown()
        return False

def main():
    """Main entry point"""
    try:
        logger.info("Starting test run...")
        success = asyncio.run(run_tests())
        
        if success:
            logger.info("All tests passed successfully")
            sys.exit(0)
        else:
            logger.error("Some tests failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Test run interrupted")
        TestEnvironment.teardown()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in test runner: {str(e)}")
        TestEnvironment.teardown()
        sys.exit(1)

if __name__ == "__main__":
    main()
