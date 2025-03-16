import os
import sys
import time
import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from auto_trader import AutoTrader
from services.strategy_selection_service import StrategySelectionService
from services.market_regime_service import MarketRegimeService

def setup_logging():
    """Setup logging configuration"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def print_status(trader, strategy_selector=None):
    """Print current trading status"""
    print("\n=== Trading Status ===")
    print(f"Active Positions: {len(trader.trade_executor.active_trades)}/{trader.config['trading_params']['max_positions']}")
    
    # Print active strategy information if available
    if strategy_selector and hasattr(strategy_selector, 'active_strategy_id'):
        print(f"Active Strategy: {strategy_selector.active_strategy_id}")
        print(f"Current Risk Profile: {strategy_selector.current_risk_profile}")
    
    # Print current market regime if available
    try:
        market_regime = trader.redis.get('current_market_regime')
        if market_regime:
            print(f"Market Regime: {market_regime}")
    except:
        pass
    
    # Print active trades
    if trader.trade_executor.active_trades:
        print("\nCurrent Positions:")
        print("-" * 80)
        print(f"{'Symbol':<10} {'Entry Price':<12} {'Current Price':<12} {'PnL %':<8} {'Duration':<15}")
        print("-" * 80)
        
        total_pnl = 0
        for symbol, trade in trader.trade_executor.active_trades.items():
            try:
                current_price = float(trader.client.get_symbol_ticker(symbol=symbol)['price'])
                entry_price = trade['entry_price']
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                entry_time = datetime.fromisoformat(trade['entry_time'])
                duration = datetime.now() - entry_time
                hours = duration.total_seconds() / 3600
                
                print(f"{symbol:<10} "
                      f"${entry_price:<11.4f} "
                      f"${current_price:<11.4f} "
                      f"{pnl_pct:>7.2f}% "
                      f"{f'{hours:.1f}h':<15}")
                
                total_pnl += (current_price - entry_price) * trade['quantity']
            except Exception as e:
                print(f"Error getting data for {symbol}: {str(e)}")
        
        print("-" * 80)
        print(f"Total PnL: ${total_pnl:.2f}")
    
    # Print current opportunities
    if not trader.opportunity_queue.empty():
        print("\nCurrent Opportunities:")
        print("-" * 80)
        print(f"{'Symbol':<10} {'Price':<12} {'Volume':<15} {'Change %':<8}")
        print("-" * 80)
        
        # Get opportunities without removing them from queue
        opportunities = []
        while not trader.opportunity_queue.empty():
            opp = trader.opportunity_queue.get()
            opportunities.append(opp)
            print(f"{opp['symbol']:<10} "
                  f"${opp['price']:<11.4f} "
                  f"${opp['volume']:>14,.0f} "
                  f"{opp['price_change']:>7.2f}%")
        
        # Put opportunities back in queue
        for opp in opportunities:
            trader.opportunity_queue.put(opp)

    # Print strategy selection metrics if available
    try:
        strategy_metrics = trader.redis.get('strategy_selection_metrics') 
        if strategy_metrics:
            metrics = json.loads(strategy_metrics)
            
            # Only print if the metrics are recent (less than 10 minutes old)
            metrics_time = datetime.fromisoformat(metrics['timestamp'])
            if datetime.now() - metrics_time < timedelta(minutes=10):
                print("\nStrategy Selection Metrics:")
                print("-" * 80)
                print(f"Optimal Strategy: {metrics['optimal_strategy_id']} (Score: {metrics['optimal_score']:.4f})")
                print(f"Current Strategy: {metrics['active_strategy_id']}")
                print(f"Market Regime: {metrics['market_regime']}")
                
                # Print factor scores if available
                if 'factor_scores' in metrics:
                    print("\nFactor Scores:")
                    for factor, score in metrics['factor_scores'].items():
                        print(f"- {factor}: {score:.4f}")
    except Exception as e:
        pass  # Silently ignore any errors here

async def run_strategy_selection_service():
    """Run the strategy selection service"""
    service = StrategySelectionService()
    await service.run()

async def run_market_regime_service():
    """Run the market regime service"""
    service = MarketRegimeService()
    await service.run()

def run_async_service(async_func):
    """Run an async service in the current thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_func())
    loop.close()

def main():
    """Main function to run the trader with status monitoring"""
    setup_logging()
    logging.info("Starting trading bot with automated strategy selection...")
    
    # Flag to control service threads
    running = True
    
    try:
        # Start market regime service in a separate thread
        regime_thread = threading.Thread(
            target=run_async_service,
            args=(run_market_regime_service,),
            daemon=True
        )
        regime_thread.start()
        logging.info("Market regime service started")
        
        # Start strategy selection service in a separate thread
        strategy_thread = threading.Thread(
            target=run_async_service,
            args=(run_strategy_selection_service,),
            daemon=True
        )
        strategy_thread.start()
        logging.info("Strategy selection service started")
        
        # Small delay to allow services to initialize
        time.sleep(3)
        
        # Initialize trader
        trader = AutoTrader()
        
        # Get strategy selection service instance for status display
        # Note: We only use this for display purposes, as the actual service runs in its own thread
        strategy_selector = StrategySelectionService()
        
        # Get initial balance
        account = trader.client.get_account()
        initial_balance = next(
            (float(asset['free']) for asset in account['balances'] 
             if asset['asset'] == 'USDC'),
            0.0
        )
        
        # Print initial configuration
        print("\n=== Trading Bot Configuration ===")
        print(f"Initial USDC Balance: ${initial_balance:.2f}")
        print(f"Reserve Ratio: {trader.config['trading_params'].get('reserve_ratio', 0.1) * 100}%")
        print(f"Max Positions: {trader.config['trading_params']['max_positions']}")
        print(f"Min Volume: ${trader.config['trading_params']['min_volume_usdc']:,}")
        print(f"Position Size: {trader.config['trading_params'].get('position_size', 0.1) * 100}% of available capital")
        print(f"Strategy Selection: Automated (Market Regime: {strategy_selector.selection_weights['market_regime']:.2f}, Performance: {strategy_selector.selection_weights['historical_performance']:.2f})")
        
        # Start the trader
        trader.start()
        
        # Monitor and print status
        while running:
            print_status(trader, strategy_selector)
            time.sleep(5)  # Update status every 5 seconds
            
    except KeyboardInterrupt:
        logging.info("Shutting down trading bot...")
        print("\nShutting down gracefully...")
        
        # Set flag to stop all services
        running = False
        
        # Stop the trader
        trader.stop()
        
        # Close all positions if needed
        if trader.trade_executor.active_trades:
            print("\nClosing all active positions...")
            for symbol, trade in trader.trade_executor.active_trades.items():
                try:
                    # Cancel all open orders
                    trader.client.cancel_open_orders(symbol=symbol)
                    # Place market sell order
                    trader.client.create_order(
                        symbol=symbol,
                        side='SELL',
                        type='MARKET',
                        quantity=trade['quantity']
                    )
                    print(f"Closed position for {symbol}")
                except Exception as e:
                    print(f"Error closing position for {symbol}: {str(e)}")
        
        print("\nTrading bot stopped.")
        
        # Give threads a moment to clean up
        time.sleep(2)
        sys.exit(0)
        
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        running = False
        raise

if __name__ == "__main__":
    main()
