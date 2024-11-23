import os
import sys
import time
import json
import logging
from datetime import datetime
from auto_trader import AutoTrader

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

def print_status(trader):
    """Print current trading status"""
    print("\n=== Trading Status ===")
    print(f"Active Positions: {len(trader.trade_executor.active_trades)}/{trader.config['trading_params']['max_positions']}")
    
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
                duration = datetime.now() - trade['entry_time']
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

def main():
    """Main function to run the trader with status monitoring"""
    setup_logging()
    logging.info("Starting trading bot...")
    
    try:
        # Initialize trader
        trader = AutoTrader()
        
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
        print(f"Reserve Ratio: {trader.config['trading_params']['reserve_ratio'] * 100}%")
        print(f"Max Positions: {trader.config['trading_params']['max_positions']}")
        print(f"Min Volume: ${trader.config['trading_params']['min_volume_usdc']:,}")
        print(f"Position Size: {trader.config['trading_params']['position_size_pct'] * 100}% of available capital")
        
        # Start the trader
        trader.start()
        
        # Monitor and print status
        while True:
            print_status(trader)
            time.sleep(5)  # Update status every 5 seconds
            
    except KeyboardInterrupt:
        logging.info("Shutting down trading bot...")
        print("\nShutting down gracefully...")
        
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
        sys.exit(0)
        
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
