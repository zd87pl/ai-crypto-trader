import abc
from typing import Dict, List, Optional, Union, Any
import logging
import os
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException

logger = logging.getLogger(__name__)

class ExchangeInterface(abc.ABC):
    """
    Abstract base class for exchange interfaces.
    Defines the common interface for all exchanges.
    """
    
    @abc.abstractmethod
    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker data for a symbol"""
        pass
    
    @abc.abstractmethod
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book data for a symbol"""
        pass
    
    @abc.abstractmethod
    def get_symbols(self) -> List[str]:
        """Get all available trading symbols"""
        pass
    
    @abc.abstractmethod
    def get_exchange_info(self) -> Dict:
        """Get exchange information"""
        pass
    
    @abc.abstractmethod
    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None) -> Dict:
        """Place an order on the exchange"""
        pass
    
    @abc.abstractmethod
    def get_balance(self, asset: str) -> float:
        """Get balance for a specific asset"""
        pass
    
    @abc.abstractmethod
    def get_all_balances(self) -> Dict:
        """Get balances for all assets"""
        pass
    
    @abc.abstractmethod
    def get_fees(self, symbol: Optional[str] = None) -> Dict:
        """Get fee information"""
        pass
    
    @abc.abstractmethod
    def get_ticker_all(self) -> Dict:
        """Get ticker data for all symbols"""
        pass
    
    @abc.abstractmethod
    def get_name(self) -> str:
        """Get the name of the exchange"""
        pass


class BinanceExchange(ExchangeInterface):
    """
    Binance exchange implementation of the ExchangeInterface.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """Initialize Binance exchange interface"""
        self.api_key = api_key or os.environ.get('BINANCE_API_KEY', '')
        self.api_secret = api_secret or os.environ.get('BINANCE_API_SECRET', '')
        
        try:
            self.client = BinanceClient(self.api_key, self.api_secret)
            logger.info("Binance client initialized")
        except Exception as e:
            logger.error(f"Error initializing Binance client: {str(e)}")
            raise
    
    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker data for a symbol"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return {
                'symbol': ticker['symbol'],
                'price': float(ticker['price'])
            }
        except BinanceAPIException as e:
            logger.error(f"Error getting ticker for {symbol}: {str(e)}")
            raise
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book data for a symbol"""
        try:
            order_book = self.client.get_order_book(symbol=symbol, limit=limit)
            return {
                'symbol': symbol,
                'bids': [[float(price), float(qty)] for price, qty in order_book['bids']],
                'asks': [[float(price), float(qty)] for price, qty in order_book['asks']],
                'timestamp': order_book['lastUpdateId']
            }
        except BinanceAPIException as e:
            logger.error(f"Error getting order book for {symbol}: {str(e)}")
            raise
    
    def get_symbols(self) -> List[str]:
        """Get all available trading symbols"""
        try:
            exchange_info = self.client.get_exchange_info()
            return [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']
        except BinanceAPIException as e:
            logger.error(f"Error getting symbols: {str(e)}")
            raise
    
    def get_exchange_info(self) -> Dict:
        """Get exchange information"""
        try:
            return self.client.get_exchange_info()
        except BinanceAPIException as e:
            logger.error(f"Error getting exchange info: {str(e)}")
            raise
    
    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None) -> Dict:
        """Place an order on the exchange"""
        try:
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity
            }
            
            if price is not None and order_type != 'MARKET':
                params['price'] = price
                params['timeInForce'] = 'GTC'
            
            order = self.client.create_order(**params)
            return order
        except BinanceAPIException as e:
            logger.error(f"Error placing order for {symbol}: {str(e)}")
            raise
    
    def get_balance(self, asset: str) -> float:
        """Get balance for a specific asset"""
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == asset:
                    return float(balance['free'])
            return 0.0
        except BinanceAPIException as e:
            logger.error(f"Error getting balance for {asset}: {str(e)}")
            raise
    
    def get_all_balances(self) -> Dict:
        """Get balances for all assets"""
        try:
            account = self.client.get_account()
            balances = {}
            for balance in account['balances']:
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                if total > 0:  # Only include non-zero balances
                    balances[balance['asset']] = {
                        'free': free,
                        'locked': locked,
                        'total': total
                    }
            return balances
        except BinanceAPIException as e:
            logger.error(f"Error getting all balances: {str(e)}")
            raise
    
    def get_fees(self, symbol: Optional[str] = None) -> Dict:
        """Get fee information"""
        try:
            if symbol:
                fees = self.client.get_trade_fee(symbol=symbol)
            else:
                fees = self.client.get_trade_fee()
            return fees
        except BinanceAPIException as e:
            logger.error(f"Error getting fees: {str(e)}")
            # If not available through API, return default values
            return {
                'maker': 0.001,  # 0.1%
                'taker': 0.001   # 0.1%
            }
    
    def get_ticker_all(self) -> Dict:
        """Get ticker data for all symbols"""
        try:
            tickers = self.client.get_all_tickers()
            return {ticker['symbol']: float(ticker['price']) for ticker in tickers}
        except BinanceAPIException as e:
            logger.error(f"Error getting all tickers: {str(e)}")
            raise
    
    def get_name(self) -> str:
        """Get the name of the exchange"""
        return "Binance"


class ExchangeFactory:
    """
    Factory class for creating exchange interfaces.
    """
    
    @staticmethod
    def create_exchange(exchange_name: str, api_key: Optional[str] = None, api_secret: Optional[str] = None) -> ExchangeInterface:
        """Create an exchange interface instance"""
        if exchange_name.lower() == 'binance':
            return BinanceExchange(api_key, api_secret)
        else:
            raise ValueError(f"Unsupported exchange: {exchange_name}")