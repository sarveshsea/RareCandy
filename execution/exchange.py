import ccxt
import time
from typing import Dict, Any, Optional
from rare_candy.core.types import OrderSide, Signal, RiskDecision

class ExchangeAdapter:
    """
    Unified Exchange Interface.
    Currently hardcoded for Coinbase via CCXT.
    """
    def __init__(self, api_key: str, api_secret: str, sandbox: bool = False):
        self.client = ccxt.coinbase({
            'apiKey': api_key,
            'secret': api_secret,
            'options': {'defaultType': 'spot'}
        })
        if sandbox:
            self.client.set_sandbox_mode(True)
            
    def get_ticker(self, symbol: str) -> Dict[str, float]:
        """Fetch current price."""
        try:
            ticker = self.client.fetch_ticker(symbol)
            return {
                'price': float(ticker['last']),
                'bid': float(ticker['bid']),
                'ask': float(ticker['ask']),
            }
        except Exception as e:
            print(f"[Exchange] Ticker Error: {e}")
            return {'price': 0.0}

    def fetch_balance(self) -> float:
        """Fetch total USD equity."""
        try:
            bal = self.client.fetch_balance()
            # Assuming USD quote for total equity
            return float(bal['total'].get('USD', 0.0))
        except Exception as e:
            print(f"[Exchange] Balance Error: {e}")
            return 0.0

    def get_positions(self) -> Dict[str, float]:
        """
        Fetch open positions. 
        For Spot, this is just positive balances of assets.
        Returns: { 'BTC/USD': 0.12, 'ETH/USD': 1.5 }
        """
        pos = {}
        try:
            bal = self.client.fetch_balance()
            total = bal['total']
            for coin, qty in total.items():
                if coin == 'USD' or qty <= 0:
                    continue
                # Map coin to pair (Simple assumption for now)
                pair = f"{coin}/USD"
                pos[pair] = float(qty)
        except Exception as e:
            print(f"[Exchange] Positions Error: {e}")
        return pos

    def execute_order(self, decision: RiskDecision, symbol: str, side: OrderSide) -> Optional[Dict]:
        """
        Execute a market order based on Risk Decision.
        """
        if not decision.approved or decision.quantity <= 0:
            return None
            
        print(f"[Exchange] Executing {side} {decision.quantity} {symbol}...")
        try:
            order = self.client.create_order(
                symbol=symbol,
                type='market',
                side=side.value.lower(),
                amount=decision.quantity
            )
            print(f"[Exchange] Order Filled: {order['id']}")
            return order
        except Exception as e:
            print(f"[Exchange] Order Failed: {e}")
            return None
