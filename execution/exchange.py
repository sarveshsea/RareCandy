import ccxt
import time
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from rare_candy.core.types import OrderSide, Signal, RiskDecision

class ExchangeAdapter:
    """
    Unified Exchange Interface.
    Currently hardcoded for Coinbase via CCXT.
    """
    def __init__(self, api_key: str, api_secret: str, sandbox: bool = False, paper_mode: bool = False):
        self.paper_mode = paper_mode
        self.paper_file = "paper_state.json"
        
        # Default Paper State
        self.paper_state = {
            "balance": {"USD": 10000.0},
            "positions": {},
            "history": []
        }
        
        if self.paper_mode:
            self._load_paper_state()
        
        self.client = ccxt.coinbase({
            'apiKey': api_key,
            'secret': api_secret,
            'options': {'defaultType': 'spot'}
        })
        if sandbox:
            self.client.set_sandbox_mode(True)

    def _load_paper_state(self):
        """Load paper state from disk."""
        if os.path.exists(self.paper_file):
            try:
                with open(self.paper_file, 'r') as f:
                    self.paper_state = json.load(f)
                print(f"[Paper] Loaded State. Equity: ${self.paper_state['balance']['USD']:.2f}")
            except Exception as e:
                print(f"[Paper] Load Error: {e}")

    def _save_paper_state(self):
        """Save paper state to disk."""
        try:
            with open(self.paper_file, 'w') as f:
                json.dump(self.paper_state, f, indent=2)
        except Exception as e:
            print(f"[Paper] Save Error: {e}")

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
        if self.paper_mode:
            return self.paper_state['balance'].get("USD", 0.0)
            
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
        if self.paper_mode:
            return self.paper_state['positions']

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
            
        print(f"[Exchange] Executing {side} {decision.quantity} {symbol} (Paper={self.paper_mode})...")
        
        if self.paper_mode:
            # --- REALISTIC PAPER EXECUTION ---
            ticker = self.get_ticker(symbol)
            raw_price = ticker['price']
            
            # 1. SLIPPAGE MODEL (0.05%)
            # Buy = Pay more, Sell = Get less
            slippage = 0.0005 
            if side == OrderSide.BUY:
                fill_price = raw_price * (1 + slippage)
            else:
                fill_price = raw_price * (1 - slippage)
                
            # 2. FEE MODEL (0.6% Taker)
            fee_rate = 0.006
            gross_cost = fill_price * decision.quantity
            fee_amt = gross_cost * fee_rate
            
            # Execution
            if side == OrderSide.BUY:
                total_cost = gross_cost + fee_amt
                usd_bal = self.paper_state['balance'].get("USD", 0.0)
                
                if usd_bal >= total_cost:
                    self.paper_state['balance']["USD"] -= total_cost
                    current_pos = self.paper_state['positions'].get(symbol, 0.0)
                    self.paper_state['positions'][symbol] = current_pos + decision.quantity
                    
                    self._log_paper_trade(symbol, side, decision.quantity, fill_price, fee_amt)
                    self._save_paper_state()
                    return {'id': f'p-{int(time.time())}', 'status': 'filled', 'filled': decision.quantity, 'price': fill_price}
                else:
                    print(f"📝 PAPER FAIL: Insufficient Funds. Need ${total_cost:.2f}, Have ${usd_bal:.2f}")
                    return None
            
            elif side == OrderSide.SELL:
                current_qty = self.paper_state['positions'].get(symbol, 0.0)
                if current_qty >= decision.quantity:
                    revenue = gross_cost - fee_amt
                    self.paper_state['positions'][symbol] -= decision.quantity
                    if self.paper_state['positions'][symbol] <= 1e-6: # Dust cleanup
                        del self.paper_state['positions'][symbol]
                        
                    self.paper_state['balance']["USD"] += revenue
                    
                    self._log_paper_trade(symbol, side, decision.quantity, fill_price, fee_amt)
                    self._save_paper_state()
                    return {'id': f'p-{int(time.time())}', 'status': 'filled', 'filled': decision.quantity, 'price': fill_price}
                else:
                    print(f"📝 PAPER FAIL: Insufficient Asset. Need {decision.quantity}, Have {current_qty}")
                    return None
                    
        # LIVE EXECUTION
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

    def _log_paper_trade(self, symbol, side, qty, price, fee):
        print(f"✅ PAPER {side}: {qty} {symbol} @ ${price:.2f} (Fee: ${fee:.2f})")
        trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "side": side.value,
            "quantity": qty,
            "price": price,
            "fee": fee
        }
        self.paper_state['history'].append(trade)
