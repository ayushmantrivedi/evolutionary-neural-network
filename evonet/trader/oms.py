
import logging
import uuid
from enum import Enum

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

class OrderState(Enum):
    OPEN = "OPEN"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

class Order:
    def __init__(self, symbol, side, qty, type, price=None):
        self.id = str(uuid.uuid4())[:8]
        self.symbol = symbol
        self.side = side # 'BUY' or 'SELL'
        self.qty = qty
        self.type = type
        self.price = price # Limit Price
        self.state = OrderState.OPEN
        self.fill_price = 0.0
        self.created_at_tick = 0
        
    def __repr__(self):
        return f"<Order {self.id} {self.side} {self.qty} @ {self.price} | {self.state.value}>"

class OrderManagementSystem:
    """
    Simulates a Real-World Exchange Connection.
    - Handles Latency (Orders don't fill instantly).
    - Handles Slippage (Market orders pay the spread).
    - Tracks Open Orders.
    """
    def __init__(self, slippage_model="ATR"):
        self.open_orders = []
        self.filled_orders = []
        self.slippage_model = slippage_model
        # Latency Simulation: Orders take 1 tick to arrive at exchange, 1 tick to fill?
        # For H1 simulation, we can assume 'Next Tick Execution' or 'Close Price Execution'.
        
    def submit_order(self, order):
        # logging.info(f"OMS: Received {order}")
        self.open_orders.append(order)
        return order
        
    def match_orders(self, current_bar):
        """
        Matches open orders against the current market bar (High/Low/Close).
        Should be called on every Environment Step.
        """
        fills = []
        kept_orders = []
        
        current_price = current_bar['Close']
        high = current_bar['High']
        low = current_bar['Low']
        atr = current_bar['ATR']
        
        for order in self.open_orders:
            filled = False
            avg_price = 0.0
            
            # --- MARKET ORDERS ---
            if order.type == OrderType.MARKET:
                # Fill immediately at current price +/- Slippage
                slippage = 0.0
                if self.slippage_model == "ATR":
                    # Slippage is roughly 10% of the volatility of that candle
                    # If ATR is $100, you might slip $10.
                    slippage = atr * 0.10
                
                # Buy pays more, Sell gets less
                if order.side == "BUY":
                    avg_price = current_price + slippage
                else:
                    avg_price = current_price - slippage
                    
                filled = True
                
            # --- LIMIT ORDERS ---
            elif order.type == OrderType.LIMIT:
                # Check if price touched the limit
                if order.side == "BUY":
                    if low <= order.price:
                        # Filled!
                        avg_price = order.price # Assert we got limit (or better)
                        filled = True
                elif order.side == "SELL":
                    if high >= order.price:
                        # Filled!
                        avg_price = order.price
                        filled = True
            
            if filled:
                order.state = OrderState.FILLED
                order.fill_price = avg_price
                self.filled_orders.append(order)
                fills.append(order)
            else:
                kept_orders.append(order)
                
        self.open_orders = kept_orders
        return fills
