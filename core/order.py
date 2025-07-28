"""
order.py
-----------

Enhanced order management system supporting all major order types used in 
modern electronic trading. This module defines the core data structures 
used by the order book simulator, including advanced order types like 
iceberg, stop-limit, and various time-in-force policies.

The simulator supports:
- Market and Limit orders
- Stop and Stop-Limit orders  
- Iceberg orders with hidden quantity
- Time-in-force policies (GTC, IOC, FOK, DAY)
- Order state management and lifecycle tracking
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, time
from typing import Optional, Dict, Any
import uuid


class Side(Enum):
    BUY = auto()
    SELL = auto()


class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()
    ICEBERG = auto()


class OrderStatus(Enum):
    NEW = auto()
    PENDING = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()


class TimeInForce(Enum):
    """Enumerates supported time-in-force policies.

    GTC: Good Till Cancelled. The order remains on the book until
         cancelled by the user or filled.
    IOC: Immediate Or Cancel. Execute any portion that can be
         immediately filled; cancel the remainder.
    FOK: Fill Or Kill. Execute the entire order immediately or
         cancel it entirely.
    DAY: Good for Day. Order expires at market close.
    """
    GTC = auto()
    IOC = auto()
    FOK = auto()
    DAY = auto()


@dataclass
class Order:
    """Enhanced order representation supporting advanced order types.

    Attributes:
        id: A unique identifier for the order (assigned by the client).
        symbol: The trading symbol this order applies to.
        side: Either Side.BUY or Side.SELL.
        type: Order type (MARKET, LIMIT, STOP, STOP_LIMIT, ICEBERG).
        quantity: The total quantity of the order.
        price: The limit price for limit orders; ignored for market orders.
        stop_price: Trigger price for stop orders.
        display_qty: For iceberg orders, the visible quantity.
        timestamp: When the order was received.
        tif: Time-in-force policy.
        owner: User/trader who submitted the order.
        status: Current order status.
        session_id: Trading session identifier.
        tags: Custom metadata for the order.
    """

    id: int
    symbol: str
    side: Side
    type: OrderType
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    display_qty: Optional[int] = None  # For iceberg orders
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tif: TimeInForce = field(default=TimeInForce.GTC)
    owner: Optional[str] = None
    status: OrderStatus = field(default=OrderStatus.NEW)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tags: Dict[str, Any] = field(default_factory=dict)
    
    # Internal tracking fields
    remaining: int = field(init=False)
    filled_qty: int = field(default=0)
    avg_fill_price: float = field(default=0.0)
    last_update: datetime = field(init=False)
    parent_order_id: Optional[int] = None  # For iceberg slices
    hidden_qty: int = field(init=False, default=0)  # Remaining hidden quantity for iceberg

    def __post_init__(self):
        self.remaining = self.quantity
        self.last_update = self.timestamp
        
        # Set up iceberg order specifics
        if self.type == OrderType.ICEBERG:
            if self.display_qty is None:
                self.display_qty = min(100, self.quantity)  # Default display size
            self.hidden_qty = self.quantity - self.display_qty
            self.remaining = self.display_qty  # Only display qty is initially available

    def fill(self, qty: int, price: float) -> None:
        """Fill the order with the specified quantity at given price."""
        if qty < 0:
            raise ValueError("Fill quantity must be non-negative")
        if qty > self.remaining:
            raise ValueError("Cannot fill more than remaining quantity")
        
        # Update fill statistics
        total_filled_value = self.filled_qty * self.avg_fill_price
        total_filled_value += qty * price
        self.filled_qty += qty
        self.avg_fill_price = total_filled_value / self.filled_qty if self.filled_qty > 0 else 0.0
        
        self.remaining -= qty
        self.last_update = datetime.utcnow()
        
        # Update status
        if self.remaining == 0:
            if self.type == OrderType.ICEBERG and self.hidden_qty > 0:
                # Refresh iceberg display
                self._refresh_iceberg_display()
                self.status = OrderStatus.PARTIALLY_FILLED
            else:
                self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def _refresh_iceberg_display(self) -> None:
        """Refresh the display quantity for iceberg orders."""
        if self.type != OrderType.ICEBERG or self.hidden_qty <= 0:
            return
            
        # Calculate next display slice
        next_display = min(self.display_qty, self.hidden_qty)
        self.hidden_qty -= next_display
        self.remaining = next_display
        self.timestamp = datetime.utcnow()  # Reset time priority

    def cancel(self) -> None:
        """Cancel the order."""
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            raise ValueError(f"Cannot cancel order in status {self.status}")
        self.status = OrderStatus.CANCELLED
        self.last_update = datetime.utcnow()

    def reject(self, reason: str = "") -> None:
        """Reject the order."""
        self.status = OrderStatus.REJECTED
        self.last_update = datetime.utcnow()
        if reason:
            self.tags['rejection_reason'] = reason

    def is_triggered(self, current_price: float) -> bool:
        """Check if a stop order should be triggered."""
        if self.type not in [OrderType.STOP, OrderType.STOP_LIMIT]:
            return True
            
        if self.stop_price is None:
            return True
            
        if self.side == Side.BUY:
            return current_price >= self.stop_price
        else:
            return current_price <= self.stop_price

    @property
    def is_filled(self) -> bool:
        """Return True if order is completely filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_active(self) -> bool:
        """Return True if order is still active in the market."""
        return self.status in [OrderStatus.NEW, OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]

    @property
    def total_quantity_remaining(self) -> int:
        """For iceberg orders, return total remaining including hidden."""
        if self.type == OrderType.ICEBERG:
            return self.remaining + self.hidden_qty
        return self.remaining

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary for serialization."""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side.name,
            'type': self.type.name,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'display_qty': self.display_qty,
            'timestamp': self.timestamp.isoformat(),
            'tif': self.tif.name,
            'owner': self.owner,
            'status': self.status.name,
            'remaining': self.remaining,
            'filled_qty': self.filled_qty,
            'avg_fill_price': self.avg_fill_price,
            'session_id': self.session_id,
            'tags': self.tags
        }


@dataclass
class Trade:
    """Represents a trade execution resulting from matching orders.

    Attributes:
        id: Unique trade identifier.
        price: The execution price.
        quantity: The quantity traded.
        timestamp: When the trade occurred.
        buy_order_id: Identifier of the buy order involved.
        sell_order_id: Identifier of the sell order involved.
        symbol: Trading symbol.
        aggressor_side: Which side was the aggressor (liquidity taker).
        trade_type: Type of trade (normal, opening, closing, etc.).
    """
    id: str
    price: float
    quantity: int
    timestamp: datetime
    buy_order_id: int
    sell_order_id: int
    symbol: str
    aggressor_side: Side
    trade_type: str = "NORMAL"
    settlement_date: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary for serialization."""
        return {
            'id': self.id,
            'price': self.price,
            'quantity': self.quantity,
            'timestamp': self.timestamp.isoformat(),
            'buy_order_id': self.buy_order_id,
            'sell_order_id': self.sell_order_id,
            'symbol': self.symbol,
            'aggressor_side': self.aggressor_side.name,
            'trade_type': self.trade_type
        }


@dataclass
class MarketDataSnapshot:
    """Market data snapshot containing order book and trade information."""
    symbol: str
    timestamp: datetime
    best_bid: Optional[float]
    best_ask: Optional[float]
    bid_size: int
    ask_size: int
    last_trade_price: Optional[float]
    last_trade_qty: Optional[int]
    total_bid_qty: int
    total_ask_qty: int
    trade_count: int
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid-market price."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None


# Market quality metrics
@dataclass
class MarketQualityMetrics:
    """Metrics for analyzing market quality and microstructure."""
    symbol: str
    timestamp: datetime
    
    # Liquidity metrics
    avg_spread_bps: float
    depth_at_touch: int
    depth_5_levels: int
    
    # Price discovery metrics
    price_efficiency: float
    volatility: float
    
    # Trading activity
    trade_count: int
    total_volume: int
    avg_trade_size: float
    
    # Market impact
    temp_impact_bps: float
    perm_impact_bps: float