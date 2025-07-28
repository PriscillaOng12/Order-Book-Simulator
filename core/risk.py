"""
risk.py
-------

Comprehensive risk management system for the order book simulator.
Implements sophisticated risk controls commonly found in modern trading systems:

- Position limits per trader/symbol
- Daily loss limits and circuit breakers
- Fat finger protection with NBBO checks
- Self-trade prevention algorithms
- Market impact limits
- Concentration risk monitoring
- Pre-trade and post-trade risk validation
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum, auto
import statistics
from collections import defaultdict, deque

from order import Order, Side, OrderType, Trade, MarketDataSnapshot


class RiskViolationType(Enum):
    POSITION_LIMIT = "position_limit"
    LOSS_LIMIT = "loss_limit"
    FAT_FINGER = "fat_finger"
    SELF_TRADE = "self_trade"
    MARKET_IMPACT = "market_impact"
    CONCENTRATION = "concentration"
    LIQUIDITY = "liquidity"
    VOLATILITY = "volatility"


@dataclass
class RiskLimits:
    """Risk limits configuration for a user or symbol."""
    max_position: int = 100000
    max_order_size: int = 10000
    max_daily_loss: float = 50000.0
    max_concentration_pct: float = 25.0  # % of total portfolio
    max_market_impact_bps: int = 50  # basis points
    fat_finger_multiplier: float = 2.0  # multiple of recent average price
    max_velocity: int = 1000  # max orders per minute


@dataclass
class PositionInfo:
    """Tracks position and P&L for a user/symbol combination."""
    quantity: int = 0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    daily_trades: int = 0
    last_update: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RiskViolation:
    """Represents a risk rule violation."""
    type: RiskViolationType
    message: str
    severity: str  # 'WARNING', 'ERROR', 'CRITICAL'
    user: str
    symbol: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CircuitBreaker:
    """Circuit breaker to halt trading during extreme market conditions."""
    
    def __init__(self, price_change_threshold: float = 0.10, 
                 time_window: int = 300, min_trades: int = 10):
        self.price_change_threshold = price_change_threshold
        self.time_window = time_window  # seconds
        self.min_trades = min_trades
        self.recent_trades: deque = deque()
        self.is_triggered = False
        self.trigger_time: Optional[datetime] = None
        
    def check_trigger(self, trade: Trade, market_open_price: float) -> bool:
        """Check if circuit breaker should trigger based on recent trades."""
        now = datetime.utcnow()
        
        # Clean old trades outside time window
        cutoff = now - timedelta(seconds=self.time_window)
        while self.recent_trades and self.recent_trades[0].timestamp < cutoff:
            self.recent_trades.popleft()
            
        self.recent_trades.append(trade)
        
        if len(self.recent_trades) < self.min_trades:
            return False
            
        # Calculate price change from market open
        price_change = abs(trade.price - market_open_price) / market_open_price
        
        if price_change > self.price_change_threshold:
            self.is_triggered = True
            self.trigger_time = now
            return True
            
        return False
        
    def reset(self):
        """Reset circuit breaker."""
        self.is_triggered = False
        self.trigger_time = None
        self.recent_trades.clear()


class NBBOTracker:
    """Tracks National Best Bid and Offer for fat finger detection."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
    def update(self, symbol: str, bid: Optional[float], ask: Optional[float]):
        """Update NBBO for a symbol."""
        if bid is not None and ask is not None:
            mid_price = (bid + ask) / 2.0
            self.price_history[symbol].append((datetime.utcnow(), mid_price))
            
    def get_recent_average(self, symbol: str, minutes: int = 5) -> Optional[float]:
        """Get recent average price for fat finger checks."""
        if symbol not in self.price_history:
            return None
            
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        recent_prices = [price for timestamp, price in self.price_history[symbol] 
                        if timestamp >= cutoff]
        
        if len(recent_prices) < 3:  # Need minimum data points
            return None
            
        return statistics.mean(recent_prices)
        
    def get_volatility(self, symbol: str, minutes: int = 30) -> float:
        """Calculate recent price volatility."""
        if symbol not in self.price_history:
            return 0.0
            
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        recent_prices = [price for timestamp, price in self.price_history[symbol] 
                        if timestamp >= cutoff]
        
        if len(recent_prices) < 10:
            return 0.0
            
        return statistics.stdev(recent_prices) / statistics.mean(recent_prices)


@dataclass
class RiskManager:
    """Comprehensive risk management system."""
    
    # Global limits
    global_limits: RiskLimits = field(default_factory=RiskLimits)
    
    # Per-user limits
    user_limits: Dict[str, RiskLimits] = field(default_factory=dict)
    
    # Per-symbol limits  
    symbol_limits: Dict[str, RiskLimits] = field(default_factory=dict)
    
    # Position tracking: user -> symbol -> position
    positions: Dict[str, Dict[str, PositionInfo]] = field(default_factory=lambda: defaultdict(dict))
    
    # Order velocity tracking: user -> timestamp list
    order_velocity: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=1000)))
    
    # Risk violations log
    violations: List[RiskViolation] = field(default_factory=list)
    
    # Market data for risk calculations
    nbbo_tracker: NBBOTracker = field(default_factory=NBBOTracker)
    
    # Circuit breakers per symbol
    circuit_breakers: Dict[str, CircuitBreaker] = field(default_factory=lambda: defaultdict(CircuitBreaker))
    
    # Self-trade prevention
    user_orders: Dict[str, Set[int]] = field(default_factory=lambda: defaultdict(set))
    
    def get_effective_limits(self, user: str, symbol: str) -> RiskLimits:
        """Get the most restrictive limits for user/symbol combination."""
        limits = [self.global_limits]
        
        if user in self.user_limits:
            limits.append(self.user_limits[user])
            
        if symbol in self.symbol_limits:
            limits.append(self.symbol_limits[symbol])
            
        # Take the minimum of all applicable limits
        return RiskLimits(
            max_position=min(l.max_position for l in limits),
            max_order_size=min(l.max_order_size for l in limits),
            max_daily_loss=min(l.max_daily_loss for l in limits),
            max_concentration_pct=min(l.max_concentration_pct for l in limits),
            max_market_impact_bps=min(l.max_market_impact_bps for l in limits),
            fat_finger_multiplier=min(l.fat_finger_multiplier for l in limits),
            max_velocity=min(l.max_velocity for l in limits)
        )

    def check_pre_trade_risk(self, order: Order, market_snapshot: Optional[MarketDataSnapshot] = None) -> Optional[RiskViolation]:
        """Comprehensive pre-trade risk validation."""
        user = order.owner or 'anonymous'
        limits = self.get_effective_limits(user, order.symbol)
        
        # 1. Order size validation
        if order.quantity <= 0 or order.quantity > limits.max_order_size:
            return RiskViolation(
                type=RiskViolationType.FAT_FINGER,
                message=f"Order size {order.quantity} exceeds limit {limits.max_order_size}",
                severity="ERROR",
                user=user,
                symbol=order.symbol
            )
        
        # 2. Fat finger price checks
        if order.price is not None and market_snapshot:
            violation = self._check_fat_finger(order, market_snapshot, limits)
            if violation:
                return violation
        
        # 3. Position limit checks
        violation = self._check_position_limits(order, limits)
        if violation:
            return violation
            
        # 4. Order velocity checks
        violation = self._check_order_velocity(order, limits)
        if violation:
            return violation
            
        # 5. Self-trade prevention
        violation = self._check_self_trade(order)
        if violation:
            return violation
            
        # 6. Circuit breaker check
        if self.circuit_breakers[order.symbol].is_triggered:
            return RiskViolation(
                type=RiskViolationType.VOLATILITY,
                message=f"Circuit breaker active for {order.symbol}",
                severity="CRITICAL",
                user=user,
                symbol=order.symbol
            )
        
        return None

    def _check_fat_finger(self, order: Order, market_snapshot: MarketDataSnapshot, 
                         limits: RiskLimits) -> Optional[RiskViolation]:
        """Check for fat finger price errors."""
        if order.type == OrderType.MARKET:
            return None
            
        recent_avg = self.nbbo_tracker.get_recent_average(order.symbol)
        if recent_avg is None:
            return None
            
        price_deviation = abs(order.price - recent_avg) / recent_avg
        max_deviation = (limits.fat_finger_multiplier - 1.0)
        
        if price_deviation > max_deviation:
            return RiskViolation(
                type=RiskViolationType.FAT_FINGER,
                message=f"Price {order.price} deviates {price_deviation:.2%} from recent average {recent_avg:.2f}",
                severity="ERROR",
                user=order.owner or 'anonymous',
                symbol=order.symbol
            )
        
        return None

    def _check_position_limits(self, order: Order, limits: RiskLimits) -> Optional[RiskViolation]:
        """Check position limits."""
        user = order.owner or 'anonymous'
        
        if user not in self.positions or order.symbol not in self.positions[user]:
            current_position = 0
        else:
            current_position = self.positions[user][order.symbol].quantity
            
        # Calculate new position after order
        if order.side == Side.BUY:
            new_position = current_position + order.quantity
        else:
            new_position = current_position - order.quantity
            
        if abs(new_position) > limits.max_position:
            return RiskViolation(
                type=RiskViolationType.POSITION_LIMIT,
                message=f"Position limit exceeded. Current: {current_position}, New: {new_position}, Limit: {limits.max_position}",
                severity="ERROR",
                user=user,
                symbol=order.symbol
            )
        
        return None

    def _check_order_velocity(self, order: Order, limits: RiskLimits) -> Optional[RiskViolation]:
        """Check order submission velocity."""
        user = order.owner or 'anonymous'
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=1)
        
        # Clean old timestamps
        velocity_queue = self.order_velocity[user]
        while velocity_queue and velocity_queue[0] < cutoff:
            velocity_queue.popleft()
            
        # Add current order timestamp
        velocity_queue.append(now)
        
        if len(velocity_queue) > limits.max_velocity:
            return RiskViolation(
                type=RiskViolationType.LIQUIDITY,
                message=f"Order velocity {len(velocity_queue)} exceeds limit {limits.max_velocity} per minute",
                severity="WARNING",
                user=user,
                symbol=order.symbol
            )
        
        return None

    def _check_self_trade(self, order: Order) -> Optional[RiskViolation]:
        """Check for potential self-trades."""
        user = order.owner or 'anonymous'
        
        # This is a simplified check - in practice you'd check against 
        # opposite side orders for the same user
        if order.id in self.user_orders[user]:
            return RiskViolation(
                type=RiskViolationType.SELF_TRADE,
                message=f"Potential self-trade detected for order {order.id}",
                severity="WARNING",
                user=user,
                symbol=order.symbol
            )
        
        return None

    def update_position(self, user: str, symbol: str, trade: Trade, is_buyer: bool):
        """Update position and P&L after a trade."""
        if user not in self.positions:
            self.positions[user] = {}
            
        if symbol not in self.positions[user]:
            self.positions[user][symbol] = PositionInfo()
            
        position = self.positions[user][symbol]
        
        # Update position and average cost
        if is_buyer:
            # Buying - add to position
            if position.quantity >= 0:
                # Same direction - update average cost
                total_cost = position.quantity * position.avg_cost + trade.quantity * trade.price
                position.quantity += trade.quantity
                position.avg_cost = total_cost / position.quantity if position.quantity > 0 else 0.0
            else:
                # Covering short position
                if trade.quantity >= abs(position.quantity):
                    # Full cover plus additional long
                    realized = abs(position.quantity) * (position.avg_cost - trade.price)
                    position.realized_pnl += realized
                    remaining_qty = trade.quantity - abs(position.quantity)
                    position.quantity = remaining_qty
                    position.avg_cost = trade.price if remaining_qty > 0 else 0.0
                else:
                    # Partial cover
                    realized = trade.quantity * (position.avg_cost - trade.price)
                    position.realized_pnl += realized
                    position.quantity += trade.quantity  # position.quantity is negative
        else:
            # Selling - subtract from position
            if position.quantity <= 0:
                # Same direction (short) - update average cost
                total_cost = abs(position.quantity) * position.avg_cost + trade.quantity * trade.price
                position.quantity -= trade.quantity
                position.avg_cost = total_cost / abs(position.quantity) if position.quantity < 0 else 0.0
            else:
                # Selling long position
                if trade.quantity >= position.quantity:
                    # Full sale plus additional short
                    realized = position.quantity * (trade.price - position.avg_cost)
                    position.realized_pnl += realized
                    remaining_qty = trade.quantity - position.quantity
                    position.quantity = -remaining_qty
                    position.avg_cost = trade.price if remaining_qty > 0 else 0.0
                else:
                    # Partial sale
                    realized = trade.quantity * (trade.price - position.avg_cost)
                    position.realized_pnl += realized
                    position.quantity -= trade.quantity
        
        position.daily_trades += 1
        position.last_update = datetime.utcnow()

    def check_post_trade_risk(self, trade: Trade, market_open_price: float) -> List[RiskViolation]:
        """Post-trade risk checks including circuit breakers."""
        violations = []
        
        # Check circuit breaker
        breaker = self.circuit_breakers[trade.symbol]
        if breaker.check_trigger(trade, market_open_price):
            violations.append(RiskViolation(
                type=RiskViolationType.VOLATILITY,
                message=f"Circuit breaker triggered for {trade.symbol} - trading halted",
                severity="CRITICAL",
                user="SYSTEM",
                symbol=trade.symbol
            ))
        
        return violations

    def update_market_data(self, snapshot: MarketDataSnapshot):
        """Update market data for risk calculations."""
        self.nbbo_tracker.update(snapshot.symbol, snapshot.best_bid, snapshot.best_ask)

    def get_risk_summary(self, user: str) -> Dict:
        """Get comprehensive risk summary for a user."""
        user_positions = self.positions.get(user, {})
        
        total_pnl = sum(pos.realized_pnl + pos.unrealized_pnl for pos in user_positions.values())
        total_exposure = sum(abs(pos.quantity * pos.avg_cost) for pos in user_positions.values())
        
        recent_violations = [v for v in self.violations 
                           if v.user == user and 
                           (datetime.utcnow() - v.timestamp).total_seconds() < 3600]
        
        return {
            'user': user,
            'total_pnl': total_pnl,
            'total_exposure': total_exposure,
            'position_count': len(user_positions),
            'recent_violations': len(recent_violations),
            'positions': {symbol: {
                'quantity': pos.quantity,
                'avg_cost': pos.avg_cost,
                'realized_pnl': pos.realized_pnl,
                'unrealized_pnl': pos.unrealized_pnl,
                'daily_trades': pos.daily_trades
            } for symbol, pos in user_positions.items()}
        }