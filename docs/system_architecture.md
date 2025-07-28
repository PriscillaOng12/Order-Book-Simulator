# 🏗️ System Architecture & Component Design

This document outlines the high-level architecture of the order book simulator, explaining how different components interact to create a cohesive trading system. The design emphasizes modularity, performance, and reliability - key requirements for financial systems.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Component Design](#component-design)
- [Data Flow](#data-flow)
- [API Design](#api-design)
- [Performance Architecture](#performance-architecture)
- [Deployment Strategy](#deployment-strategy)
- [Fault Tolerance](#fault-tolerance)
- [Monitoring & Observability](#monitoring--observability) 

## Architecture Overview

The system follows a **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────┐
│                CLIENT LAYER                 │
│  Web UI, REST API clients, WebSocket feeds  │
└─────────────────┬───────────────────────────┘
                  │ HTTP/WebSocket
┌─────────────────▼───────────────────────────┐
│              API GATEWAY                    │
│    Flask server, request routing, CORS      │
└─────────────────┬───────────────────────────┘
                  │ Internal API calls
┌─────────────────▼───────────────────────────┐
│             SERVICE LAYER                   │
│ ┌─────────────┬─────────────┬─────────────┐ │
│ │    Risk     │   Market    │ Performance │ │
│ │  Manager    │   Maker     │  Analyzer   │ │
│ └─────────────┴─────────────┴─────────────┘ │
└─────────────────┬───────────────────────────┘
                  │ Order processing
┌─────────────────▼───────────────────────────┐
│              CORE ENGINE                    │
│         Matching Engine + Order Books       │
└─────────────────┬───────────────────────────┘
                  │ Data persistence
┌─────────────────▼───────────────────────────┐
│              DATA LAYER                     │
│    In-memory data, Event logs, Metrics      │
└─────────────────────────────────────────────┘
```

### Design Principles

1. **Single Responsibility**: Each component has one clear purpose
2. **Loose Coupling**: Components communicate through well-defined interfaces
3. **High Cohesion**: Related functionality is grouped together
4. **Performance First**: Sub-millisecond latency requirements drive design decisions
5. **Fail-Safe**: System degrades gracefully under failure conditions

## Component Design

### Core Components

#### 1. Matching Engine (`engine.py`)
The heart of the system - processes all order matching logic.

```python
class MatchingEngine:
    """
    Single-threaded matching engine for deterministic execution
    """
    def __init__(self):
        # Order books for each symbol
        self.order_books = {
            symbol: OrderBook(symbol) 
            for symbol in ["AAPL", "GOOGL", "MSFT", "TSLA"]
        }
        
        # Performance optimization
        self.order_pool = OrderPool(100000)
        self.trade_pool = TradePool(50000)
        
        # Event sourcing for audit trail
        self.event_store = EventStore()
        
        # Metrics collection
        self.metrics = MetricsCollector()
    
    def process_order(self, order: Order) -> ProcessingResult:
        """Main entry point for order processing"""
        start_time = time.perf_counter_ns()
        
        try:
            # Get the appropriate order book
            book = self.order_books[order.symbol]
            
            # Execute matching
            trades = book.match_order(order)
            
            # Record events for audit
            self.event_store.record_order(order)
            for trade in trades:
                self.event_store.record_trade(trade)
            
            # Update metrics
            latency = time.perf_counter_ns() - start_time
            self.metrics.record_latency("order_processing", latency)
            
            return ProcessingResult(
                success=True,
                trades=trades,
                latency_ns=latency
            )
            
        except Exception as e:
            self.metrics.increment_counter("processing_errors")
            return ProcessingResult(
                success=False,
                error=str(e),
                latency_ns=time.perf_counter_ns() - start_time
            )
```

**Key Design Decisions:**
- **Single-threaded**: Eliminates race conditions and ensures deterministic execution
- **Symbol isolation**: Each symbol has its own order book to prevent cross-contamination
- **Object pooling**: Reuses objects to minimize garbage collection pressure
- **Event sourcing**: Every order and trade is logged for recovery and audit

#### 2. API Gateway (`web_server.py`)
Handles all external communication and request routing.

```python
class APIGateway:
    """
    Flask-based API gateway with WebSocket support
    """
    def __init__(self, matching_engine: MatchingEngine):
        self.app = Flask(__name__)
        self.matching_engine = matching_engine
        self.websocket_manager = WebSocketManager()
        
        # CORS setup for web clients
        CORS(self.app)
        
        # Rate limiting
        self.rate_limiter = RateLimiter(
            default_limits=["1000 per minute", "100 per second"]
        )
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Configure REST endpoints"""
        
        @self.app.route('/api/submit', methods=['POST'])
        @self.rate_limiter.limit("10 per second")
        def submit_order():
            try:
                # Parse and validate request
                order_data = request.get_json()
                order = self._parse_order(order_data)
                
                # Process through matching engine
                result = self.matching_engine.process_order(order)
                
                # Broadcast market data updates
                if result.trades:
                    self._broadcast_trades(result.trades)
                
                return jsonify({
                    "success": True,
                    "order_id": order.order_id,
                    "trades": [trade.to_dict() for trade in result.trades],
                    "latency_us": result.latency_ns // 1000
                })
                
            except ValidationError as e:
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                return jsonify({"error": "Internal server error"}), 500
```

#### 3. Risk Manager (`risk.py`)
Implements pre-trade and post-trade risk controls.

```python
class RiskManager:
    """
    Real-time risk management for trading operations
    """
    def __init__(self):
        # Risk limits per symbol
        self.position_limits = {
            "AAPL": {"max_position": 10000, "max_order_size": 1000},
            "GOOGL": {"max_position": 5000, "max_order_size": 500},
            # ... other symbols
        }
        
        # Current positions per user
        self.positions = defaultdict(lambda: defaultdict(int))
        
        # Risk metrics tracking
        self.daily_volume = defaultdict(int)
        self.recent_trades = defaultdict(deque)
        
    def validate_order(self, order: Order, user_id: str) -> RiskDecision:
        """Pre-trade risk validation (must complete in <50μs)"""
        
        # 1. Position limit check
        current_position = self.positions[user_id][order.symbol]
        new_position = current_position + order.signed_quantity()
        
        max_position = self.position_limits[order.symbol]["max_position"]
        if abs(new_position) > max_position:
            return RiskDecision(
                decision="REJECT",
                reason=f"Position limit exceeded: {abs(new_position)} > {max_position}"
            )
        
        # 2. Order size check (fat finger protection)
        max_order = self.position_limits[order.symbol]["max_order_size"]
        if order.quantity > max_order:
            return RiskDecision(
                decision="REJECT", 
                reason=f"Order size too large: {order.quantity} > {max_order}"
            )
        
        # 3. Percentage of volume check
        daily_vol = self.daily_volume[order.symbol]
        if daily_vol > 0 and order.quantity > 0.05 * daily_vol:
            return RiskDecision(
                decision="HOLD",
                reason="Order size exceeds 5% of daily volume"
            )
        
        # 4. Self-trade prevention
        if self._would_self_trade(order, user_id):
            return RiskDecision(
                decision="REJECT",
                reason="Self-trade prevention triggered"
            )
        
        return RiskDecision(decision="ACCEPT")
    
    def update_position(self, trade: Trade):
        """Post-trade position updates"""
        buyer_id = trade.buy_order.user_id
        seller_id = trade.sell_order.user_id
        
        # Update positions
        self.positions[buyer_id][trade.symbol] += trade.quantity
        self.positions[seller_id][trade.symbol] -= trade.quantity
        
        # Update daily volume
        self.daily_volume[trade.symbol] += trade.quantity
        
        # Check for post-trade risk breaches
        self._check_post_trade_risk(buyer_id, trade.symbol)
        self._check_post_trade_risk(seller_id, trade.symbol)
```

#### 4. Market Maker (`market_maker.py`)
Provides automated liquidity with adaptive strategies.

```python
class MarketMaker:
    """
    Automated market making with risk management
    """
    def __init__(self, matching_engine: MatchingEngine):
        self.matching_engine = matching_engine
        self.positions = defaultdict(int)
        
        # Strategy parameters
        self.base_spread = 0.01  # 1 cent
        self.max_inventory = 1000
        self.quote_size = 100
        
        # Market analysis
        self.volatility_tracker = VolatilityTracker()
        self.flow_analyzer = OrderFlowAnalyzer()
        
        # Active orders tracking
        self.active_quotes = {}  # symbol -> {bid_order_id, ask_order_id}
    
    def generate_quotes(self, symbol: str) -> Tuple[Order, Order]:
        """Generate bid/ask quotes based on current market conditions"""
        
        # Get current market state
        book = self.matching_engine.order_books[symbol]
        mid_price = book.mid_price()
        
        if not mid_price:
            return None, None  # No market data available
        
        # Calculate adaptive spread
        base_spread = self.base_spread
        volatility = self.volatility_tracker.get_volatility(symbol)
        vol_adjustment = volatility * 0.5  # Widen spread in high volatility
        
        # Inventory skew
        inventory = self.positions[symbol]
        inventory_skew = (inventory / self.max_inventory) * 0.005
        
        # Final spread calculation
        spread = base_spread + vol_adjustment
        
        # Generate quotes
        bid_price = mid_price - (spread / 2) + inventory_skew
        ask_price = mid_price + (spread / 2) + inventory_skew
        
        bid_order = Order(
            symbol=symbol,
            side=Side.BUY,
            price=round(bid_price, 2),
            quantity=self.quote_size,
            order_type=OrderType.LIMIT,
            user_id="market_maker"
        )
        
        ask_order = Order(
            symbol=symbol,
            side=Side.SELL,
            price=round(ask_price, 2),
            quantity=self.quote_size,
            order_type=OrderType.LIMIT,
            user_id="market_maker"
        )
        
        return bid_order, ask_order
```

## Data Flow

The system processes orders through a well-defined pipeline:

```
Order Submission Flow:
┌────────────┐    ┌─────────────┐    ┌──────────────┐
│   Client   │───▶│ API Gateway │───▶│ Rate Limiter │
└────────────┘    └─────────────┘    └──────┬───────┘
                                             │
┌────────────┐    ┌─────────────┐    ┌──────▼───────┐
│Event Store │◄───│Risk Manager │◄───│ Request      │
└────────────┘    └─────────────┘    │ Validator    │
                                     └──────┬───────┘
┌────────────┐    ┌─────────────┐    ┌──────▼───────┐
│Market Data │◄───│   Order     │◄───│ Matching     │
│Broadcast   │    │   Book      │    │ Engine       │
└────────────┘    └─────────────┘    └──────────────┘

Typical Processing Time:
- Request validation: 20μs
- Risk check: 45μs  
- Order matching: 180μs
- Market data update: 30μs
- Response generation: 15μs
Total: ~290μs average
```

### WebSocket Data Flow

Real-time market data is distributed via WebSocket connections:

```python
class WebSocketManager:
    """Manages WebSocket connections for real-time data"""
    
    def __init__(self):
        self.connections = set()
        self.subscriptions = defaultdict(set)  # symbol -> {connections}
        
    async def broadcast_trade(self, trade: Trade):
        """Broadcast trade to all subscribers"""
        message = {
            "type": "trade",
            "symbol": trade.symbol,
            "price": trade.price,
            "quantity": trade.quantity,
            "timestamp": trade.timestamp,
            "side": "BUY" if trade.aggressor_side == Side.BUY else "SELL"
        }
        
        # Send to all subscribers of this symbol
        subscribers = self.subscriptions[trade.symbol]
        if subscribers:
            await asyncio.gather(*[
                ws.send(json.dumps(message)) 
                for ws in subscribers
            ], return_exceptions=True)
    
    async def broadcast_level2_update(self, symbol: str, book_data: Dict):
        """Broadcast order book depth updates"""
        message = {
            "type": "level2_update",
            "symbol": symbol,
            "bids": book_data["bids"][:10],  # Top 10 levels
            "asks": book_data["asks"][:10],
            "timestamp": time.time_ns()
        }
        
        subscribers = self.subscriptions[symbol]
        if subscribers:
            await asyncio.gather(*[
                ws.send(json.dumps(message))
                for ws in subscribers
            ], return_exceptions=True)
```

## API Design

### REST Endpoints

The system exposes a clean REST API for order management and data retrieval:

| Endpoint | Method | Purpose | Response Time |
|----------|--------|---------|---------------|
| `/api/submit` | POST | Submit new order | <1ms |
| `/api/cancel/{order_id}` | DELETE | Cancel existing order | <500μs |
| `/api/book/{symbol}` | GET | Get order book depth | <100μs |
| `/api/trades/{symbol}` | GET | Get recent trades | <200μs |
| `/api/performance/system` | GET | System metrics | <50μs |
| `/api/user/{user_id}/positions` | GET | User positions | <100μs |

### Example API Usage

```python
# Submit a limit order
response = requests.post('/api/submit', json={
    'symbol': 'AAPL',
    'side': 'buy',
    'order_type': 'limit',
    'price': 150.25,
    'quantity': 100,
    'user_id': 'trader123'
})

# Response format
{
    "success": true,
    "order_id": "ORD-123456789",
    "status": "FILLED",
    "trades": [
        {
            "trade_id": "TRD-987654321",
            "price": 150.24,
            "quantity": 100,
            "timestamp": "2024-01-15T09:30:01.123456Z"
        }
    ],
    "remaining_quantity": 0,
    "latency_us": 287
}
```

### WebSocket API

```javascript
// Connect to real-time feed
const ws = new WebSocket('ws://localhost:8765');

// Subscribe to AAPL data
ws.send(JSON.stringify({
    action: 'subscribe',
    symbol: 'AAPL',
    feed_types: ['trades', 'level2']
}));

// Handle incoming messages
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'trade':
            console.log(`Trade: ${data.quantity} @ ${data.price}`);
            break;
            
        case 'level2_update':
            updateOrderBookDisplay(data.bids, data.asks);
            break;
    }
};
```

## Performance Architecture

### Single-Threaded Design

The system uses a **single-threaded core** with async I/O at the boundaries:

```
Threading Model:
┌─────────────────────────────────────────┐
│              MAIN THREAD                │
│  ┌───────────────────────────────────┐  │
│  │        MATCHING ENGINE            │  │
│  │    (Single-threaded, lock-free)   │  │
│  └───────────────────────────────────┘  │
│                    ▲                    │
│                    │                    │
│  ┌─────────────────┴──────────────--─┐  │
│  │         ASYNC I/O POOL            │  │
│  │  ┌──────────┐  ┌──────────────┐   │  │
│  │  │ WebSocket│  │ HTTP Request │   │  │
│  │  │ Handler  │  │   Handler    │   │  │
│  │  └──────────┘  └──────────────┘   │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘

Advantages:
- No lock contention
- Deterministic execution order
- Simplified debugging
- Predictable performance
```

### Memory Architecture

```python
class MemoryManager:
    """Optimized memory management for low-latency trading"""
    
    def __init__(self):
        # Pre-allocated object pools
        self.order_pool = ObjectPool(Order, size=100000)
        self.trade_pool = ObjectPool(Trade, size=50000)
        
        # Memory-mapped files for persistent data
        self.trade_log = MemoryMappedFile("trades.log", size=1024*1024*100)  # 100MB
        
        # Circular buffers for real-time data
        self.price_buffer = CircularBuffer(size=10000)
        self.volume_buffer = CircularBuffer(size=10000)
    
    def get_memory_stats(self) -> Dict:
        """Get current memory usage statistics"""
        return {
            "order_pool_utilization": self.order_pool.utilization(),
            "trade_pool_utilization": self.trade_pool.utilization(),
            "heap_size_mb": self._get_heap_size() / (1024*1024),
            "cache_hit_rate": self._get_cache_hit_rate()
        }
```

## Deployment Strategy

### Single-Node Deployment (Current)

```yaml
# docker-compose.yml
version: '3.8'
services:
  orderbook:
    build: .
    ports:
      - "8080:8080"  # HTTP API
      - "8765:8765"  # WebSocket
    environment:
      - LOG_LEVEL=INFO
      - MAX_CONNECTIONS=1000
      - ENABLE_METRICS=true
    resources:
      limits:
        memory: 2G
        cpus: '2.0'
      reservations:
        memory: 512M
        cpus: '1.0'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Horizontal Scaling (Future)

```
Distributed Architecture:
┌─────────────────────────────────────────┐
│            LOAD BALANCER                │
│         (Symbol-based routing)          │
└─────────────┬───────────────────────────┘
              │
    ┌─────────┼─────────┐
    │         │         │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐
│Shard 1│ │Shard 2│ │Shard 3│
│ AAPL  │ │ GOOGL │ │ MSFT  │
│ TSLA  │ │ NVDA  │ │ NFLX  │
└───┬───┘ └───┬───┘ └───┬───┘
    │         │         │
    └─────────┼─────────┘
              │
    ┌─────────▼─────────┐
    │ CONSENSUS LAYER   │
    │ (Cross-shard      │
    │  coordination)    │
    └───────────────────┘
    
Projected Capacity:
- Single node: 114K orders/sec
- 3-node cluster: 300K+ orders/sec  
- 10-node cluster: 1M+ orders/sec
```

## Fault Tolerance

### Circuit Breakers

```python
class CircuitBreaker:
    """Prevent cascade failures in distributed system"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset failure count
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e
```

### Graceful Degradation

```python
class GracefulDegradation:
    """Handle system failures with reduced functionality"""
    
    def __init__(self):
        self.degraded_services = set()
    
    def handle_service_failure(self, service_name: str):
        """React to service failure with appropriate fallback"""
        
        self.degraded_services.add(service_name)
        
        if service_name == "websocket_feed":
            # Fall back to HTTP polling
            self.enable_polling_mode()
            self.notify_clients("Switched to polling mode due to WebSocket issues")
            
        elif service_name == "risk_manager":
            # Use conservative risk settings
            self.enable_conservative_mode()
            self.alert_operations("Risk manager degraded - using conservative limits")
            
        elif service_name == "market_maker":
            # Continue with user orders only
            self.disable_market_making()
            self.log_warning("Market maker disabled - liquidity may be reduced")
    
    def attempt_recovery(self):
        """Try to restore failed services"""
        for service in list(self.degraded_services):
            if self.test_service_health(service):
                self.restore_service(service)
                self.degraded_services.remove(service)
```

## Monitoring & Observability

### Metrics Collection

```python
class MetricsCollector:
    """Comprehensive system metrics collection"""
    
    def __init__(self):
        # Performance metrics
        self.latency_histogram = LatencyHistogram()
        self.throughput_counter = ThroughputCounter()
        
        # Business metrics
        self.order_count = Counter()
        self.trade_count = Counter() 
        self.volume_tracker = VolumeTracker()
        
        # System health metrics
        self.memory_usage = MemoryGauge()
        self.cpu_usage = CPUGauge()
        self.connection_count = ConnectionGauge()
    
    def record_order_latency(self, latency_ns: int):
        """Record order processing latency"""
        latency_us = latency_ns // 1000
        self.latency_histogram.record(latency_us)
        
        # Alert if latency exceeds SLA
        if latency_us > 1000:  # 1ms SLA
            self.alert_manager.send_alert(
                "High latency detected",
                f"Order processing took {latency_us}μs"
            )
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        return {
            "throughput": {
                "orders_per_second": self.throughput_counter.rate(),
                "trades_per_second": self.trade_count.rate(),
                "total_orders": self.order_count.value(),
                "total_trades": self.trade_count.value()
            },
            "latency": {
                "p50_us": self.latency_histogram.percentile(0.5),
                "p95_us": self.latency_histogram.percentile(0.95),
                "p99_us": self.latency_histogram.percentile(0.99),
                "p99_9_us": self.latency_histogram.percentile(0.999)
            },
            "system": {
                "memory_mb": self.memory_usage.value(),
                "cpu_percent": self.cpu_usage.value(),
                "active_connections": self.connection_count.value()
            }
        }
```

### Health Checks

```python
@app.route('/health')
def health_check():
    """Comprehensive system health check"""
    
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {}
    }
    
    # Check matching engine
    try:
        engine_latency = test_engine_response_time()
        health_status["components"]["matching_engine"] = {
            "status": "healthy" if engine_latency < 1000 else "degraded",
            "latency_us": engine_latency
        }
    except Exception as e:
        health_status["components"]["matching_engine"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Check memory usage
    memory_usage = get_memory_usage_percent()
    health_status["components"]["memory"] = {
        "status": "healthy" if memory_usage < 80 else "warning",
        "usage_percent": memory_usage
    }
    
    # Overall status
    component_statuses = [comp["status"] for comp in health_status["components"].values()]
    if "unhealthy" in component_statuses:
        health_status["status"] = "unhealthy"
    elif "degraded" in component_statuses or "warning" in component_statuses:
        health_status["status"] = "degraded"
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return jsonify(health_status), status_code
```

## Related Documentation

- **[Order Book Mechanics](order_book_mechanics.md)**: Core algorithms and data structures
- **[Performance Engineering](performance_engineering.md)**: Optimization techniques and benchmarks  
- **[Market Microstructure](market_microstructure.md)**: Financial concepts and regulatory requirements

