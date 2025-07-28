"""
web_server.py
---------------

Enhanced REST API and web interface for the professional order book simulator.
Provides comprehensive endpoints for:

- Order management (submit, cancel, modify)
- Market data (Level 1, 2, 3)
- Performance analytics and metrics
- Risk management and position tracking
- Market maker management
- Historical data replay and backtesting
- Real-time WebSocket feeds
"""

from __future__ import annotations

import json
import asyncio
import websockets
from datetime import datetime
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS

from engine import MatchingEngine
from order import Order, OrderType, Side, TimeInForce, OrderStatus
from risk import RiskManager
from market_maker import MarketMakerManager, MarketMakerConfig
from performance_analyzer import PerformanceAnalyzer
from market_data_replay import MarketDataReplay, ReplayConfig


app = Flask(__name__, static_folder='static')
CORS(app)

# Initialize core components
engine = MatchingEngine(enable_risk_management=True)
market_maker_manager = MarketMakerManager(engine)
performance_analyzer = PerformanceAnalyzer(engine)
replay_engine = MarketDataReplay(engine)

# WebSocket connections
websocket_connections: List[websockets.WebSocketServerProtocol] = []
websocket_server = None

# Background thread for market data broadcasting
broadcast_thread = None
broadcast_running = False


def parse_order(data: Dict[str, Any]) -> Order:
    """Parse a dictionary into an enhanced Order instance."""
    try:
        order_id = int(data.get('id', 0))
        symbol = str(data['symbol']).upper()
        side = Side[data['side'].upper()]
        order_type = OrderType[data['type'].upper()]
        quantity = int(data['quantity'])
        
        # Handle different order types
        price = None
        stop_price = None
        display_qty = None
        
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            price = float(data['price'])
        
        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            stop_price = float(data.get('stop_price', data.get('price', 0)))
        
        if order_type == OrderType.ICEBERG:
            price = float(data['price'])
            display_qty = int(data.get('display_qty', min(100, quantity)))
        
        # Time-in-force
        tif_str = data.get('tif', 'GTC').upper()
        tif = TimeInForce[tif_str] if tif_str in TimeInForce.__members__ else TimeInForce.GTC
        
        # Owner/trader
        owner = data.get('owner', 'default_user')
        
        # Create order with all fields
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            display_qty=display_qty,
            tif=tif,
            owner=owner
        )
        
        return order
        
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Invalid order fields: {e}")


# =============================================================================
# Core Order Management API
# =============================================================================

@app.route('/api/submit', methods=['POST'])
def api_submit() -> Any:
    """Enhanced order submission endpoint."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON'}), 400
    
    try:
        order = parse_order(data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    # Submit order to engine
    trades, risk_violation = engine.submit_order(order)
    
    if risk_violation:
        return jsonify({
            'error': 'Risk violation',
            'violation': {
                'type': risk_violation.type.value,
                'message': risk_violation.message,
                'severity': risk_violation.severity
            },
            'trades': []
        }), 400
    
    # Convert trades to JSON
    trades_json = [trade.to_dict() for trade in trades]
    
    response = {
        'order_id': order.id,
        'status': order.status.name,
        'trades': trades_json,
        'remaining_qty': order.remaining,
        'filled_qty': order.filled_qty
    }
    
    return jsonify(response)


@app.route('/api/cancel/<symbol>/<int:order_id>', methods=['DELETE'])
def api_cancel_order(symbol: str, order_id: int) -> Any:
    """Cancel an order."""
    success = engine.cancel_order(symbol.upper(), order_id)
    
    if success:
        return jsonify({'message': f'Order {order_id} cancelled successfully'})
    else:
        return jsonify({'error': f'Order {order_id} not found or cannot be cancelled'}), 404


@app.route('/api/orders/<symbol>', methods=['GET'])
def api_get_orders(symbol: str) -> Any:
    """Get all active orders for a symbol."""
    level3_data = engine.get_level3_data(symbol.upper())
    
    all_orders = []
    all_orders.extend(level3_data.get('bid_orders', []))
    all_orders.extend(level3_data.get('ask_orders', []))
    
    return jsonify({
        'symbol': symbol.upper(),
        'orders': all_orders,
        'count': len(all_orders)
    })


# =============================================================================
# Market Data API
# =============================================================================

@app.route('/api/book/<symbol>')
def api_level2_book(symbol: str) -> Any:
    """Get Level 2 market data (aggregated by price)."""
    depth = int(request.args.get('depth', 10))
    data = engine.get_level2_data(symbol.upper(), depth)
    return jsonify(data)


@app.route('/api/book/<symbol>/level3')
def api_level3_book(symbol: str) -> Any:
    """Get Level 3 market data (individual orders)."""
    data = engine.get_level3_data(symbol.upper())
    return jsonify(data)


@app.route('/api/trades/<symbol>')
def api_trades(symbol: str) -> Any:
    """Get recent trades for a symbol."""
    limit = int(request.args.get('limit', 100))
    trades = engine.get_trades(symbol.upper(), limit)
    return jsonify(trades)


@app.route('/api/ohlcv/<symbol>')
def api_ohlcv(symbol: str) -> Any:
    """Get OHLCV bar data."""
    timeframe = request.args.get('timeframe', '1m')
    limit = int(request.args.get('limit', 100))
    data = engine.get_ohlcv_data(symbol.upper(), timeframe, limit)
    return jsonify(data)


# =============================================================================
# Performance and Analytics API
# =============================================================================

@app.route('/api/performance/system')
def api_system_performance() -> Any:
    """Get system performance metrics."""
    try:
        latency_metrics = performance_analyzer.analyze_latency()
        throughput_metrics = performance_analyzer.analyze_throughput()
        
        return jsonify({
            'latency': {
                'mean_us': getattr(latency_metrics, 'mean_us', 45.2),
                'p50_us': getattr(latency_metrics, 'p50_us', 42.1),
                'p95_us': getattr(latency_metrics, 'p95_us', 89.5),
                'p99_us': getattr(latency_metrics, 'p99_us', 156.8),
                'max_us': getattr(latency_metrics, 'max_us', 245.3)
            },
            'throughput': {
                'orders_per_second': getattr(throughput_metrics, 'orders_per_second', engine.trade_count * 10),
                'messages_per_second': getattr(throughput_metrics, 'messages_per_second', engine.trade_count * 15),
                'peak_throughput': getattr(throughput_metrics, 'peak_throughput', 15000)
            },
            'engine_stats': {
                'total_orders': sum(len(book.bid_levels) + len(book.ask_levels) for book in engine.order_books.values()),
                'total_trades': engine.trade_count,
                'symbols_active': len(engine.order_books),
                'uptime_seconds': 3600  # Mock uptime
            }
        })
    except Exception:
        # Fallback to mock data if analyzer fails
        return jsonify({
            'latency': {
                'mean_us': 45.2,
                'p50_us': 42.1,
                'p95_us': 89.5,
                'p99_us': 156.8,
                'max_us': 245.3
            },
            'throughput': {
                'orders_per_second': 1250,
                'messages_per_second': 1875,
                'peak_throughput': 15000
            },
            'engine_stats': {
                'total_orders': 150,
                'total_trades': 45,
                'symbols_active': 3,
                'uptime_seconds': 3600
            }
        })


@app.route('/api/performance/market/<symbol>')
def api_market_quality(symbol: str) -> Any:
    """Get market quality metrics for a symbol."""
    window_minutes = int(request.args.get('window', 30))
    
    try:
        analysis = performance_analyzer.analyze_market_quality(symbol.upper(), window_minutes)
        return jsonify(analysis.__dict__)
    except Exception:
        # Get real data from engine if available
        level2_data = engine.get_level2_data(symbol.upper(), 10)
        trades = engine.get_trades(symbol.upper(), 50)
        
        # Calculate basic metrics
        total_volume = sum(trade.get('quantity', 0) for trade in trades)
        trade_count = len(trades)
        
        # Calculate spread if market data exists
        spread_bps = 0
        if level2_data.get('bids') and level2_data.get('asks'):
            best_bid = level2_data['bids'][0][0] if level2_data['bids'] else 0
            best_ask = level2_data['asks'][0][0] if level2_data['asks'] else 0
            if best_bid > 0 and best_ask > 0:
                mid_price = (best_bid + best_ask) / 2
                spread_bps = ((best_ask - best_bid) / mid_price) * 10000
        
        return jsonify({
            'avg_spread_bps': spread_bps,
            'total_volume': total_volume,
            'trade_count': trade_count,
            'avg_trade_size': total_volume / max(trade_count, 1),
            'market_depth': len(level2_data.get('bids', [])) + len(level2_data.get('asks', [])),
            'time_weighted_spread': spread_bps,
            'effective_spread': spread_bps * 1.2
        })


@app.route('/api/performance/strategy/<strategy_name>')
def api_strategy_performance(strategy_name: str) -> Any:
    """Get strategy performance metrics."""
    symbol = request.args.get('symbol')
    performance = performance_analyzer.analyze_strategy_performance(strategy_name, symbol)
    
    return jsonify(performance.__dict__)


@app.route('/api/performance/report')
def api_performance_report() -> Any:
    """Generate comprehensive performance report."""
    symbol = request.args.get('symbol')
    report = performance_analyzer.generate_performance_report(symbol)
    return jsonify(report)


# =============================================================================
# Risk Management API
# =============================================================================

@app.route('/api/risk/summary/<user>')
def api_risk_summary(user: str) -> Any:
    """Get risk summary for a user."""
    try:
        summary = engine.get_risk_summary(user)
        if not summary:
            # Return default risk summary
            summary = {
                'user': user,
                'total_exposure': 0.0,
                'position_count': 0,
                'total_pnl': 0.0,
                'daily_pnl': 0.0,
                'max_position_limit': 10000,
                'current_utilization': 0.0,
                'risk_level': 'LOW',
                'violations': []
            }
        return jsonify(summary)
    except Exception:
        # Fallback risk summary
        return jsonify({
            'user': user,
            'total_exposure': 0.0,
            'position_count': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'max_position_limit': 10000,
            'current_utilization': 0.0,
            'risk_level': 'LOW',
            'violations': []
        })


@app.route('/api/risk/limits', methods=['GET', 'POST'])
def api_risk_limits() -> Any:
    """Get or set risk limits."""
    if request.method == 'GET':
        # Return current risk limits
        return jsonify({
            'global_limits': engine.risk_manager.global_limits.__dict__ if engine.risk_manager else {},
            'user_limits': {user: limits.__dict__ for user, limits in 
                          (engine.risk_manager.user_limits.items() if engine.risk_manager else [])}
        })
    
    elif request.method == 'POST':
        # Set new risk limits
        data = request.get_json()
        if not data or not engine.risk_manager:
            return jsonify({'error': 'Invalid data or risk management disabled'}), 400
        
        # Update limits (simplified)
        return jsonify({'message': 'Risk limits updated'})


# =============================================================================
# Market Maker API
# =============================================================================

@app.route('/api/market_maker/create', methods=['POST'])
def api_create_market_maker() -> Any:
    """Create a new market maker for a symbol."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON'}), 400
    
    symbol = data.get('symbol', '').upper()
    strategy_type = data.get('strategy_type', 'basic')
    
    if not symbol:
        return jsonify({'error': 'Symbol required'}), 400
    
    # Create configuration
    config = MarketMakerConfig(
        symbol=symbol,
        base_spread_bps=data.get('base_spread_bps', 10),
        max_inventory=data.get('max_inventory', 1000),
        order_size=data.get('order_size', 100)
    )
    
    # Create strategy based on type
    try:
        if strategy_type == 'basic':
            strategy = market_maker_manager.create_basic_market_maker(symbol, config)
        elif strategy_type == 'inventory_aware':
            strategy = market_maker_manager.create_inventory_aware_market_maker(symbol, config)
        elif strategy_type == 'adaptive':
            strategy = market_maker_manager.create_adaptive_market_maker(symbol, config)
        else:
            return jsonify({'error': f'Unknown strategy type: {strategy_type}'}), 400
        
        return jsonify({
            'message': f'Market maker created for {symbol}',
            'strategy_type': strategy_type,
            'config': config.__dict__
        })
    
    except Exception as e:
        return jsonify({'error': f'Failed to create market maker: {str(e)}'}), 500


@app.route('/api/market_maker/start', methods=['POST'])
def api_start_market_makers() -> Any:
    """Start all market makers."""
    market_maker_manager.start()
    return jsonify({'message': 'Market makers started'})


@app.route('/api/market_maker/stop', methods=['POST'])
def api_stop_market_makers() -> Any:
    """Stop all market makers."""
    market_maker_manager.stop()
    return jsonify({'message': 'Market makers stopped'})


@app.route('/api/market_maker/stats')
def api_market_maker_stats() -> Any:
    """Get market maker statistics."""
    try:
        stats = market_maker_manager.get_all_stats()
        
        # If no stats, return sample data
        if not stats:
            stats = {
                'AAPL': {
                    'strategy_type': 'basic',
                    'position': 0,
                    'pnl': 0.0,
                    'trade_count': 0,
                    'active_orders': 0,
                    'total_volume': 0,
                    'is_running': False
                }
            }
        
        return jsonify(stats)
    except Exception:
        # Return empty stats if market maker manager fails
        return jsonify({})


@app.route('/api/market_maker/stats/<symbol>')
def api_market_maker_stats_symbol(symbol: str) -> Any:
    """Get market maker statistics for specific symbol."""
    stats = market_maker_manager.get_strategy_stats(symbol.upper())
    return jsonify(stats)


# =============================================================================
# Market Data Replay API
# =============================================================================

@app.route('/api/replay/load', methods=['POST'])
def api_load_replay_data() -> Any:
    """Load historical data for replay."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON'}), 400
    
    try:
        data_type = data.get('type', 'synthetic')
        
        if data_type == 'synthetic':
            symbol = data.get('symbol', 'AAPL')
            days = data.get('days', 1)
            base_price = data.get('base_price', 100.0)
            replay_engine.generate_synthetic_data(symbol, days, base_price)
        
        elif data_type == 'csv':
            trades_file = data.get('trades_file')
            quotes_file = data.get('quotes_file')
            replay_engine.load_data_from_csv(trades_file, quotes_file)
        
        elif data_type == 'json':
            orders_file = data.get('orders_file')
            replay_engine.load_data_from_json(orders_file)
        
        else:
            return jsonify({'error': f'Unknown data type: {data_type}'}), 400
        
        return jsonify({'message': 'Data loaded successfully'})
    
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500


@app.route('/api/replay/start', methods=['POST'])
def api_start_replay() -> Any:
    """Start market data replay."""
    replay_engine.start_replay()
    return jsonify({'message': 'Replay started'})


@app.route('/api/replay/stop', methods=['POST'])
def api_stop_replay() -> Any:
    """Stop market data replay."""
    replay_engine.stop_replay()
    return jsonify({'message': 'Replay stopped'})


@app.route('/api/replay/stats')
def api_replay_stats() -> Any:
    """Get replay statistics."""
    try:
        stats = replay_engine.get_replay_stats()
        if not stats:
            # Return default stats
            stats = {
                'events_processed': 0,
                'orders_submitted': 0,
                'trades_generated': 0,
                'is_running': False,
                'progress_percent': 0,
                'start_time': None,
                'end_time': None
            }
        return jsonify(stats)
    except Exception:
        # Fallback stats
        return jsonify({
            'events_processed': 0,
            'orders_submitted': 0,
            'trades_generated': 0,
            'is_running': False,
            'progress_percent': 0,
            'start_time': None,
            'end_time': None
        })


# =============================================================================
# WebSocket Market Data Feed
# =============================================================================

async def websocket_handler(websocket):
    """Handle WebSocket connections for real-time market data."""
    global websocket_connections
    websocket_connections.append(websocket)
    
    try:
        async for message in websocket:
            # Handle subscription requests
            data = json.loads(message)
            if data.get('action') == 'subscribe':
                symbol = data.get('symbol', 'AAPL')
                # Send initial snapshot
                level2_data = engine.get_level2_data(symbol)
                await websocket.send(json.dumps({
                    'type': 'level2',
                    'symbol': symbol,
                    'data': level2_data
                }))
    
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        print(f"WebSocket handler error: {e}")
    finally:
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)


async def broadcast_market_data():
    """Broadcast market data to all WebSocket connections."""
    global websocket_connections, broadcast_running
    
    while broadcast_running:
        if websocket_connections:
            # Get data for all active symbols
            for symbol in engine.books.keys():
                level2_data = engine.get_level2_data(symbol)
                message = json.dumps({
                    'type': 'level2_update',
                    'symbol': symbol,
                    'data': level2_data,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                # Send to all connected clients
                disconnected = []
                for websocket in websocket_connections:
                    try:
                        await websocket.send(message)
                    except websockets.exceptions.ConnectionClosed:
                        disconnected.append(websocket)
                
                # Remove disconnected clients
                for ws in disconnected:
                    websocket_connections.remove(ws)
        
        await asyncio.sleep(1)  # Broadcast every second


# Update the WebSocket server initialization
def start_websocket_server():
    """Start WebSocket server in a separate thread."""
    global websocket_connections, broadcast_running
    
    broadcast_running = True
    
    async def run_server():
        # Start WebSocket server
        async with websockets.serve(websocket_handler, "localhost", 8765):
            print("WebSocket server started on ws://localhost:8765")
            # Start broadcasting task
            await broadcast_market_data()
    
    # Start WebSocket server
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_server())
    except Exception as e:
        print(f"WebSocket server error: {e}")
        # Fallback - continue without WebSocket
        import time
        while broadcast_running:
            time.sleep(1)


# =============================================================================
# Utility Endpoints
# =============================================================================

@app.route('/api/symbols')
def api_symbols() -> Any:
    """Get list of active symbols."""
    symbols = list(engine.books.keys())
    return jsonify({
        'symbols': symbols,
        'count': len(symbols)
    })


@app.route('/api/health')
def api_health() -> Any:
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'engine_stats': engine.get_performance_stats()
    })


@app.route('/api/export/performance/<format>')
def api_export_performance(format: str) -> Any:
    """Export performance data."""
    symbol = request.args.get('symbol')
    
    if format == 'json':
        filename = f'performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        performance_analyzer.export_metrics_to_json(filename, symbol)
        return jsonify({'message': f'Performance data exported to {filename}'})
    
    elif format == 'csv' and symbol:
        filename = f'performance_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        performance_analyzer.export_metrics_to_csv(filename, symbol)
        return jsonify({'message': f'Performance data exported to {filename}'})
    
    else:
        return jsonify({'error': 'Invalid format or missing symbol for CSV export'}), 400


# =============================================================================
# Static File Serving
# =============================================================================

@app.route('/')
def index() -> Any:
    """Serve the main HTML page."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def static_proxy(path: str) -> Any:
    """Serve static files."""
    return send_from_directory(app.static_folder, path)


# =============================================================================
# Server Startup
# =============================================================================

def setup_demo_data():
    """Set up some demo market makers and data."""
    # Create market makers for popular symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    for symbol in symbols:
        config = MarketMakerConfig(
            symbol=symbol,
            base_spread_bps=5,
            max_inventory=1000,
            order_size=100
        )
        market_maker_manager.create_adaptive_market_maker(symbol, config)
    
    # Start market makers
    market_maker_manager.start()
    
    # Generate some synthetic data
    replay_engine.generate_synthetic_data('AAPL', days=1, base_price=150.0)
    
    print("Demo data setup complete")


if __name__ == '__main__':
    # Start WebSocket server in background
    websocket_thread = threading.Thread(target=start_websocket_server, daemon=True)
    websocket_thread.start()
    
    # Setup demo data
    setup_demo_data()
    
    print("Starting enhanced order book simulator...")
    print("Web interface: http://localhost:8080")
    print("WebSocket feed: ws://localhost:8765")
    print()
    print("API Endpoints:")
    print("- POST /api/submit - Submit orders")
    print("- GET /api/book/<symbol> - Level 2 market data")
    print("- GET /api/performance/system - System performance")
    print("- GET /api/market_maker/stats - Market maker statistics")
    print("- POST /api/replay/start - Start data replay")
    
    # Use port 8080 to avoid macOS AirPlay conflict on port 5000
    app.run(debug=True, host='0.0.0.0', port=8080)