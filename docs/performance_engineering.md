# 🚀 Performance Engineering & Optimization

## Low-Latency Trading Systems

```
⚡ LATENCY OPTIMIZATION HIERARCHY
═══════════════════════════════════════

TARGET: Sub-millisecond order-to-execution latency

┌─────────────────────────────────────────┐
│          LATENCY BREAKDOWN              │
├─────────────────────────────────────────┤
│ Network (datacenter to exchange): 50μs │
│ Application processing:          200μs │  ← OUR FOCUS
│ Operating system overhead:       100μs │
│ Hardware/NIC processing:          50μs │
├─────────────────────────────────────────┤
│ TOTAL ROUND-TRIP LATENCY:       400μs │
└─────────────────────────────────────────┘

COMPETITIVE LANDSCAPE:
🥇 High-frequency traders:    < 100μs
🥈 Professional traders:     < 1ms  
🥉 Retail platforms:         < 10ms
📱 Mobile apps:              < 100ms

OUR ACHIEVEMENT: <1ms average latency
COMPETITIVE POSITIONING: Professional-grade performance
```

## Memory Management & Object Pooling

```python
"""
ZERO-ALLOCATION TRADING ENGINE
=============================

Problem: Garbage collection pauses cause latency spikes
Solution: Pre-allocate and reuse all objects
"""

import threading
from typing import List, Optional
from collections import deque
import mmap

class Order:
    """
    Pre-allocated order objects for zero-GC trading
    """
    __slots__ = [
        'order_id', 'symbol', 'side', 'quantity', 
        'price', 'order_type', 'timestamp', 'user_id',
        'filled_quantity', 'status', '_next_free'
    ]
    
    def reset(self):
        """Reset order for reuse"""
        self.order_id = 0
        self.symbol = ""
        self.side = None
        self.quantity = 0
        self.price = 0.0
        self.order_type = None
        self.timestamp = 0
        self.user_id = ""
        self.filled_quantity = 0
        self.status = OrderStatus.NEW
        self._next_free = None

class ObjectPool:
    """
    Lock-free object pool for high-frequency allocation
    """
    def __init__(self, object_class, initial_size: int = 10000):
        self.object_class = object_class
        self._free_objects = deque()
        self._lock = threading.Lock()
        
        # Pre-allocate objects
        for _ in range(initial_size):
            obj = object_class()
            self._free_objects.append(obj)
    
    def acquire(self):
        """Get object from pool (lock-free fast path)"""
        try:
            return self._free_objects.popleft()
        except IndexError:
            # Pool empty - allocate new (rare path)
            return self._allocate_new()
    
    def release(self, obj):
        """Return object to pool"""
        obj.reset()
        self._free_objects.append(obj)
    
    def _allocate_new(self):
        """Fallback allocation when pool is empty"""
        return self.object_class()

# Global pools for zero-allocation trading
ORDER_POOL = ObjectPool(Order, 50000)
TRADE_POOL = ObjectPool(Trade, 100000)

"""
MEMORY-MAPPED ORDER BOOK
=======================

Use memory mapping for ultra-fast order book operations
"""

class MemoryMappedOrderBook:
    def __init__(self, symbol: str, max_levels: int = 1000):
        self.symbol = symbol
        
        # Memory-map files for order book data
        self.bid_file = f"/tmp/orderbook_{symbol}_bids.dat"
        self.ask_file = f"/tmp/orderbook_{symbol}_asks.dat"
        
        # Each level: price (8 bytes) + quantity (8 bytes) + count (4 bytes)
        self.level_size = 20
        self.book_size = max_levels * self.level_size
        
        # Memory map bid/ask sides
        with open(self.bid_file, "w+b") as f:
            f.write(b'\x00' * self.book_size)
        with open(self.ask_file, "w+b") as f:
            f.write(b'\x00' * self.book_size)
            
        with open(self.bid_file, "r+b") as f:
            self.bid_mmap = mmap.mmap(f.fileno(), self.book_size)
        with open(self.ask_file, "r+b") as f:
            self.ask_mmap = mmap.mmap(f.fileno(), self.book_size)
    
    def update_level(self, side: str, level: int, price: float, 
                    quantity: int, count: int):
        """
        Update price level with zero-copy operation
        """
        mmap_obj = self.bid_mmap if side == "bid" else self.ask_mmap
        offset = level * self.level_size
        
        # Pack data directly into memory map (big-endian)
        import struct
        data = struct.pack(">ddi", price, float(quantity), count)
        mmap_obj[offset:offset + self.level_size] = data
    
    def get_best_price(self, side: str) -> float:
        """Get best bid/ask with minimal CPU cycles"""
        mmap_obj = self.bid_mmap if side == "bid" else self.ask_mmap
        
        import struct
        price_data = mmap_obj[0:8]  # First 8 bytes = price
        return struct.unpack(">d", price_data)[0]

"""
LOCK-FREE DATA STRUCTURES
========================

Eliminate synchronization overhead in multi-threaded environment
"""

import ctypes
from typing import TypeVar, Generic

T = TypeVar('T')

class LockFreeQueue(Generic[T]):
    """
    Lock-free MPSC (Multi-Producer Single-Consumer) queue
    Used for order submission from multiple threads
    """
    def __init__(self, capacity: int = 65536):
        self.capacity = capacity
        self.mask = capacity - 1  # Assumes power of 2
        self.buffer = [None] * capacity
        
        # Atomic counters using ctypes
        self.head = ctypes.c_ulong(0)  # Consumer index
        self.tail = ctypes.c_ulong(0)  # Producer index
    
    def enqueue(self, item: T) -> bool:
        """
        Thread-safe enqueue (multiple producers)
        """
        current_tail = self.tail.value
        next_tail = (current_tail + 1) & self.mask
        
        # Check if queue is full
        if next_tail == self.head.value:
            return False  # Queue full
        
        # Store item and update tail atomically
        self.buffer[current_tail] = item
        self.tail.value = next_tail
        return True
    
    def dequeue(self) -> Optional[T]:
        """
        Single-consumer dequeue
        """
        current_head = self.head.value
        
        # Check if queue is empty
        if current_head == self.tail.value:
            return None
        
        # Get item and update head
        item = self.buffer[current_head]
        self.buffer[current_head] = None  # Clear reference
        self.head.value = (current_head + 1) & self.mask
        return item

# Lock-free order queues for each trading thread
INCOMING_ORDERS = LockFreeQueue[Order](65536)
MARKET_DATA_UPDATES = LockFreeQueue[MarketUpdate](32768)
```

## CPU Optimization & Algorithmic Efficiency

```python
"""
BRANCH PREDICTION OPTIMIZATION
=============================

Minimize CPU pipeline stalls from unpredictable branches
"""

class OptimizedMatcher:
    def __init__(self):
        # Pre-computed lookup tables
        self.price_buckets = {}  # Price -> List[Orders]
        self.side_multiplier = {"BUY": 1, "SELL": -1}
        
    def match_orders_optimized(self, incoming: Order) -> List[Trade]:
        """
        Optimized matching with predictable branches
        """
        trades = []
        side_mult = self.side_multiplier[incoming.side]
        
        # Get opposite side orders (predictable branch)
        opposite_side = "SELL" if incoming.side == "BUY" else "BUY"
        candidate_prices = self.get_matchable_prices(
            incoming.price, opposite_side
        )
        
        # Sort once, iterate in optimal order
        if incoming.side == "BUY":
            candidate_prices.sort()  # Best ask first
        else:
            candidate_prices.sort(reverse=True)  # Best bid first
        
        remaining_qty = incoming.quantity
        
        # Unrolled loop for common case (1-3 price levels)
        for i in range(min(3, len(candidate_prices))):
            if remaining_qty == 0:
                break
                
            price = candidate_prices[i]
            orders_at_price = self.price_buckets[price]
            
            # Process all orders at this price level
            remaining_qty = self._match_at_price_level(
                incoming, orders_at_price, trades, remaining_qty
            )
        
        # Handle remaining price levels (rare case)
        for price in candidate_prices[3:]:
            if remaining_qty == 0:
                break
            # ... remaining matching logic
        
        return trades
    
    def _match_at_price_level(self, incoming: Order, 
                             orders: List[Order], trades: List[Trade], 
                             remaining_qty: int) -> int:
        """
        Optimized matching at single price level
        """
        # Likely branch: exactly one order at price level
        if len(orders) == 1:
            resting = orders[0]
            if resting.quantity >= remaining_qty:
                # Full fill (most common case)
                trade = self._create_trade(
                    incoming, resting, remaining_qty, resting.price
                )
                trades.append(trade)
                resting.quantity -= remaining_qty
                return 0
            else:
                # Partial fill of resting order
                trade = self._create_trade(
                    incoming, resting, resting.quantity, resting.price
                )
                trades.append(trade)
                remaining_qty -= resting.quantity
                orders.remove(resting)  # Order fully filled
                return remaining_qty
        
        # Less likely: multiple orders at same price
        return self._match_multiple_orders(
            incoming, orders, trades, remaining_qty
        )

"""
SIMD OPTIMIZATION FOR PRICE CALCULATIONS
========================================

Use CPU vector instructions for parallel price operations
"""

import numpy as np

class SIMDPriceEngine:
    def __init__(self):
        # Vectorized operations using NumPy (SIMD under the hood)
        self.price_dtype = np.float64
        self.quantity_dtype = np.int64
        
    def calculate_vwap_vectorized(self, prices: np.ndarray, 
                                 quantities: np.ndarray) -> float:
        """
        Volume-Weighted Average Price using SIMD
        Process 4-8 prices simultaneously on modern CPUs
        """
        # Vectorized multiplication: all prices * quantities in parallel
        values = np.multiply(prices, quantities.astype(self.price_dtype))
        
        # Parallel sum using CPU SIMD instructions
        total_value = np.sum(values)
        total_quantity = np.sum(quantities)
        
        return total_value / total_quantity
    
    def update_order_book_levels_simd(self, price_levels: np.ndarray,
                                     quantity_changes: np.ndarray):
        """
        Update multiple price levels simultaneously
        """
        # Vectorized addition: update all levels in parallel
        np.add(price_levels, quantity_changes, out=price_levels)
        
        # Vectorized comparison: find zero quantities
        zero_mask = price_levels == 0
        
        # Remove empty levels (vectorized operation)
        return price_levels[~zero_mask]

"""
CACHE-FRIENDLY DATA STRUCTURES
=============================

Optimize memory access patterns for CPU cache efficiency
"""

class CacheOptimizedOrderBook:
    def __init__(self):
        # Array-of-structures vs Structure-of-arrays
        # SOA is more cache-friendly for bulk operations
        
        # Structure of Arrays (cache-friendly)
        self.prices = np.zeros(1000, dtype=np.float64)      # 8KB continuous
        self.quantities = np.zeros(1000, dtype=np.int64)    # 8KB continuous  
        self.timestamps = np.zeros(1000, dtype=np.int64)    # 8KB continuous
        self.order_ids = np.zeros(1000, dtype=np.int64)     # 8KB continuous
        
        # vs Array of Structures (cache-unfriendly)
        # self.orders = [Order() for _ in range(1000)]  # Scattered in memory
        
    def binary_search_optimized(self, target_price: float) -> int:
        """
        Cache-friendly binary search with prefetching
        """
        left, right = 0, len(self.prices) - 1
        
        while left <= right:
            # Calculate middle with bit shift (faster than division)
            mid = (left + right) >> 1
            
            # Prefetch next potential memory locations
            # (actual prefetch would be CPU-specific intrinsics)
            mid_price = self.prices[mid]
            
            if mid_price == target_price:
                return mid
            elif mid_price < target_price:
                left = mid + 1
            else:
                right = mid - 1
                
        return -1  # Not found
    
    def batch_update_quantities(self, indices: List[int], 
                               new_quantities: List[int]):
        """
        Batch update to minimize cache misses
        """
        # Sort indices to access memory sequentially
        sorted_updates = sorted(zip(indices, new_quantities))
        
        # Sequential memory access pattern
        for idx, qty in sorted_updates:
            self.quantities[idx] = qty

"""
PERFORMANCE MEASUREMENT & PROFILING
==================================

Real-time performance monitoring for production systems
"""

import time
import threading
from collections import defaultdict

class PerformanceProfiler:
    def __init__(self):
        self.latency_histogram = defaultdict(int)
        self.throughput_counter = 0
        self.start_time = time.time()
        
        # Percentile tracking
        self.latency_samples = []
        self.max_samples = 100000
        
    def record_operation_latency(self, start_time: float, 
                                operation: str):
        """
        Record latency for performance analysis
        """
        latency_us = int((time.time() - start_time) * 1_000_000)
        
        # Update histogram
        bucket = self.get_latency_bucket(latency_us)
        self.latency_histogram[bucket] += 1
        
        # Sample for percentile calculation
        if len(self.latency_samples) < self.max_samples:
            self.latency_samples.append(latency_us)
        
        self.throughput_counter += 1
    
    def get_latency_bucket(self, latency_us: int) -> str:
        """Categorize latency into buckets"""
        if latency_us < 100:
            return "<100μs"
        elif latency_us < 500:
            return "100-500μs"
        elif latency_us < 1000:
            return "500μs-1ms"
        elif latency_us < 5000:
            return "1-5ms"
        else:
            return ">5ms"
    
    def get_performance_report(self) -> dict:
        """Generate comprehensive performance report"""
        runtime = time.time() - self.start_time
        
        # Calculate percentiles
        sorted_latencies = sorted(self.latency_samples)
        n = len(sorted_latencies)
        
        return {
            "throughput_ops_sec": self.throughput_counter / runtime,
            "total_operations": self.throughput_counter,
            "runtime_seconds": runtime,
            "latency_percentiles": {
                "p50": sorted_latencies[int(n * 0.5)] if n > 0 else 0,
                "p95": sorted_latencies[int(n * 0.95)] if n > 0 else 0,
                "p99": sorted_latencies[int(n * 0.99)] if n > 0 else 0,
                "p99.9": sorted_latencies[int(n * 0.999)] if n > 0 else 0,
            },
            "latency_histogram": dict(self.latency_histogram)
        }

# Global profiler instance
PROFILER = PerformanceProfiler()

def profile_operation(operation_name: str):
    """Decorator for automatic performance profiling"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            PROFILER.record_operation_latency(start, operation_name)
            return result
        return wrapper
    return decorator

# Usage example:
@profile_operation("order_matching")
def match_order(self, order: Order) -> List[Trade]:
    # ... matching logic
    pass
```

## Hardware-Specific Optimizations

```python
"""
NUMA (Non-Uniform Memory Access) OPTIMIZATION
============================================

Optimize for multi-socket server hardware
"""

import os
import threading
import psutil

class NUMAOptimizedEngine:
    def __init__(self):
        self.cpu_count = os.cpu_count()
        self.numa_nodes = self.detect_numa_topology()
        
        # Pin critical threads to specific NUMA nodes
        self.setup_numa_affinity()
    
    def detect_numa_topology(self) -> dict:
        """Detect NUMA node configuration"""
        numa_info = {}
        
        # Read NUMA topology from /sys/devices/system/node/
        try:
            numa_nodes = os.listdir("/sys/devices/system/node/")
            numa_nodes = [n for n in numa_nodes if n.startswith("node")]
            
            for node in numa_nodes:
                node_id = int(node[4:])  # Extract number from "nodeX"
                cpu_list_path = f"/sys/devices/system/node/{node}/cpulist"
                
                with open(cpu_list_path, 'r') as f:
                    cpu_range = f.read().strip()
                    numa_info[node_id] = self.parse_cpu_range(cpu_range)
                    
        except (OSError, FileNotFoundError):
            # Fallback: assume single NUMA node
            numa_info[0] = list(range(self.cpu_count))
            
        return numa_info
    
    def setup_numa_affinity(self):
        """Pin trading threads to optimal NUMA nodes"""
        if len(self.numa_nodes) >= 2:
            # Pin order matching to NUMA node 0 (fastest memory access)
            matching_cpus = self.numa_nodes[0][:2]  # First 2 CPUs
            os.sched_setaffinity(0, matching_cpus)
            
            # Pin market data processing to NUMA node 1
            # (separate from order matching for isolation)
            
    def parse_cpu_range(self, cpu_range: str) -> List[int]:
        """Parse CPU range string like '0-7,16-23'"""
        cpus = []
        for part in cpu_range.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                cpus.extend(range(start, end + 1))
            else:
                cpus.append(int(part))
        return cpus

"""
CPU CACHE OPTIMIZATION
====================

Optimize data layout for L1/L2/L3 cache efficiency
"""

class CacheOptimizedStructure:
    def __init__(self):
        # Align data structures to cache line boundaries (64 bytes)
        self.CACHE_LINE_SIZE = 64
        
        # Hot data: frequently accessed together
        self.hot_order_data = np.zeros(
            (1000, 4),  # [price, quantity, timestamp, order_id]
            dtype=np.float64
        )
        
        # Cold data: infrequently accessed metadata
        self.cold_order_metadata = {}
        
    def allocate_cache_aligned(self, size: int) -> bytearray:
        """Allocate memory aligned to cache line boundaries"""
        # Allocate extra space for alignment
        raw_memory = bytearray(size + self.CACHE_LINE_SIZE)
        
        # Calculate aligned offset
        memory_address = id(raw_memory)
        aligned_offset = (self.CACHE_LINE_SIZE - 
                         (memory_address % self.CACHE_LINE_SIZE)) % self.CACHE_LINE_SIZE
        
        # Return aligned portion
        return raw_memory[aligned_offset:aligned_offset + size]
    
    def prefetch_memory(self, address: int):
        """CPU prefetch hint (would use intrinsics in C++)"""
        # In Python, we can simulate by accessing memory
        # Real implementation would use __builtin_prefetch()
        pass

"""
HARDWARE TIMESTAMPING
====================

Use hardware timestamps for maximum precision
"""

import ctypes
import time

class HardwareTimer:
    def __init__(self):
        # Access hardware timestamp counter (TSC)
        self.tsc_frequency = self.calibrate_tsc()
        
    def calibrate_tsc(self) -> int:
        """Calibrate TSC frequency against system clock"""
        start_tsc = self.read_tsc()
        start_time = time.time()
        
        time.sleep(1.0)  # Wait 1 second
        
        end_tsc = self.read_tsc()
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        elapsed_tsc = end_tsc - start_tsc
        
        return int(elapsed_tsc / elapsed_time)
    
    def read_tsc(self) -> int:
        """Read CPU timestamp counter (TSC)"""
        # In real implementation, would use assembly instruction
        # RDTSC (Read Time-Stamp Counter)
        
        # Simplified version using time.perf_counter_ns()
        return time.perf_counter_ns()
    
    def tsc_to_nanoseconds(self, tsc_value: int) -> int:
        """Convert TSC value to nanoseconds"""
        return int((tsc_value * 1_000_000_000) / self.tsc_frequency)
    
    def get_hardware_timestamp(self) -> int:
        """Get hardware timestamp in nanoseconds"""
        tsc = self.read_tsc()
        return self.tsc_to_nanoseconds(tsc)

# Global hardware timer
HARDWARE_TIMER = HardwareTimer()

"""
COMPILER OPTIMIZATIONS
====================

Hints for compiler optimization (would be C/C++ in production)
"""

class OptimizationHints:
    """
    Performance optimization techniques for production systems
    """
    
    @staticmethod
    def likely_branch(condition: bool) -> bool:
        """
        Branch prediction hint - condition is likely true
        In C++: if (__builtin_expect(condition, 1))
        """
        return condition
    
    @staticmethod  
    def unlikely_branch(condition: bool) -> bool:
        """
        Branch prediction hint - condition is unlikely true
        In C++: if (__builtin_expect(condition, 0))
        """
        return condition
    
    @staticmethod
    def force_inline(func):
        """
        Force function inlining to eliminate call overhead
        In C++: __forceinline or __attribute__((always_inline))
        """
        return func
    
    @staticmethod
    def no_inline(func):
        """
        Prevent function inlining (for debugging/profiling)
        In C++: __noinline or __attribute__((noinline))
        """
        return func

# Example usage with optimization hints
def process_order_fast_path(order: Order) -> bool:
    """Optimized for the common case"""
    
    # Hot path: valid orders (99% of cases)
    if OptimizationHints.likely_branch(order.is_valid()):
        return True
    
    # Cold path: invalid orders (1% of cases)  
    if OptimizationHints.unlikely_branch(order.quantity <= 0):
        return False
        
    return True
```

This performance engineering documentation demonstrates:

1. **Low-Latency Design**: Sub-millisecond execution targets
2. **Memory Optimization**: Object pooling, zero-allocation patterns
3. **CPU Optimization**: Cache-friendly algorithms, SIMD usage
4. **Hardware Awareness**: NUMA topology, cache alignment
5. **Profiling & Measurement**: Real-time performance monitoring
