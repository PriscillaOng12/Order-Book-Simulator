/**
 * Professional Order Book Simulator - Client Side Application
 * 
 * Features:
 * - Advanced order types (Market, Limit, Stop, Iceberg)
 * - Real-time WebSocket market data
 * - Performance analytics and metrics
 * - Market maker management
 * - Risk management dashboard
 * - Historical data replay
 * - Level 3 market data visualization
 */

class OrderBookSimulator {
  constructor() {
    this.currentSymbol = 'AAPL';
    this.nextOrderId = 1;
    this.websocket = null;
    this.charts = {};
    this.isConnected = false;
    
    this.initializeApp();
  }

  initializeApp() {
    this.setupEventListeners();
    this.initializeCharts();
    this.connectWebSocket();
    this.startPerformanceMonitoring();
    this.loadInitialData();
  }

  setupEventListeners() {
    // Order form submission
    const orderForm = document.getElementById('orderForm');
    orderForm.addEventListener('submit', (e) => this.handleOrderSubmission(e));

    // Order type changes
    const typeSelect = document.getElementById('type');
    typeSelect.addEventListener('change', () => this.handleOrderTypeChange());

    // Symbol changes
    const symbolSelect = document.getElementById('symbol');
    symbolSelect.addEventListener('change', (e) => {
      this.currentSymbol = e.target.value;
      this.updateMarketData();
      document.getElementById('currentSymbol').textContent = `${this.currentSymbol} Market Data`;
    });

    // Tab navigation
    this.setupTabNavigation();

    // Market maker controls
    this.setupMarketMakerControls();

    // Replay controls
    this.setupReplayControls();

    // Initial order type setup
    this.handleOrderTypeChange();
  }

  setupTabNavigation() {
    // Add click handlers directly to buttons
    const buttons = {
      'trading': () => this.showTab('trading'),
      'analytics': () => this.showTab('analytics'),
      'market-maker': () => this.showTab('market-maker'),
      'risk': () => this.showTab('risk'),
      'replay': () => this.showTab('replay')
    };

    // Add event listeners to tab buttons
    Object.keys(buttons).forEach(tabName => {
      const button = document.querySelector(`[onclick="showTab('${tabName}')"]`);
      if (button) {
        button.addEventListener('click', (e) => {
          e.preventDefault();
          buttons[tabName]();
        });
      }
    });

    // Make showTab globally available for onclick handlers
    window.showTab = (tabName) => this.showTab(tabName);
  }

  showTab(tabName) {
    console.log(`Switching to tab: ${tabName}`);
    
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
      content.classList.remove('active');
    });
    
    // Remove active class from all tab buttons
    document.querySelectorAll('.tab-button').forEach(button => {
      button.classList.remove('active');
    });
    
    // Show the selected tab
    const targetTab = document.getElementById(tabName);
    if (targetTab) {
      targetTab.classList.add('active');
      console.log(`Tab ${tabName} activated`);
    } else {
      console.error(`Tab content not found: ${tabName}`);
    }
    
    // Activate the corresponding button
    const targetButton = document.querySelector(`[onclick="showTab('${tabName}')"]`);
    if (targetButton) {
      targetButton.classList.add('active');
    }
    
    // Load data for the specific tab
    setTimeout(() => {
      switch(tabName) {
        case 'market-maker':
          console.log('Loading market maker data...');
          this.loadMarketMakerStats();
          break;
        case 'risk':
          console.log('Loading risk data...');
          this.loadRiskData();
          break;
        case 'replay':
          console.log('Loading replay data...');
          this.loadReplayData();
          break;
        case 'analytics':
          console.log('Loading analytics data...');
          this.loadAnalytics();
          break;
      }
    }, 100);
  }

  setupMarketMakerControls() {
    // Market maker form
    const mmForm = document.getElementById('mmForm');
    if (mmForm) {
      mmForm.addEventListener('submit', (e) => this.handleMarketMakerCreation(e));
    }

    // Control buttons
    const startBtn = document.getElementById('startMM');
    const stopBtn = document.getElementById('stopMM');
    const refreshBtn = document.getElementById('refreshMM');

    if (startBtn) startBtn.addEventListener('click', () => this.startMarketMakers());
    if (stopBtn) stopBtn.addEventListener('click', () => this.stopMarketMakers());
    if (refreshBtn) refreshBtn.addEventListener('click', () => this.loadMarketMakerStats());
  }

  setupReplayControls() {
    const replaySetupForm = document.getElementById('replaySetupForm');
    if (replaySetupForm) {
      replaySetupForm.addEventListener('submit', (e) => this.handleReplaySetup(e));
    }

    const startReplayBtn = document.getElementById('startReplay');
    const stopReplayBtn = document.getElementById('stopReplay');
    
    if (startReplayBtn) startReplayBtn.addEventListener('click', () => this.startReplay());
    if (stopReplayBtn) stopReplayBtn.addEventListener('click', () => this.stopReplay());
  }

  async handleOrderSubmission(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const order = {
      id: this.nextOrderId++,
      symbol: this.currentSymbol,
      side: formData.get('side') || document.getElementById('side').value,
      type: formData.get('type') || document.getElementById('type').value,
      quantity: parseInt(document.getElementById('quantity').value),
      owner: document.getElementById('owner').value,
      tif: document.getElementById('tif').value
    };

    // Add price for limit orders
    const orderType = document.getElementById('type').value;
    if (['LIMIT', 'STOP_LIMIT', 'ICEBERG'].includes(orderType)) {
      order.price = parseFloat(document.getElementById('price').value);
    }

    // Add stop price for stop orders
    if (['STOP', 'STOP_LIMIT'].includes(orderType)) {
      const stopPrice = document.getElementById('stopPrice').value;
      if (stopPrice) {
        order.stop_price = parseFloat(stopPrice);
      }
    }

    // Add display quantity for iceberg orders
    if (orderType === 'ICEBERG') {
      const displayQty = document.getElementById('displayQty').value;
      if (displayQty) {
        order.display_qty = parseInt(displayQty);
      }
    }

    try {
      const response = await fetch('/api/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(order)
      });

      const result = await response.json();
      
      if (response.ok) {
        this.showFeedback(
          `✅ Order ${result.order_id} submitted. Status: ${result.status}. Executed ${result.trades.length} trade(s).`,
          'success'
        );
        this.updateMarketData();
      } else {
        this.showFeedback(`❌ Error: ${result.error}`, 'error');
        if (result.violation) {
          this.showFeedback(`🚨 Risk Violation: ${result.violation.message}`, 'error');
        }
      }
    } catch (error) {
      this.showFeedback(`❌ Network error: ${error.message}`, 'error');
    }
  }

  handleOrderTypeChange() {
    const orderType = document.getElementById('type').value;
    const priceRow = document.getElementById('priceRow');
    const stopPriceRow = document.getElementById('stopPriceRow');
    const displayQtyRow = document.getElementById('displayQtyRow');

    // Reset visibility
    priceRow.style.display = 'block';
    stopPriceRow.style.display = 'none';
    displayQtyRow.style.display = 'none';

    switch (orderType) {
      case 'MARKET':
        priceRow.style.display = 'none';
        break;
      case 'STOP':
        priceRow.style.display = 'none';
        stopPriceRow.style.display = 'block';
        break;
      case 'STOP_LIMIT':
        stopPriceRow.style.display = 'block';
        break;
      case 'ICEBERG':
        displayQtyRow.style.display = 'block';
        break;
    }
  }

  showFeedback(message, type = 'info') {
    const feedback = document.getElementById('feedback');
    feedback.textContent = message;
    feedback.className = `feedback ${type}`;
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
      feedback.textContent = '';
      feedback.className = 'feedback';
    }, 5000);
  }

  connectWebSocket() {
    console.log('Attempting to connect WebSocket...');
    
    // Use the correct WebSocket URL
    const wsUrl = `ws://${window.location.hostname}:8765`;
    console.log(`Connecting to: ${wsUrl}`);
    
    try {
      this.websocket = new WebSocket(wsUrl);
      
      this.websocket.onopen = () => {
        console.log('WebSocket connected successfully');
        this.isConnected = true;
        this.updateConnectionStatus('Connected', 'success');
      };
      
      this.websocket.onmessage = (event) => {
        console.log('WebSocket message received:', event.data);
        try {
          const data = JSON.parse(event.data);
          this.handleWebSocketMessage(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      this.websocket.onclose = () => {
        console.log('WebSocket disconnected');
        this.isConnected = false;
        this.updateConnectionStatus('Disconnected', 'error');
        
        // Attempt to reconnect after 3 seconds
        setTimeout(() => {
          if (!this.isConnected) {
            console.log('Attempting to reconnect...');
            this.connectWebSocket();
          }
        }, 3000);
      };
      
      this.websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.updateConnectionStatus('Error', 'error');
      };
      
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.updateConnectionStatus('Failed to Connect', 'error');
    }
  }

  handleWebSocketMessage(data) {
    switch (data.type) {
      case 'level2':
      case 'level2_update':
        this.updateOrderBookChart(data.data);
        this.updateMarketInfo(data.data);
        break;
      case 'trade':
        this.updateTradesChart(data.trades);
        break;
      case 'performance':
        this.updatePerformanceMetrics(data.metrics);
        break;
    }
  }

  updateConnectionStatus() {
    const wsStatus = document.getElementById('wsStatus');
    if (this.isConnected) {
      wsStatus.textContent = 'Connected';
      wsStatus.className = 'status-value connected';
    } else {
      wsStatus.textContent = 'Disconnected';
      wsStatus.className = 'status-value disconnected';
    }
  }

  initializeCharts() {
    this.initializeOrderBookChart();
    this.initializeTradesChart();
    this.initializeOHLCChart();
    this.initializePerformanceChart();
  }

  initializeOrderBookChart() {
    const ctx = document.getElementById('bookChart');
    if (!ctx) return;

    this.charts.orderBook = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Bids',
            data: [],
            backgroundColor: 'rgba(42, 157, 143, 0.8)',
            borderColor: 'rgba(42, 157, 143, 1)',
            borderWidth: 1
          },
          {
            label: 'Asks',
            data: [],
            backgroundColor: 'rgba(231, 111, 81, 0.8)',
            borderColor: 'rgba(231, 111, 81, 1)',
            borderWidth: 1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top'
          },
          title: {
            display: true,
            text: 'Order Book Depth'
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Quantity'
            }
          },
          x: {
            title: {
              display: true,
              text: 'Price'
            }
          }
        }
      }
    });
  }

  initializeTradesChart() {
    const ctx = document.getElementById('tradeChart');
    if (!ctx) return;

    this.charts.trades = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'Trades',
          data: [],
          backgroundColor: 'rgba(38, 70, 83, 0.8)',
          borderColor: 'rgba(38, 70, 83, 1)',
          pointRadius: 4,
          pointHoverRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false
          },
          title: {
            display: true,
            text: 'Trade History'
          }
        },
        scales: {
          x: {
            type: 'time',
            time: {
              unit: 'minute',
              displayFormats: {
                minute: 'HH:mm'
              }
            },
            title: {
              display: true,
              text: 'Time'
            }
          },
          y: {
            title: {
              display: true,
              text: 'Price'
            }
          }
        }
      }
    });
  }

  initializeOHLCChart() {
    const ctx = document.getElementById('ohlcChart');
    if (!ctx) return;

    this.charts.ohlc = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Price',
          data: [],
          borderColor: 'rgba(42, 157, 143, 1)',
          backgroundColor: 'rgba(42, 157, 143, 0.1)',
          fill: true,
          tension: 0.1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false
          },
          title: {
            display: true,
            text: 'Price Chart (1min)'
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Time'
            }
          },
          y: {
            title: {
              display: true,
              text: 'Price'
            }
          }
        }
      }
    });
  }

  initializePerformanceChart() {
    const ctx = document.getElementById('performanceChart');
    if (!ctx) return;

    this.charts.performance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Latency (μs)',
            data: [],
            borderColor: 'rgba(231, 111, 81, 1)',
            backgroundColor: 'rgba(231, 111, 81, 0.1)',
            yAxisID: 'y'
          },
          {
            label: 'Orders/sec',
            data: [],
            borderColor: 'rgba(42, 157, 143, 1)',
            backgroundColor: 'rgba(42, 157, 143, 0.1)',
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: 'System Performance Metrics'
          }
        },
        scales: {
          y: {
            type: 'linear',
            display: true,
            position: 'left',
            title: {
              display: true,
              text: 'Latency (μs)'
            }
          },
          y1: {
            type: 'linear',
            display: true,
            position: 'right',
            title: {
              display: true,
              text: 'Orders/sec'
            },
            grid: {
              drawOnChartArea: false,
            }
          }
        }
      }
    });
  }

  async updateMarketData() {
    try {
      // Update Level 2 data
      const level2Response = await fetch(`/api/book/${this.currentSymbol}?depth=10`);
      const level2Data = await level2Response.json();
      this.updateOrderBookChart(level2Data);
      this.updateMarketInfo(level2Data);

      // Update Level 3 data
      const level3Response = await fetch(`/api/book/${this.currentSymbol}/level3`);
      const level3Data = await level3Response.json();
      this.updateLevel3Data(level3Data);

      // Update trades
      const tradesResponse = await fetch(`/api/trades/${this.currentSymbol}?limit=50`);
      const tradesData = await tradesResponse.json();
      this.updateTradesChart(tradesData);

      // Update OHLC data
      const ohlcResponse = await fetch(`/api/ohlcv/${this.currentSymbol}?timeframe=1m&limit=50`);
      const ohlcData = await ohlcResponse.json();
      this.updateOHLCChart(ohlcData);

    } catch (error) {
      console.error('Error updating market data:', error);
    }
  }

  updateOrderBookChart(data) {
    if (!this.charts.orderBook || !data.bids || !data.asks) return;

    const prices = new Set();
    data.bids.forEach(([price]) => prices.add(price));
    data.asks.forEach(([price]) => prices.add(price));
    
    const sortedPrices = Array.from(prices).sort((a, b) => a - b);
    
    const bidsData = sortedPrices.map(price => {
      const bid = data.bids.find(([p]) => p === price);
      return bid ? bid[1] : 0;
    });
    
    const asksData = sortedPrices.map(price => {
      const ask = data.asks.find(([p]) => p === price);
      return ask ? -ask[1] : 0; // Negative for visual separation
    });

    this.charts.orderBook.data.labels = sortedPrices.map(p => p.toFixed(2));
    this.charts.orderBook.data.datasets[0].data = bidsData;
    this.charts.orderBook.data.datasets[1].data = asksData;
    this.charts.orderBook.update('none');
  }

  updateMarketInfo(data) {
    const bestBid = data.bids && data.bids.length > 0 ? data.bids[0][0] : null;
    const bestAsk = data.asks && data.asks.length > 0 ? data.asks[0][0] : null;
    
    document.getElementById('bestBid').textContent = bestBid ? bestBid.toFixed(2) : '-';
    document.getElementById('bestAsk').textContent = bestAsk ? bestAsk.toFixed(2) : '-';
    
    if (bestBid && bestAsk) {
      const spread = (bestAsk - bestBid).toFixed(2);
      const spreadBps = ((bestAsk - bestBid) / ((bestBid + bestAsk) / 2) * 10000).toFixed(1);
      document.getElementById('spread').textContent = `${spread} (${spreadBps} bps)`;
    } else {
      document.getElementById('spread').textContent = '-';
    }
  }

  updateLevel3Data(data) {
    const bidOrdersContainer = document.getElementById('bidOrders');
    const askOrdersContainer = document.getElementById('askOrders');
    
    if (!bidOrdersContainer || !askOrdersContainer) return;

    // Clear existing data
    bidOrdersContainer.innerHTML = '';
    askOrdersContainer.innerHTML = '';

    // Display bid orders
    if (data.bid_orders) {
      data.bid_orders.slice(0, 10).forEach(order => {
        const orderElement = document.createElement('div');
        orderElement.className = 'order-item';
        orderElement.innerHTML = `
          <span>${order.price.toFixed(2)}</span>
          <span>${order.quantity}</span>
        `;
        bidOrdersContainer.appendChild(orderElement);
      });
    }

    // Display ask orders
    if (data.ask_orders) {
      data.ask_orders.slice(0, 10).forEach(order => {
        const orderElement = document.createElement('div');
        orderElement.className = 'order-item';
        orderElement.innerHTML = `
          <span>${order.price.toFixed(2)}</span>
          <span>${order.quantity}</span>
        `;
        askOrdersContainer.appendChild(orderElement);
      });
    }
  }

  updateTradesChart(trades) {
    if (!this.charts.trades || !trades) return;

    const tradePoints = trades.slice(-50).map(trade => ({
      x: new Date(trade.timestamp),
      y: trade.price
    }));

    this.charts.trades.data.datasets[0].data = tradePoints;
    this.charts.trades.update('none');

    // Update last trade price
    if (trades.length > 0) {
      const lastTrade = trades[trades.length - 1];
      document.getElementById('lastTrade').textContent = lastTrade.price.toFixed(2);
    }
  }

  updateOHLCChart(ohlcData) {
    if (!this.charts.ohlc || !ohlcData) return;

    const labels = ohlcData.map(bar => new Date(bar.timestamp).toLocaleTimeString());
    const prices = ohlcData.map(bar => bar.close);

    this.charts.ohlc.data.labels = labels;
    this.charts.ohlc.data.datasets[0].data = prices;
    this.charts.ohlc.update('none');
  }

  async loadAnalytics() {
    try {
      const response = await fetch('/api/performance/system');
      const data = await response.json();

      // Update metric cards
      document.getElementById('latencyP99').textContent = 
        data.latency.p99_us ? `${data.latency.p99_us.toFixed(1)} μs` : '- μs';
      document.getElementById('throughput').textContent = 
        data.throughput.orders_per_second ? `${data.throughput.orders_per_second.toFixed(1)}` : '-';

      // Update performance chart
      this.updatePerformanceChart(data);

      // Load market quality metrics
      const qualityResponse = await fetch(`/api/performance/market/${this.currentSymbol}`);
      const qualityData = await qualityResponse.json();

      document.getElementById('avgSpread').textContent = 
        qualityData.avg_spread_bps ? `${qualityData.avg_spread_bps.toFixed(1)} bps` : '- bps';
      document.getElementById('totalVolume').textContent = 
        qualityData.total_volume ? qualityData.total_volume.toLocaleString() : '-';

    } catch (error) {
      console.error('Error loading analytics:', error);
    }
  }

  updatePerformanceChart(data) {
    if (!this.charts.performance) return;

    // This would typically accumulate data over time
    // For now, just show current values
    const now = new Date().toLocaleTimeString();
    
    if (this.charts.performance.data.labels.length > 20) {
      this.charts.performance.data.labels.shift();
      this.charts.performance.data.datasets[0].data.shift();
      this.charts.performance.data.datasets[1].data.shift();
    }

    this.charts.performance.data.labels.push(now);
    this.charts.performance.data.datasets[0].data.push(data.latency.p99_us || 0);
    this.charts.performance.data.datasets[1].data.push(data.throughput.orders_per_second || 0);
    
    this.charts.performance.update('none');
  }

  async handleMarketMakerCreation(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const config = {
      symbol: formData.get('symbol') || document.getElementById('mmSymbol').value,
      strategy_type: formData.get('strategy') || document.getElementById('mmStrategy').value,
      base_spread_bps: parseInt(document.getElementById('mmSpread').value),
      order_size: parseInt(document.getElementById('mmSize').value)
    };

    try {
      const response = await fetch('/api/market_maker/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });

      const result = await response.json();
      
      if (response.ok) {
        this.showFeedback(`✅ ${result.message}`, 'success');
        this.loadMarketMakerStats();
      } else {
        this.showFeedback(`❌ Error: ${result.error}`, 'error');
      }
    } catch (error) {
      this.showFeedback(`❌ Network error: ${error.message}`, 'error');
    }
  }

  async startMarketMakers() {
    try {
      const response = await fetch('/api/market_maker/start', { method: 'POST' });
      const result = await response.json();
      
      if (response.ok) {
        this.showFeedback('✅ Market makers started', 'success');
        this.loadMarketMakerStats();
      } else {
        this.showFeedback(`❌ Error: ${result.error}`, 'error');
      }
    } catch (error) {
      this.showFeedback(`❌ Network error: ${error.message}`, 'error');
    }
  }

  async stopMarketMakers() {
    try {
      const response = await fetch('/api/market_maker/stop', { method: 'POST' });
      const result = await response.json();
      
      if (response.ok) {
        this.showFeedback('⏹️ Market makers stopped', 'success');
        this.loadMarketMakerStats();
      } else {
        this.showFeedback(`❌ Error: ${result.error}`, 'error');
      }
    } catch (error) {
      this.showFeedback(`❌ Network error: ${error.message}`, 'error');
    }
  }

  async loadMarketMakerStats() {
    console.log('=== Loading market maker stats ===');
    try {
      const response = await fetch('/api/market_maker/stats');
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const stats = await response.json();
      console.log('Market maker stats received:', stats);

      const container = document.getElementById('mmStatsContainer');
      console.log('Container element:', container);
      
      if (!container) {
        console.error('ERROR: mmStatsContainer element not found!');
        return;
      }

      if (!stats || Object.keys(stats).length === 0) {
        console.log('No market maker stats available');
        container.innerHTML = '<p>No market makers active</p>';
        return;
      }

      // Clear container and add stats
      container.innerHTML = '';
      
      for (const [symbol, stat] of Object.entries(stats)) {
        console.log(`Creating card for ${symbol}:`, stat);
        
        const statElement = document.createElement('div');
        statElement.className = 'mm-stat-card';
        statElement.innerHTML = `
          <h4>${symbol}</h4>
          <div class="stat-grid">
            <div class="stat-item">
              <span class="label">Strategy:</span>
              <span class="value">${stat.strategy_type || 'Unknown'}</span>
            </div>
            <div class="stat-item">
              <span class="label">Position:</span>
              <span class="value">${stat.position || 0}</span>
            </div>
            <div class="stat-item">
              <span class="label">P&L:</span>
              <span class="value ${(stat.pnl || 0) >= 0 ? 'text-success' : 'text-danger'}">
                $${(stat.pnl || 0).toFixed(2)}
              </span>
            </div>
            <div class="stat-item">
              <span class="label">Trades:</span>
              <span class="value">${stat.trade_count || 0}</span>
            </div>
            <div class="stat-item">
              <span class="label">Active Orders:</span>
              <span class="value">${stat.active_orders || 0}</span>
            </div>
          </div>
        `;
        container.appendChild(statElement);
      }
      
      console.log('Market maker stats loaded successfully');
      console.log('Container contents:', container.innerHTML);
      
    } catch (error) {
      console.error('ERROR loading market maker stats:', error);
      const container = document.getElementById('mmStatsContainer');
      if (container) {
        container.innerHTML = `<p>Error loading market maker stats: ${error.message}</p>`;
      }
    }
  }

  async loadRiskData() {
    console.log('Loading risk data...');
    try {
      const response = await fetch('/api/risk/summary/trader1');
      const riskData = await response.json();
      console.log('Risk data received:', riskData);
      
      // Update risk displays
      const positionLimits = document.getElementById('positionLimits');
      const pnlSummary = document.getElementById('pnlSummary');
      const riskViolations = document.getElementById('riskViolations');
      
      console.log('Risk elements found:', { positionLimits, pnlSummary, riskViolations });
      
      if (positionLimits) {
        positionLimits.innerHTML = `
          <div class="stat-item">
            <span class="label">Total Exposure:</span>
            <span class="value">${riskData.total_exposure?.toFixed(2) || '0.00'}</span>
          </div>
          <div class="stat-item">
            <span class="label">Position Count:</span>
            <span class="value">${riskData.position_count || 0}</span>
          </div>
          <div class="stat-item">
            <span class="label">Max Limit:</span>
            <span class="value">${riskData.max_position_limit?.toLocaleString() || '10,000'}</span>
          </div>
          <div class="stat-item">
            <span class="label">Utilization:</span>
            <span class="value">${(riskData.current_utilization * 100)?.toFixed(1) || '0.0'}%</span>
          </div>
        `;
      }
      
      if (pnlSummary) {
        const totalPnl = riskData.total_pnl || 0;
        const dailyPnl = riskData.daily_pnl || 0;
        pnlSummary.innerHTML = `
          <div class="stat-item">
            <span class="label">Total P&L:</span>
            <span class="value ${totalPnl >= 0 ? 'text-success' : 'text-danger'}">
              $${totalPnl.toFixed(2)}
            </span>
          </div>
          <div class="stat-item">
            <span class="label">Daily P&L:</span>
            <span class="value ${dailyPnl >= 0 ? 'text-success' : 'text-danger'}">
              $${dailyPnl.toFixed(2)}
            </span>
          </div>
          <div class="stat-item">
            <span class="label">Risk Level:</span>
            <span class="value">${riskData.risk_level || 'LOW'}</span>
          </div>
        `;
      }

      if (riskViolations) {
        const violations = riskData.violations || [];
        if (violations.length > 0) {
          riskViolations.innerHTML = violations.map(v => 
            `<div class="violation-item text-warning">⚠️ ${v.message || v}</div>`
          ).join('');
        } else {
          riskViolations.innerHTML = '<div class="text-success">✅ No current violations</div>';
        }
      }
      
      console.log('Risk data loaded successfully');
    } catch (error) {
      console.error('Error loading risk data:', error);
    }
  }

  async handleReplaySetup(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const config = {
      type: formData.get('type') || document.getElementById('dataType').value,
      symbol: formData.get('symbol') || document.getElementById('replaySymbol').value,
      days: parseInt(document.getElementById('replayDays').value),
      base_price: parseFloat(document.getElementById('basePrice').value)
    };

    try {
      const response = await fetch('/api/replay/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });

      const result = await response.json();
      
      if (response.ok) {
        this.showFeedback('✅ Replay data loaded successfully', 'success');
      } else {
        this.showFeedback(`❌ Error: ${result.error}`, 'error');
      }
    } catch (error) {
      this.showFeedback(`❌ Network error: ${error.message}`, 'error');
    }
  }

  async startReplay() {
    try {
      const response = await fetch('/api/replay/start', { method: 'POST' });
      const result = await response.json();
      
      if (response.ok) {
        this.showFeedback('▶️ Replay started', 'success');
        this.monitorReplayStats();
      } else {
        this.showFeedback(`❌ Error: ${result.error}`, 'error');
      }
    } catch (error) {
      this.showFeedback(`❌ Network error: ${error.message}`, 'error');
    }
  }

  async stopReplay() {
    try {
      const response = await fetch('/api/replay/stop', { method: 'POST' });
      const result = await response.json();
      
      if (response.ok) {
        this.showFeedback('⏹️ Replay stopped', 'success');
      } else {
        this.showFeedback(`❌ Error: ${result.error}`, 'error');
      }
    } catch (error) {
      this.showFeedback(`❌ Network error: ${error.message}`, 'error');
    }
  }

  async monitorReplayStats() {
    try {
      const response = await fetch('/api/replay/stats');
      const stats = await response.json();

      document.getElementById('eventsProcessed').textContent = stats.events_processed || 0;
      document.getElementById('ordersSubmitted').textContent = stats.orders_submitted || 0;
      document.getElementById('tradesGenerated').textContent = stats.trades_generated || 0;

      // Continue monitoring if replay is running
      if (stats.is_running) {
        setTimeout(() => this.monitorReplayStats(), 2000);
      }
    } catch (error) {
      console.error('Error monitoring replay stats:', error);
    }
  }

  async loadReplayData() {
    try {
      const response = await fetch('/api/replay/stats');
      const stats = await response.json();

      // Update replay statistics
      const eventsProcessedEl = document.getElementById('eventsProcessed');
      const ordersSubmittedEl = document.getElementById('ordersSubmitted');
      const tradesGeneratedEl = document.getElementById('tradesGenerated');

      if (eventsProcessedEl) eventsProcessedEl.textContent = stats.events_processed || 0;
      if (ordersSubmittedEl) ordersSubmittedEl.textContent = stats.orders_submitted || 0;
      if (tradesGeneratedEl) tradesGeneratedEl.textContent = stats.trades_generated || 0;

      // Show replay status
      const isRunning = stats.is_running;
      this.showFeedback(
        isRunning ? 'Replay is currently running' : 'Replay is stopped',
        isRunning ? 'success' : 'info'
      );

    } catch (error) {
      console.error('Error loading replay data:', error);
      this.showFeedback('Error loading replay data', 'error');
    }
  }

  startPerformanceMonitoring() {
    // Update engine status and performance metrics every 5 seconds
    setInterval(async () => {
      try {
        const response = await fetch('/api/health');
        const health = await response.json();
        
        document.getElementById('engineStatus').textContent = health.status || 'Unknown';
        document.getElementById('ordersPerSec').textContent = 
          health.engine_stats?.orders_per_second?.toFixed(1) || '0';
      } catch (error) {
        document.getElementById('engineStatus').textContent = 'Error';
      }
    }, 5000);
  }

  loadInitialData() {
    // Load initial market data
    this.updateMarketData();
    
    // Test loading market maker data immediately
    console.log('Testing market maker data loading on startup...');
    this.loadMarketMakerStats();
    
    // Refresh data every 2 seconds
    setInterval(() => {
      if (document.querySelector('.tab-content.active').id === 'trading') {
        this.updateMarketData();
      }
    }, 2000);
  }

  // Test function to verify tabs work
  testTabs() {
    console.log('Testing all tab functions...');
    this.loadMarketMakerStats();
    this.loadRiskData();  
    this.loadReplayData();
  }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new OrderBookSimulator();
});