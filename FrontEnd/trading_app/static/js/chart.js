/**
 * TradingView Lightweight Charts - Candlestick, EMA 12/26, SMA 200, Buy markers.
 * Fetches data from /api/chart-data/?symbol=...&period=...&interval=...
 */
(function() {
  'use strict';

  var chart = null;
  var candlestickSeries = null;
  var ema12Line = null;
  var ema26Line = null;
  var sma200Line = null;

  var loadingEl = document.getElementById('chart-loading');
  var errorEl = document.getElementById('chart-error');
  var rootEl = document.getElementById('chart-root');

  function showLoading(v) {
    if (loadingEl) loadingEl.style.display = v ? 'block' : 'none';
  }
  function showError(msg) {
    if (errorEl) {
      errorEl.textContent = msg || '';
      errorEl.style.display = msg ? 'block' : 'none';
    }
  }
  function showChart(v) {
    if (rootEl) rootEl.style.display = v ? 'block' : 'none';
  }

  function ensureChart() {
    if (chart) return;
    if (!rootEl) return;
    rootEl.innerHTML = '';
    chart = LightweightCharts.createChart(rootEl, {
      layout: {
        background: { color: '#1a1a2e' },
        textColor: '#e0e0e0',
      },
      grid: {
        vertLines: { color: '#2a2a3e' },
        horzLines: { color: '#2a2a3e' },
      },
      width: Math.max((rootEl && rootEl.clientWidth) || 0, 600),
      height: 450,
      rightPriceScale: { borderColor: '#444' },
      timeScale: { borderColor: '#444', timeVisible: true, secondsVisible: false },
    });

    candlestickSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderDownColor: '#ef5350',
      borderUpColor: '#26a69a',
    });
    ema12Line = chart.addLineSeries({ color: 'rgb(255, 255, 0)', lineWidth: 2 });
    ema26Line = chart.addLineSeries({ color: 'orange', lineWidth: 2 });
    sma200Line = chart.addLineSeries({ color: 'blue', lineWidth: 2 });

    if (window.ResizeObserver && rootEl) {
      var ro = new ResizeObserver(function() {
        if (chart && rootEl) chart.applyOptions({ width: rootEl.clientWidth });
      });
      ro.observe(rootEl);
    }
  }

  function destroyChart() {
    if (chart) {
      chart.remove();
      chart = null;
      candlestickSeries = null;
      ema12Line = null;
      ema26Line = null;
      sma200Line = null;
    }
  }

  function load(symbol, period, interval) {
    symbol = (symbol || 'AAPL').trim().toUpperCase();
    period = period || '1y';
    interval = interval || '1d';

    showLoading(true);
    showError('');
    showChart(false);

    var url = '/api/chart-data/?symbol=' + encodeURIComponent(symbol) + '&period=' + encodeURIComponent(period) + '&interval=' + encodeURIComponent(interval);

    fetch(url)
      .then(function(res) { return res.json(); })
      .then(function(data) {
        showLoading(false);
        if (data.error) {
          showError(data.error);
          return;
        }

        ensureChart();

        candlestickSeries.setData(data.candlestick_data || []);
        ema12Line.setData(data.ema12_data || []);
        ema26Line.setData(data.ema26_data || []);
        sma200Line.setData(data.sma200_data || []);
        candlestickSeries.setMarkers(data.markers_data || []);

        chart.timeScale().fitContent();
        showChart(true);
      })
      .catch(function(err) {
        showLoading(false);
        showError('Failed to load chart: ' + (err && err.message ? err.message : 'Unknown error'));
      });
  }

  window.TradingChart = { load: load, destroy: destroyChart };
})();
