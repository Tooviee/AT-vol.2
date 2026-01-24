"""
API views - Chart data and other JSON endpoints.
"""

import re
import logging
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from django.core.cache import cache

logger = logging.getLogger(__name__)

# Allowed for validation
ALLOWED_PERIODS = {'1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'}
ALLOWED_INTERVALS = {'1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'}

# Cache TTL in seconds (5 minutes)
CHART_CACHE_TTL = 300


def _clean_ohlc(df):
    """Fix common yfinance OHLC issues. df must have Open, High, Low, Close."""
    if df is None or df.empty or not all(c in df.columns for c in ['Open', 'High', 'Low', 'Close']):
        return df
    out = df.copy()
    # High >= Open, Close; Low <= Open, Close
    out['High'] = out[['High', 'Open', 'Close']].max(axis=1)
    out['Low'] = out[['Low', 'Open', 'Close']].min(axis=1)
    # High >= Low
    bad = out['High'] < out['Low']
    if bad.any():
        out.loc[bad, 'High'] = out.loc[bad, ['High', 'Low', 'Open', 'Close']].max(axis=1)
        out.loc[bad, 'Low'] = out.loc[bad, ['High', 'Low', 'Open', 'Close']].min(axis=1)
    return out


def _build_chart_response(symbol: str, period: str, interval: str) -> dict:
    """Fetch OHLC via yfinance, compute indicators, return structure for LightweightCharts."""
    import pandas as pd
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)

    if hist is None or hist.empty:
        return {'error': 'No historical data for this symbol/period/interval.'}

    # Normalize columns (yfinance: Open, High, Low, Close)
    for c in ['Open', 'High', 'Low', 'Close']:
        if c not in hist.columns:
            return {'error': f'Missing OHLC column: {c}'}

    hist = _clean_ohlc(hist)
    hist = hist.dropna(subset=['Open', 'High', 'Low', 'Close'])
    if hist.empty:
        return {'error': 'No valid OHLC data after cleaning.'}

    # EMA 12, 26 and SMA 200 (match strategy.py)
    hist['EMA_12'] = hist['Close'].ewm(span=12, adjust=False).mean()
    hist['EMA_26'] = hist['Close'].ewm(span=26, adjust=False).mean()
    hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
    # MACD line for buy signal: EMA12 - EMA26; cross above zero
    hist['MACD'] = hist['EMA_12'] - hist['EMA_26']
    hist['MACD_prev'] = hist['MACD'].shift(1)
    hist['MACD_cross_above_zero'] = (hist['MACD_prev'] <= 0) & (hist['MACD'] > 0)

    candlestick_data = []
    ema12_data = []
    ema26_data = []
    sma200_data = []
    markers_data = []

    for ts, row in hist.iterrows():
        # LightweightCharts: time as 'YYYY-MM-DD' for daily; for intraday can use 'YYYY-MM-DD' or ISO
        if hasattr(ts, 'strftime'):
            t_str = ts.strftime('%Y-%m-%d') if interval in ('1d', '5d', '1wk', '1mo', '3mo') else ts.isoformat()
        else:
            t_str = str(ts)[:10]

        o, h, l, c = float(row['Open']), float(row['High']), float(row['Low']), float(row['Close'])
        candlestick_data.append({'time': t_str, 'open': o, 'high': h, 'low': l, 'close': c})

        if pd.notna(row.get('EMA_12')):
            ema12_data.append({'time': t_str, 'value': float(row['EMA_12'])})
        if pd.notna(row.get('EMA_26')):
            ema26_data.append({'time': t_str, 'value': float(row['EMA_26'])})
        if pd.notna(row.get('SMA_200')):
            sma200_data.append({'time': t_str, 'value': float(row['SMA_200'])})

        if row.get('MACD_cross_above_zero') and pd.notna(row.get('MACD_cross_above_zero')):
            markers_data.append({
                'time': t_str,
                'position': 'belowBar',
                'color': 'rgba(0, 255, 0, 0.7)',
                'shape': 'arrowUp',
                'text': 'Buy',
            })

    return {
        'candlestick_data': candlestick_data,
        'ema12_data': ema12_data,
        'ema26_data': ema26_data,
        'sma200_data': sma200_data,
        'markers_data': markers_data,
        'symbol': symbol,
        'period': period,
        'interval': interval,
    }


@require_GET
def chart_data_api(request):
    """
    GET /api/chart-data/?symbol=AAPL&period=1y&interval=1d

    Returns JSON for TradingView Lightweight Charts:
    - candlestick_data: [{ time, open, high, low, close }]
    - ema12_data, ema26_data, sma200_data: [{ time, value }]
    - markers_data: [{ time, position, color, shape, text }] (buy signals)
    """
    symbol = (request.GET.get('symbol') or 'AAPL').strip().upper()
    period = (request.GET.get('period') or '1y').strip().lower()
    interval = (request.GET.get('interval') or '1d').strip().lower()

    # Validate symbol: alphanumeric, 1–10 chars
    if not symbol or not re.match(r'^[A-Z0-9.]{1,10}$', symbol):
        return JsonResponse({'error': 'Invalid symbol. Use 1–10 alphanumeric characters.'}, status=400)

    if period not in ALLOWED_PERIODS:
        period = '1y'
    if interval not in ALLOWED_INTERVALS:
        interval = '1d'

    cache_key = f'chart_data:{symbol}:{period}:{interval}'
    data = cache.get(cache_key)
    if data is not None:
        return JsonResponse(data)

    try:
        data = _build_chart_response(symbol, period, interval)
    except Exception as e:
        logger.exception('chart_data_api error')
        return JsonResponse({'error': f'Failed to build chart data: {str(e)}'}, status=500)

    if 'error' in data:
        return JsonResponse(data, status=404)

    cache.set(cache_key, data, CHART_CACHE_TTL)
    return JsonResponse(data)
