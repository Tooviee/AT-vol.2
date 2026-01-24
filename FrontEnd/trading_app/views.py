"""
Views for trading_app - Core and Enhanced features
"""

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.db.models import Sum, Count, Q
from datetime import datetime, date, timedelta
from typing import Dict, Any

from .db_adapter import get_db_adapter


@login_required
def dashboard(request):
    """Main dashboard view"""
    db = get_db_adapter()
    
    try:
        # Get current positions
        positions = db.get_current_positions()
        
        # Get today's P&L
        daily_pnl = db.get_daily_pnl()
        
        # Get recent trades (last 10)
        recent_trades = db.get_recent_trades(limit=10)
        
        # Calculate totals from positions
        total_positions_value_usd = sum(
            (pos.get('current_price_usd', 0) or 0) * pos.get('quantity', 0)
            for pos in positions
        )
        total_unrealized_pnl_krw = sum(
            pos.get('unrealized_pnl_krw', 0) or 0
            for pos in positions
        )
        
        # Get P&L history for chart (last 30 days)
        pnl_history = db.get_pnl_history(days=30)
        
        context = {
            'positions': positions,
            'position_count': len(positions),
            'daily_pnl': daily_pnl or {},
            'recent_trades': recent_trades,
            'total_positions_value_usd': total_positions_value_usd,
            'total_unrealized_pnl_krw': total_unrealized_pnl_krw,
            'pnl_history': pnl_history,
        }
        
        return render(request, 'trading_app/dashboard.html', context)
    
    except Exception as e:
        context = {
            'error': str(e),
            'positions': [],
            'daily_pnl': {},
            'recent_trades': [],
        }
        return render(request, 'trading_app/dashboard.html', context)


@login_required
def trading_log(request):
    """Trading log view - shows all executed trades"""
    db = get_db_adapter()
    
    # Get filter parameters
    symbol_filter = request.GET.get('symbol', '')
    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    limit = int(request.GET.get('limit', 100))
    
    try:
        # Get trades
        if date_from and date_to:
            try:
                from_date = datetime.strptime(date_from, '%Y-%m-%d').date()
                to_date = datetime.strptime(date_to, '%Y-%m-%d').date()
                trades = db.get_trades_by_date_range(from_date, to_date)
            except ValueError:
                trades = db.get_recent_trades(limit=limit, symbol=symbol_filter if symbol_filter else None)
        else:
            trades = db.get_recent_trades(limit=limit, symbol=symbol_filter if symbol_filter else None)
        
        # Filter by symbol if provided
        if symbol_filter:
            trades = [t for t in trades if t['symbol'] == symbol_filter]
        
        # Get unique symbols for filter dropdown
        all_trades = db.get_recent_trades(limit=1000)
        symbols = sorted(set(t['symbol'] for t in all_trades))
        
        context = {
            'trades': trades,
            'symbols': symbols,
            'selected_symbol': symbol_filter,
            'date_from': date_from,
            'date_to': date_to,
            'limit': limit,
        }
        
        return render(request, 'trading_app/trading_log.html', context)
    
    except Exception as e:
        context = {
            'error': str(e),
            'trades': [],
            'symbols': [],
        }
        return render(request, 'trading_app/trading_log.html', context)


@login_required
def positions(request):
    """Positions view - shows current holdings"""
    db = get_db_adapter()
    
    try:
        positions = db.get_current_positions()
        
        # Calculate summary statistics
        total_quantity = sum(pos.get('quantity', 0) for pos in positions)
        total_value_usd = sum(
            (pos.get('current_price_usd', 0) or 0) * pos.get('quantity', 0)
            for pos in positions
        )
        total_unrealized_pnl_krw = sum(
            pos.get('unrealized_pnl_krw', 0) or 0
            for pos in positions
        )
        total_unrealized_pnl_usd = sum(
            pos.get('unrealized_pnl_usd', 0) or 0
            for pos in positions
        )
        
        # Sort by unrealized P&L (descending)
        positions_sorted = sorted(
            positions,
            key=lambda x: x.get('unrealized_pnl_krw', 0) or 0,
            reverse=True
        )
        
        context = {
            'positions': positions_sorted,
            'position_count': len(positions),
            'total_quantity': total_quantity,
            'total_value_usd': total_value_usd,
            'total_unrealized_pnl_krw': total_unrealized_pnl_krw,
            'total_unrealized_pnl_usd': total_unrealized_pnl_usd,
        }
        
        return render(request, 'trading_app/positions.html', context)
    
    except Exception as e:
        context = {
            'error': str(e),
            'positions': [],
        }
        return render(request, 'trading_app/positions.html', context)


@login_required
def performance(request):
    """Performance analytics view"""
    db = get_db_adapter()
    
    # Get time period filter
    days = int(request.GET.get('days', 30))
    
    try:
        # Get P&L history
        pnl_history = db.get_pnl_history(days=days)
        
        # Get trade statistics
        stats = db.get_trade_statistics(days=days)
        
        # Calculate performance metrics
        if pnl_history:
            total_realized_pnl = sum(pnl.get('realized_pnl_krw', 0) for pnl in pnl_history)
            total_trades = sum(pnl.get('total_trades', 0) for pnl in pnl_history)
            avg_daily_pnl = total_realized_pnl / len(pnl_history) if pnl_history else 0
            
            # Calculate win rate
            total_wins = sum(
                pnl.get('win_rate', 0) * pnl.get('total_trades', 0) / 100
                for pnl in pnl_history
            )
            win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        else:
            total_realized_pnl = 0
            total_trades = 0
            avg_daily_pnl = 0
            win_rate = 0
        
        # Get best and worst days
        if pnl_history:
            best_day = max(pnl_history, key=lambda x: x.get('realized_pnl_krw', 0))
            worst_day = min(pnl_history, key=lambda x: x.get('realized_pnl_krw', 0))
        else:
            best_day = None
            worst_day = None
        
        context = {
            'pnl_history': pnl_history,
            'stats': stats,
            'days': days,
            'total_realized_pnl': total_realized_pnl,
            'total_trades': total_trades,
            'avg_daily_pnl': avg_daily_pnl,
            'win_rate': win_rate,
            'best_day': best_day,
            'worst_day': worst_day,
        }
        
        return render(request, 'trading_app/performance.html', context)
    
    except Exception as e:
        context = {
            'error': str(e),
            'pnl_history': [],
            'stats': {},
        }
        return render(request, 'trading_app/performance.html', context)


@login_required
def risk_dashboard(request):
    """Risk management dashboard"""
    db = get_db_adapter()
    
    try:
        # Get current positions
        positions = db.get_current_positions()
        
        # Get circuit breaker events
        cb_events = db.get_circuit_breaker_events(limit=10)
        
        # Calculate risk metrics
        total_positions_value_usd = sum(
            (pos.get('current_price_usd', 0) or 0) * pos.get('quantity', 0)
            for pos in positions
        )
        
        # Get today's P&L
        daily_pnl = db.get_daily_pnl()
        
        # Get recent trades to calculate consecutive losses
        recent_trades = db.get_recent_trades(limit=50)
        
        # Count consecutive losses (simplified - would need order matching)
        consecutive_losses = 0
        if recent_trades:
            # This is a simplified calculation
            # In reality, you'd need to match buy/sell pairs
            pass
        
        context = {
            'positions': positions,
            'position_count': len(positions),
            'total_positions_value_usd': total_positions_value_usd,
            'daily_pnl': daily_pnl or {},
            'circuit_breaker_events': cb_events,
            'consecutive_losses': consecutive_losses,
        }
        
        return render(request, 'trading_app/risk_dashboard.html', context)
    
    except Exception as e:
        context = {
            'error': str(e),
            'positions': [],
            'circuit_breaker_events': [],
        }
        return render(request, 'trading_app/risk_dashboard.html', context)


@login_required
def market_status(request):
    """Market status and system health view"""
    # This would ideally connect to the running trading system
    # For now, we'll show database status and recent activity
    
    db = get_db_adapter()
    
    try:
        # Get recent activity
        recent_trades = db.get_recent_trades(limit=10)
        recent_orders = db.get_recent_orders(limit=10)
        active_orders = db.get_active_orders()
        
        # Get today's P&L
        daily_pnl = db.get_daily_pnl()
        
        context = {
            'recent_trades': recent_trades,
            'recent_orders': recent_orders,
            'active_orders': active_orders,
            'daily_pnl': daily_pnl or {},
            'database_status': 'Connected',  # Would check actual connection
        }
        
        return render(request, 'trading_app/market_status.html', context)
    
    except Exception as e:
        context = {
            'error': str(e),
            'database_status': 'Error',
        }
        return render(request, 'trading_app/market_status.html', context)


@login_required
def ml_insights(request):
    """ML model insights view"""
    # This would show ML model status, confidence scores, etc.
    # For now, placeholder
    
    db = get_db_adapter()
    
    try:
        # Get recent trades (would include ML confidence if available)
        recent_trades = db.get_recent_trades(limit=50)
        
        context = {
            'recent_trades': recent_trades,
            'ml_enabled': True,  # Would check from config
            'model_status': 'Trained',  # Would check actual model status
        }
        
        return render(request, 'trading_app/ml_insights.html', context)
    
    except Exception as e:
        context = {
            'error': str(e),
            'ml_enabled': False,
        }
        return render(request, 'trading_app/ml_insights.html', context)


@login_required
def chart(request):
    """TradingView Lightweight Charts - OHLC, EMA 12/26, SMA 200, buy markers.
    Defaults to first position's symbol when the user has positions."""
    db = get_db_adapter()
    positions = db.get_current_positions()
    default_symbol = (positions[0].get('symbol') if positions else None) or 'AAPL'
    symbol = (request.GET.get('symbol') or default_symbol).strip().upper()[:10]
    period = request.GET.get('period') or '1y'
    interval = request.GET.get('interval') or '1d'
    return render(request, 'trading_app/chart.html', {
        'symbol': symbol,
        'period': period,
        'interval': interval,
    })


# API endpoints for AJAX updates
@login_required
def api_balance(request):
    """API endpoint for balance data"""
    db = get_db_adapter()
    
    try:
        positions = db.get_current_positions()
        daily_pnl = db.get_daily_pnl()
        
        total_positions_value_usd = sum(
            (pos.get('current_price_usd', 0) or 0) * pos.get('quantity', 0)
            for pos in positions
        )
        total_unrealized_pnl_krw = sum(
            pos.get('unrealized_pnl_krw', 0) or 0
            for pos in positions
        )
        
        return JsonResponse({
            'total_positions_value_usd': total_positions_value_usd,
            'total_unrealized_pnl_krw': total_unrealized_pnl_krw,
            'realized_pnl_krw': daily_pnl.get('realized_pnl_krw', 0) if daily_pnl else 0,
            'position_count': len(positions),
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
