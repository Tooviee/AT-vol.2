"""
Dashboard - FastAPI web dashboard for monitoring and control.
Includes session-based authentication and real-time updates.
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import uvicorn

# Session storage (in production, use Redis or database)
sessions: Dict[str, Dict[str, Any]] = {}

# Security
security = HTTPBasic()

app = FastAPI(
    title="USA Auto Trader Dashboard",
    description="Trading system monitoring dashboard",
    version="1.0.0"
)

# Templates directory
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
if os.path.exists(templates_dir):
    templates = Jinja2Templates(directory=templates_dir)
else:
    templates = None


class DashboardConfig:
    """Dashboard configuration"""
    def __init__(self):
        self.host = os.getenv("DASHBOARD_HOST", "127.0.0.1")
        self.port = int(os.getenv("DASHBOARD_PORT", "8000"))
        self.username = os.getenv("DASHBOARD_USER", "admin")
        self.password = os.getenv("DASHBOARD_PASS", "changeme")
        self.allowed_ips = os.getenv("DASHBOARD_ALLOWED_IPS", "127.0.0.1").split(",")
        self.session_timeout = int(os.getenv("DASHBOARD_SESSION_TIMEOUT", "30"))  # minutes


config = DashboardConfig()


# Trader reference (set from main.py)
trader = None


def set_trader(trader_instance: Any) -> None:
    """Set the trader instance for dashboard access"""
    global trader
    trader = trader_instance


def verify_ip(request: Request) -> bool:
    """Verify client IP is allowed"""
    client_ip = request.client.host
    return client_ip in config.allowed_ips or "0.0.0.0" in config.allowed_ips


def create_session(username: str) -> str:
    """Create a new session"""
    session_id = secrets.token_urlsafe(32)
    sessions[session_id] = {
        "username": username,
        "created_at": datetime.now(),
        "expires_at": datetime.now() + timedelta(minutes=config.session_timeout)
    }
    return session_id


def verify_session(request: Request) -> Optional[Dict]:
    """Verify session from cookie"""
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in sessions:
        return None
    
    session = sessions[session_id]
    if datetime.now() > session["expires_at"]:
        del sessions[session_id]
        return None
    
    # Extend session
    session["expires_at"] = datetime.now() + timedelta(minutes=config.session_timeout)
    return session


def require_auth(request: Request) -> Dict:
    """Dependency to require authentication"""
    session = verify_session(request)
    if not session:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return session


# ===== Routes =====

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Dashboard home page"""
    session = verify_session(request)
    if not session:
        return RedirectResponse(url="/login")
    
    if templates:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "title": "Dashboard"
        })
    
    return HTMLResponse(content=generate_dashboard_html())


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page"""
    return HTMLResponse(content=generate_login_html())


@app.post("/login")
async def login(request: Request, response: Response):
    """Handle login"""
    form_data = await request.form()
    username = form_data.get("username")
    password = form_data.get("password")
    
    if username == config.username and password == config.password:
        session_id = create_session(username)
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            max_age=config.session_timeout * 60
        )
        return response
    
    raise HTTPException(status_code=401, detail="Invalid credentials")


@app.get("/logout")
async def logout(request: Request):
    """Handle logout"""
    session_id = request.cookies.get("session_id")
    if session_id and session_id in sessions:
        del sessions[session_id]
    
    response = RedirectResponse(url="/login")
    response.delete_cookie("session_id")
    return response


# ===== API Endpoints =====

@app.get("/api/status")
async def get_status(session: Dict = Depends(require_auth)):
    """Get system status"""
    if not trader:
        return {"error": "Trader not connected"}
    
    try:
        return {
            "mode": trader.config.mode if trader.config else "unknown",
            "is_running": trader.is_running,
            "market_status": trader.market_hours.get_status_dict() if trader.market_hours else {},
            "balance": trader.balance_tracker.get_summary() if trader.balance_tracker else {},
            "circuit_breaker": trader.circuit_breaker.get_status() if trader.circuit_breaker else {},
            "health": trader.health_monitor.get_status() if trader.health_monitor else {}
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/positions")
async def get_positions(session: Dict = Depends(require_auth)):
    """Get current positions"""
    if not trader or not trader.balance_tracker:
        return {"positions": []}
    
    try:
        positions = trader.balance_tracker.get_all_positions()
        return {
            "positions": [
                {
                    "symbol": symbol,
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_percent": pos.unrealized_pnl_percent
                }
                for symbol, pos in positions.items()
            ]
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/orders")
async def get_orders(session: Dict = Depends(require_auth)):
    """Get recent orders"""
    if not trader or not trader.order_manager:
        return {"orders": []}
    
    try:
        orders = trader.order_manager.active_orders
        return {
            "orders": [
                order.to_dict() for order in orders.values()
            ]
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/pnl")
async def get_pnl(session: Dict = Depends(require_auth)):
    """Get P&L data"""
    if not trader or not trader.database:
        return {"pnl": {}}
    
    try:
        daily_pnl = trader.database.get_daily_pnl()
        return {
            "realized_pnl_krw": daily_pnl.realized_pnl_krw if daily_pnl else 0,
            "unrealized_pnl_krw": trader.balance_tracker.get_unrealized_pnl() if trader.balance_tracker else 0,
            "total_trades": daily_pnl.total_trades if daily_pnl else 0,
            "win_rate": daily_pnl.win_rate if daily_pnl else 0
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/chart-data")
async def get_chart_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    session: Dict = Depends(require_auth),
):
    """
    Export strategy data in JSON format for TradingView Lightweight Charts.

    - **symbol** (required): Stock symbol (e.g. AAPL)
    - **period** (optional): Time period (default: 1y)
    - **interval** (optional): Data interval (default: 1d)

    Returns candlestick_data, ema12_data, ema26_data, sma200_data, markers_data.
    Times are UNIX timestamps in seconds. Markers come from Buy_Signal column when present.
    """
    if not trader:
        raise HTTPException(status_code=503, detail="Trader not connected")
    if not trader.kis_api or not trader.strategy:
        raise HTTPException(status_code=503, detail="Trader components not available")

    symbol = symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol is required")

    hist = trader.kis_api.get_historical_data(symbol, period=period, interval=interval)
    if hist is None or hist.empty:
        raise HTTPException(status_code=404, detail=f"No data for symbol {symbol}")

    df = trader.strategy.calculate_indicators(hist.copy())

    def _unix_sec(ts) -> int:
        t = ts if hasattr(ts, "timestamp") else pd.Timestamp(ts)
        return int(t.timestamp())

    candlestick_data = []
    for ts, row in df.iterrows():
        o, h, l, c = row.get("Open"), row.get("High"), row.get("Low"), row.get("Close")
        if pd.notna(o) and pd.notna(h) and pd.notna(l) and pd.notna(c):
            candlestick_data.append({
                "time": _unix_sec(ts),
                "open": round(float(o), 4),
                "high": round(float(h), 4),
                "low": round(float(l), 4),
                "close": round(float(c), 4),
            })

    ema12_data = [
        {"time": _unix_sec(ts), "value": round(float(row["EMA_12"]), 4)}
        for ts, row in df.iterrows()
        if "EMA_12" in row and pd.notna(row.get("EMA_12"))
    ]
    ema26_data = [
        {"time": _unix_sec(ts), "value": round(float(row["EMA_26"]), 4)}
        for ts, row in df.iterrows()
        if "EMA_26" in row and pd.notna(row.get("EMA_26"))
    ]
    sma200_data = [
        {"time": _unix_sec(ts), "value": round(float(row["SMA_200"]), 4)}
        for ts, row in df.iterrows()
        if "SMA_200" in row and pd.notna(row.get("SMA_200"))
    ]

    markers_data = []
    if "Buy_Signal" in df.columns:
        buy = df.loc[df["Buy_Signal"] == 1]
        for ts in buy.index:
            markers_data.append({
                "time": _unix_sec(ts),
                "position": "belowBar",
                "color": "green",
                "shape": "arrowUp",
                "text": "BUY",
            })

    return {
        "candlestick_data": candlestick_data,
        "ema12_data": ema12_data,
        "ema26_data": ema26_data,
        "sma200_data": sma200_data,
        "markers_data": markers_data,
    }


@app.post("/api/circuit-breaker/reset")
async def reset_circuit_breaker(session: Dict = Depends(require_auth)):
    """Reset circuit breaker"""
    if not trader or not trader.circuit_breaker:
        raise HTTPException(status_code=400, detail="Circuit breaker not available")
    
    trader.circuit_breaker.reset(force=True)
    return {"success": True, "message": "Circuit breaker reset"}


# ===== HTML Templates =====

def generate_login_html() -> str:
    """Generate login page HTML"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Trading Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
        }
        .login-container {
            background: rgba(255,255,255,0.1);
            padding: 2rem;
            border-radius: 1rem;
            backdrop-filter: blur(10px);
            width: 100%;
            max-width: 400px;
        }
        h1 { text-align: center; margin-bottom: 2rem; }
        .form-group { margin-bottom: 1rem; }
        label { display: block; margin-bottom: 0.5rem; opacity: 0.8; }
        input {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 0.5rem;
            background: rgba(255,255,255,0.1);
            color: #fff;
            font-size: 1rem;
        }
        input:focus { outline: none; border-color: #4ade80; }
        button {
            width: 100%;
            padding: 0.75rem;
            background: #4ade80;
            color: #1a1a2e;
            border: none;
            border-radius: 0.5rem;
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            margin-top: 1rem;
        }
        button:hover { background: #22c55e; }
    </style>
</head>
<body>
    <div class="login-container">
        <h1>🚀 Trading Dashboard</h1>
        <form action="/login" method="post">
            <div class="form-group">
                <label for="username">Username</label>
                <input type="text" id="username" name="username" required>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required>
            </div>
            <button type="submit">Login</button>
        </form>
    </div>
</body>
</html>
"""


def generate_dashboard_html() -> str:
    """Generate dashboard page HTML"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }
        .header {
            background: rgba(0,0,0,0.3);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { font-size: 1.5rem; }
        .header a { color: #4ade80; text-decoration: none; }
        .container { padding: 2rem; max-width: 1400px; margin: 0 auto; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; }
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 1rem;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
        }
        .card h2 { font-size: 1rem; opacity: 0.8; margin-bottom: 1rem; }
        .stat { font-size: 2rem; font-weight: bold; }
        .stat.positive { color: #4ade80; }
        .stat.negative { color: #f87171; }
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.875rem;
        }
        .status-open { background: #4ade80; color: #000; }
        .status-closed { background: #6b7280; color: #fff; }
        table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        th, td { padding: 0.75rem; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); }
        th { opacity: 0.7; font-weight: normal; }
        .refresh-btn {
            background: #4ade80;
            color: #000;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Trading Dashboard</h1>
        <div>
            <button class="refresh-btn" onclick="refreshData()">Refresh</button>
            <a href="/logout" style="margin-left: 1rem;">Logout</a>
        </div>
    </div>
    
    <div class="container">
        <div class="grid">
            <div class="card">
                <h2>BALANCE</h2>
                <div class="stat" id="balance">Loading...</div>
            </div>
            <div class="card">
                <h2>TODAY'S P&L</h2>
                <div class="stat" id="pnl">Loading...</div>
            </div>
            <div class="card">
                <h2>MARKET STATUS</h2>
                <div id="market-status">Loading...</div>
            </div>
            <div class="card">
                <h2>CIRCUIT BREAKER</h2>
                <div id="circuit-breaker">Loading...</div>
            </div>
        </div>
        
        <div class="card" style="margin-top: 1.5rem;">
            <h2>POSITIONS</h2>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Qty</th>
                        <th>Avg Price</th>
                        <th>Current</th>
                        <th>P&L</th>
                    </tr>
                </thead>
                <tbody id="positions-table">
                    <tr><td colspan="5">Loading...</td></tr>
                </tbody>
            </table>
        </div>

        <div class="card" style="margin-top: 1.5rem;">
            <h2>PRICE CHART</h2>
            <div style="display:flex; gap:0.5rem; flex-wrap:wrap; align-items:center; margin-bottom:1rem;">
                <label>Symbol</label>
                <select id="chart-symbol" style="padding:0.35rem 0.5rem; border-radius:0.5rem; background:rgba(255,255,255,0.1); color:#fff; border:1px solid rgba(255,255,255,0.2); min-width:100px;"></select>
                <label>Period</label>
                <select id="chart-period" style="padding:0.35rem 0.5rem; border-radius:0.5rem; background:rgba(255,255,255,0.1); color:#fff; border:1px solid rgba(255,255,255,0.2);">
                    <option value="1y">1y</option><option value="6mo">6mo</option><option value="3mo">3mo</option><option value="1mo">1mo</option>
                </select>
                <button type="button" id="chart-load" class="refresh-btn">Load</button>
            </div>
            <div id="chart-loading" style="display:none;">Loading chart…</div>
            <div id="chart-error" style="display:none; color:#f87171;"></div>
            <div id="chart-root" style="min-height:400px;"></div>
        </div>
    </div>

    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <script>
        var dChart = null, dCandle = null, dEma12 = null, dEma26 = null, dSma200 = null;
        window.chartDataLoaded = false;

        function ensureChart() {
            if (dChart) return;
            var root = document.getElementById('chart-root');
            if (!root) return;
            root.innerHTML = '';
            dChart = LightweightCharts.createChart(root, {
                layout: { background: { color: '#1a1a2e' }, textColor: '#e0e0e0' },
                grid: { vertLines: { color: '#2a2a3e' }, horzLines: { color: '#2a2a3e' } },
                width: Math.max(root.clientWidth || 0, 600), height: 400,
                rightPriceScale: { borderColor: '#444' }, timeScale: { borderColor: '#444', timeVisible: true, secondsVisible: false },
            });
            dCandle = dChart.addCandlestickSeries({ upColor: '#26a69a', downColor: '#ef5350', borderDownColor: '#ef5350', borderUpColor: '#26a69a' });
            dEma12 = dChart.addLineSeries({ color: 'rgb(255,255,0)', lineWidth: 2 });
            dEma26 = dChart.addLineSeries({ color: 'orange', lineWidth: 2 });
            dSma200 = dChart.addLineSeries({ color: 'blue', lineWidth: 2 });
            if (window.ResizeObserver && root) new ResizeObserver(function(){ if (dChart && root) dChart.applyOptions({ width: root.clientWidth }); }).observe(root);
        }

        async function loadChart(sym, period, interval) {
            sym = (sym || 'AAPL').trim().toUpperCase();
            period = period || '1y';
            interval = interval || '1d';
            var loading = document.getElementById('chart-loading'), err = document.getElementById('chart-error'), root = document.getElementById('chart-root');
            if (loading) loading.style.display = 'block';
            if (err) { err.textContent = ''; err.style.display = 'none'; }
            if (root) root.style.display = 'none';
            try {
                var res = await fetch('/api/chart-data?symbol=' + encodeURIComponent(sym) + '&period=' + encodeURIComponent(period) + '&interval=' + encodeURIComponent(interval));
                var data = res.ok ? await res.json() : {};
                if (!res.ok) { if (err) { err.textContent = (data.detail || res.statusText || 'Request failed'); err.style.display = 'block'; } return; }
                if (data.error) { if (err) { err.textContent = data.error; err.style.display = 'block'; } return; }
                ensureChart();
                dCandle.setData(data.candlestick_data || []);
                dEma12.setData(data.ema12_data || []);
                dEma26.setData(data.ema26_data || []);
                dSma200.setData(data.sma200_data || []);
                dCandle.setMarkers(data.markers_data || []);
                dChart.timeScale().fitContent();
                if (root) root.style.display = 'block';
                window.chartDataLoaded = true;
            } catch (e) { if (err) { err.textContent = 'Failed to load chart: ' + (e.message || 'Unknown'); err.style.display = 'block'; } }
            if (loading) loading.style.display = 'none';
        }

        async function refreshData() {
            try {
                const status = await fetch('/api/status').then(r => r.json());
                const pnl = await fetch('/api/pnl').then(r => r.json());
                const positions = await fetch('/api/positions').then(r => r.json());
                
                // Update balance
                if (status.balance) {
                    document.getElementById('balance').textContent = 
                        new Intl.NumberFormat('ko-KR').format(status.balance.total_balance_krw || 0) + ' KRW';
                }
                
                // Update P&L
                const totalPnl = (pnl.realized_pnl_krw || 0) + (pnl.unrealized_pnl_krw || 0);
                const pnlEl = document.getElementById('pnl');
                pnlEl.textContent = (totalPnl >= 0 ? '+' : '') + 
                    new Intl.NumberFormat('ko-KR').format(totalPnl) + ' KRW';
                pnlEl.className = 'stat ' + (totalPnl >= 0 ? 'positive' : 'negative');
                
                // Update market status
                if (status.market_status) {
                    const isOpen = status.market_status.can_trade;
                    document.getElementById('market-status').innerHTML = 
                        `<span class="status-badge ${isOpen ? 'status-open' : 'status-closed'}">
                            ${isOpen ? 'OPEN' : 'CLOSED'}
                        </span>
                        <div style="margin-top: 0.5rem; opacity: 0.8;">${status.market_status.message || ''}</div>`;
                }
                
                // Update circuit breaker
                if (status.circuit_breaker) {
                    const canTrade = status.circuit_breaker.can_trade;
                    document.getElementById('circuit-breaker').innerHTML = 
                        `<span class="status-badge ${canTrade ? 'status-open' : 'status-closed'}">
                            ${canTrade ? 'OK' : 'TRIPPED'}
                        </span>
                        <div style="margin-top: 0.5rem; opacity: 0.8;">
                            Losses: ${status.circuit_breaker.consecutive_losses || 0}/${status.circuit_breaker.max_consecutive_losses || 3}
                        </div>`;
                }
                
                // Update positions table
                const tbody = document.getElementById('positions-table');
                if (positions.positions && positions.positions.length > 0) {
                    tbody.innerHTML = positions.positions.map(p => `
                        <tr>
                            <td><strong>${p.symbol}</strong></td>
                            <td>${p.quantity}</td>
                            <td>$${p.avg_price.toFixed(2)}</td>
                            <td>$${p.current_price.toFixed(2)}</td>
                            <td class="${p.unrealized_pnl >= 0 ? 'positive' : 'negative'}">
                                ${p.unrealized_pnl >= 0 ? '+' : ''}${p.unrealized_pnl.toFixed(2)} 
                                (${p.unrealized_pnl_percent.toFixed(1)}%)
                            </td>
                        </tr>
                    `).join('');
                } else {
                    tbody.innerHTML = '<tr><td colspan="5" style="opacity: 0.5;">No positions</td></tr>';
                }

                // Chart: fill symbol select (positions first, else AAPL); auto-load first position when you have positions
                var sel = document.getElementById('chart-symbol');
                if (sel) {
                    var syms = (positions.positions || []).map(function(p){ return p.symbol; }).filter(Boolean);
                    if (!syms.length) syms = ['AAPL'];
                    sel.innerHTML = '';
                    syms.forEach(function(s){ var o = document.createElement('option'); o.value = s; o.textContent = s; sel.appendChild(o); });
                    sel.value = syms[0];
                    if ((positions.positions || []).length && !window.chartDataLoaded) {
                        loadChart(positions.positions[0].symbol, (document.getElementById('chart-period') || {}).value || '1y', '1d');
                    }
                }
            } catch (error) {
                console.error('Error refreshing data:', error);
            }
        }

        document.getElementById('chart-load') && document.getElementById('chart-load').addEventListener('click', function() {
            loadChart(document.getElementById('chart-symbol').value, (document.getElementById('chart-period') || {}).value || '1y', '1d');
        });
        
        // Initial load
        refreshData();
        
        // Auto-refresh every 10 seconds
        setInterval(refreshData, 10000);
    </script>
</body>
</html>
"""


def run_dashboard(host: str = "127.0.0.1", port: int = 8000):
    """Run the dashboard server"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_dashboard()


