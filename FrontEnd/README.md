# Django Trading Dashboard - FrontEnd

This is the Django web interface for the AT vol.2 Trading System.

## Setup

### 1. Install Dependencies

Django and Django REST Framework should already be installed in the virtual environment. If not:

```powershell
cd "C:\Users\PRO\Desktop\AT vol.2"
& "venv\Scripts\python.exe" -m pip install django djangorestframework
```

### 2. Run Migrations

Already done! But if you need to run again:

```powershell
cd "C:\Users\PRO\Desktop\AT vol.2\FrontEnd"
& "..\venv\Scripts\python.exe" manage.py migrate
```

### 3. Create Superuser

Create an admin user to access the dashboard:

```powershell
cd "C:\Users\PRO\Desktop\AT vol.2\FrontEnd"
& "..\venv\Scripts\python.exe" manage.py createsuperuser
```

Follow the prompts to create your username and password.

### 4. Run Development Server

```powershell
cd "C:\Users\PRO\Desktop\AT vol.2\FrontEnd"
& "..\venv\Scripts\python.exe" manage.py runserver
```

Then open your browser to: http://127.0.0.1:8000

## Features Implemented

### Core Features ✅
- **Dashboard**: Overview with balance, P&L, positions, and charts
- **Trading Log**: Complete trade history with filters
- **Positions**: Current holdings with P&L tracking

### Enhanced Features ✅
- **Performance Analytics**: Win rate, statistics, best/worst days
- **Risk Dashboard**: Circuit breaker events, exposure tracking
- **Market Status**: System health, active orders
- **ML Insights**: ML model status (placeholder for future ML data)

## Architecture

### Database Connection
- Uses `db_adapter.py` to connect to existing SQLite database at `BackEnd/data/trading.db`
- Read-only access to prevent conflicts with trading system
- Leverages SQLite WAL mode for concurrent reads

### Structure
```
FrontEnd/
├── trading_web/          # Django project settings
├── trading_app/          # Main app
│   ├── db_adapter.py     # Database connection adapter
│   ├── views.py          # View functions
│   ├── urls.py           # URL routing
│   ├── templates/        # HTML templates
│   └── static/           # CSS, JS, images
└── api/                  # REST API (optional, for future use)
```

## URLs

- `/` - Dashboard
- `/trading-log/` - Trading log
- `/positions/` - Current positions
- `/performance/` - Performance analytics
- `/risk/` - Risk dashboard
- `/market-status/` - Market status
- `/ml-insights/` - ML insights
- `/admin/` - Django admin (requires superuser)
- `/login/` - Login page
- `/logout/` - Logout

## Notes

- The dashboard reads from the same database as the trading system
- No write operations from Django (read-only)
- Real-time updates via page refresh (auto-refresh every 30s on dashboard)
- Future: Can add WebSocket support for true real-time updates

## Troubleshooting

### Database Not Found
If you get "Database not found" error:
- Ensure `BackEnd/data/trading.db` exists
- Check that the trading system has created the database

### Import Errors
If you get import errors for BackEnd modules:
- Ensure you're running from the FrontEnd directory
- The `db_adapter.py` adds BackEnd to sys.path automatically

### Static Files Not Loading
Run collectstatic (for production):
```powershell
& "..\venv\Scripts\python.exe" manage.py collectstatic
```

## Next Steps

1. Create superuser and test login
2. Verify database connection works
3. Test all views
4. Add more charts and visualizations
5. Implement WebSocket for real-time updates
6. Add export functionality (CSV, PDF)
