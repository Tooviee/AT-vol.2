# Path Fixes Required After Moving Files to BackEnd/

## Summary
After moving files to `BackEnd/`, the following files need path adjustments to function correctly when running from the root directory (`AT vol.2/`).

## Files That Need Fixes

### 1. **main.py** (BackEnd/main.py)
**Current:**
```python
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
```

**Issue:** After move, `__file__` will be `BackEnd/main.py`, so `PROJECT_ROOT` will be `BackEnd/`. This is correct for imports, but we need to ensure config file can be found.

**Fix:** Keep as is - it should work since config_loader will handle finding the config file.

---

### 2. **modules/config_loader.py** (BackEnd/modules/config_loader.py)
**Current:**
```python
possible_paths = [
    "usa_stock_trading_config.yaml",
    "config.yaml",
    Path(__file__).parent.parent / "usa_stock_trading_config.yaml",
]
```

**Issue:** After move:
- `Path(__file__).parent.parent` from `BackEnd/modules/config_loader.py` = `BackEnd/` (wrong, should be root)
- Need to go up one more level: `Path(__file__).parent.parent.parent`

**Fix:** Change to:
```python
possible_paths = [
    "usa_stock_trading_config.yaml",  # Current directory (root when running from root)
    "config.yaml",
    Path(__file__).parent.parent.parent / "usa_stock_trading_config.yaml",  # From BackEnd/modules/
]
```

---

### 3. **check_ml_data.py** (BackEnd/check_ml_data.py)
**Current:**
```python
sys.path.insert(0, str(Path(__file__).parent))
```

**Issue:** After move, this adds `BackEnd/` to path, which is correct for imports.

**Fix:** Keep as is - should work fine.

---

### 4. **diagnose_signals.py** (BackEnd/diagnose_signals.py)
**Current:**
```python
sys.path.insert(0, str(Path(__file__).parent))
```

**Issue:** Same as check_ml_data.py

**Fix:** Keep as is - should work fine.

---

### 5. **ml/trainer.py** (BackEnd/ml/trainer.py)
**Current:**
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```

**Issue:** After move:
- `Path(__file__).parent.parent` from `BackEnd/ml/trainer.py` = `BackEnd/` (correct for imports)
- But if running from root, might need to adjust

**Fix:** Keep as is - should work since it's adding BackEnd to path.

---

### 6. **usa_stock_trading_config.yaml** (stays at root)
**Current paths:**
```yaml
database:
  path: "data/trading.db"
  backup_path: "data/backups/"

ml:
  model_path: "ml/models/confidence_model.pkl"
  data_path: "ml/training_data"
```

**Issue:** These are relative paths. When running from root:
- `data/trading.db` → `AT vol.2/data/trading.db` ✅ (correct)
- `ml/models/` → `AT vol.2/ml/models/` ❌ (should be `BackEnd/ml/models/`)

**Fix:** Update paths to be relative to BackEnd:
```yaml
database:
  path: "BackEnd/data/trading.db"
  backup_path: "BackEnd/data/backups/"

ml:
  model_path: "BackEnd/ml/models/confidence_model.pkl"
  data_path: "BackEnd/ml/training_data"
```

**OR** keep relative and ensure scripts run from BackEnd directory.

---

### 7. **modules/logger.py** (BackEnd/modules/logger.py)
**Current:**
```python
log_dir: str = "logs"
```

**Issue:** When running from root, `logs/` will be created at root level.

**Fix:** Update to `BackEnd/logs/` or use absolute path based on project root.

---

### 8. **main.py** - log_dir parameter
**Current:**
```python
trading_logger = init_logging(
    self.config.logging.model_dump(),
    log_dir="logs"
)
```

**Issue:** Same as logger.py

**Fix:** Update to `BackEnd/logs/` or use config-based path.

---

## Recommended Approach

### Option 1: Update config paths (Recommended)
Update `usa_stock_trading_config.yaml` to use `BackEnd/` prefix for all relative paths:
- `data/` → `BackEnd/data/`
- `ml/models/` → `BackEnd/ml/models/`
- `ml/training_data` → `BackEnd/ml/training_data`
- `logs/` → `BackEnd/logs/` (via code update)

### Option 2: Run from BackEnd directory
Keep paths as-is, but always run scripts from `BackEnd/` directory:
```powershell
cd "C:\Users\PRO\Desktop\AT vol.2\BackEnd"
python main.py
```

### Option 3: Use absolute paths
Calculate paths relative to project root dynamically.

---

## Files That DON'T Need Changes
- `dashboard/app.py` - Uses `os.path.dirname(__file__)` which is relative
- `data_persistence/database.py` - Uses paths from config
- `ml/training_data_manager.py` - Uses paths from config
- `ml/confidence_booster.py` - Uses paths from config
- Test files - Should work with updated paths

---

## Action Items
1. ✅ Update `modules/config_loader.py` - Fix config file path lookup
2. ✅ Update `usa_stock_trading_config.yaml` - Update all relative paths
3. ✅ Update `main.py` - Update log_dir to use BackEnd prefix
4. ✅ Update `modules/logger.py` - Consider path updates (or keep relative)
5. ⚠️ Test all scripts after move to ensure paths work
