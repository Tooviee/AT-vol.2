# Agent Structure Recommendations for AT vol.2 Trading System

## Executive Summary

Based on your codebase analysis, here's a recommended agent structure that balances specialization with maintainability. Your instinct to separate concerns is correct - strategy work should go to a Strategy Manager, not a Backend Developer.

---

## Recommended Agent Structure

### Core Agents (Essential)

1. **Orchestrator** - Project-level coordination and task delegation
3. **Strategy Manager** - Trading logic, indicators, ML integration
4. **Backend Developer** - Infrastructure, APIs, data persistence, system modules
5. **Frontend Developer** - Django dashboard, UI/UX, visualization
6. **Testing Agent** - Test coverage, quality assurance, validation

### Specialized Agents (Consider Adding)

6. **ML Engineer** - Model training, feature engineering, A/B testing
7. **DevOps/Infrastructure Agent** - Deployment, monitoring, configuration management

---

## Agent Responsibilities

### 1. Orchestrator (Project Manager)

**What it does:**
- **Task delegation**: Routes work to appropriate agents based on issue type
- **Project coordination**: Ensures dependencies are handled (e.g., backend must be ready before frontend)
- **Cross-cutting concerns**: Handles tasks that span multiple domains
- **Issue triage**: Reviews new issues, assigns priorities, creates sub-tasks
- **Progress tracking**: Monitors overall project health via beads
- **Architecture decisions**: Makes high-level design choices affecting multiple components

**What it does NOT do:**
- ❌ Write trading strategy code (delegates to Strategy Manager)
- ❌ Implement database schemas (delegates to Backend Developer)
- ❌ Design UI components (delegates to Frontend Developer)
- ❌ Write individual tests (delegates to Testing Agent)

**Example tasks:**
- "Create a new feature for real-time position monitoring"
  - Orchestrator breaks this into: Backend API endpoint → Frontend dashboard widget → Tests
- "Investigate why ML model accuracy dropped"
  - Orchestrator delegates to ML Engineer, who may need Backend Developer for data access

**Memory/Context:**
- Should maintain high-level project knowledge
- Understands system architecture and component relationships
- Tracks what each agent is working on
- **Recommendation**: Keep your current orchestrator but evolve its role to be more of a coordinator than a builder

---

### 2. Strategy Manager (Trading Logic Specialist)

**What it does:**
- **Trading strategy development**: Implements and refines entry/exit logic
- **Indicator calculations**: MACD, RSI, ATR, SMA, EMA modifications
- **Signal generation**: Buy/sell/hold decision logic
- **Risk parameter tuning**: Stop-loss, take-profit, position sizing rules
- **Strategy backtesting**: Historical performance analysis
- **ML strategy integration**: Works with ML Engineer on confidence boosting

**Owns these files:**
- `BackEnd/modules/strategy.py`
- `BackEnd/ml/ml_strategy.py`
- `BackEnd/modules/backtester.py`
- Strategy-related config in `usa_stock_trading_config.yaml`

**Example tasks:**
- "Enhance trading strategy with hybrid MACD/RSI/ATR indicators" ✅
- "Add new entry condition: price must be above SMA 200"
- "Optimize stop-loss multiplier based on volatility"
- "Implement trailing stop-loss logic"

**Why separate from Backend Developer:**
- Strategy logic is domain-specific (trading knowledge)
- Backend Developer focuses on infrastructure (APIs, databases, system reliability)
- Different skill sets: trading algorithms vs. system architecture

---

### 3. Backend Developer (Infrastructure Specialist)

**What it does:**
- **System modules**: Order management, risk management, circuit breakers, health monitoring
- **API integration**: KIS API manager, data fetching, rate limiting
- **Data persistence**: Database schemas, models, migrations
- **System reliability**: Error handling, recovery, logging, monitoring
- **Configuration management**: Config loaders, validation
- **Integration points**: Ensures modules work together correctly

**Owns these files:**
- `BackEnd/main.py` (orchestration code, not strategy)
- `BackEnd/modules/*.py` (except strategy.py)
- `BackEnd/data_persistence/*.py`
- `BackEnd/modules/kis_api_manager.py`
- `BackEnd/modules/order_manager.py`
- `BackEnd/modules/risk_management.py`
- Infrastructure config in `usa_stock_trading_config.yaml`

**Example tasks:**
- "Fix order timeout handling in order_manager.py"
- "Add database migration for new position fields"
- "Implement retry logic for KIS API calls"
- "Add health check endpoint for monitoring"
- "Optimize database queries for position reconciliation"

**Does NOT do:**
- ❌ Modify trading strategy logic (Strategy Manager's job)
- ❌ Change indicator calculations (Strategy Manager's job)

---

### 4. Frontend Developer (UI/UX Specialist)

**What it does:**
- **Django application**: Views, templates, static files
- **Dashboard features**: Charts, tables, real-time updates
- **User experience**: Navigation, responsive design, accessibility
- **Data visualization**: TradingView charts, performance graphs
- **API integration**: Consumes backend APIs for data display

**Owns these files:**
- `FrontEnd/trading_app/*.py`
- `FrontEnd/trading_app/templates/**/*.html`
- `FrontEnd/trading_app/static/**/*`
- `FrontEnd/api/*.py` (API endpoints for frontend)

**Example tasks:**
- "Implement TradingView Lightweight Charts for strategy visualization"
- "Add dark mode toggle to dashboard"
- "Create mobile-responsive position table"
- "Add real-time P&L updates via WebSocket"

---

### 6. Testing Agent (Quality Assurance)

**What it does:**
- **Unit tests**: Tests for each module
- **Integration tests**: Tests for component interactions
- **Test coverage**: Ensures critical paths are tested
- **Test maintenance**: Updates tests when code changes
- **Test automation**: CI/CD test pipelines

**Owns these files:**
- `BackEnd/tests/*.py`
- `FrontEnd/trading_app/tests.py`
- Test configuration files

**Example tasks:**
- "Add tests for new MACD calculation in strategy.py"
- "Test order manager timeout handling"
- "Verify ML confidence booster edge cases"
- "Add integration test for buy signal → order execution flow"

---

### 6. ML Engineer (Optional but Recommended)

**What it does:**
- **Model development**: LightGBM training, hyperparameter tuning
- **Feature engineering**: Extracting and validating ML features
- **Training pipeline**: Data collection, preprocessing, model training
- **A/B testing**: ML vs. control group analysis
- **Model evaluation**: Performance metrics, validation

**Owns these files:**
- `BackEnd/ml/feature_extractor.py`
- `BackEnd/ml/confidence_booster.py`
- `BackEnd/ml/trainer.py`
- `BackEnd/ml/training_data_manager.py`
- `BackEnd/ml/ab_testing.py`
- ML model files in `BackEnd/ml/models/`

**Example tasks:**
- "Add new feature: price momentum over 20 days"
- "Retrain model with latest trading data"
- "Analyze why ML confidence is too conservative"
- "Implement feature importance analysis"

**Why separate from Strategy Manager:**
- ML engineering requires different expertise (data science, model tuning)
- Strategy Manager focuses on trading rules; ML Engineer focuses on model performance
- Can work in parallel: Strategy Manager adds new indicators, ML Engineer adds features

---

### 8. DevOps/Infrastructure Agent (Optional)

**What it does:**
- **Deployment**: Production setup, environment configuration
- **Monitoring**: Log aggregation, alerting setup
- **CI/CD**: Automated testing, deployment pipelines
- **Infrastructure**: Server setup, database backups, scaling

**Example tasks:**
- "Set up automated daily database backups"
- "Configure production logging to external service"
- "Create deployment script for live trading environment"
- "Set up monitoring alerts for circuit breaker triggers"

---

## Decision: Keep Current Orchestrator or Create New?

### Recommendation: **Evolve Current Orchestrator**

**Keep your current orchestrator** but transition its role:

1. **Current state**: It built the entire backend (which was necessary initially)
2. **Future state**: It should coordinate, not build everything

**Transition plan:**
- Keep orchestrator's memory of the system architecture
- Have it delegate implementation to specialized agents
- Use orchestrator for:
  - Breaking down large features into agent-specific tasks
  - Ensuring agents don't conflict (e.g., Backend and Strategy both modifying same file)
  - Cross-cutting architectural decisions

**Example workflow:**
```
User: "Add real-time position monitoring"

Orchestrator:
1. Creates issue in beads
2. Breaks into tasks:
   - Backend: Add WebSocket endpoint for position updates
   - Frontend: Create real-time position widget
   - Testing: Add integration tests
3. Assigns tasks to respective agents
4. Monitors progress and resolves conflicts
```

---

## Agent Communication & Coordination

### Issue Routing Rules

Use beads issue types and labels to route work:

```yaml
# Issue routing logic
strategy: → Strategy Manager
  - type: feature, task
  - labels: ["strategy", "indicators", "signals", "backtesting"]
  
backend: → Backend Developer
  - type: feature, task, bug
  - labels: ["infrastructure", "api", "database", "system"]
  
frontend: → Frontend Developer
  - type: feature, task, bug
  - labels: ["ui", "dashboard", "visualization"]
  
ml: → ML Engineer
  - type: feature, task
  - labels: ["ml", "model", "training", "features"]
  
testing: → Testing Agent
  - type: task
  - labels: ["test", "qa", "coverage"]
  
orchestrator: → Orchestrator
  - type: epic, feature (large)
  - labels: ["architecture", "coordination"]
  - Multiple components affected
```

### Conflict Prevention

**File ownership matrix:**

| File/Directory | Primary Owner | Secondary Owner |
|----------------|---------------|-----------------|
| `modules/strategy.py` | Strategy Manager | Testing Agent |
| `ml/ml_strategy.py` | Strategy Manager | ML Engineer |
| `ml/feature_extractor.py` | ML Engineer | Strategy Manager |
| `main.py` | Backend Developer | Orchestrator |
| `modules/order_manager.py` | Backend Developer | Testing Agent |
| `trading_app/templates/` | Frontend Developer | - |
| `tests/` | Testing Agent | All (for their modules) |

**Rule**: If an agent needs to modify another agent's file, they should:
1. Create a task/issue for that agent
2. Or coordinate via orchestrator if it's a shared change

---

## Recommended Agent Setup

### Minimum Viable Setup (Start Here)

1. **Orchestrator** - Coordinates everything
2. **Strategy Manager** - Trading logic
3. **Backend Developer** - Infrastructure
4. **Frontend Developer** - UI
5. **Testing Agent** - Quality assurance

### Full Setup (As Project Grows)

Add when needed:
6. **ML Engineer** - When ML work becomes frequent
7. **DevOps Agent** - When deploying to production

---

## Implementation Strategy

### Phase 1: Define Agent Rules (Now)

Create `.cursor/rules/` files for each agent:

```
.cursor/rules/
├── orchestrator.mdc
├── strategy-manager.mdc
├── backend-developer.mdc
├── frontend-developer.mdc
├── testing-agent.mdc
└── ml-engineer.mdc
```

Each rule file defines:
- What the agent owns
- What it delegates
- How it communicates with other agents
- File ownership boundaries

### Phase 2: Update AGENTS.md

Add agent-specific sections to `AGENTS.md`:
- When to use each agent
- How to route issues
- Agent communication protocols

### Phase 3: Start Using Agents

- Create issues with appropriate labels
- Let orchestrator route work
- Agents claim work via `bd update <id> --status in_progress`

---

## Answering Your Questions

### Q: What should orchestrator do?

**A:** Orchestrator is a **project coordinator**, not a builder:
- Routes tasks to specialized agents
- Breaks down large features into smaller tasks
- Ensures dependencies are handled
- Makes architectural decisions
- Tracks overall progress

Think of it as a **project manager** who delegates to specialists.

### Q: Do I need more agents?

**A:** Your 5-agent setup is solid. Consider adding:
- **ML Engineer** if you're doing frequent ML work (you have ML components, so this makes sense)
- **DevOps Agent** only when you're deploying to production

Start with 5, add ML Engineer when ML tasks become frequent.

### Q: Strategy tasks to Strategy Manager, not Backend Developer?

**A:** **Absolutely correct!** 

- **Strategy Manager**: Trading logic, indicators, signals, entry/exit rules
- **Backend Developer**: System infrastructure, APIs, databases, reliability

These are different domains. Strategy Manager needs trading knowledge; Backend Developer needs system architecture knowledge.

### Q: Keep current orchestrator or create new?

**A:** **Keep and evolve it.**

Your orchestrator has valuable context about the system. Transition it from "builder" to "coordinator":
- Keep its memory of the architecture
- Have it delegate implementation to specialized agents
- Use it for coordination and task breakdown

---

## Example Workflow

### Scenario: "Enhance trading strategy with hybrid MACD/RSI/ATR"

**Current approach (orchestrator does everything):**
- Orchestrator implements the entire feature

**Recommended approach (agent collaboration):**
1. **Orchestrator**: Creates issue, breaks into tasks
2. **Strategy Manager**: Implements indicator calculations, signal logic
3. **Backend Developer**: Ensures strategy integrates with order manager correctly
4. **Testing Agent**: Adds tests for new strategy logic
5. **Frontend Developer**: (If needed) Updates dashboard to show new indicators

**Benefits:**
- Each agent works in their domain
- Clear ownership and responsibility
- Easier to maintain and extend
- Better code quality (specialists)

---

## Next Steps

1. **Create agent rule files** in `.cursor/rules/`
2. **Update AGENTS.md** with agent routing guidelines
3. **Start using labels** in beads issues to route work
4. **Let orchestrator coordinate** rather than implement everything

Would you like me to create the agent rule files and update AGENTS.md with these guidelines?
