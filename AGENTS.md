# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Agent Structure

This project uses a specialized agent structure. See `AGENT_STRUCTURE.md` for detailed information.

### Available Agents

1. **Agent Resource Manager** - Assigns work to appropriate agents (first point of contact)
2. **Orchestrator** - Project coordination and task delegation
3. **Strategy Manager** - Trading logic, indicators, signals
4. **Backend Developer** - Infrastructure, APIs, data persistence
5. **Frontend Developer** - Django dashboard, UI/UX, visualization
6. **ML Engineer** - Model training, feature engineering, A/B testing
7. **Testing Agent** - Test coverage, quality assurance

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Issue Routing Guidelines

### Agent Resource Manager Role

The **Agent Resource Manager** is responsible for assigning work to the appropriate specialist agent. When you receive a task, analyze it and route it based on:

#### Route to Strategy Manager when:
- Trading strategy, indicators (MACD, RSI, ATR, SMA, EMA)
- Signal generation, entry/exit conditions
- Risk parameter tuning (stop-loss, take-profit)
- Strategy backtesting
- **Labels**: `strategy`, `indicators`, `signals`, `backtesting`

#### Route to Backend Developer when:
- Infrastructure, APIs, databases
- Order management, risk management systems
- Error handling, reliability, monitoring
- Database migrations, data persistence
- **Labels**: `infrastructure`, `api`, `database`, `system`

#### Route to Frontend Developer when:
- UI, dashboard, visualization
- Django templates, charts, user experience
- Responsive design, accessibility
- **Labels**: `ui`, `dashboard`, `visualization`, `frontend`

#### Route to ML Engineer when:
- ML models, training, features
- Model evaluation, A/B testing
- Feature engineering
- **Labels**: `ml`, `model`, `training`, `features`

#### Route to Testing Agent when:
- Tests, quality assurance, coverage
- Test maintenance, automation
- **Labels**: `test`, `qa`, `coverage`

#### Route to Orchestrator when:
- Large features affecting multiple components
- Requires coordination across agents
- Architectural decisions
- **Labels**: `architecture`, `coordination`, `epic`

### Assignment Workflow

1. **Agent Resource Manager** analyzes incoming task
2. Determines which agent(s) should handle it
3. Assigns work based on issue type and labels
4. If multiple agents needed → **Orchestrator** breaks into sub-tasks
5. Specialized agents claim work via `bd update <id> --status in_progress`

### File Ownership

See `AGENT_STRUCTURE.md` for detailed file ownership matrix. Key rules:
- **Strategy Manager** owns: `BackEnd/modules/strategy.py`, `BackEnd/ml/ml_strategy.py`
- **Backend Developer** owns: `BackEnd/main.py`, `BackEnd/modules/*.py` (except strategy)
- **Frontend Developer** owns: `FrontEnd/trading_app/*`
- **ML Engineer** owns: `BackEnd/ml/*.py` (except ml_strategy.py)
- **Testing Agent** owns: `BackEnd/tests/*.py`

If an agent needs to modify another agent's file, they should:
1. Create a task/issue for that agent, OR
2. Coordinate via Orchestrator if it's a shared change

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

## Agent-Specific Rules

Each agent has detailed rules in `.cursor/rules/`:
- `.cursor/rules/agent-resource-manager.mdc` - Work assignment
- `.cursor/rules/orchestrator.mdc` - Project coordination
- `.cursor/rules/strategy-manager.mdc` - Trading logic
- `.cursor/rules/backend-developer.mdc` - Infrastructure
- `.cursor/rules/frontend-developer.mdc` - UI/UX
- `.cursor/rules/ml-engineer.mdc` - Machine learning
- `.cursor/rules/testing-agent.mdc` - Quality assurance

Read the appropriate rule file when acting as that agent.

