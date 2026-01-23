# How to Use Beads for Your Trading System Tasks

This guide shows how to store your work items in Beads so the AI agent can remember and work on them across sessions.

## Your Current Tasks

### Task 1: Enhance Trading Strategy
### Task 2: Update Website with TradingView Charts

## Step 1: Create Issues in Beads

Run these commands in PowerShell from your project directory:

```powershell
cd "C:\Users\PRO\Desktop\AT vol.2"

# Create Issue 1: Strategy Enhancement
.\bd.exe create `
  --title="Enhance trading strategy with hybrid MACD/RSI/ATR indicators" `
  --type=feature `
  --priority=1 `
  --description="Refactor trading logic with new parameters and hybrid indicator strategy.

Parameters:
- MACD (Fast: 12, Slow: 26, Signal: 9)
- RSI (Period: 14, Overbought: 70, Oversold: 30)
- ATR (Period: 14)
- New Indicators: SMA 50 (Medium-term), SMA 200 (Long-term)

Entry Logic (Long Position Only):
1. Trend Filter: Current Price must be above SMA 200
2. Signal: EMA 12 must cross above EMA 26 (equivalent to MACD Line crossing above 0)
3. Overbought Protection: RSI(14) must be below 70 to avoid buying at the peak
4. Final Check: Ensure SMA 50 is trending upwards or Price is above SMA 50

Exit & Risk Management:
1. Stop-Loss: Trailing stop-loss based on ATR (2 * ATR)
2. Take-Profit: 1.5:1 or 2:1 Risk-Reward ratio using ATR-based stop-loss

Files: BackEnd/modules/strategy.py" `
  --notes="Use pandas or 'ta' library for indicator calculations"

# Create Issue 2: Website Visualization
.\bd.exe create `
  --title="Implement TradingView Lightweight Charts for strategy visualization" `
  --type=feature `
  --priority=1 `
  --description="Implement Method B using TradingView Lightweight Charts library.

Backend:
- Convert Pandas DataFrame to JSON format for Lightweight Charts
- Time Format: UNIX timestamps (seconds)
- Structure: candlestick_data, ema12_data, ema26_data, sma200_data, markers_data
- Markers: Extract rows where Buy_Signal == 1

Frontend:
- Load lightweight-charts library via CDN
- Dark-themed chart with Candlestick Series
- Line Series: EMA 12 (Yellow), EMA 26 (Orange), SMA 200 (Blue)
- Use setMarkers() to visualize Buy Signals
- Support zooming and scrolling" `
  --notes="Assume data.json file or API endpoint ready to fetch"
```

## Step 2: View Your Issues

```powershell
# List all open issues
.\bd.exe list --status=open

# Show available work (no blockers)
.\bd.exe ready

# Show details of a specific issue
.\bd.exe show <issue-id>
```

## Step 3: Work on Issues

When you or the AI agent starts working:

```powershell
# Claim an issue (mark as in progress)
.\bd.exe update <issue-id> --status=in_progress

# Add notes as you work
.\bd.exe update <issue-id> --notes="Started implementing MACD calculation"

# Close when complete
.\bd.exe close <issue-id> --reason="Strategy enhancement completed and tested"
```

## Step 4: Link Dependencies (Optional)

If Task 2 depends on Task 1 being completed first:

```powershell
# First, get the issue IDs
.\bd.exe list --status=open

# Then link them (website depends on strategy)
.\bd.exe dep add <website-issue-id> <strategy-issue-id>
```

This means the website task is blocked until the strategy is done.

## Step 5: Sync with Git

At the end of your session:

```powershell
# Sync Beads changes to git
.\bd.exe sync

# Then commit your code changes
git add .
git commit -m "Your commit message"
git push
```

## Example Workflow

1. **Create issues** (done above)
2. **Ask AI agent**: "What tasks do we have?"
   - Agent runs: `.\bd.exe ready`
   - Agent sees your two tasks
3. **Start work**: "Let's work on the strategy enhancement"
   - Agent runs: `.\bd.exe update <id> --status=in_progress`
   - Agent implements the code
4. **Complete**: "The strategy is done"
   - Agent runs: `.\bd.exe close <id>`
   - Agent runs: `.\bd.exe sync`
5. **Next session**: Agent remembers what was done and what's left

## Benefits

- ✅ **Persistent Memory**: Issues stored in `.beads/issues.jsonl` (git-tracked)
- ✅ **Cross-Session**: Agent remembers tasks across different Cursor sessions
- ✅ **Dependencies**: Track what blocks what
- ✅ **History**: See what was done and when
- ✅ **Collaboration**: Works with git branches and team members

## Quick Reference

```powershell
.\bd.exe create --title="..." --type=feature|task|bug --priority=0-4
.\bd.exe list --status=open|in_progress|closed
.\bd.exe ready                    # Show available work
.\bd.exe show <id>                # Detailed view
.\bd.exe update <id> --status=in_progress
.\bd.exe close <id>
.\bd.exe dep add <issue> <depends-on>
.\bd.exe sync                     # Sync with git
```
