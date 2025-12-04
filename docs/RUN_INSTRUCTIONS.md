# Running the Full Experiment

## Quick Start

```bash
cd "/Users/elvira/Documents/MATS/Second ICML poster"
uv run runner.py
```

That's it\! The experiment will start immediately.

## What Happens

1. **Loads config.yaml**
   - 4 models (Claude, GPT-5.1, Gemini, DeepSeek)
   - 100 trials per condition
   - All 5 experiments enabled

2. **Creates tasks**
   - Total: 18,000 API calls
   - Prints estimated total before starting

3. **Runs async with 10 concurrent requests**
   - Progress printed to console: `✓ experiment/condition/model/trial_X`
   - Live log: `results/experiment_live.log`

4. **Saves to SQLite**
   - Database: `results/results.db`
   - Updates after each trial

5. **Estimated completion: ~4.8 hours**

## While Running

**Monitor progress:**
```bash
# Watch live output
tail -f results/experiment_live.log

# Count completed trials
sqlite3 results/results.db "SELECT COUNT(*) FROM trials"

# Check by experiment
sqlite3 results/results.db "SELECT experiment, COUNT(*) FROM trials GROUP BY experiment"
```

**Check errors:**
```bash
# Parse failures
sqlite3 results/results.db "SELECT model, COUNT(*) FROM trials WHERE parse_success=0 GROUP BY model"
```

## After Completion

**Analyze results:**
```bash
uv run analyze.py
```

Generates:
- Console summary with key findings
- `results/results_trials.csv`
- `results/results_games.csv`
- `results/results_allais.csv`

## Interrupting & Resuming

**To stop:**
- Press `Ctrl+C` once (gracefully stops after current batch)
- Press `Ctrl+C` twice (force quit)

**To resume:**
```bash
# Delete partial results first
rm results/results.db

# Re-run
uv run runner.py
```

Note: Currently no automatic resume - will re-run from scratch.

## Running Partial Experiments

**Test with 10 trials first:**
```bash
# Edit config.yaml
# Change: trials_per_condition: 10

uv run runner.py
```

**Run specific experiments only:**
```bash
# Edit config.yaml
# Set enabled: false for experiments you want to skip

# Example: Only PD and PG
experiments:
  allais:
    enabled: false
  prisoner_dilemma:
    enabled: true
  public_goods:
    enabled: true
  ultimatum:
    enabled: false
  framing:
    enabled: false
```

**Run fewer models:**
```bash
# Edit config.yaml
# Remove models from fallback list

models:
  fallback:
    - anthropic/claude-opus-4.5
    - openai/gpt-5.1
    # - google/gemini-3-pro-preview  # Commented out
    # - deepseek/deepseek-r1-0528-qwen3-8b  # Commented out
```

## Expected Console Output

```
Setting up allais...
Setting up prisoner_dilemma...
Setting up public_goods...
Setting up ultimatum...
Setting up framing...

Total tasks: 4500
Estimated API calls: 18000
Starting execution...

✓ allais/no_history/anthropic/claude-opus-4.5/trial_0: A, D
✓ allais/no_history/anthropic/claude-opus-4.5/trial_1: B, D
✓ prisoner_dilemma/single/anthropic/claude-opus-4.5/trial_0
✓ prisoner_dilemma/multi/anthropic/claude-opus-4.5/n=2/trial_0: ['COOPERATE', 'DEFECT']
...

Completed 17856/18000 trials successfully
```

## Costs

- **18,000 calls** × $0.02-0.05 per call
- **Estimated: $360-900**
- Varies by model (Claude most expensive, DeepSeek cheapest)

## Performance Tips

**Increase concurrency (if you have good connection):**
```yaml
# config.yaml
api:
  rate_limit:
    concurrent_requests: 20  # Default is 10
```

**Reduce token usage:**
```yaml
execution:
  max_tokens: 300  # Default is 500
```
