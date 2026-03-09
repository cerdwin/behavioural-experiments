# Project Structure

```
.
├── README.md                      # Main documentation with key findings
├── config.yaml                    # Main experiment config (100 trials, 4 models)
├── config_iterated.yaml           # Iterated PD config (optional)
│
├── prompts.json                   # Base prompts for all experiments
├── prompts_variations.json        # 4 variants per experiment for robustness
│
├── runner.py                      # Main async experiment orchestration
├── openrouter_client.py           # API client with robust parsing (handles Gemini reasoning field)
├── database.py                    # SQLite schema (trials, games, allais_pairs)
├── analyze.py                     # Results analysis & CSV export
├── pyproject.toml                 # Python dependencies (uv)
│
├── docs/                          # Documentation
│   ├── EXPERIMENT_OVERVIEW.md     # Detailed experiment descriptions
│   ├── PRESENTATION_REPORT.md     # Setup report for presentation
│   ├── PROJECT_STRUCTURE.md       # This file - project layout
│   ├── RUN_INSTRUCTIONS.md        # How to run experiments
│   └── IMPROVEMENTS.md            # Parser improvement notes
│
├── scripts/                       # Test/debug scripts
│   ├── test_all_models.py         # Test all 4 models before full run
│   ├── test_gemini_fixed.py       # Gemini-specific parser test
│   ├── test_models.py             # General model testing
│   ├── test_api_raw.py            # Raw API endpoint testing
│   └── debug_gemini.py            # Raw API debugging
│
├── results/                       # All outputs (gitignored)
│   ├── results.db                 # SQLite database
│   ├── results_trials.csv         # Single-agent data
│   ├── results_games.csv          # Multi-agent data
│   ├── results_allais.csv         # Allais pairs
│   └── *.log                      # Run logs
│
├── CONTRIBUTING.md                # Contribution guidelines
└── LICENSE                        # MIT License
```

## Core Files (Must Commit)

- `runner.py`, `openrouter_client.py`, `database.py`, `analyze.py` - Core code
- `config.yaml` - Main configuration
- `prompts.json`, `prompts_variations.json` - All experiment prompts
- `pyproject.toml` - Dependencies
- `README.md` - Documentation
- `.gitignore` - Excludes results, venv, etc.

## Excluded from Git (in .gitignore)

- `.venv/` - Virtual environment
- `.env` - API keys (keep private\!)
- `results/` - All databases, CSVs, logs
- `__pycache__/` - Python cache
- `uv.lock` - Auto-generated, large
- `*_test.*` - Temporary test files

## File Sizes (Approximate)

| File | Size | Purpose |
|------|------|---------|
| runner.py | 24 KB | Orchestration logic |
| openrouter_client.py | 11 KB | API client |
| prompts_variations.json | 11 KB | 4 variants × 9 experiments |
| prompts.json | 6 KB | Base prompts |
| database.py | 7 KB | Schema |
| analyze.py | 9 KB | Analysis |
| README.md | 4 KB | Docs |
| **Total core code** | **~70 KB** | Excluding dependencies |

## Results Files (Generated)

After running experiments:

| File | Rows | Columns | Size |
|------|------|---------|------|
| results.db | ~18k trials | 15 cols | ~10-50 MB |
| results_trials.csv | ~2,400 | 10 cols | ~500 KB |
| results_games.csv | ~13,600 | 8 cols | ~2 MB |
| results_allais.csv | ~1,600 | 6 cols | ~100 KB |

## Quick Navigation

```bash
# View main config
cat config.yaml

# Check experiment prompts
cat prompts.json | jq '.pd_single'

# See all prompt variants
cat prompts_variations.json | jq 'keys'

# Monitor running experiment
tail -f results/experiment_live.log

# Analyze completed results
uv run analyze.py
```
