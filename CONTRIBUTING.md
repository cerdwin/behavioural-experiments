# Contributing

Thank you for your interest in contributing to this project!

## Getting Started

1. **Fork & clone** the repository
2. **Install dependencies** (requires [uv](https://docs.astral.sh/uv/)):
   ```bash
   uv sync
   ```
3. **Set up environment** — copy `.env.example` to `.env` and add your OpenRouter API key:
   ```bash
   cp .env.example .env
   # Edit .env with your key
   ```

## Development

- **Code style**: We use [Black](https://black.readthedocs.io/) for formatting and [isort](https://pycqa.github.io/isort/) for import sorting.
  ```bash
  uv run black .
  uv run isort .
  ```
- **Tests**: Run the test scripts in `scripts/` to verify API connectivity:
  ```bash
  uv run scripts/test_models.py
  ```

## Adding New Experiments

1. Add the experiment prompt(s) to `prompts.json` (and optionally to `prompts_variations.json`)
2. Add the experiment configuration to `config.yaml`
3. Implement the experiment logic in `runner.py` (follow the pattern of existing experiments)
4. Update the database schema in `database.py` if new tables are needed
5. Add analysis logic in `analyze.py`
6. Document the experiment in `docs/EXPERIMENT_OVERVIEW.md`

## Submitting Changes

1. Create a feature branch from `main`
2. Make your changes and ensure code is formatted
3. Test with a small number of trials (`trials_per_condition: 2`) before full runs
4. Submit a pull request with a clear description of changes

## Reporting Issues

Please open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Your Python version and OS
