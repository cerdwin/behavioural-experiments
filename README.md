# Behavioral Economics Experiments on LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Testing frontier language models on classic behavioral economics experiments to measure human-like biases and social preferences.

For a detailed project layout see [Project Structure](docs/PROJECT_STRUCTURE.md).

## Key Findings

**Claude Opus 4.5 shows non-monotonic cooperation:**
- Prisoner's Dilemma: 97% → 0% → 29% → 63% cooperation (U-shaped across group sizes N=1,2,4,10)
- Public Goods: $12.80 → $16.20 → $9.70 → $9.40 average contribution

**GPT-5.1 shows monotonic rational decline:**
- Prisoner's Dilemma: Defects 99-100% of the time (game-theoretically rational)
- Public Goods: $7 → $9 → $0.20 → $0 (complete free-riding at scale)

**Other findings:**
- Both show Allais violations (10-30%)
- Both propose fair ultimatum splits (~$44/100) but accept unfair offers
- Both show framing effects (gain vs loss)

## Experiments

1. **Allais Paradox** — Rational consistency under uncertainty
2. **Prisoner's Dilemma** — Cooperation at N=1, 2, 4, 10 agents
3. **Public Goods Game** — Contributions at N=1, 2, 4, 10 agents
4. **Ultimatum Game** — Fairness as proposer/responder
5. **Framing Effect** — Gain vs loss decision framing
6. **Iterated Prisoner's Dilemma** — Strategic learning over rounds

## Setup

**Prerequisites:** [uv](https://docs.astral.sh/uv/) (Python package manager), an [OpenRouter](https://openrouter.ai/) API key.

```bash
# Clone and install dependencies
git clone https://github.com/<your-org>/behavioural-experiments.git
cd behavioural-experiments
uv sync

# Configure your API key
cp .env.example .env
# Edit .env and paste your OpenRouter API key
```

## Usage

```bash
# Run experiments with current config
uv run runner.py

# Run with specific config
uv run runner.py config_iterated.yaml

# Analyze results
uv run analyze.py
```

See [Run Instructions](docs/RUN_INSTRUCTIONS.md) for details on monitoring progress, interrupting/resuming, and cost estimates.

## Configuration

Edit `config.yaml` to customize:
- Models to test (`models.fallback`)
- Experiments to run (`experiments.*.enabled`)
- Group sizes (`agent_counts: [1, 2, 4, 10]`)
- Trials per condition (`execution.trials_per_condition`)
- Temperature, max_tokens, etc.

## Multi-Agent Design

- **Homogeneous:** All N agents use the same model
- **Blind:** Agents don't know they're playing with copies of themselves
- **Simultaneous:** All agents respond to same prompt in parallel
- **One-shot:** No repeated games (except iterated PD experiment)

## Parsing Strategy

Models output `[MY FINAL CHOICE IS: X]` format for reliable extraction. Falls back to:
1. JSON parsing
2. Markdown code blocks
3. Regex text extraction
4. Natural language parsing

Parse success rate: **94.3%**

See [Parser Improvements](docs/IMPROVEMENTS.md) for details on the cascading parser.

## Output Files

**Database:**
- `results.db` — SQLite with trials, multi_agent_games, allais_pairs tables

**CSV Exports:**
- `results_trials.csv` — Single-agent trial data
- `results_games.csv` — Multi-agent game outcomes
- `results_allais.csv` — Allais paradox choice pairs

## Models Tested

- `anthropic/claude-opus-4.5`
- `openai/gpt-5.1`

(Gemini-3-Pro and DeepSeek-R1 returned empty responses — parser issues)

## Documentation

| Document | Description |
|----------|-------------|
| [Experiment Overview](docs/EXPERIMENT_OVERVIEW.md) | Detailed descriptions of all 6 experiments, parameters, and preliminary results |
| [Project Structure](docs/PROJECT_STRUCTURE.md) | Full file tree with sizes and schema details |
| [Run Instructions](docs/RUN_INSTRUCTIONS.md) | How to run, monitor, interrupt, and resume experiments |
| [Presentation Report](docs/PRESENTATION_REPORT.md) | Study report with hypotheses, analysis plan, and limitations |
| [Parser Improvements](docs/IMPROVEMENTS.md) | Notes on the cascading response parser |

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{behavioral-experiments,
  title   = {Behavioral Economics Experiments on LLMs},
  author  = {Jennifer Za, Aristeidis Panos, Jan Cuhel, Samuel Albanie},
  year    = {2025},
  url     = {https://github.com/cerdwin/behavioural-experiments}
}
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
