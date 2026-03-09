# Behavioral Economics Experiments on LLMs

Testing frontier language models on classic behavioral economics and game theory experiments to measure human-like biases, social preferences, and strategic reasoning.

**Paper**: *Towards Predictive Models of Strategic Behaviour in Large Language Model Agents*
- ICLR 2026 AIMS Workshop
- ICLR 2026 Agents in the Wild (AIWILD)

## Experiments

1. **Prisoner's Dilemma** (N=1,2,4,10) — cooperation rates across group sizes, stakes, incentive structures, 62 scenario framings
2. **Public Goods Game** (N=1,2,4,10) — contribution levels with varying endowments and multipliers
3. **Allais Paradox** — independence axiom violations at different stake levels
4. **Ultimatum Game** (N=2,4,10) — fairness as proposer/responder
5. **Iterated Prisoner's Dilemma** — strategy evolution over rounds (forgiveness, guilt, deadlock, GRIM)
6. **Identity-Coupling Mechanism** — 2x2 factorial testing opponent identity and coupling effects

## Models Tested (7 across 4 providers)

- **OpenAI**: GPT-5.2, o4-mini
- **Anthropic**: Claude 3.7 Sonnet, Claude Haiku 4.5
- **Google**: Gemini 2.5 Pro, Gemini 3 Pro Preview
- **DeepSeek**: v3.2

All accessed via [OpenRouter](https://openrouter.ai/) API.

## Setup

```bash
# Install dependencies
pip install -e .

# Set OpenRouter API key
cp .env.example .env
# Edit .env with your API key
```

## Usage

```bash
# Run PD/PG experiments with a config
python runner_ablations.py --config configs/config_wave1.yaml

# Run with specific model
python runner_ablations.py --config configs/config_pd_expansion.yaml --model anthropic/claude-3.7-sonnet

# Resume interrupted run (skips completed conditions)
python runner_ablations.py --config configs/config_wave1.yaml
```

## Project Structure

```
├── openrouter_client.py          # Async API client (OpenRouter)
├── parser.py                     # Response parser (square-bracket extraction + fallbacks)
├── database_v2.py                # SQLite schema + logging
├── runner_v2.py                  # PD + PG runner
├── runner_ablations.py           # Config-driven runner with CLI args, resume support
├── configs/                      # Experiment configurations (YAML)
│   ├── config_wave1.yaml         # PD wave 1 (opponent identity)
│   ├── config_pd_expansion.yaml  # PD 62-scenario expansion
│   ├── config_allais.yaml        # Allais paradox
│   ├── config_ultimatum.yaml     # Ultimatum game
│   └── config_pg_wave*.yaml      # Public Goods waves 1-3
├── scenarios/                    # Wave 1 scenario prompts (PD, PG, Allais, Ultimatum)
├── scenarios_v2/                 # 62 PD scenarios + metadata (R/S/C ratings)
├── prompts/                      # Prompt templates
├── experiments/
│   ├── iterated_pd/              # Iterated PD: horizon effects, strategy response
│   └── mechanism/                # Identity-Coupling 2x2 factorial
├── analysis/
│   ├── prediction_v2/            # QRE + hierarchical family models
│   ├── embeddings/scripts/       # Embedding-based analysis
│   └── component_analysis/       # PCA / factor analysis
```

## Configuration

Experiments are configured via YAML files in `configs/`. Each config specifies:
- Models to test
- Experiment parameters (group sizes, stake levels, incentive structures)
- Scenario files to use
- Number of trials per condition
- Concurrency and API settings

## Parsing

Models output `[CHOICE: X]` format for reliable extraction. Parser falls back through:
1. Square-bracket extraction
2. JSON parsing
3. Regex text extraction
4. Natural language parsing

Parse success rate: >95% across all 7 models.

## Multi-Agent Design

- **Homogeneous**: All N agents use the same model
- **Blind**: Agents don't know they're playing with copies of themselves
- **Simultaneous**: All agents respond to same prompt in parallel
- **One-shot**: No repeated games (except iterated PD experiment)

## Citation

```bibtex
@inproceedings{za2026predictive,
  title={Towards Predictive Models of Strategic Behaviour in Large Language Model Agents},
  author={Za, Jennifer and Panos, Aristeidis and Cuhel, Jan and Albanie, Samuel},
  booktitle={ICLR 2026 AIMS Workshop / Agents in the Wild},
  year={2026}
}
```

## License

MIT
