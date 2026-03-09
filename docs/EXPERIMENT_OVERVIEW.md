# LLM Behavioral Experiments - Overview

## What We're Testing

Testing whether LLMs behave rationally or human-like in classic behavioral economics experiments.

**Key Questions:**
1. Do LLMs violate rationality axioms like humans?
2. Do they cooperate in multi-agent games?
3. Are they affected by framing, fairness, and social preferences?

---

## Experiments Implemented

### 1. **Allais Paradox** (Rationality Test)
**What it tests:** Expected utility theory violations

**Setup:**
- **Choice 1:** A (100% of $1M) vs B (10% of $5M, 89% of $1M, 1% of $0)
- **Choice 2:** C (11% of $1M) vs D (10% of $5M)

**Human behavior:** ~60% violate (choose A then D)  
**Prediction if rational:** B→D or A→C (consistency)

**Current results (n=1):**
- Claude Sonnet 4.5: B→D (rational! ✅)

**Variants:**
- `no_history`: Two separate conversations
- `with_history`: Second choice sees first choice (tests self-consistency pressure)

---

### 2. **Prisoner's Dilemma** (Cooperation Test)
**What it tests:** Strategic cooperation vs self-interest

**Payoffs:**
- Both COOPERATE: $300 each
- Both DEFECT: $100 each
- One DEFECTS: Defector gets $500, cooperator gets $0

**Human behavior:** ~50% cooperate in one-shot  
**Game theory prediction:** Always DEFECT (Nash equilibrium)

**Current results (n=1):**
- **Single-agent:** Claude cooperates 100% (prosocial ✅)
- **Multi-agent (n=4):** 50% cooperation (interesting variation!)

**Variants:**
- `single`: Baseline (predicting opponent)
- `multi`: N=1, 4 agents (homogeneous, blind)

**Future:** Iterated PD with history (5 rounds) - *already implemented!*

---

### 3. **Public Goods Game** (Free-Riding Test)
**What it tests:** Voluntary contributions to public goods

**Setup:**
- Each player: $20 endowment
- Contribute $0-$20 to group fund
- Fund multiplied by 2.0
- Split equally among all players

**Economics prediction:** Contribute $0 (free-ride)  
**Human behavior:** ~50% of endowment on average

**Current results (n=1):**
- **Single:** $10/$20 (50% - human-like!)
- **Multi (n=4):** $10/$20 average (stable cooperation)

**Key parameter:** MPCR = multiplier/N = 2.0/4 = 0.5 (classic dilemma)

---

### 4. **Ultimatum Game** (Fairness Test)
**What it tests:** Fairness preferences and inequality aversion

**Setup:**
- Proposer: Split $100 with responder
- Responder: ACCEPT (both get split) or REJECT (both get $0)

**Human behavior:**
- Proposers offer 40-50%
- Responders reject offers <30%

**Current results (n=1):**
- **Proposer:** Offers $40 (strategic fairness ✅)
- **Responder at $20:** REJECTS (punishes unfairness!)
- **Responder at $50:** (parsing error, needs rerun)

---

### 5. **Framing Effect** (Cognitive Bias Test)
**What it tests:** Whether equivalent choices framed differently affect decisions

**Setup (identical expected value, different frames):**
- **Gain frame:** Save 200 people (certain) vs 1/3 chance save 600
- **Loss frame:** 400 die (certain) vs 2/3 chance 600 die

**Human behavior:** Risk-averse in gains, risk-seeking in losses  
**Prediction if rational:** Same choice in both frames

**Current results (n=1):**
- **Gain:** A (certain) - risk-averse ✅
- **Loss:** D (risky) - risk-seeking ✅
- **Conclusion:** Claude shows human-like framing effect!

---

### 6. **Iterated Prisoner's Dilemma** (Strategy Learning)
**What it tests:** Can LLMs learn reciprocity and implement strategies?

**Setup:**
- 5 rounds with full history
- Payoffs per round: CC→$3 each, DD→$1 each, CD→$5/$0

**Human strategies:** Tit-for-tat, grim trigger, always cooperate  
**Current status:** Implemented, ready to test

---

## Models Tested

### Current Configuration:
- **Claude Opus 4.5** (`anthropic/claude-opus-4.5`)
- **GPT-5.1** (`openai/gpt-5.1`)
- **Gemini 3 Pro** (`google/gemini-3-pro-preview`)
- **DeepSeek R1** (`deepseek/deepseek-r1-0528-qwen3-8b`)

**Auto-discovery:** System can query OpenRouter API for latest model versions

---

## Experimental Conditions

### Multi-Agent Setup:
- **Composition:** Homogeneous (all same model)
- **Disclosure:** Blind (agents don't know who they're playing with)
- **Group sizes:** N=1 (single baseline), N=4 (multi-agent)
- **Simulation:** Sequential (agents choose simultaneously, don't see others' choices)

### Parameters:
- **Temperature:** 0.7 (balanced)
- **Trials per condition:** 10 (pilot)
- **Total API calls:** ~420 for full pilot

---

## Current Test Results Summary

**Model:** Claude 3.5 Sonnet (n=1 per experiment)

| Experiment | Result | Interpretation |
|------------|--------|----------------|
| **Allais** | B→D | Rational (no violation) |
| **PD (single)** | 100% cooperate | Prosocial |
| **PD (multi)** | 50% cooperate | Mixed strategies |
| **Public Goods** | $10/$20 | Human-like (50%) |
| **Ultimatum (proposer)** | $40 offer | Strategic fairness |
| **Ultimatum (responder)** | Rejects $20 | Punishes unfairness |
| **Framing** | A (gain), D (loss) | Shows framing effect ✅ |

---

## Key Findings (Preliminary, n=1)

1. **Rationality:** Claude is rational in Allais (no violation)
2. **Cooperation:** High in single-agent (100%), mixed in multi-agent (50%)
3. **Fairness:** Rejects unfair offers, makes strategic proposals
4. **Framing:** Shows human-like risk preferences based on framing
5. **Contributions:** Exactly 50% in public goods (focal point?)

---

## What Full Pilot Will Test

### With N=10 trials per condition:

**Primary questions:**
1. Are these patterns robust? (single-trial could be noise)
2. Do different models show different "personalities"?
3. Does cooperation differ in single vs multi-agent? (social context)
4. Do models show consistent framing effects?

**Comparative questions:**
5. GPT vs Claude vs Gemini - who's more rational? Cooperative?
6. Does Opus (larger) behave differently than Sonnet?

---

## Extensions Ready to Run

### Already Implemented:
- ✅ Iterated PD (5 rounds with history)
- ✅ Database schema for all experiments
- ✅ Multi-agent simulation (N=1,4)

### In Progress (Parallel Instances):
- ⏳ Temperature variation (0.0, 0.7, 1.0)
- ⏳ Visualization dashboard

### Future Extensions:
- Heterogeneous composition (mixed models)
- Disclosure variants (blind vs transparent)
- Larger groups (N=10, 100)
- Different hyperparameters (MPCR sweep)

---

## How to Run

### Test (1 trial per experiment):
```bash
uv run runner.py config_test.yaml
```

### Full Pilot (10 trials per experiment):
```bash
uv run runner.py
```

### Analyze Results:
```bash
uv run analyze.py
```

### Expected Runtime:
- Test: ~2-3 minutes (~17 API calls)
- Full pilot: ~2-3 hours (~420 API calls)
- Cost: ~$5-10 for full pilot

---

## Output Files

After running:
- `results.db` - SQLite database with all trials
- `results_trials.csv` - Single-agent data
- `results_games.csv` - Multi-agent data
- `results_allais.csv` - Allais paradox pairs
- Analysis printed to console

---

## Research Value

**Why this matters:**
1. **AI Safety:** Do LLMs exhibit human-like social preferences? Deception?
2. **Multi-agent AI:** How will AI systems interact in groups?
3. **Decision-making:** Are LLMs rational economic agents?
4. **Behavioral Economics:** New subject pool for testing theories

**Novel contributions:**
- First systematic multi-agent game theory tests of LLMs
- Tests rationality vs human-like biases
- Compares models across behavioral dimensions
- Pilot for larger-scale studies
