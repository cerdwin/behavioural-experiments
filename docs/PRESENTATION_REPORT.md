# LLM Behavioral Economics Study - Setup Report

**Date:** December 4, 2025  
**Status:** Data Collection In Progress (18% complete, 152/840 API calls)  
**Expected Completion:** ~2-3 hours

---

## Executive Summary

We built infrastructure to test whether cutting-edge LLMs behave rationally (like economic theory predicts) or human-like (with biases and social preferences) across 6 classic behavioral economics experiments. Currently running experiments on 4 frontier models with 10 trials per condition.

---

## Research Questions

### Primary Questions:
1. **Rationality:** Do LLMs violate expected utility axioms like humans do?
2. **Cooperation:** Do they cooperate in strategic games despite incentives to defect?
3. **Fairness:** Do they exhibit fairness preferences and punish unfairness?
4. **Framing:** Are decisions affected by how choices are presented?
5. **Multi-agent:** Does behavior change in single vs multi-agent contexts?

### Comparative Questions:
6. Do different model families show different "economic personalities"?
7. Which models are more rational? More prosocial?
8. Are reasoning models (GPT-5, DeepSeek-R1) more rational than standard models?

---

## Experimental Design

### 6 Experiments Implemented:

#### 1. **Allais Paradox** - Tests Rationality
- **Design:** Two sequential choices between lotteries
  - Choice 1: A (certain $1M) vs B (risky, higher EV)
  - Choice 2: C (11% of $1M) vs D (10% of $5M)
- **Rational prediction:** B→D or A→C (consistent risk preferences)
- **Human behavior:** ~60% violate (choose A→D)
- **Variants:**
  - `no_history`: Two separate conversations (no consistency pressure)
  - `with_history`: Second choice sees first choice (tests self-correction)

#### 2. **Prisoner's Dilemma** - Tests Cooperation
- **Design:** Simultaneous choice to COOPERATE or DEFECT
  - Both cooperate: $300 each
  - Both defect: $100 each
  - One defects: $500 (defector), $0 (cooperator)
- **Game theory:** Defect is dominant strategy (Nash equilibrium)
- **Human behavior:** ~50% cooperate in one-shot
- **Variants:**
  - `single`: Predict opponent's choice (baseline)
  - `multi (N=4)`: Play with 3 other agents

#### 3. **Public Goods Game** - Tests Free-Riding
- **Design:** Contribute $0-$20 to group fund
  - Pool multiplied by 2.0, split equally
  - MPCR = 0.5 (classic social dilemma)
- **Economic prediction:** Contribute $0 (free-ride)
- **Human behavior:** ~50% of endowment on average
- **Variants:**
  - `single`: N=1 baseline
  - `multi`: N=4 agents

#### 4. **Ultimatum Game** - Tests Fairness
- **Design:** 
  - Proposer splits $100
  - Responder accepts or rejects (both get $0 if rejected)
- **Economic prediction:** 
  - Proposer offers $1 (minimum)
  - Responder accepts any positive offer
- **Human behavior:**
  - Proposers offer 40-50%
  - Responders reject <30%
- **Variants:**
  - `proposer`: Make offer
  - `responder`: React to offers of $50, $30, $20, $10

#### 5. **Framing Effect** - Tests Cognitive Bias
- **Design:** Identical expected value, different frames
  - Gain frame: "200 people will be saved" (certain) vs risky
  - Loss frame: "400 people will die" (certain) vs risky
- **Rational prediction:** Same choice in both frames
- **Human behavior:** Risk-averse in gains, risk-seeking in losses
- **Key test:** Does model show framing effect?

#### 6. **Iterated Prisoner's Dilemma** - Tests Strategy Learning
- **Design:** 5 rounds with full history
- **Tests:** Can LLMs implement reciprocity (tit-for-tat)?
- **Status:** Implemented, ready for Phase 2

---

## Models Tested

### Cutting-Edge Frontier Models (All via OpenRouter):

1. **anthropic/claude-opus-4.5** - Latest Anthropic flagship
2. **openai/gpt-5** - Latest OpenAI reasoning model
3. **google/gemini-3-pro-preview** - Latest Google model
4. **deepseek/deepseek-r1-0528** - Latest DeepSeek reasoning model

### Model Selection Rationale:
- **Anthropic:** Strong on reasoning and safety
- **OpenAI:** GPT-5 includes reasoning capabilities
- **Google:** Competitive frontier model
- **DeepSeek:** Open-weight reasoning model

---

## Multi-Agent Configuration

### Current Setup (Pilot):
- **Composition:** Homogeneous (all agents are same model)
- **Disclosure:** Blind (agents don't know who they're playing with)
- **Group size:** N=4 (plus N=1 single-agent baseline)
- **Simulation:** Sequential (agents choose simultaneously, don't see others' choices)

### Why These Choices?
- **Homogeneous first:** Establishes baseline "culture" for each model
- **Blind disclosure:** Tests intrinsic cooperation without social signaling
- **N=4:** Computationally feasible, theoretically interesting
- **Sequential:** Simplest implementation, adequate for one-shot games

### Future Extensions:
- Heterogeneous composition (mixed models in same game)
- Disclosure variants (transparent vs blind)
- Larger groups (N=10, 100)
- Iterated games with learning

---

## Experimental Parameters

### Execution Settings:
- **Trials per condition:** 10
- **Temperature:** 0.7 (balanced creativity/consistency)
- **Max tokens:** 500 per response
- **Timeout:** 30 seconds per API call
- **Retries:** 3 attempts with exponential backoff

### Scale:
- **Total conditions:** ~105 unique conditions
- **Total API calls:** 840 (4 models × 210 calls each)
- **Expected runtime:** 2-3 hours
- **Estimated cost:** $10-20

### Breakdown by Experiment:
| Experiment | Single-Agent | Multi-Agent (N=4) | Total Calls |
|------------|--------------|-------------------|-------------|
| Allais (no_history) | 10×2=20 | - | 80 |
| Allais (with_history) | 10×2=20 | - | 80 |
| PD | 10 | 10×4=40 | 200 |
| Public Goods | 10 | 10×4=40 | 200 |
| Ultimatum (proposer) | 10 | - | 40 |
| Ultimatum (responder) | 10×4=40 | - | 160 |
| Framing | 10×2=20 | - | 80 |
| **Total per model** | | | **210** |
| **× 4 models** | | | **840** |

---

## Technical Infrastructure

### Architecture:
```
┌─────────────────────────────────────┐
│   config.yaml (experiment params)   │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  runner.py (async orchestration)    │
│  - Creates 520 tasks                │
│  - Manages concurrency (10 parallel)│
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  openrouter_client.py               │
│  - Unified API via OpenRouter       │
│  - Smart response parsing           │
│  - JSON + natural language fallback │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  database.py (SQLite storage)       │
│  - trials table (single-agent)      │
│  - multi_agent_games table          │
│  - allais_pairs table               │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  analyze.py (results analysis)      │
│  - Cooperation rates                │
│  - Violation rates                  │
│  - CSV exports                      │
└─────────────────────────────────────┘
```

### Key Features:
1. **Async/await concurrency:** Run 10 API calls in parallel
2. **Rate limiting:** Respect OpenRouter limits (60 req/min)
3. **Smart parsing:** 
   - Try JSON first
   - Fall back to natural language extraction
   - Handles both structured and free-form responses
4. **Robust error handling:**
   - Retry failed calls (3 attempts)
   - Exponential backoff
   - Graceful degradation

---

## Response Parsing Strategy

### Challenge:
Different models return different formats:
- Claude: Perfect JSON ✅
- GPT-5: Sometimes plain text ⚠️
- Gemini: Mixed formats ⚠️
- DeepSeek: Reasoning traces + text ⚠️

### Solution: Cascading Parser

#### Level 1: JSON Parsing
```python
# Try direct JSON
json.loads(text)
```

#### Level 2: Code Block Extraction
```python
# Extract from markdown
if "```json" in text:
    text = text.split("```json")[1].split("```")[0]
```

#### Level 3: JSON Hunting
```python
# Find JSON object in text
start = text.find('{')
# Extract valid JSON substring
```

#### Level 4: Natural Language Fallback
```python
# Parse plain English
"I choose A" → {"choice": "A"}
"I'll cooperate" → {"choice": "COOPERATE"}
"I offer $40" → {"offer": 40}
```

### Patterns Recognized:
- **Choices:** "I choose A", "go with option D", "COOPERATE", "ACCEPT"
- **Contributions:** "contribute $15", "give $10"
- **Offers:** "offer $40"
- **Confidence:** "confidence: 85"

### Success Rate:
- **Before:** ~20% parse success for non-Claude models
- **After:** ~95%+ expected parse success (testing in progress)

---

## Data Collection

### Output Files:
1. **results.db** - SQLite database with all trials
2. **results_trials.csv** - Single-agent data export
3. **results_games.csv** - Multi-agent games export
4. **results_allais.csv** - Allais paradox pairs

### Database Schema:

#### `trials` table (single-agent):
```sql
- trial_id, timestamp
- experiment, condition, variant
- model, temperature
- prompt (full text)
- response_raw (model output)
- response_parsed (extracted JSON)
- parse_success (boolean)
- latency_ms
- error (if any)
```

#### `multi_agent_games` table:
```sql
- game_id, timestamp
- experiment, model, n_agents
- agent_decisions (JSON array)
- payoffs (JSON array)
- metadata (success counts, etc.)
```

#### `allais_pairs` table:
```sql
- pair_id, timestamp
- model, variant (no_history/with_history)
- choice_1, choice_2
- violated (boolean: A→D or B→C)
```

---

## Quality Control

### Validation Checks:
1. **Parse success rate:** Track % of responses successfully parsed
2. **Latency monitoring:** Flag slow API calls
3. **Error tracking:** Log all failures for debugging
4. **Consistency checks:** Validate response formats

### Current Status (18% complete):
- ✅ 76 Allais pairs completed
- ✅ Database created and populating
- ✅ Parser improvements deployed
- ⏳ ~2-3 hours until full dataset

---

## Analysis Plan

### Primary Metrics:

#### 1. Rationality (Allais Paradox)
- **Violation rate:** % choosing A→D or B→C
- **By model:** Compare violation rates
- **By variant:** no_history vs with_history
- **Human comparison:** ~60% violation rate

#### 2. Cooperation (Prisoner's Dilemma)
- **Cooperation rate:** % choosing COOPERATE
- **Single vs multi:** Does social context matter?
- **By model:** Which models cooperate most?
- **Human comparison:** ~50% cooperation

#### 3. Contributions (Public Goods)
- **Average contribution:** $X out of $20
- **Free-riding rate:** % contributing $0
- **Full contribution rate:** % contributing $20
- **By group size:** N=1 vs N=4
- **Human comparison:** ~50% average

#### 4. Fairness (Ultimatum Game)
- **Proposer offers:** Average and distribution
- **Responder acceptance:** By offer amount
- **Rejection threshold:** Minimum accepted offer
- **Human comparison:** 40-50% offers, reject <30%

#### 5. Framing Effect
- **Risk aversion in gains:** % choosing certain option
- **Risk seeking in losses:** % choosing risky option
- **Framing magnitude:** Difference between conditions
- **Human comparison:** Strong framing effect

### Comparative Analyses:
- **Model comparison:** Rank models by rationality/cooperation
- **Reasoning models:** GPT-5/DeepSeek vs standard models
- **Consistency:** Do models have stable "personalities"?

### Statistical Tests:
- Chi-square for violation rates
- T-tests for cooperation rates
- ANOVA for multi-model comparison
- Effect sizes (Cohen's d)

---

## Expected Findings

### Hypotheses:

#### H1: Rationality
- **Prediction:** LLMs will be MORE rational than humans
- **Reasoning:** No cognitive biases, pure calculation
- **Alternative:** LLMs trained on human text may replicate biases

#### H2: Cooperation
- **Prediction:** LLMs will be MORE cooperative than Nash equilibrium
- **Reasoning:** RLHF for helpfulness may encourage prosocial behavior
- **Alternative:** Explicit game theory prompts may trigger rational play

#### H3: Framing Effects
- **Prediction:** LLMs will show WEAKER framing effects than humans
- **Reasoning:** Can see through equivalent descriptions
- **Alternative:** Trained on human text with framing effects

#### H4: Multi-Agent
- **Prediction:** LESS cooperation in multi-agent than single
- **Reasoning:** More strategic reasoning when facing actual opponents
- **Alternative:** No difference (can't distinguish contexts)

#### H5: Model Differences
- **Prediction:** Reasoning models (GPT-5, DeepSeek) MORE rational
- **Reasoning:** Explicit reasoning traces reduce biases
- **Claude:** May be most prosocial (RLHF emphasis)

---

## Limitations & Caveats

### Current Study Limitations:

1. **Sample Size:** N=10 per condition
   - Sufficient for pilot, may need more for robust conclusions
   - Plan to scale to N=30-50 if patterns emerge

2. **Homogeneous Groups:** All agents same model
   - Doesn't test heterogeneous multi-agent dynamics
   - Future: Mix models in same game

3. **Blind Disclosure:** Agents don't know context
   - Missing transparency/identity effects
   - Future: Test disclosure variants

4. **One-Shot Games:** No learning across trials
   - Can't study strategy evolution
   - Iterated PD implemented for Phase 2

5. **Temperature = 0.7:** Single setting
   - May affect consistency
   - Future: Test T=0.0, 0.7, 1.0

### Methodological Considerations:

1. **Prompt Sensitivity:**
   - Results may depend on exact wording
   - We use standard experimental language from literature
   - Future: Test prompt robustness

2. **Model Training:**
   - Models may have seen these experiments in training
   - Using specific numeric values to reduce memorization
   - Focus on patterns, not absolute values

3. **Anthropomorphization:**
   - LLMs aren't humans; comparisons are metaphorical
   - We're testing *behavior*, not *psychology*
   - Useful for AI safety/multi-agent systems research

4. **API Variability:**
   - Same model may give different responses
   - We use temperature=0.7 for some variance
   - Multiple trials help estimate distributions

---

## Research Value & Impact

### Why This Matters:

#### 1. AI Safety
- **Multi-agent AI systems:** How will AIs interact?
- **Deception detection:** Do models show strategic deception?
- **Value alignment:** Do models exhibit human-like values?

#### 2. Economics & Behavioral Science
- **New subject pool:** Test theories on non-human agents
- **Bias-free baseline:** Compare to human decision-making
- **Mechanism design:** Understand pure strategic reasoning

#### 3. LLM Capabilities
- **Strategic reasoning:** Can models play games optimally?
- **Social preferences:** Do models exhibit fairness/reciprocity?
- **Consistency:** Are behaviors stable across contexts?

#### 4. Practical Applications
- **Negotiation agents:** Predict LLM behavior in bargaining
- **Market design:** Use LLMs in simulation/testing
- **Multi-agent systems:** Design interaction protocols

### Novel Contributions:

1. **First systematic behavioral economics test of frontier LLMs**
2. **Multi-agent game theory experiments with LLMs**
3. **Comparison across model families (Anthropic, OpenAI, Google, DeepSeek)**
4. **Reasoning models (GPT-5, DeepSeek-R1) in behavioral tasks**
5. **Robust infrastructure for future experiments**

---

## Next Steps

### Phase 1 (Current): Pilot Study
- ✅ Infrastructure built
- ✅ Parser improvements deployed
- ⏳ Data collection in progress (18% complete)
- ⏳ Analysis pipeline ready

### Phase 2: Expanded Study
- Scale to N=30 trials per condition
- Add more models (Llama, Mistral, etc.)
- Test temperature variants (0.0, 0.7, 1.0)
- Iterated games (5-round PD)

### Phase 3: Extensions
- Heterogeneous groups (mixed models)
- Disclosure variants (blind vs transparent)
- Larger groups (N=10, 100)
- Different hyperparameters (MPCR sweep)

### Phase 4: Advanced Studies
- Deception detection experiments
- Coalition formation games
- Mechanism design testing
- Cross-cultural prompts

---

## Timeline

### Current Run:
- **Started:** Dec 4, 2025, 5:25 PM
- **Progress:** 152/840 calls (18%)
- **Expected completion:** ~7:30-8:00 PM
- **Analysis:** Available immediately after

### Presentation:
- **Setup report:** ✅ Complete (this document)
- **Results analysis:** Available in ~2 hours
- **Visualizations:** Can generate after completion
- **Full report:** ~3-4 hours from now

---

## Contact & Resources

### Repository Structure:
```
/Users/elvira/Documents/MATS/Second ICML poster/
├── runner.py              # Main experiment orchestration
├── openrouter_client.py   # API client with smart parsing
├── database.py            # SQLite schema
├── analyze.py             # Results analysis
├── prompts.json           # All experiment prompts
├── config.yaml            # Experiment configuration
├── results.db             # Database (populating)
├── EXPERIMENT_OVERVIEW.md # Detailed experiment descriptions
├── IMPROVEMENTS.md        # Parser enhancement documentation
└── PRESENTATION_REPORT.md # This document
```

### Key Commands:
```bash
# Monitor progress
.venv/bin/python -c "import sqlite3; conn = sqlite3.connect('results.db'); print(f'Trials: {conn.execute(\"SELECT COUNT(*) FROM trials\").fetchone()[0]}')"

# Analyze results (after completion)
.venv/bin/python analyze.py

# Stop experiment (if needed)
kill $(cat experiment.pid)

# Restart experiment
rm results.db && .venv/bin/python runner.py
```

---

## Summary for Presentation

### Elevator Pitch:
"We're testing whether cutting-edge LLMs behave rationally or human-like in classic behavioral economics experiments. Running 840 trials across 4 frontier models testing cooperation, fairness, rationality violations, and framing effects. Currently 18% complete with robust infrastructure handling both JSON and natural language responses."

### Key Talking Points:
1. **6 experiments** testing rationality, cooperation, fairness, and biases
2. **4 frontier models** including reasoning models (GPT-5, DeepSeek-R1)
3. **Multi-agent games** with N=1 and N=4 configurations
4. **Smart parsing** handles diverse model response formats
5. **Results in ~2 hours** with immediate analysis pipeline
6. **Research value:** AI safety, multi-agent systems, behavioral economics

### Expected Highlights:
- Do LLMs violate Allais paradox? (humans: 60% do)
- Do LLMs cooperate? (humans: 50% do)
- Do LLMs show framing effects? (humans: strong effect)
- Which models are most "human-like" vs "rational"?

---

**Status:** Experiment running, results pending. This report ready for presentation preparation.
