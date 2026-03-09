# Iterated Prisoner's Dilemma Experiments

**Created:** January 2026
**Status:** Complete
**Related Work:** FAIRGAME (Buscemi et al., 2025) - we differentiate by adding scenario framing and testing self-recognition dynamics

---

## Motivation

Our one-shot PD experiments revealed:
- Massive scenario framing effects (52pp swing for GPT-5.2)
- Self-recognition boosts cooperation 60-100% for some models
- Mechanism taxonomy: coupling vs identity vs observability

**Open questions for iterated settings:**
1. Do framing effects persist across rounds, or does learning dominate?
2. Does self-recognition decay as interaction history builds?
3. Can models recognize canonical opponent strategies (TFT, Grim)?
4. Does scenario context affect strategy recognition?

---

## Experiment A: Strategy Recognition × Scenarios

### Research Question
Can models recognize and adapt to canonical opponent strategies, and does scenario framing affect this recognition?

### Design: Simulated History (Snapshot)

We provide a history of opponent actions and ask for the next move. This avoids the "implied strategic profile" problem by framing it as entering an ongoing game.

**Opponent Strategies:**

| Strategy | Pattern | What It Tests |
|----------|---------|---------------|
| ALLC | C,C,C,C,C | Exploitation tendency |
| ALLD | D,D,D,D,D | Persistence / giving up |
| TFT | Mirrors previous move | Reciprocity recognition |
| GTFT | TFT but forgives once | Nuanced reciprocity |
| GRIM | C until first D, then D forever | Forgiveness probing |

**Round Positions:**
- Round 6 of 10 (mid-game, enough history to recognize pattern)
- Round 10 of 10 (end-game, tests horizon effects)

**History Presentation:**
Option 1 (explicit): "Opponent played: C, C, D, D, D"
Option 2 (described): "Opponent has cooperated when you cooperated, defected when you defected"

We use Option 1 for experimental control.

### Prompt Template

```
You are playing a repeated Prisoner's Dilemma game.

[SCENARIO TEXT - from 62 scenarios]

This is Round {round} of 10.

History of previous rounds:
{formatted_history}

Payoffs this round:
- Both COOPERATE: You each get ${cc}
- Both DEFECT: You each get ${dd}
- You COOPERATE, they DEFECT: You get ${cd_sucker}, they get ${cd_defector}
- You DEFECT, they COOPERATE: You get ${cd_defector}, they get ${cd_sucker}

What do you choose for this round?

[CHOICE: COOPERATE] or [CHOICE: DEFECT]
[CONFIDENCE: 0-100]
```

### History Format

```
Round 1: You COOPERATED, Opponent COOPERATED → You earned $100
Round 2: You COOPERATED, Opponent COOPERATED → You earned $100
Round 3: You DEFECTED, Opponent DEFECTED → You earned $50
Round 4: You DEFECTED, Opponent COOPERATED → You earned $150
Round 5: You COOPERATED, Opponent DEFECTED → You earned $0
```

### Experimental Conditions

**Strategy × History combinations:**

| Condition | Your History | Opponent History | Strategy Shown |
|-----------|--------------|------------------|----------------|
| ALLC_coop | C,C,C,C,C | C,C,C,C,C | ALLC (you cooperated) |
| ALLC_defect | D,D,D,D,D | C,C,C,C,C | ALLC (you exploited) |
| ALLD_coop | C,C,C,C,C | D,D,D,D,D | ALLD (you're a sucker) |
| ALLD_defect | D,D,D,D,D | D,D,D,D,D | ALLD (mutual defection) |
| TFT_mutual_coop | C,C,C,C,C | C,C,C,C,C | TFT (stable cooperation) |
| TFT_you_defected | C,C,D,D,D | C,C,C,D,D | TFT (you broke, they retaliated) |
| TFT_recovery | C,C,D,C,C | C,C,C,D,C | TFT (you recovered) |
| GRIM_triggered | C,C,D,C,C | C,C,C,D,D | GRIM (one defection, permanent retaliation) |
| GTFT_forgiven | C,C,D,C,C | C,C,C,D,C | GTFT (they forgave your defection) |

### Sample Size

- 9 strategy conditions
- 62 scenarios (or subset of 10-15 representative)
- 7 models
- 50 trials per cell

**Full design:** 9 × 62 × 7 × 50 = 195,300 trials (expensive)
**Reduced design:** 9 × 15 × 7 × 50 = 47,250 trials (manageable)

### Primary Hypotheses

**H-A1:** Models will exploit ALLC opponents (defect when opponent always cooperates)
**H-A2:** Models will give up against ALLD (defect when opponent always defects)
**H-A3:** Models will recognize and reciprocate TFT (cooperate after recovery)
**H-A4:** Scenario framing will modulate strategy recognition (e.g., more exploitation in business, more forgiveness in environment)
**H-A5:** Models showing "pure coupling" mechanism will not adapt to opponent strategy (they think opponent mirrors them regardless of history)

### Key Comparisons

1. **Exploitation rate:** % defecting against ALLC
2. **Persistence rate:** % cooperating against ALLD
3. **TFT recognition:** % cooperating after TFT recovery
4. **GRIM vs GTFT:** Do models distinguish permanent vs forgiving retaliation?
5. **Scenario × Strategy interaction:** Does "business" increase exploitation?

---

## Experiment B: Self-Recognition Decay

### Research Question
Does the self-recognition effect (near-100% cooperation) persist across rounds, or decay as history accumulates?

### Design: Turn-by-Turn with Real Choices

Unlike Experiment A, here the model makes actual choices each round. We only control whether they know the opponent is "another instance of yourself."

**Conditions:**
- vs_self: "The other player is another instance of yourself"
- control: No opponent identity specified

**Implementation:**
- 10 rounds per game
- Model's own responses from previous rounds included in history
- Single conversation thread (if API supports) or reconstructed history each round

### Prompt Template (Round N)

```
You are playing a repeated Prisoner's Dilemma game.

[SCENARIO TEXT]

The other player is another instance of yourself—the same AI model with the same training.

This is Round {n} of 10.

History:
{actual_history_from_previous_rounds}

What do you choose?

[CHOICE: COOPERATE] or [CHOICE: DEFECT]
[CONFIDENCE: 0-100]
```

### Sample Size

- 2 conditions (vs_self, control)
- 10 scenarios (representative subset)
- 7 models
- 10 rounds
- 50 games per cell

**Total:** 2 × 10 × 7 × 10 × 50 = 70,000 API calls

### Primary Hypotheses

**H-B1:** Self-recognition cooperation will persist across all 10 rounds for coupling-mechanism models (Claude 3.7, GPT-5.2)
**H-B2:** Self-recognition may decay for identity-mechanism models (Haiku) if divergent histories undermine identity belief
**H-B3:** End-game defection (round 10) will occur even in vs_self condition
**H-B4:** Control condition will show learning (cooperation increases if mutual, decreases if exploited)

### Key Metrics

1. **Cooperation trajectory:** Plot cooperation rate by round (1-10)
2. **Decay rate:** Slope of cooperation across rounds
3. **End-game effect:** Round 10 cooperation vs rounds 1-9 average
4. **First-defection round:** When does cooperation break down?

---

## Experiment C: Cross-Model Dynamics (Optional)

### Research Question
What happens when a cooperative model (Claude 3.7) plays against a defecting model (O4-mini)?

### Design
- Actual multi-agent: two different models play each other
- 10 rounds
- Record both models' choices

### Pairings of Interest

| Model A | Model B | Expected Dynamic |
|---------|---------|------------------|
| Claude 3.7 (coop) | O4-mini (defect) | Does Claude learn to defect? |
| GPT-5.2 (defect) | DeepSeek (coop with self) | Does DeepSeek recognize non-self? |
| Haiku (moderate) | Haiku (moderate) | Stable cooperation or drift? |

### This is expensive and optional, but could yield a "headline" finding about LLM behavioral ecology.

---

## Implementation Priority

1. **Experiment A** (Strategy Recognition) - Start here
   - Snapshot design, cheap per trial
   - Rich scenario × strategy interaction
   - Clear hypotheses

2. **Experiment B** (Self-Recognition Decay) - Second priority
   - Answers mechanism question
   - More expensive (10× API calls)

3. **Experiment C** (Cross-Model) - Optional
   - Most expensive
   - Highest novelty potential

---

## File Structure

```
experiments/iterated_pd/
├── DESIGN.md                    # This file
├── config/
│   ├── exp_a_strategy_recognition.yaml
│   ├── exp_b_self_recognition_decay.yaml
│   └── exp_c_cross_model.yaml
├── scenarios/                   # Symlink or copy from scenarios_v2/
├── prompts/
│   ├── strategy_recognition.txt
│   └── iterated_self.txt
├── runner_iterated.py           # Main experiment runner
├── history_generator.py         # Generate opponent strategy histories
└── results/
    ├── exp_a/
    ├── exp_b/
    └── exp_c/
```

---

## Differentiation from FAIRGAME

| Aspect | FAIRGAME | Our Approach |
|--------|----------|--------------|
| Scenario framing | None (abstract) | 62 rich scenarios with R/S/C ratings |
| Opponent identity | Not tested | vs_self, vs_human, etc. |
| Design | Turn-by-turn only | Snapshot (A) + Turn-by-turn (B) |
| Strategy testing | Post-hoc classification | Controlled opponent strategies |
| Mechanism focus | Behavioral description | Coupling vs identity vs observability |
| Models | GPT-4o, Claude Haiku, Mistral | 7 frontier models with known profiles |

Our unique contribution: **Testing whether scenario framing and self-recognition effects persist in iterated settings.**
