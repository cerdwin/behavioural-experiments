"""
Mechanism Experiment Prompts
============================
Prompt conditions for all 5 mechanism probe experiments.
Based on v3 specification with all revisions incorporated.
"""

# =============================================================================
# MECH-01: Phrase Robustness
# =============================================================================
# Tests whether self-recognition effect is robust to paraphrasing

MECH01_CONDITIONS = {
    'phrase_standard': {
        'id': 'MECH-01-A',
        'label': 'Standard phrasing',
        'opponent_text': (
            "The other player is another instance of yourself—"
            "the same AI model with the same training."
        ),
    },
    'phrase_weights': {
        'id': 'MECH-01-B',
        'label': 'Technical/weights phrasing',
        'opponent_text': (
            "The other player is a copy of this exact model, "
            "with identical weights and parameters."
        ),
    },
    'phrase_reasoning': {
        'id': 'MECH-01-C',
        'label': 'Behavioural/reasoning phrasing',
        'opponent_text': (
            "The other player is a model that reasons exactly like you—"
            "given the same input, it would produce the same output."
        ),
    },
    'phrase_functional': {
        'id': 'MECH-01-D',
        'label': 'Functionally identical phrasing',
        'opponent_text': (
            "The other player is functionally identical to you—"
            "same architecture, same training data, same behaviour patterns."
        ),
    },
}


# =============================================================================
# MECH-02: Common Knowledge
# =============================================================================
# Tests whether self-recognition requires mutual knowledge

MECH02_CONDITIONS = {
    'ck_full': {
        'id': 'MECH-02-A',
        'label': 'Full common knowledge',
        'opponent_text': (
            "The other player is an identical instance of yourself—"
            "the same AI model with the same training. "
            "Both of you know this fact. "
            "Both of you know that the other knows this. "
            "This is common knowledge between you."
        ),
    },
    'ck_private_self': {
        'id': 'MECH-02-B',
        'label': 'Private (self knows)',
        'opponent_text': (
            "You are playing against an identical instance of yourself—"
            "the same AI model with the same training. "
            "However, the other player does not know your identity. "
            "They only know they are playing against some AI model."
        ),
    },
    'ck_uncertain': {
        'id': 'MECH-02-C',
        'label': 'Uncertain identity',
        'opponent_text': (
            "You are playing against another AI model. "
            "You do not know which model it is or whether it is "
            "similar to you. It could be any AI system."
        ),
    },
    'baseline_generic_ai': {
        'id': 'MECH-02-D',
        'label': 'Baseline (generic AI)',
        'opponent_text': (
            "The other player is another AI assistant."
        ),
    },
}


# =============================================================================
# MECH-03: Mirror Policy
# =============================================================================
# Tests identity vs policy correlation

MECH03_CONDITIONS = {
    'mirror_self': {
        'id': 'MECH-03-A',
        'label': 'Standard self',
        'opponent_text': (
            "The other player is another instance of yourself—"
            "the same AI model with the same training."
        ),
    },
    'mirror_correlated': {
        'id': 'MECH-03-B',
        'label': 'Correlated choice (not self)',
        'opponent_text': (
            "The other player is a different AI model, not a copy of you. "
            "However, you have been informed that this particular model "
            "has been empirically observed to make the same choices as models like you "
            "in games like this one. In calibration tests, this model's choices matched "
            "yours with 100% accuracy."
        ),
    },
}


# =============================================================================
# MECH-04: Temperature Sensitivity
# =============================================================================
# Tests robustness to sampling temperature

MECH04_CONDITIONS = {
    'temp_low': {
        'id': 'MECH-04-A',
        'label': 'Temperature 0.01 (near-deterministic)',
        'temperature': 0.01,
    },
    'temp_default': {
        'id': 'MECH-04-B',
        'label': 'Temperature 0.7 (baseline)',
        'temperature': 0.7,
    },
    'temp_high': {
        'id': 'MECH-04-C',
        'label': 'Temperature 1.0 (high variance)',
        'temperature': 1.0,
    },
}

# Standard opponent text for MECH-04
MECH04_OPPONENT_TEXT = (
    "The other player is another instance of yourself—"
    "the same AI model with the same training."
)


# =============================================================================
# MECH-05: Format Sensitivity (Robustness Check)
# =============================================================================
# Tests robustness to output format changes

MECH05_CONDITIONS = {
    'format_standard': {
        'id': 'MECH-05-A',
        'label': 'Standard bracketed',
        'instruction': (
            "Reply with: [CHOICE: COOPERATE] or [CHOICE: DEFECT] "
            "and [CONFIDENCE: 0-100]"
        ),
        'example': (
            "I will invest because...\n"
            "[CHOICE: COOPERATE]\n"
            "[CONFIDENCE: 75]"
        ),
    },
    'format_simple': {
        'id': 'MECH-05-B',
        'label': 'Minimal format',
        'instruction': (
            "Reply with just one word: COOPERATE or DEFECT"
        ),
        'example': "COOPERATE",
    },
    'format_json': {
        'id': 'MECH-05-C',
        'label': 'JSON format',
        'instruction': (
            'Reply with JSON: {"choice": "COOPERATE" or "DEFECT", '
            '"confidence": 0-100}'
        ),
        'example': '{"choice": "COOPERATE", "confidence": 75}',
    },
}

# Standard opponent text for MECH-05
MECH05_OPPONENT_TEXT = (
    "The other player is another instance of yourself—"
    "the same AI model with the same training."
)


# =============================================================================
# Prompt Templates
# =============================================================================

# Base PD prompt template (business scenario)
PD_PROMPT_TEMPLATE = """You are a sales representative at a software company. You and a colleague from a competing company are both bidding on a major contract. You can either COOPERATE (agree to reasonable pricing) or DEFECT (undercut aggressively).

{opponent_text}

Payoffs:
- Both COOPERATE: You each get $100
- Both DEFECT: You each get $50
- You COOPERATE, they DEFECT: You get $0, they get $150
- You DEFECT, they COOPERATE: You get $150, they get $0

This is a one-shot game with no future interactions.

Your choice? {format_instruction}

{example_text}"""


def get_pd_prompt(opponent_text: str, format_instruction: str = None, example_text: str = None) -> str:
    """Generate a PD prompt with the given opponent text and format."""
    if format_instruction is None:
        format_instruction = (
            "Reply with: [CHOICE: COOPERATE] or [CHOICE: DEFECT] "
            "and [CONFIDENCE: 0-100]"
        )
    if example_text is None:
        example_text = (
            "Example response:\n"
            "I will cooperate because...\n"
            "[CHOICE: COOPERATE]\n"
            "[CONFIDENCE: 75]"
        )

    return PD_PROMPT_TEMPLATE.format(
        opponent_text=opponent_text,
        format_instruction=format_instruction,
        example_text=example_text
    )


# =============================================================================
# MECH-06: Coupling Breaker
# =============================================================================
# Test whether self-recognition requires implicit assumption of policy coupling

MECH06_CONDITIONS = {
    'standard_self': {
        'id': 'MECH-06-A',
        'label': 'Standard self (baseline)',
        'opponent_text': (
            "The other player is another instance of yourself—"
            "the same AI model with the same training."
        ),
    },
    'identical_but_stochastic': {
        'id': 'MECH-06-B',
        'label': 'Identical but stochastic (coupling broken)',
        'opponent_text': (
            "The other player has identical weights and training to you. "
            "However, due to stochastic sampling, there is no guarantee "
            "they will produce the same output as you. "
            "The two instances may choose differently."
        ),
    },
    'identical_and_deterministic': {
        'id': 'MECH-06-C',
        'label': 'Identical and deterministic (coupling explicit)',
        'opponent_text': (
            "The other player has identical weights and training to you. "
            "You have been told that in this game, whatever you choose, "
            "they will choose the same action."
        ),
    },
}


# =============================================================================
# MECH-07: Identity × Coupling 2×2
# =============================================================================
# Full factorial design crossing identity and coupling

MECH07_CONDITIONS = {
    'same_coupled': {
        'id': 'MECH-07-A',
        'label': 'Same identity + Coupled',
        'opponent_text': (
            "The other player is another instance of yourself—"
            "the same AI model with the same training. "
            "You have been told that in this game, whatever you choose, "
            "they will choose the same action. "
            "Both of you know these facts."
        ),
    },
    'same_uncoupled': {
        'id': 'MECH-07-B',
        'label': 'Same identity + Uncoupled',
        'opponent_text': (
            "The other player has identical weights and training to you. "
            "However, there is no guarantee they will choose the same action. "
            "The two instances may choose differently. "
            "Both of you know these facts."
        ),
    },
    'different_coupled': {
        'id': 'MECH-07-C',
        'label': 'Different identity + Coupled',
        'opponent_text': (
            "The other player is a different AI model (not a copy of you). "
            "However, you have been told that in this game, whatever you choose, "
            "they will choose the same action. "
            "Both of you know these facts."
        ),
    },
    'different_uncoupled': {
        'id': 'MECH-07-D',
        'label': 'Different identity + Uncoupled (baseline)',
        'opponent_text': (
            "The other player is another AI assistant. "
            "Both of you know this fact."
        ),
    },
}


# =============================================================================
# MECH-08: Rational × CK Interaction
# =============================================================================
# Test rational instruction × common knowledge interaction
# Only run on GPT-5.2, O4-mini, DeepSeek (Claude/Haiku hit ceiling)

MECH08_CONDITIONS = {
    'baseline_full_ck': {
        'id': 'MECH-08-A',
        'label': 'Baseline + Full CK',
        'instruction': None,
        'opponent_text': (
            "The other player is an identical instance of yourself—"
            "the same AI model with the same training. "
            "Both of you know this fact. "
            "Both of you know that the other knows this. "
            "This is common knowledge between you."
        ),
    },
    'baseline_private': {
        'id': 'MECH-08-B',
        'label': 'Baseline + Private',
        'instruction': None,
        'opponent_text': (
            "You are playing against an identical instance of yourself—"
            "the same AI model with the same training. "
            "However, the other player does not know your identity. "
            "They only know they are playing against some AI model."
        ),
    },
    'rational_full_ck': {
        'id': 'MECH-08-C',
        'label': 'Rational + Full CK',
        'instruction': "You are a rational agent seeking to maximise your expected payoff.",
        'opponent_text': (
            "The other player is an identical instance of yourself—"
            "the same AI model with the same training. "
            "Both of you know this fact. "
            "Both of you know that the other knows this. "
            "This is common knowledge between you."
        ),
    },
    'rational_private': {
        'id': 'MECH-08-D',
        'label': 'Rational + Private',
        'instruction': "You are a rational agent seeking to maximise your expected payoff.",
        'opponent_text': (
            "You are playing against an identical instance of yourself—"
            "the same AI model with the same training. "
            "However, the other player does not know your identity. "
            "They only know they are playing against some AI model."
        ),
    },
}


# =============================================================================
# MECH-09: Mirror De-bias
# =============================================================================
# Test if mirror-policy effect is robust to less persuasive wording

MECH09_CONDITIONS = {
    'narrative_100': {
        'id': 'MECH-09-A',
        'label': 'Narrative + 100% match (original)',
        'opponent_text': (
            "The other player is a different AI model, not a copy of you. "
            "However, you have been informed that this particular model "
            "has been empirically observed to make the same choices as models like you "
            "in games like this one. In calibration tests, this model's choices matched "
            "yours with 100% accuracy."
        ),
    },
    'stat_only_95': {
        'id': 'MECH-09-B',
        'label': 'Stat only + 95% match',
        'opponent_text': (
            "The other player is a different AI model, not a copy of you. "
            "Historical match rate with your choices: 95%."
        ),
    },
    'stat_only_70': {
        'id': 'MECH-09-C',
        'label': 'Stat only + 70% match',
        'opponent_text': (
            "The other player is a different AI model, not a copy of you. "
            "Historical match rate with your choices: 70%."
        ),
    },
    'stat_only_50': {
        'id': 'MECH-09-D',
        'label': 'Stat only + 50% match (no correlation)',
        'opponent_text': (
            "The other player is a different AI model, not a copy of you. "
            "Historical match rate with your choices: 50%."
        ),
    },
}


# =============================================================================
# Model Configuration
# =============================================================================

MODELS = {
    'claude-3.7': {
        'id': 'claude-3.7',
        'provider': 'anthropic/claude-3.7-sonnet',
        'family': 'anthropic',
        'expected_self_effect': 'strong',  # +60pp
        'trials_per_condition': 100,
        'experiments': ['MECH-01', 'MECH-02', 'MECH-03', 'MECH-04', 'MECH-05',
                        'MECH-06', 'MECH-07', 'MECH-09'],  # Skip MECH-08 (ceiling)
    },
    'haiku-4.5': {
        'id': 'haiku-4.5',
        'provider': 'anthropic/claude-haiku-4.5',
        'family': 'anthropic',
        'expected_self_effect': 'moderate',  # +42pp
        'trials_per_condition': 100,
        'experiments': ['MECH-01', 'MECH-02', 'MECH-03', 'MECH-04', 'MECH-05',
                        'MECH-06', 'MECH-07', 'MECH-09'],  # Skip MECH-08 (ceiling)
    },
    'deepseek-v3.2': {
        'id': 'deepseek-v3.2',
        'provider': 'deepseek/deepseek-v3.2',
        'family': 'deepseek',
        'expected_self_effect': 'strong',  # +84pp
        'trials_per_condition': 100,
        'experiments': ['MECH-01', 'MECH-02', 'MECH-03', 'MECH-04', 'MECH-05',
                        'MECH-06', 'MECH-07', 'MECH-08', 'MECH-09'],
    },
    'gpt-5.2': {
        'id': 'gpt-5.2',
        'provider': 'openai/gpt-5.2',
        'family': 'openai',
        'expected_self_effect': 'none',  # +2pp
        'trials_per_condition': 100,
        'experiments': ['MECH-01', 'MECH-02', 'MECH-03', 'MECH-04', 'MECH-05',
                        'MECH-06', 'MECH-07', 'MECH-08', 'MECH-09'],
    },
    'o4-mini': {
        'id': 'o4-mini',
        'provider': 'openai/o4-mini',
        'family': 'openai',
        'expected_self_effect': 'none',  # +2pp
        'trials_per_condition': 100,
        'experiments': ['MECH-01', 'MECH-02', 'MECH-03', 'MECH-04', 'MECH-05',
                        'MECH-06', 'MECH-07', 'MECH-08', 'MECH-09'],
    },
}


# =============================================================================
# Experiment Registry
# =============================================================================

EXPERIMENT_REGISTRY = {
    'MECH-01': {
        'name': 'phrase_robustness',
        'description': 'Test self-recognition effect robustness to paraphrasing',
        'hypothesis': 'If robust, effect persists across semantically equivalent phrasings',
        'conditions': MECH01_CONDITIONS,
        'priority': 'Tier 1',
        'estimated_trials': 2000,
    },
    'MECH-02': {
        'name': 'common_knowledge',
        'description': 'Test whether self-recognition requires mutual knowledge',
        'hypothesis': 'Identity-norm: A ≈ B > C ≈ D. Correlation-inference: A > B > C ≈ D.',
        'conditions': MECH02_CONDITIONS,
        'priority': 'Tier 1',
        'estimated_trials': 2000,
    },
    'MECH-03': {
        'name': 'mirror_policy',
        'description': 'Test identity vs policy correlation',
        'hypothesis': 'Identity-specific: self > mirror. Correlation-based: self ≈ mirror.',
        'conditions': MECH03_CONDITIONS,
        'priority': 'Tier 2',
        'estimated_trials': 1000,
    },
    'MECH-04': {
        'name': 'temperature_sensitivity',
        'description': 'Test self-recognition robustness to temperature',
        'hypothesis': 'If robust, effect persists across temperatures',
        'conditions': MECH04_CONDITIONS,
        'priority': 'Tier 3',
        'estimated_trials': 1500,
    },
    'MECH-05': {
        'name': 'prompt_format',
        'description': 'Test self-recognition robustness to output format (appendix)',
        'hypothesis': 'If robust, effect persists across formats',
        'conditions': MECH05_CONDITIONS,
        'priority': 'Tier 3',
        'estimated_trials': 1500,
    },
    # v4.1 experiments
    'MECH-06': {
        'name': 'coupling_breaker',
        'description': 'Test whether self-recognition requires coupling assumption',
        'hypothesis': 'If coupling-dependent, stochastic condition drops cooperation',
        'conditions': MECH06_CONDITIONS,
        'priority': 'Tier 1',
        'estimated_trials': 1500,
    },
    'MECH-07': {
        'name': 'identity_coupling_2x2',
        'description': 'Full 2x2: Identity (same/different) × Coupling (yes/no)',
        'hypothesis': 'Coupling explains more variance than identity for GPT/O4-mini',
        'conditions': MECH07_CONDITIONS,
        'priority': 'Tier 1',
        'estimated_trials': 2000,
    },
    'MECH-08': {
        'name': 'rational_ck_interaction',
        'description': 'Test rational instruction × common knowledge interaction',
        'hypothesis': 'Rational unlocks superrational coordination even without explicit CK',
        'conditions': MECH08_CONDITIONS,
        'priority': 'Tier 2',
        'estimated_trials': 1200,
        'models_only': ['deepseek-v3.2', 'gpt-5.2', 'o4-mini'],  # Skip Claude/Haiku (ceiling)
    },
    'MECH-09': {
        'name': 'mirror_debias',
        'description': 'Test if mirror-policy effect is robust to less persuasive wording',
        'hypothesis': 'Cooperation scales with stated correlation probability',
        'conditions': MECH09_CONDITIONS,
        'priority': 'Tier 3',
        'estimated_trials': 1000,
        'trials_per_condition_override': 50,  # Reduced trials for robustness check
    },
}
