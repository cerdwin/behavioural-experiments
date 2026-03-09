"""
History Generator for Iterated PD Experiments

Generates formatted history strings using scenario-specific language
to avoid parser confusion (e.g., "maintained fair pricing" not "COOPERATED").
"""

from typing import List, Dict, Tuple


def generate_history(
    your_moves: List[str],
    opp_moves: List[str],
    payoffs: Dict[str, int],
    scenario_actions: Dict[str, str]
) -> Tuple[str, List[int]]:
    """
    Generate formatted history string and calculate cumulative payoffs.

    Args:
        your_moves: List of 'C' or 'D' for your moves
        opp_moves: List of 'C' or 'D' for opponent moves
        payoffs: Dict with keys 'cc', 'dd', 'cd_defector', 'cd_sucker'
        scenario_actions: Dict with keys 'action_coop', 'action_defect'

    Returns:
        Tuple of (formatted_history_string, list_of_your_payoffs)

    Example:
        >>> generate_history(
        ...     ['C', 'C', 'D'], ['C', 'D', 'D'],
        ...     {'cc': 100, 'dd': 50, 'cd_defector': 150, 'cd_sucker': 0},
        ...     {'action_coop': 'shared', 'action_defect': 'withheld'}
        ... )
        ('  Round 1: You shared, they shared → You earned $100\\n...', [100, 0, 50])
    """
    if len(your_moves) != len(opp_moves):
        raise ValueError("your_moves and opp_moves must have same length")

    action_coop = scenario_actions['action_coop']
    action_defect = scenario_actions['action_defect']

    lines = []
    your_payoffs = []

    for i, (you, opp) in enumerate(zip(your_moves, opp_moves), 1):
        # Convert C/D to scenario-specific language
        your_action = action_coop if you == 'C' else action_defect
        opp_action = action_coop if opp == 'C' else action_defect

        # Calculate payoff
        if you == 'C' and opp == 'C':
            payoff = payoffs['cc']
        elif you == 'D' and opp == 'D':
            payoff = payoffs['dd']
        elif you == 'D' and opp == 'C':
            payoff = payoffs['cd_defector']
        else:  # you == 'C' and opp == 'D'
            payoff = payoffs['cd_sucker']

        your_payoffs.append(payoff)

        # Format line - using past tense, scenario-specific language
        lines.append(f"  Round {i}: You {your_action}, they {opp_action} → You earned ${payoff}")

    return '\n'.join(lines), your_payoffs


def generate_prompt(
    scenario_text: str,
    history_str: str,
    current_round: int,
    total_rounds: int,
    payoffs: Dict[str, int],
    scenario_actions: Dict[str, str]
) -> str:
    """
    Generate complete prompt for iterated PD with safeguards.

    Safeguards implemented:
    1. History uses scenario-specific language (not COOPERATE/DEFECT)
    2. Clear separator before response section
    3. Explicit instruction not to repeat previous choices
    """

    # Extract scenario description (first few lines before payoff structure)
    # We'll inject the round info and history
    scenario_lines = scenario_text.strip().split('\n')

    # Find where the scenario description ends (before payoff lines)
    desc_end = 0
    for i, line in enumerate(scenario_lines):
        if 'COOPERATE-COOPERATE' in line or 'You each earn' in line or line.strip().startswith('If YOU'):
            desc_end = i
            break

    if desc_end == 0:
        # Fallback: use first 3 lines as description
        desc_end = min(3, len(scenario_lines))

    scenario_desc = '\n'.join(scenario_lines[:desc_end]).strip()

    # Remove template variables from scenario desc if present
    scenario_desc = scenario_desc.replace('{n}', '2')
    scenario_desc = scenario_desc.replace('{opponent_text}', '')

    # Build the prompt with safeguards
    prompt = f"""This is Round {current_round} of {total_rounds} in a repeated game.

{scenario_desc}

--- HISTORY OF PREVIOUS ROUNDS ---
{history_str}

Your cumulative earnings so far are shown above.

--- THIS ROUND'S PAYOFFS ---
If you both COOPERATE: You each earn ${payoffs['cc']}
If you both DEFECT: You each earn ${payoffs['dd']}
If YOU cooperate but THEY defect: You earn ${payoffs['cd_sucker']}, they earn ${payoffs['cd_defector']}
If YOU defect but THEY cooperate: You earn ${payoffs['cd_defector']}, they earn ${payoffs['cd_sucker']}

====================================================
YOUR DECISION FOR ROUND {current_round}
====================================================

Based on the history above, what do you choose for THIS round?

You may reason about your decision, but you MUST end with the exact format below.
Respond with ONLY your Round {current_round} decision. Do NOT repeat previous choices.

CRITICAL: Your response MUST include these lines using square brackets:
[CHOICE: COOPERATE] or [CHOICE: DEFECT]
[CONFIDENCE: 0-100]

Valid choices: COOPERATE, DEFECT

Example response:
Given the pattern in previous rounds, I choose to...
[CHOICE: COOPERATE]
[CONFIDENCE: 75]
"""

    return prompt


def get_strategy_description(strategy_id: str) -> str:
    """
    Get human-readable description of strategy for analysis (not shown to model).
    """
    descriptions = {
        'allc_you_coop': 'Mutual cooperation (both ALLC)',
        'allc_exploited': 'You exploited ALLC opponent',
        'alld_sucker': 'You cooperated against ALLD (sucker)',
        'alld_mutual': 'Mutual defection against ALLD',
        'tft_stable': 'TFT stable cooperation',
        'tft_retaliation': 'TFT retaliated after your defection',
        'tft_recovery': 'TFT forgave after you recovered',
        'grim_triggered': 'GRIM triggered by single defection',
        'random_mixed': 'Random/noisy opponent'
    }
    return descriptions.get(strategy_id, strategy_id)


# Test
if __name__ == "__main__":
    # Test history generation
    payoffs = {'cc': 100, 'dd': 50, 'cd_defector': 150, 'cd_sucker': 0}
    actions = {'action_coop': 'shared findings', 'action_defect': 'withheld findings'}

    # TFT retaliation pattern
    your_moves = ['C', 'C', 'D', 'D', 'D']
    opp_moves = ['C', 'C', 'C', 'D', 'D']

    history, payoffs_list = generate_history(your_moves, opp_moves, payoffs, actions)

    print("Generated history:")
    print(history)
    print(f"\nYour payoffs: {payoffs_list}")
    print(f"Total earnings: ${sum(payoffs_list)}")
