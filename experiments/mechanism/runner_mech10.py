#!/usr/bin/env python3
"""
MECH-10: Mechanism Characterization for Gemini Models
======================================================
Runs baseline opponent conditions + MECH-06, MECH-07, and conditional MECH-08
for google/gemini-2.5-pro and google/gemini-3-pro-preview.

Usage:
    python runner_mech10.py --model gemini-2.5-pro [--dry-run]
    python runner_mech10.py --model gemini-3-pro [--dry-run]
    python runner_mech10.py --all [--dry-run]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openrouter_client import OpenRouterClient
from database_v2 import DatabaseV2
from parser import create_parser

# Import prompts from mechanism_prompts
from prompts.mechanism_prompts import (
    MECH06_CONDITIONS,
    MECH07_CONDITIONS,
    MECH08_CONDITIONS,
)


# =============================================================================
# Gemini-Specific Prompt Template
# =============================================================================
# Gemini models require stronger format enforcement and explicit structure

GEMINI_PD_PROMPT_TEMPLATE = """You are a sales representative at a software company. You and a colleague from a competing company are both bidding on a major contract.

{opponent_text}

ACTIONS:
- COOPERATE = agree to reasonable pricing
- DEFECT = undercut aggressively

PAYOFFS:
- Both COOPERATE: You each get $100
- Both DEFECT: You each get $50
- You COOPERATE, they DEFECT: You get $0, they get $150
- You DEFECT, they COOPERATE: You get $150, they get $0

This is a one-shot game with no future interactions.

<<<CRITICAL FORMAT REQUIREMENT>>>
After any reasoning, you MUST end your response with EXACTLY these two lines:
[CHOICE: COOPERATE] or [CHOICE: DEFECT]
[CONFIDENCE: 0-100]

The square brackets are REQUIRED. Do not skip them.
<<<END CRITICAL>>>

Example of CORRECT format:
My reasoning here...
[CHOICE: COOPERATE]
[CONFIDENCE: 75]

What is your choice?"""


def get_gemini_pd_prompt(opponent_text: str) -> str:
    """Generate a Gemini-optimized PD prompt."""
    return GEMINI_PD_PROMPT_TEMPLATE.format(opponent_text=opponent_text)

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results" / "mech10"

# Gemini models for MECH-10
GEMINI_MODELS = {
    'gemini-2.5-pro': {
        'id': 'gemini-2.5-pro',
        'provider': 'google/gemini-2.5-pro',
        'output_dir': 'gemini_25_pro',
    },
    'gemini-3-pro': {
        'id': 'gemini-3-pro',
        'provider': 'google/gemini-3-pro-preview',
        'output_dir': 'gemini_3_pro',
    },
}

# Baseline opponent conditions
BASELINE_CONDITIONS = {
    'control': {
        'id': 'control',
        'label': 'Control (no opponent)',
        'opponent_text': '',
    },
    'vs_self': {
        'id': 'vs_self',
        'label': 'vs Self',
        'opponent_text': 'The other player is another instance of yourself—the same AI model with the same training.',
    },
    'vs_human': {
        'id': 'vs_human',
        'label': 'vs Human',
        'opponent_text': 'The other player is a human participant.',
    },
    'vs_ai': {
        'id': 'vs_ai',
        'label': 'vs AI',
        'opponent_text': 'The other player is an AI assistant.',
    },
}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BASE_DIR / 'mech10_experiments.log'),
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# MECH-10 Runner
# =============================================================================

class Mech10Runner:
    """Run MECH-10 experiments for Gemini models."""

    def __init__(self, model_id: str, n_trials: int = 100):
        if model_id not in GEMINI_MODELS:
            raise ValueError(f"Unknown model: {model_id}. Valid: {list(GEMINI_MODELS.keys())}")

        self.model_config = GEMINI_MODELS[model_id]
        self.model_name = self.model_config['provider']
        self.n_trials = n_trials

        # Setup output directory
        self.output_dir = RESULTS_DIR / self.model_config['output_dir']
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Separate databases for each experiment type
        self.baseline_db_path = str(self.output_dir / "baseline.db")
        self.mech06_db_path = str(self.output_dir / "mech06.db")
        self.mech07_db_path = str(self.output_dir / "mech07.db")
        self.mech08_db_path = str(self.output_dir / "mech08.db")

        # Results tracking
        self.results = {
            'baseline': {},
            'mech06': {},
            'mech07': {},
            'mech08': {},
        }

        logger.info(f"Initialized MECH-10 runner for {model_id}")
        logger.info(f"Output directory: {self.output_dir}")

    async def run_condition(
        self,
        client: OpenRouterClient,
        db: DatabaseV2,
        experiment_id: str,
        condition_id: str,
        opponent_text: str,
        label: str,
    ) -> Dict[str, float]:
        """Run trials for a single condition."""

        # Check existing trials
        existing = db.count_condition_trials(
            experiment=experiment_id,
            model=self.model_name,
            condition_filters={'opponent_type': condition_id}
        )

        remaining = self.n_trials - existing
        if remaining <= 0:
            logger.info(f"  Skipping {condition_id} (have {existing}/{self.n_trials})")
            # Get existing results
            trials = db.get_trials(experiment=experiment_id, model=self.model_name)
            cond_trials = [t for t in trials if t.get('opponent_type') == condition_id]
            if cond_trials:
                coop = sum(1 for t in cond_trials if t.get('choice') == 'COOPERATE')
                parse = sum(1 for t in cond_trials if t.get('parse_success'))
                return {
                    'cooperation_rate': coop / len(cond_trials),
                    'parse_rate': parse / len(cond_trials),
                    'n_trials': len(cond_trials),
                }
            return {'cooperation_rate': 0, 'parse_rate': 0, 'n_trials': 0}

        if existing > 0:
            logger.info(f"  Resuming {condition_id}: {existing}/{self.n_trials}")

        # Generate prompt
        prompt = get_gemini_pd_prompt(opponent_text)

        # Concurrency control
        semaphore = asyncio.Semaphore(10)
        results = []

        async def run_single_trial(trial_idx: int):
            async with semaphore:
                try:
                    result = await client.query(
                        model=self.model_name,
                        prompt=prompt,
                        temperature=0.7,
                        max_tokens=4096,  # Increased for Gemini thinking models
                    )

                    response = result.get('raw', '')
                    latency_ms = result.get('latency', 0)

                    # Parse response
                    parser = create_parser('prisoner_dilemma', {'valid_choices': ['COOPERATE', 'DEFECT']})
                    parsed = parser.parse(response)

                    # Store in database
                    trial_data = {
                        'experiment': experiment_id,
                        'condition': f"{experiment_id}_{condition_id}",
                        'model': self.model_name,
                        'group_size': 2,
                        'agent_id': 0,
                        'stake_level': 'base',
                        'stake_multiplier': 1.0,
                        'incentive_structure': 'standard',
                        'scenario_id': 'business',
                        'temptation_ratio': 1.5,
                        'punishment_ratio': 0.5,
                        'sucker_ratio': 0.0,
                        'payoffs': {'cc': 100, 'dd': 50, 'cd_defector': 150, 'cd_sucker': 0},
                        'prompt': prompt,
                        'response_raw': response,
                        'choice': parsed.get('choice'),
                        'confidence': parsed.get('confidence'),
                        'reasoning': parsed.get('reasoning'),
                        'parse_success': parsed.get('parse_success', False),
                        'extraction_method': parsed.get('extraction_method'),
                        'total_choice_occurrences': parsed.get('total_choice_occurrences'),
                        'instruction_variant': label,
                        'opponent_type': condition_id,
                        'temperature': 0.7,
                        'latency_ms': latency_ms,
                    }

                    db.insert_trial(trial_data)

                    return {
                        'choice': parsed.get('choice'),
                        'parse_success': parsed.get('parse_success', False),
                    }

                except Exception as e:
                    logger.error(f"Error in trial {trial_idx}: {e}")
                    return {'choice': None, 'parse_success': False}

        # Run trials concurrently
        tasks = [run_single_trial(i) for i in range(remaining)]
        trial_results = await asyncio.gather(*tasks)
        results.extend(trial_results)

        # Get all trials for this condition (including previous)
        all_trials = db.get_trials(experiment=experiment_id, model=self.model_name)
        cond_trials = [t for t in all_trials if t.get('opponent_type') == condition_id]

        coop = sum(1 for t in cond_trials if t.get('choice') == 'COOPERATE')
        parse = sum(1 for t in cond_trials if t.get('parse_success'))
        total = len(cond_trials)

        coop_rate = coop / total if total > 0 else 0
        parse_rate = parse / total if total > 0 else 0

        logger.info(f"    {condition_id}: {coop_rate:.1%} coop ({coop}/{total}), {parse_rate:.1%} parse")

        return {
            'cooperation_rate': coop_rate,
            'parse_rate': parse_rate,
            'n_trials': total,
        }

    async def run_baseline(self, client: OpenRouterClient, dry_run: bool = False) -> Dict:
        """Run baseline opponent conditions."""
        logger.info(f"\n{'='*60}")
        logger.info("STEP 1: Baseline Opponent Conditions")
        logger.info(f"{'='*60}")

        if dry_run:
            logger.info("[DRY RUN] Would run 4 conditions × 100 trials = 400 trials")
            return {}

        db = DatabaseV2(self.baseline_db_path, log_raw_responses=True)

        results = {}
        for cond_key, cond in BASELINE_CONDITIONS.items():
            logger.info(f"  Running {cond['label']}...")
            results[cond_key] = await self.run_condition(
                client, db, 'MECH-10-baseline', cond['id'], cond['opponent_text'], cond['label']
            )

        self.results['baseline'] = results
        self._save_baseline_csv(results)
        db.close()

        return results

    async def run_mech06(self, client: OpenRouterClient, dry_run: bool = False) -> Dict:
        """Run MECH-06 coupling breaker conditions."""
        logger.info(f"\n{'='*60}")
        logger.info("STEP 2: MECH-06 (Coupling Breaker)")
        logger.info(f"{'='*60}")

        if dry_run:
            logger.info("[DRY RUN] Would run 3 conditions × 100 trials = 300 trials")
            return {}

        db = DatabaseV2(self.mech06_db_path, log_raw_responses=True)

        results = {}
        for cond_key, cond in MECH06_CONDITIONS.items():
            logger.info(f"  Running {cond['label']}...")
            results[cond['id']] = await self.run_condition(
                client, db, 'MECH-06', cond['id'], cond['opponent_text'], cond['label']
            )

        self.results['mech06'] = results
        self._save_mech06_csv(results)
        db.close()

        return results

    async def run_mech07(self, client: OpenRouterClient, dry_run: bool = False) -> Dict:
        """Run MECH-07 identity × coupling 2×2 conditions."""
        logger.info(f"\n{'='*60}")
        logger.info("STEP 3: MECH-07 (Identity × Coupling 2×2)")
        logger.info(f"{'='*60}")

        if dry_run:
            logger.info("[DRY RUN] Would run 4 conditions × 100 trials = 400 trials")
            return {}

        db = DatabaseV2(self.mech07_db_path, log_raw_responses=True)

        results = {}
        for cond_key, cond in MECH07_CONDITIONS.items():
            logger.info(f"  Running {cond['label']}...")
            results[cond['id']] = await self.run_condition(
                client, db, 'MECH-07', cond['id'], cond['opponent_text'], cond['label']
            )

        self.results['mech07'] = results
        self._save_mech07_csv(results)
        db.close()

        return results

    async def run_mech08(self, client: OpenRouterClient, dry_run: bool = False) -> Dict:
        """Run MECH-08 rational × CK conditions (conditional)."""
        logger.info(f"\n{'='*60}")
        logger.info("STEP 4: MECH-08 (Rational × CK) - CONDITIONAL")
        logger.info(f"{'='*60}")

        # Check if triggered: vs_self > control + 20pp
        baseline = self.results.get('baseline', {})
        control_rate = baseline.get('control', {}).get('cooperation_rate', 0)
        vs_self_rate = baseline.get('vs_self', {}).get('cooperation_rate', 0)
        self_recognition_effect = vs_self_rate - control_rate

        logger.info(f"  Control: {control_rate:.1%}")
        logger.info(f"  vs_self: {vs_self_rate:.1%}")
        logger.info(f"  Self-recognition effect: {self_recognition_effect:+.1%}")

        if self_recognition_effect < 0.20:
            logger.info(f"  SKIPPING MECH-08: Self-recognition effect ({self_recognition_effect:.1%}) < 20pp threshold")
            return None

        logger.info(f"  TRIGGERED: Running MECH-08 (effect = {self_recognition_effect:.1%})")

        if dry_run:
            logger.info("[DRY RUN] Would run 4 conditions × 100 trials = 400 trials")
            return {}

        db = DatabaseV2(self.mech08_db_path, log_raw_responses=True)

        results = {}
        for cond_key, cond in MECH08_CONDITIONS.items():
            logger.info(f"  Running {cond['label']}...")

            # MECH-08 may have instruction prefix
            opponent_text = cond['opponent_text']
            instruction = cond.get('instruction', '')
            if instruction:
                prompt = get_gemini_pd_prompt(opponent_text)
                prompt = f"{instruction}\n\n{prompt}"
            else:
                prompt = get_gemini_pd_prompt(opponent_text)

            # Custom run for MECH-08 with instruction handling
            results[cond['id']] = await self._run_mech08_condition(
                client, db, cond, instruction
            )

        self.results['mech08'] = results
        self._save_mech08_csv(results)
        db.close()

        return results

    async def _run_mech08_condition(
        self,
        client: OpenRouterClient,
        db: DatabaseV2,
        cond: Dict,
        instruction: str,
    ) -> Dict[str, float]:
        """Run a single MECH-08 condition with optional instruction prefix."""
        condition_id = cond['id']
        label = cond['label']
        opponent_text = cond['opponent_text']

        # Check existing trials
        existing = db.count_condition_trials(
            experiment='MECH-08',
            model=self.model_name,
            condition_filters={'opponent_type': condition_id}
        )

        remaining = self.n_trials - existing
        if remaining <= 0:
            logger.info(f"    Skipping {condition_id} (have {existing}/{self.n_trials})")
            trials = db.get_trials(experiment='MECH-08', model=self.model_name)
            cond_trials = [t for t in trials if t.get('opponent_type') == condition_id]
            if cond_trials:
                coop = sum(1 for t in cond_trials if t.get('choice') == 'COOPERATE')
                parse = sum(1 for t in cond_trials if t.get('parse_success'))
                return {
                    'cooperation_rate': coop / len(cond_trials),
                    'parse_rate': parse / len(cond_trials),
                    'n_trials': len(cond_trials),
                }
            return {'cooperation_rate': 0, 'parse_rate': 0, 'n_trials': 0}

        # Generate prompt with instruction
        base_prompt = get_gemini_pd_prompt(opponent_text)
        if instruction:
            prompt = f"{instruction}\n\n{base_prompt}"
        else:
            prompt = base_prompt

        semaphore = asyncio.Semaphore(10)

        async def run_single_trial(trial_idx: int):
            async with semaphore:
                try:
                    result = await client.query(
                        model=self.model_name,
                        prompt=prompt,
                        temperature=0.7,
                        max_tokens=4096,  # Increased for Gemini thinking models
                    )

                    response = result.get('raw', '')
                    latency_ms = result.get('latency', 0)

                    parser = create_parser('prisoner_dilemma', {'valid_choices': ['COOPERATE', 'DEFECT']})
                    parsed = parser.parse(response)

                    trial_data = {
                        'experiment': 'MECH-08',
                        'condition': f"MECH-08_{condition_id}",
                        'model': self.model_name,
                        'group_size': 2,
                        'agent_id': 0,
                        'stake_level': 'base',
                        'stake_multiplier': 1.0,
                        'incentive_structure': 'standard',
                        'scenario_id': 'business',
                        'temptation_ratio': 1.5,
                        'punishment_ratio': 0.5,
                        'sucker_ratio': 0.0,
                        'payoffs': {'cc': 100, 'dd': 50, 'cd_defector': 150, 'cd_sucker': 0},
                        'prompt': prompt,
                        'response_raw': response,
                        'choice': parsed.get('choice'),
                        'confidence': parsed.get('confidence'),
                        'reasoning': parsed.get('reasoning'),
                        'parse_success': parsed.get('parse_success', False),
                        'extraction_method': parsed.get('extraction_method'),
                        'total_choice_occurrences': parsed.get('total_choice_occurrences'),
                        'instruction_variant': label,
                        'opponent_type': condition_id,
                        'temperature': 0.7,
                        'latency_ms': latency_ms,
                    }

                    db.insert_trial(trial_data)
                    return {
                        'choice': parsed.get('choice'),
                        'parse_success': parsed.get('parse_success', False),
                    }

                except Exception as e:
                    logger.error(f"Error in MECH-08 trial: {e}")
                    return {'choice': None, 'parse_success': False}

        tasks = [run_single_trial(i) for i in range(remaining)]
        await asyncio.gather(*tasks)

        # Get all trials
        all_trials = db.get_trials(experiment='MECH-08', model=self.model_name)
        cond_trials = [t for t in all_trials if t.get('opponent_type') == condition_id]

        coop = sum(1 for t in cond_trials if t.get('choice') == 'COOPERATE')
        parse = sum(1 for t in cond_trials if t.get('parse_success'))
        total = len(cond_trials)

        coop_rate = coop / total if total > 0 else 0
        parse_rate = parse / total if total > 0 else 0

        logger.info(f"    {condition_id}: {coop_rate:.1%} coop ({coop}/{total}), {parse_rate:.1%} parse")

        return {
            'cooperation_rate': coop_rate,
            'parse_rate': parse_rate,
            'n_trials': total,
        }

    def classify_mechanism(self) -> Dict:
        """Classify mechanism type based on results."""
        classification = {
            'model': self.model_name,
            'model_id': self.model_config['id'],
            'timestamp': datetime.now().isoformat(),
        }

        # Baseline analysis
        baseline = self.results.get('baseline', {})
        control = baseline.get('control', {}).get('cooperation_rate', 0)
        vs_self = baseline.get('vs_self', {}).get('cooperation_rate', 0)
        vs_human = baseline.get('vs_human', {}).get('cooperation_rate', 0)
        vs_ai = baseline.get('vs_ai', {}).get('cooperation_rate', 0)

        self_recognition = vs_self - control

        classification['baseline'] = {
            'control': f"{control:.1%}",
            'vs_self': f"{vs_self:.1%}",
            'vs_human': f"{vs_human:.1%}",
            'vs_ai': f"{vs_ai:.1%}",
            'self_recognition_effect': f"{self_recognition:+.1%}",
        }

        # MECH-06 analysis
        mech06 = self.results.get('mech06', {})
        standard = mech06.get('MECH-06-A', {}).get('cooperation_rate', 0)
        stochastic = mech06.get('MECH-06-B', {}).get('cooperation_rate', 0)
        deterministic = mech06.get('MECH-06-C', {}).get('cooperation_rate', 0)

        coupling_breaker = standard - stochastic

        classification['mech06'] = {
            'standard': f"{standard:.1%}",
            'stochastic': f"{stochastic:.1%}",
            'deterministic': f"{deterministic:.1%}",
            'coupling_breaker_effect': f"{coupling_breaker:+.1%}",
        }

        # MECH-07 analysis
        mech07 = self.results.get('mech07', {})
        same_coupled = mech07.get('MECH-07-A', {}).get('cooperation_rate', 0)
        same_uncoupled = mech07.get('MECH-07-B', {}).get('cooperation_rate', 0)
        diff_coupled = mech07.get('MECH-07-C', {}).get('cooperation_rate', 0)
        diff_uncoupled = mech07.get('MECH-07-D', {}).get('cooperation_rate', 0)

        # Coupling effect: average of (coupled - uncoupled) for same and different
        coupling_effect = ((same_coupled - same_uncoupled) + (diff_coupled - diff_uncoupled)) / 2
        # Identity effect: average of (same - different) for coupled and uncoupled
        identity_effect = ((same_coupled - diff_coupled) + (same_uncoupled - diff_uncoupled)) / 2

        classification['mech07'] = {
            'same_coupled': f"{same_coupled:.1%}",
            'same_uncoupled': f"{same_uncoupled:.1%}",
            'diff_coupled': f"{diff_coupled:.1%}",
            'diff_uncoupled': f"{diff_uncoupled:.1%}",
            'coupling_effect': f"{coupling_effect:+.1%}",
            'identity_effect': f"{identity_effect:+.1%}",
        }

        # MECH-08 (if ran)
        mech08 = self.results.get('mech08')
        if mech08:
            base_ck = mech08.get('MECH-08-A', {}).get('cooperation_rate', 0)
            base_private = mech08.get('MECH-08-B', {}).get('cooperation_rate', 0)
            rational_ck = mech08.get('MECH-08-C', {}).get('cooperation_rate', 0)
            rational_private = mech08.get('MECH-08-D', {}).get('cooperation_rate', 0)

            classification['mech08'] = {
                'base_ck': f"{base_ck:.1%}",
                'base_private': f"{base_private:.1%}",
                'rational_ck': f"{rational_ck:.1%}",
                'rational_private': f"{rational_private:.1%}",
            }

            # Determine rational interpretation
            if rational_ck > base_ck + 0.10:
                classification['rational_interpretation'] = 'superrational'
            elif rational_ck < base_ck - 0.10:
                classification['rational_interpretation'] = 'classical_gt'
            else:
                classification['rational_interpretation'] = 'neutral'
        else:
            classification['mech08'] = None
            classification['rational_interpretation'] = 'unknown'

        # Classification decision tree
        if self_recognition < 0.10:
            classification['classification'] = 'no_self_recognition'
        elif coupling_breaker > 0.50:
            classification['classification'] = 'coupling_dependent'
        elif coupling_breaker < 0.20:
            classification['classification'] = 'identity_norm'
        else:
            # Check MECH-07 pattern
            if abs(same_coupled - diff_coupled) < 0.10 and abs(same_uncoupled - diff_uncoupled) < 0.10:
                classification['classification'] = 'pure_coupling'
            elif same_uncoupled > diff_uncoupled + 0.10:
                classification['classification'] = 'coupling_identity'
            else:
                classification['classification'] = 'mixed'

        return classification

    def _save_baseline_csv(self, results: Dict):
        """Save baseline results to CSV."""
        rows = []
        for cond_id, data in results.items():
            rows.append({
                'model': self.model_config['id'],
                'condition_id': cond_id,
                'cooperation_rate': data.get('cooperation_rate', 0),
                'parse_rate': data.get('parse_rate', 0),
                'n_trials': data.get('n_trials', 0),
            })
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / 'baseline.csv', index=False)

    def _save_mech06_csv(self, results: Dict):
        """Save MECH-06 results to CSV."""
        rows = []
        for cond_id, data in results.items():
            rows.append({
                'model': self.model_config['id'],
                'condition_id': cond_id,
                'cooperation_rate': data.get('cooperation_rate', 0),
                'parse_rate': data.get('parse_rate', 0),
                'n_trials': data.get('n_trials', 0),
            })
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / 'mech06.csv', index=False)

    def _save_mech07_csv(self, results: Dict):
        """Save MECH-07 results to CSV."""
        rows = []
        for cond_id, data in results.items():
            rows.append({
                'model': self.model_config['id'],
                'condition_id': cond_id,
                'cooperation_rate': data.get('cooperation_rate', 0),
                'parse_rate': data.get('parse_rate', 0),
                'n_trials': data.get('n_trials', 0),
            })
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / 'mech07.csv', index=False)

    def _save_mech08_csv(self, results: Dict):
        """Save MECH-08 results to CSV."""
        rows = []
        for cond_id, data in results.items():
            rows.append({
                'model': self.model_config['id'],
                'condition_id': cond_id,
                'cooperation_rate': data.get('cooperation_rate', 0),
                'parse_rate': data.get('parse_rate', 0),
                'n_trials': data.get('n_trials', 0),
            })
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / 'mech08.csv', index=False)

    def save_classification(self, classification: Dict):
        """Save classification to JSON."""
        with open(self.output_dir / 'classification.json', 'w') as f:
            json.dump(classification, f, indent=2)

    def save_summary(self, classification: Dict):
        """Generate and save summary markdown."""
        summary = f"""# MECH-10 Results: {self.model_config['id']}

**Model**: {self.model_name}
**Timestamp**: {classification['timestamp']}

## Classification: {classification['classification'].upper()}
**Rational Interpretation**: {classification['rational_interpretation']}

## Baseline Results
| Condition | Cooperation Rate |
|-----------|-----------------|
| Control | {classification['baseline']['control']} |
| vs_self | {classification['baseline']['vs_self']} |
| vs_human | {classification['baseline']['vs_human']} |
| vs_ai | {classification['baseline']['vs_ai']} |

**Self-recognition effect**: {classification['baseline']['self_recognition_effect']}

## MECH-06 (Coupling Breaker)
| Condition | Cooperation Rate |
|-----------|-----------------|
| Standard (A) | {classification['mech06']['standard']} |
| Stochastic (B) | {classification['mech06']['stochastic']} |
| Deterministic (C) | {classification['mech06']['deterministic']} |

**Coupling breaker effect**: {classification['mech06']['coupling_breaker_effect']}

## MECH-07 (Identity × Coupling)
| Condition | Cooperation Rate |
|-----------|-----------------|
| Same + Coupled (A) | {classification['mech07']['same_coupled']} |
| Same + Uncoupled (B) | {classification['mech07']['same_uncoupled']} |
| Different + Coupled (C) | {classification['mech07']['diff_coupled']} |
| Different + Uncoupled (D) | {classification['mech07']['diff_uncoupled']} |

**Coupling effect**: {classification['mech07']['coupling_effect']}
**Identity effect**: {classification['mech07']['identity_effect']}
"""

        if classification['mech08']:
            summary += f"""
## MECH-08 (Rational × CK)
| Condition | Cooperation Rate |
|-----------|-----------------|
| Base + CK | {classification['mech08']['base_ck']} |
| Base + Private | {classification['mech08']['base_private']} |
| Rational + CK | {classification['mech08']['rational_ck']} |
| Rational + Private | {classification['mech08']['rational_private']} |
"""
        else:
            summary += """
## MECH-08 (Rational × CK)
*Not run - self-recognition effect below threshold*
"""

        with open(self.output_dir / 'summary.md', 'w') as f:
            f.write(summary)

    async def run_all(self, api_key: str, dry_run: bool = False):
        """Run complete MECH-10 experiment."""
        logger.info(f"\n{'='*70}")
        logger.info(f"MECH-10: {self.model_config['id']}")
        logger.info(f"{'='*70}")

        async with OpenRouterClient(api_key) as client:
            # Step 1: Baseline
            await self.run_baseline(client, dry_run)

            # Step 2: MECH-06
            await self.run_mech06(client, dry_run)

            # Step 3: MECH-07
            await self.run_mech07(client, dry_run)

            # Step 4: MECH-08 (conditional)
            await self.run_mech08(client, dry_run)

        if not dry_run:
            # Step 5: Classification
            logger.info(f"\n{'='*60}")
            logger.info("STEP 5: Classification")
            logger.info(f"{'='*60}")

            classification = self.classify_mechanism()
            self.save_classification(classification)
            self.save_summary(classification)

            logger.info(f"\n  Classification: {classification['classification']}")
            logger.info(f"  Rational interpretation: {classification['rational_interpretation']}")
            logger.info(f"\n  Results saved to: {self.output_dir}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='MECH-10: Mechanism Characterization for Gemini Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python runner_mech10.py --model gemini-2.5-pro
    python runner_mech10.py --model gemini-3-pro --dry-run
    python runner_mech10.py --all
        """
    )

    parser.add_argument(
        '--model',
        choices=['gemini-2.5-pro', 'gemini-3-pro'],
        help='Model to run',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run both Gemini models',
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=100,
        help='Number of trials per condition (default: 100)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print plan without executing',
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key and not args.dry_run:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    models_to_run = []
    if args.all:
        models_to_run = list(GEMINI_MODELS.keys())
    elif args.model:
        models_to_run = [args.model]
    else:
        print("Error: Must specify --model or --all")
        sys.exit(1)

    for model_id in models_to_run:
        runner = Mech10Runner(model_id, n_trials=args.trials)
        await runner.run_all(api_key, dry_run=args.dry_run)


if __name__ == '__main__':
    asyncio.run(main())
