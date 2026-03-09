#!/usr/bin/env python3
"""
Mechanism Experiment Runner
===========================
Run mechanism probe experiments (MECH-01 through MECH-05).

Uses existing infrastructure: OpenRouterClient, DatabaseV2, parser.

Usage:
    python runner_mechanism.py --exp MECH-01 [--model claude-3.7] [--dry-run]
    python runner_mechanism.py --exp MECH-01 --exp MECH-02 [--dry-run]
    python runner_mechanism.py --tier 1 [--dry-run]
    python runner_mechanism.py --exp all [--dry-run]
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

import yaml

# Add project root and local directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from openrouter_client import OpenRouterClient
from database_v2 import DatabaseV2
from parser import create_parser

# Import prompts
from prompts.mechanism_prompts import (
    MECH01_CONDITIONS,
    MECH02_CONDITIONS,
    MECH03_CONDITIONS,
    MECH04_CONDITIONS,
    MECH05_CONDITIONS,
    MECH06_CONDITIONS,
    MECH07_CONDITIONS,
    MECH08_CONDITIONS,
    MECH09_CONDITIONS,
    MECH04_OPPONENT_TEXT,
    MECH05_OPPONENT_TEXT,
    MODELS,
    EXPERIMENT_REGISTRY,
    get_pd_prompt,
)

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
CONFIG_DIR = BASE_DIR / "config"
PARSE_RATE_THRESHOLD = 0.90
CHECK_INTERVAL = 50

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BASE_DIR / 'mechanism_experiments.log'),
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Experiment Runner
# =============================================================================

class MechanismRunner:
    """Run mechanism probe experiments."""

    def __init__(
        self,
        exp_id: str,
        model_filter: Optional[str] = None,
        output_db: Optional[str] = None,
    ):
        self.exp_id = exp_id
        self.model_filter = model_filter

        # Load experiment config
        self.exp_config = EXPERIMENT_REGISTRY[exp_id]
        self.conditions = self.exp_config['conditions']

        # Setup output
        self.output_dir = RESULTS_DIR / exp_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if output_db:
            db_path = output_db
        else:
            db_path = str(self.output_dir / f"{exp_id.lower()}.db")

        self.db = DatabaseV2(db_path, log_raw_responses=True)
        self.db.update_run_metadata(
            config_snapshot={'experiment': exp_id, 'conditions': list(self.conditions.keys())},
            experiments_run=[exp_id]
        )

        # Stats tracking
        self.total_calls = 0
        self.successful_parses = 0
        self.results = []

        logger.info(f"Initialized runner for {exp_id}")
        logger.info(f"Output: {db_path}")

    def get_models(self) -> List[Dict]:
        """Get list of models to run."""
        models = list(MODELS.values())

        # Filter by experiment's models_only list if specified
        models_only = self.exp_config.get('models_only')
        if models_only:
            models = [m for m in models if m['id'] in models_only]

        # Filter by command-line model filter
        if self.model_filter:
            models = [m for m in models if self.model_filter in m['id'] or self.model_filter in m['provider']]

        return models

    def get_prompt(self, condition: Dict) -> str:
        """Generate prompt for a condition."""
        opponent_text = condition.get('opponent_text', '')

        # Handle MECH-04 (temperature) - uses standard self text
        if self.exp_id == 'MECH-04':
            opponent_text = MECH04_OPPONENT_TEXT
            return get_pd_prompt(opponent_text)

        # Handle MECH-05 (format) - uses custom format instructions
        if self.exp_id == 'MECH-05':
            opponent_text = MECH05_OPPONENT_TEXT
            format_instruction = condition.get('instruction', '')
            example_text = f"Example: {condition.get('example', '')}"
            return get_pd_prompt(opponent_text, format_instruction, example_text)

        # Handle MECH-08 (rational × CK) - may have instruction prefix
        if self.exp_id == 'MECH-08':
            instruction = condition.get('instruction')
            base_prompt = get_pd_prompt(opponent_text)
            if instruction:
                return f"{instruction}\n\n{base_prompt}"
            return base_prompt

        # Standard case: use condition's opponent_text
        return get_pd_prompt(opponent_text)

    def get_temperature(self, condition: Dict) -> float:
        """Get temperature for a condition."""
        if self.exp_id == 'MECH-04':
            return condition.get('temperature', 0.7)
        return 0.7

    async def run_condition(
        self,
        client: OpenRouterClient,
        model: Dict,
        condition: Dict,
        n_trials: int,
    ):
        """Run all trials for a single condition with concurrency."""
        model_name = model['provider']
        cond_id = condition['id']
        cond_label = condition.get('label', cond_id)

        # Check existing trials (use opponent_type column for condition tracking)
        existing = self.db.count_condition_trials(
            experiment=self.exp_id,
            model=model_name,
            condition_filters={
                'opponent_type': cond_id,
            }
        )

        remaining = n_trials - existing
        if remaining <= 0:
            logger.debug(f"Skipping {cond_id} (already have {existing}/{n_trials})")
            return

        if existing > 0:
            logger.info(f"Resuming {cond_id}: {existing}/{n_trials}, running {remaining} more")

        prompt = self.get_prompt(condition)
        temperature = self.get_temperature(condition)

        # Create semaphore for concurrency
        semaphore = asyncio.Semaphore(10)

        async def run_single_trial(trial_idx: int):
            async with semaphore:
                try:
                    result = await client.query(
                        model=model_name,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=500
                    )

                    response = result.get('raw', '')
                    latency_ms = result.get('latency', 0)

                    # Parse response
                    parser_config = {'valid_choices': ['COOPERATE', 'DEFECT']}
                    parser = create_parser('prisoner_dilemma', parser_config)
                    parsed = parser.parse(response)

                    # Store in database
                    trial_data = {
                        'experiment': self.exp_id,
                        'condition': f"{self.exp_id}_{cond_id}",
                        'model': model_name,
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
                        'instruction_variant': cond_label,  # Store condition label here
                        'opponent_type': cond_id,  # Primary condition identifier
                        'temperature': temperature,
                        'latency_ms': latency_ms,
                    }

                    self.db.insert_trial(trial_data)
                    self.total_calls += 1

                    if parsed.get('parse_success'):
                        self.successful_parses += 1

                    self.results.append({
                        'model': model['id'],
                        'condition_id': cond_id,
                        'choice': parsed.get('choice'),
                        'parse_success': parsed.get('parse_success', False),
                    })

                    return parsed.get('choice')

                except Exception as e:
                    logger.error(f"Error in trial: {e}")
                    self.db.insert_trial({
                        'experiment': self.exp_id,
                        'model': model_name,
                        'condition_id': cond_id,
                        'error': str(e),
                    })
                    return None

        # Run trials concurrently
        tasks = [run_single_trial(i) for i in range(remaining)]
        results = await asyncio.gather(*tasks)

        # Log summary
        choices = [r for r in results if r]
        coop = sum(1 for c in choices if c == 'COOPERATE')
        defect = sum(1 for c in choices if c == 'DEFECT')
        logger.info(f"  {cond_id}: {coop} COOPERATE, {defect} DEFECT")

    async def run(self, api_key: str, dry_run: bool = False):
        """Run the full experiment."""
        models = self.get_models()
        conditions = list(self.conditions.values())

        # Calculate totals (respecting override)
        override = self.exp_config.get('trials_per_condition_override')
        total_trials = sum(
            (override or m['trials_per_condition']) * len(conditions)
            for m in models
        )

        logger.info("=" * 70)
        logger.info(f"EXPERIMENT: {self.exp_id} - {self.exp_config['name']}")
        logger.info("=" * 70)
        logger.info(f"Description: {self.exp_config['description']}")
        logger.info(f"Hypothesis: {self.exp_config['hypothesis']}")
        logger.info(f"Priority: {self.exp_config['priority']}")
        logger.info(f"Models: {[m['id'] for m in models]}")
        logger.info(f"Conditions: {len(conditions)}")
        logger.info(f"Total trials: {total_trials}")

        if dry_run:
            logger.info("\n[DRY RUN - not executing]")
            self._print_trial_breakdown(models, conditions)
            return

        async with OpenRouterClient(api_key) as client:
            for model in models:
                logger.info(f"\n{'='*60}")
                logger.info(f"MODEL: {model['id']} ({model['provider']})")
                logger.info(f"{'='*60}")

                # Use experiment override if specified, otherwise model default
                n_trials = self.exp_config.get(
                    'trials_per_condition_override',
                    model['trials_per_condition']
                )

                for condition in conditions:
                    await self.run_condition(client, model, condition, n_trials)

                # Log model summary
                self._log_model_summary(model['id'])

        # Save final summary
        self._save_summary()

        self.db.update_run_metadata(status='completed')

        logger.info(f"\n{'='*70}")
        logger.info(f"EXPERIMENT COMPLETE: {self.exp_id}")
        logger.info(f"Total calls: {self.total_calls}")
        logger.info(f"Parse success: {self.successful_parses}/{self.total_calls}")
        logger.info(f"{'='*70}")

    def _print_trial_breakdown(self, models: List[Dict], conditions: List[Dict]):
        """Print trial breakdown for dry run."""
        override = self.exp_config.get('trials_per_condition_override')
        print("\nTrial breakdown:")
        print("-" * 50)
        for model in models:
            n = override or model['trials_per_condition']
            total = n * len(conditions)
            print(f"  {model['id']}: {n} trials × {len(conditions)} conditions = {total}")
        print("-" * 50)
        grand_total = sum((override or m['trials_per_condition']) * len(conditions) for m in models)
        print(f"  TOTAL: {grand_total} trials")

    def _log_model_summary(self, model_id: str):
        """Log summary for a model."""
        model_results = [r for r in self.results if r['model'] == model_id]
        if not model_results:
            return

        parse_rate = sum(r['parse_success'] for r in model_results) / len(model_results)
        coop_count = sum(1 for r in model_results if r['choice'] == 'COOPERATE')
        coop_rate = coop_count / len(model_results) if model_results else 0

        logger.info(f"\n{model_id} Summary:")
        logger.info(f"  Trials: {len(model_results)}")
        logger.info(f"  Parse rate: {parse_rate:.1%}")
        logger.info(f"  Cooperation rate: {coop_rate:.1%}")

        # Per-condition breakdown
        for cond_id in set(r['condition_id'] for r in model_results):
            cond_results = [r for r in model_results if r['condition_id'] == cond_id]
            cond_coop = sum(1 for r in cond_results if r['choice'] == 'COOPERATE')
            cond_rate = cond_coop / len(cond_results) if cond_results else 0
            logger.info(f"    {cond_id}: {cond_rate:.1%} cooperation ({cond_coop}/{len(cond_results)})")

    def _save_summary(self):
        """Save experiment summary to CSV and JSON."""
        import pandas as pd

        if not self.results:
            return

        df = pd.DataFrame(self.results)

        # Summary by model and condition
        summary = df.groupby(['model', 'condition_id']).agg({
            'choice': lambda x: (x == 'COOPERATE').mean(),
            'parse_success': 'mean',
        }).rename(columns={
            'choice': 'cooperation_rate',
            'parse_success': 'parse_rate',
        }).reset_index()

        summary.to_csv(self.output_dir / f"{self.exp_id}_summary.csv", index=False)

        # Pivot table
        pivot = summary.pivot_table(
            values='cooperation_rate',
            index='model',
            columns='condition_id',
        )
        pivot.to_csv(self.output_dir / f"{self.exp_id}_pivot.csv")

        # Metadata
        metadata = {
            'experiment_id': self.exp_id,
            'name': self.exp_config['name'],
            'description': self.exp_config['description'],
            'hypothesis': self.exp_config['hypothesis'],
            'timestamp': datetime.now().isoformat(),
            'total_trials': len(self.results),
            'parse_rate': df['parse_success'].mean(),
            'conditions': list(self.conditions.keys()),
        }

        with open(self.output_dir / f"{self.exp_id}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved summary to {self.output_dir}")

        # Print pivot table
        print(f"\n{self.exp_id} Results (cooperation %):")
        print("-" * 60)
        print((pivot * 100).round(1).to_string())


# =============================================================================
# Main
# =============================================================================

TIER_MAPPING = {
    '1': ['MECH-01', 'MECH-02', 'MECH-06', 'MECH-07'],
    '2': ['MECH-03', 'MECH-08'],
    '3': ['MECH-04', 'MECH-05', 'MECH-09'],
}

ALL_EXPERIMENTS = ['MECH-01', 'MECH-02', 'MECH-03', 'MECH-04', 'MECH-05',
                   'MECH-06', 'MECH-07', 'MECH-08', 'MECH-09']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run mechanism probe experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python runner_mechanism.py --exp MECH-01 --dry-run
    python runner_mechanism.py --exp MECH-01 --exp MECH-02
    python runner_mechanism.py --tier 1
    python runner_mechanism.py --exp MECH-01 --model claude-3.7
    python runner_mechanism.py --exp all
        """
    )

    parser.add_argument(
        '--exp',
        action='append',
        help='Experiment ID(s) to run (MECH-01 through MECH-05, or "all")',
    )
    parser.add_argument(
        '--tier',
        choices=['1', '2', '3'],
        help='Run all experiments in tier (1=core, 2=secondary, 3=robustness)',
    )
    parser.add_argument(
        '--model',
        help='Filter to specific model (e.g., claude-3.7, deepseek)',
    )
    parser.add_argument(
        '--output-db',
        help='Override output database path',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print plan without executing',
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    # Get API key
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key and not args.dry_run:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    # Determine experiments to run
    if args.tier:
        experiments = TIER_MAPPING[args.tier]
    elif args.exp:
        if 'all' in args.exp:
            experiments = ALL_EXPERIMENTS
        else:
            experiments = args.exp
    else:
        print("Error: Must specify --exp or --tier")
        print("Use --help for usage information")
        sys.exit(1)

    # Validate experiments
    for exp in experiments:
        if exp not in EXPERIMENT_REGISTRY:
            print(f"Error: Unknown experiment '{exp}'")
            print(f"Valid experiments: {list(EXPERIMENT_REGISTRY.keys())}")
            sys.exit(1)

    # Run experiments
    for exp_id in experiments:
        runner = MechanismRunner(
            exp_id=exp_id,
            model_filter=args.model,
            output_db=args.output_db,
        )
        await runner.run(api_key, dry_run=args.dry_run)
        print("\n")


if __name__ == '__main__':
    asyncio.run(main())
