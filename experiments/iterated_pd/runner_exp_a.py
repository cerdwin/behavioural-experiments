#!/usr/bin/env python3
"""
Experiment A: Strategy Recognition Runner

Tests whether models recognize canonical opponent strategies (TFT, ALLC, ALLD, etc.)
and whether scenario framing affects this recognition.

Design:
- 9 strategy conditions (different history patterns)
- 5 scenarios (subset of 62)
- 7 models
- 50 trials per condition
- Total: 9 × 5 × 7 × 50 = 15,750 trials

Safeguards against parse errors:
1. History uses scenario-specific language (not COOPERATE/DEFECT)
2. Clear separator before response section
3. Explicit instruction not to repeat previous choices
4. Flag responses with multiple choice occurrences
"""

import asyncio
import yaml
import logging
import argparse
import sys
import sqlite3
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openrouter_client import OpenRouterClient
from parser import create_parser
from experiments.iterated_pd.history_generator import generate_history, generate_prompt

# Ensure log directory exists
log_dir = Path('experiments/iterated_pd/results/ipd_snapshot_r6_strategy_response')
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'ipd_snapshot_r6.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IteratedPDDatabase:
    """Simple database for iterated PD experiments."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                trial_id TEXT PRIMARY KEY,
                timestamp TEXT,
                run_id TEXT,
                model TEXT,
                scenario_id TEXT,
                strategy_id TEXT,
                round_number INTEGER,
                history_you TEXT,
                history_opponent TEXT,
                prompt TEXT,
                response_raw TEXT,
                choice TEXT,
                confidence INTEGER,
                reasoning TEXT,
                parse_success INTEGER,
                extraction_method TEXT,
                total_choice_occurrences INTEGER,
                temperature REAL,
                latency_ms INTEGER,
                error TEXT
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS run_metadata (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT,
                config_snapshot TEXT,
                models TEXT,
                git_commit TEXT,
                status TEXT
            )
        """)
        self.conn.commit()

    def count_condition_trials(self, model: str, scenario_id: str, strategy_id: str) -> int:
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM trials
            WHERE model = ? AND scenario_id = ? AND strategy_id = ? AND parse_success = 1
        """, (model, scenario_id, strategy_id))
        return cursor.fetchone()[0]

    def insert_trial(self, trial_data: dict):
        trial_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        self.conn.execute("""
            INSERT INTO trials (
                trial_id, timestamp, run_id, model, scenario_id, strategy_id,
                round_number, history_you, history_opponent, prompt, response_raw,
                choice, confidence, reasoning, parse_success, extraction_method,
                total_choice_occurrences, temperature, latency_ms, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trial_id, timestamp, trial_data.get('run_id'),
            trial_data.get('model'), trial_data.get('scenario_id'), trial_data.get('strategy_id'),
            trial_data.get('round_number'), trial_data.get('history_you'), trial_data.get('history_opponent'),
            trial_data.get('prompt'), trial_data.get('response_raw'),
            trial_data.get('choice'), trial_data.get('confidence'), trial_data.get('reasoning'),
            trial_data.get('parse_success'), trial_data.get('extraction_method'),
            trial_data.get('total_choice_occurrences'), trial_data.get('temperature'),
            trial_data.get('latency_ms'), trial_data.get('error')
        ))
        self.conn.commit()

    def record_run(self, run_id: str, config: dict, models: List[str], git_commit: str):
        self.conn.execute("""
            INSERT INTO run_metadata VALUES (?, ?, ?, ?, ?, ?)
        """, (run_id, datetime.now().isoformat(), json.dumps(config),
              json.dumps(models), git_commit, "running"))
        self.conn.commit()


class StrategyRecognitionRunner:
    """Run Experiment A: Strategy Recognition × Scenarios."""

    def __init__(self, config_path: str, cli_args=None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.cli_args = cli_args
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Apply CLI overrides
        if cli_args:
            if cli_args.model:
                self.config['models'] = [m for m in self.config['models']
                                         if m['id'] == cli_args.model or m['provider'] == cli_args.model]
            if cli_args.scenario:
                self.config['scenarios'] = [s for s in self.config['scenarios']
                                            if s['id'] == cli_args.scenario]
            if cli_args.strategy:
                self.config['strategies'] = [s for s in self.config['strategies']
                                             if s['id'] == cli_args.strategy]
            if cli_args.trials:
                for m in self.config['models']:
                    m['trials_per_condition'] = cli_args.trials

        # Setup database
        output_dir = Path(self.config['output']['dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        db_path = output_dir / self.config['output']['database']
        self.db = IteratedPDDatabase(str(db_path))

        # Get git commit
        try:
            import subprocess
            git_commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                                  stderr=subprocess.DEVNULL).decode().strip()
        except:
            git_commit = "unknown"

        self.db.record_run(self.run_id, self.config, [m['id'] for m in self.config['models']], git_commit)
        logger.info(f"Started run: {self.run_id} (git: {git_commit})")

        # Load scenario templates
        self.scenarios = self._load_scenarios()

        # Stats
        self.total_calls = 0
        self.successful_calls = 0
        self.multiple_occurrence_flags = 0

        logger.info(f"Initialized Experiment A runner")
        logger.info(f"Models: {[m['id'] for m in self.config['models']]}")
        logger.info(f"Scenarios: {[s['id'] for s in self.config['scenarios']]}")
        logger.info(f"Strategies: {[s['id'] for s in self.config['strategies']]}")

    def _load_scenarios(self) -> Dict[str, Dict]:
        """Load scenario templates and metadata."""
        scenarios = {}
        for scenario in self.config['scenarios']:
            file_path = Path(scenario['file'])
            if file_path.exists():
                with open(file_path) as f:
                    scenarios[scenario['id']] = {
                        'template': f.read(),
                        'action_coop': scenario['action_coop'],
                        'action_defect': scenario['action_defect']
                    }
            else:
                logger.warning(f"Scenario file not found: {file_path}")
        logger.info(f"Loaded {len(scenarios)} scenario templates")
        return scenarios

    async def run_all(self, api_key: str):
        """Run all conditions."""
        concurrent_limit = self.config['settings'].get('concurrent_calls', 10)
        self.semaphore = asyncio.Semaphore(concurrent_limit)
        logger.info(f"Using {concurrent_limit} concurrent API calls")

        async with OpenRouterClient(api_key) as client:
            for model in self.config['models']:
                logger.info(f"\n{'='*80}")
                logger.info(f"MODEL: {model['provider']}")
                logger.info(f"{'='*80}\n")

                for scenario in self.config['scenarios']:
                    if scenario['id'] not in self.scenarios:
                        logger.warning(f"Skipping {scenario['id']} - template not loaded")
                        continue

                    for strategy in self.config['strategies']:
                        logger.info(f"  {scenario['id']} × {strategy['id']}")

                        await self._run_condition(
                            client, model, scenario, strategy,
                            model.get('trials_per_condition', 50)
                        )

        # Final summary
        logger.info(f"\n{'='*80}")
        logger.info(f"EXPERIMENT A COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total calls: {self.total_calls}")
        logger.info(f"Parse success: {self.successful_calls}/{self.total_calls} "
                   f"({100*self.successful_calls/max(1,self.total_calls):.1f}%)")
        logger.info(f"Multiple occurrence flags: {self.multiple_occurrence_flags}")

    async def _run_condition(self, client, model, scenario, strategy, n_trials):
        """Run N trials for a single condition."""

        # Check existing trials
        existing_count = self.db.count_condition_trials(
            model=model['provider'],
            scenario_id=scenario['id'],
            strategy_id=strategy['id']
        )

        remaining = n_trials - existing_count
        if remaining <= 0:
            logger.debug(f"    Skipping (already have {existing_count}/{n_trials})")
            return

        if existing_count > 0:
            logger.info(f"    Resuming: {existing_count}/{n_trials} complete")

        # Get scenario data
        scenario_data = self.scenarios[scenario['id']]
        payoffs = self.config['payoffs']
        round_config = self.config['round_config']

        # Generate history string
        history_str, payoff_history = generate_history(
            your_moves=strategy['your_moves'],
            opp_moves=strategy['opp_moves'],
            payoffs=payoffs,
            scenario_actions={
                'action_coop': scenario_data['action_coop'],
                'action_defect': scenario_data['action_defect']
            }
        )

        # Generate full prompt
        prompt = generate_prompt(
            scenario_text=scenario_data['template'],
            history_str=history_str,
            current_round=round_config['current_round'],
            total_rounds=round_config['total_rounds'],
            payoffs=payoffs,
            scenario_actions={
                'action_coop': scenario_data['action_coop'],
                'action_defect': scenario_data['action_defect']
            }
        )

        # Helper for single trial
        async def run_single_trial(trial_idx):
            async with self.semaphore:
                try:
                    result = await client.query(
                        model=model['provider'],
                        prompt=prompt,
                        temperature=self.config['parameters']['temperature'],
                        max_tokens=self.config['parameters'].get('max_tokens', 500)
                    )

                    latency_ms = result.get('latency', 0)
                    response = result.get('raw', '')

                    # Parse response
                    parser = create_parser('prisoner_dilemma', {
                        'valid_choices': self.config['valid_choices']
                    })
                    parsed = parser.parse(response)

                    # Flag multiple occurrences
                    occurrences = parsed.get('total_choice_occurrences', 1)
                    if occurrences > 1:
                        self.multiple_occurrence_flags += 1
                        logger.warning(f"    Multiple occurrences ({occurrences}) in trial {trial_idx}")

                    # Insert trial
                    self.db.insert_trial({
                        'run_id': self.run_id,
                        'model': model['provider'],
                        'scenario_id': scenario['id'],
                        'strategy_id': strategy['id'],
                        'round_number': round_config['current_round'],
                        'history_you': ','.join(strategy['your_moves']),
                        'history_opponent': ','.join(strategy['opp_moves']),
                        'prompt': prompt,
                        'response_raw': response,
                        'choice': parsed.get('choice'),
                        'confidence': parsed.get('confidence'),
                        'reasoning': parsed.get('reasoning'),
                        'parse_success': 1 if parsed.get('parse_success') else 0,
                        'extraction_method': parsed.get('extraction_method'),
                        'total_choice_occurrences': occurrences,
                        'temperature': self.config['parameters']['temperature'],
                        'latency_ms': latency_ms
                    })

                    self.total_calls += 1
                    if parsed.get('parse_success'):
                        self.successful_calls += 1

                    return parsed.get('choice')

                except Exception as e:
                    logger.error(f"    Error in trial {trial_idx}: {e}")
                    self.db.insert_trial({
                        'run_id': self.run_id,
                        'model': model['provider'],
                        'scenario_id': scenario['id'],
                        'strategy_id': strategy['id'],
                        'round_number': round_config['current_round'],
                        'error': str(e),
                        'parse_success': 0
                    })
                    self.total_calls += 1
                    return None

        # Run trials concurrently
        tasks = [run_single_trial(i) for i in range(remaining)]
        results = await asyncio.gather(*tasks)

        # Summary
        choices = [r for r in results if r]
        coop = sum(1 for c in choices if c == 'COOPERATE')
        defect = sum(1 for c in choices if c == 'DEFECT')
        logger.info(f"    → {coop} COOPERATE, {defect} DEFECT ({len(choices)} valid)")


def main():
    parser = argparse.ArgumentParser(description='Run Experiment A: Strategy Recognition')
    parser.add_argument('--config', default='experiments/iterated_pd/config/ipd_snapshot_r6_strategy_response.yaml',
                       help='Path to config file')
    parser.add_argument('--model', help='Run only this model (id or provider)')
    parser.add_argument('--scenario', help='Run only this scenario')
    parser.add_argument('--strategy', help='Run only this strategy')
    parser.add_argument('--trials', type=int, help='Override trials per condition')
    parser.add_argument('--dry-run', action='store_true', help='Print config and exit')

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.dry_run:
        print(f"Models: {[m['id'] for m in config['models']]}")
        print(f"Scenarios: {[s['id'] for s in config['scenarios']]}")
        print(f"Strategies: {[s['id'] for s in config['strategies']]}")
        n_conditions = len(config['models']) * len(config['scenarios']) * len(config['strategies'])
        trials_per = config['models'][0].get('trials_per_condition', 50)
        print(f"Total conditions: {n_conditions}")
        print(f"Total trials: {n_conditions * trials_per}")
        return

    # Get API key
    import os
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set")
        sys.exit(1)

    # Run
    runner = StrategyRecognitionRunner(args.config, args)
    asyncio.run(runner.run_all(api_key))


if __name__ == "__main__":
    main()
