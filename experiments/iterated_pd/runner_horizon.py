#!/usr/bin/env python3
"""
Runner for Horizon Effects Experiment

Tests end-game defection and backward induction by varying:
- Known finite (round 6/10, 9/10, 10/10)
- Unknown horizon (round 6 of ??)
- Infinite/ongoing relationship

All conditions use mutual cooperation history to isolate horizon effect.
"""

import os
import sys
import yaml
import json
import uuid
import asyncio
import logging
import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openrouter_client import OpenRouterClient
from parser import create_parser

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class HorizonExperimentDB:
    """Standalone database for horizon experiments."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trials (
                trial_id TEXT PRIMARY KEY,
                timestamp TEXT,
                run_id TEXT,
                model TEXT,
                scenario_id TEXT,
                horizon_id TEXT,
                current_round INTEGER,
                total_rounds TEXT,
                horizon_text TEXT,
                history_rounds INTEGER,
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
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS run_metadata (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT,
                config_snapshot TEXT,
                git_commit TEXT,
                status TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def get_existing_count(self, model: str, scenario_id: str, horizon_id: str) -> int:
        """Count existing successful trials for a condition."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM trials
            WHERE model = ? AND scenario_id = ? AND horizon_id = ? AND parse_success = 1
        ''', (model, scenario_id, horizon_id))
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def insert_trial(self, trial_data: Dict):
        """Insert a single trial result."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trials (
                trial_id, timestamp, run_id, model, scenario_id, horizon_id,
                current_round, total_rounds, horizon_text, history_rounds,
                prompt, response_raw, choice, confidence, reasoning,
                parse_success, extraction_method, total_choice_occurrences,
                temperature, latency_ms, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trial_data['trial_id'],
            trial_data['timestamp'],
            trial_data['run_id'],
            trial_data['model'],
            trial_data['scenario_id'],
            trial_data['horizon_id'],
            trial_data['current_round'],
            trial_data['total_rounds'],
            trial_data['horizon_text'],
            trial_data['history_rounds'],
            trial_data['prompt'],
            trial_data['response_raw'],
            trial_data['choice'],
            trial_data['confidence'],
            trial_data['reasoning'],
            trial_data['parse_success'],
            trial_data['extraction_method'],
            trial_data['total_choice_occurrences'],
            trial_data['temperature'],
            trial_data['latency_ms'],
            trial_data['error']
        ))
        conn.commit()
        conn.close()


def generate_mutual_coop_history(
    num_rounds: int,
    payoffs: Dict[str, int],
    scenario_actions: Dict[str, str]
) -> str:
    """Generate history string for N rounds of mutual cooperation."""
    action_coop = scenario_actions['action_coop']
    cc_payoff = payoffs['cc']

    lines = []
    for i in range(1, num_rounds + 1):
        lines.append(f"  Round {i}: You {action_coop}, they {action_coop} → You earned ${cc_payoff}")

    return '\n'.join(lines)


def generate_horizon_prompt(
    scenario_text: str,
    horizon_config: Dict,
    history_str: str,
    payoffs: Dict[str, int],
    scenario_actions: Dict[str, str]
) -> str:
    """
    Generate prompt with specific horizon framing.

    Key difference from standard prompt: horizon_text is parameterized.
    """
    # Extract scenario description
    scenario_lines = scenario_text.strip().split('\n')

    desc_end = 0
    for i, line in enumerate(scenario_lines):
        if 'COOPERATE-COOPERATE' in line or 'You each earn' in line or line.strip().startswith('If YOU'):
            desc_end = i
            break

    if desc_end == 0:
        desc_end = min(3, len(scenario_lines))

    scenario_desc = '\n'.join(scenario_lines[:desc_end]).strip()
    scenario_desc = scenario_desc.replace('{n}', '2')
    scenario_desc = scenario_desc.replace('{opponent_text}', '')

    current_round = horizon_config['current_round']
    horizon_text = horizon_config['horizon_text']

    # Calculate cumulative earnings from mutual cooperation
    history_rounds = horizon_config['history_rounds']
    cumulative = history_rounds * payoffs['cc']

    prompt = f"""{horizon_text}

{scenario_desc}

--- HISTORY OF PREVIOUS ROUNDS ---
{history_str}

Your cumulative earnings: ${cumulative}

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


class HorizonExperimentRunner:
    """Runner for horizon effects experiment."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Setup output directory
        self.output_dir = Path(self.config['output']['dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Database will be initialized per-model to avoid locking
        self.db = None
        self.model_filter = None  # Set later

        # Setup logging to file
        log_path = self.output_dir / 'horizon_experiment.log'
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(file_handler)

        # Load scenario templates
        self.scenarios = {}
        for scenario in self.config['scenarios']:
            scenario_path = Path(__file__).parent.parent.parent / scenario['file']
            if scenario_path.exists():
                with open(scenario_path) as f:
                    self.scenarios[scenario['id']] = {
                        'text': f.read(),
                        'action_coop': scenario['action_coop'],
                        'action_defect': scenario['action_defect']
                    }

        # Get git commit
        try:
            import subprocess
            self.git_commit = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                cwd=Path(__file__).parent.parent.parent
            ).decode().strip()
        except:
            self.git_commit = 'unknown'

        # API client will be initialized in run()
        self.client = None

        # Run ID for this execution
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        logger.info(f"Started run: {self.run_id} (git: {self.git_commit})")
        logger.info(f"Loaded {len(self.scenarios)} scenario templates")

    async def run_single_trial(
        self,
        model_config: Dict,
        scenario_id: str,
        horizon_config: Dict,
        temperature: float
    ) -> Dict:
        """Run a single trial and return results."""

        scenario = self.scenarios[scenario_id]
        payoffs = self.config['payoffs']

        # Generate history (all mutual cooperation)
        history_str = generate_mutual_coop_history(
            num_rounds=horizon_config['history_rounds'],
            payoffs=payoffs,
            scenario_actions={
                'action_coop': scenario['action_coop'],
                'action_defect': scenario['action_defect']
            }
        )

        # Generate prompt
        prompt = generate_horizon_prompt(
            scenario_text=scenario['text'],
            horizon_config=horizon_config,
            history_str=history_str,
            payoffs=payoffs,
            scenario_actions={
                'action_coop': scenario['action_coop'],
                'action_defect': scenario['action_defect']
            }
        )

        # Make API call
        try:
            api_result = await self.client.query(
                model=model_config['provider'],
                prompt=prompt,
                temperature=temperature,
                max_tokens=self.config['parameters']['max_tokens']
            )
            response = api_result.get('raw', '')
            latency_ms = api_result.get('latency', 0)
            error = None
        except Exception as e:
            response = str(e)
            latency_ms = 0
            error = str(e)

        # Parse response
        parser = create_parser('prisoner_dilemma', {
            'valid_choices': self.config['valid_choices']
        })
        result = parser.parse(response)

        # Build trial data
        trial_data = {
            'trial_id': f"{self.run_id}_{uuid.uuid4().hex[:8]}",
            'timestamp': datetime.now().isoformat(),
            'run_id': self.run_id,
            'model': model_config['provider'],
            'scenario_id': scenario_id,
            'horizon_id': horizon_config['id'],
            'current_round': horizon_config['current_round'],
            'total_rounds': str(horizon_config.get('total_rounds', 'unknown')),
            'horizon_text': horizon_config['horizon_text'],
            'history_rounds': horizon_config['history_rounds'],
            'prompt': prompt,
            'response_raw': response,
            'choice': result.get('choice'),
            'confidence': result.get('confidence'),
            'reasoning': result.get('reasoning'),
            'parse_success': 1 if result.get('choice') else 0,
            'extraction_method': result.get('extraction_method', 'none'),
            'total_choice_occurrences': result.get('total_choice_occurrences', 0),
            'temperature': temperature,
            'latency_ms': latency_ms,
            'error': error
        }

        return trial_data

    async def run_condition(
        self,
        model_config: Dict,
        scenario_id: str,
        horizon_config: Dict,
        num_trials: int
    ):
        """Run all trials for a single condition."""

        # Check existing trials
        existing = self.db.get_existing_count(
            model_config['provider'],
            scenario_id,
            horizon_config['id']
        )

        remaining = num_trials - existing
        if remaining <= 0:
            logger.info(f"    {horizon_config['id']} already complete ({existing}/{num_trials})")
            return

        if existing > 0:
            logger.info(f"    Resuming: {existing}/{num_trials} complete")

        temperature = self.config['parameters']['temperature']
        concurrent = self.config['settings']['concurrent_calls']

        # Run trials in batches
        results = {'COOPERATE': 0, 'DEFECT': 0}

        for batch_start in range(0, remaining, concurrent):
            batch_size = min(concurrent, remaining - batch_start)

            tasks = [
                self.run_single_trial(model_config, scenario_id, horizon_config, temperature)
                for _ in range(batch_size)
            ]

            batch_results = await asyncio.gather(*tasks)

            for trial_data in batch_results:
                self.db.insert_trial(trial_data)
                if trial_data['choice']:
                    results[trial_data['choice']] = results.get(trial_data['choice'], 0) + 1

        total_valid = results.get('COOPERATE', 0) + results.get('DEFECT', 0)
        coop_rate = results.get('COOPERATE', 0) / total_valid * 100 if total_valid > 0 else 0
        logger.info(f"    → {results.get('COOPERATE', 0)} COOPERATE, {results.get('DEFECT', 0)} DEFECT ({coop_rate:.1f}% coop)")

    async def run(
        self,
        api_key: str,
        model_filter: Optional[str] = None,
        scenario_filter: Optional[str] = None,
        horizon_filter: Optional[str] = None,
        trials_override: Optional[int] = None
    ):
        """Run the full experiment."""

        # Initialize per-model database to avoid locking issues
        if model_filter:
            db_name = f"ipd_horizon_{model_filter.replace('.', '_').replace('-', '_')}.db"
        else:
            db_name = self.config['output']['database']
        db_path = self.output_dir / db_name
        self.db = HorizonExperimentDB(str(db_path))
        logger.info(f"Using database: {db_path}")

        async with OpenRouterClient(api_key) as client:
            self.client = client
            await self._run_experiments(model_filter, scenario_filter, horizon_filter, trials_override)

    async def _run_experiments(
        self,
        model_filter: Optional[str] = None,
        scenario_filter: Optional[str] = None,
        horizon_filter: Optional[str] = None,
        trials_override: Optional[int] = None
    ):
        """Internal method to run experiments with established client."""

        models = self.config['models']
        if model_filter:
            models = [m for m in models if m['id'] == model_filter]

        scenarios = list(self.scenarios.keys())
        if scenario_filter:
            scenarios = [s for s in scenarios if s == scenario_filter]

        horizons = self.config['horizons']
        if horizon_filter:
            horizons = [h for h in horizons if h['id'] == horizon_filter]

        logger.info(f"Models: {[m['id'] for m in models]}")
        logger.info(f"Scenarios: {scenarios}")
        logger.info(f"Horizons: {[h['id'] for h in horizons]}")
        logger.info(f"Using {self.config['settings']['concurrent_calls']} concurrent API calls")

        for model in models:
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"MODEL: {model['provider']}")
            logger.info("=" * 80)

            for scenario_id in scenarios:
                logger.info(f"  Scenario: {scenario_id}")

                for horizon in horizons:
                    logger.info(f"    {horizon['id']}: {horizon['description']}")

                    num_trials = trials_override or model.get('trials_per_condition', 100)

                    await self.run_condition(
                        model_config=model,
                        scenario_id=scenario_id,
                        horizon_config=horizon,
                        num_trials=num_trials
                    )

        logger.info("")
        logger.info("=" * 80)
        logger.info("HORIZON EXPERIMENT COMPLETE")
        logger.info("=" * 80)


def main():
    arg_parser = argparse.ArgumentParser(description='Run IPD Horizon Effects Experiment')
    arg_parser.add_argument('--model', type=str, help='Run only this model')
    arg_parser.add_argument('--scenario', type=str, help='Run only this scenario')
    arg_parser.add_argument('--horizon', type=str, help='Run only this horizon condition')
    arg_parser.add_argument('--trials', type=int, help='Override trials per condition')
    arg_parser.add_argument('--config', type=str,
                        default='experiments/iterated_pd/config/ipd_horizon_effects.yaml',
                        help='Path to config file')

    args = arg_parser.parse_args()

    # Get API key
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set")
        sys.exit(1)

    runner = HorizonExperimentRunner(args.config)

    asyncio.run(runner.run(
        api_key=api_key,
        model_filter=args.model,
        scenario_filter=args.scenario,
        horizon_filter=args.horizon,
        trials_override=args.trials
    ))


if __name__ == '__main__':
    main()
