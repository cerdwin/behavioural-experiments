#!/usr/bin/env python3
"""
Ablation Study Runner

Tests two hypotheses:
1. Do rationality instructions change biased behavior? (DeepSeek, GPT-5.2)
2. Do models behave differently vs humans? (O4-mini, others)

Design:
- 3 models: DeepSeek v3.2, O4-mini, GPT-5.2
- 2 experiments: Prisoner's Dilemma, Allais
- 3 instruction variants: baseline, rational_agent, calculate_ev
- 2 opponent types: unspecified, human
- 50 trials per condition
- Total: 3 models × 2 exp × 3 instructions × 2 opponents × 50 = 1,800 trials
"""

import asyncio
import yaml
import logging
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime
import itertools

from openrouter_client import OpenRouterClient
from database_v2 import DatabaseV2
from parser import create_parser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('results/logs/ablations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AblationRunner:
    """Run ablation experiments with instruction and opponent variations."""

    def __init__(self, config_path: str = "config_ablations.yaml", cli_args=None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Store CLI args for filtering
        self.cli_args = cli_args

        # Apply CLI overrides
        if cli_args:
            # Override output database
            if cli_args.output_db:
                db_path = cli_args.output_db
            else:
                db_path = f"results/{self.config['output']['database']}"

            # Override model
            if cli_args.model:
                self.config['models'] = [{
                    'id': cli_args.model.split('/')[-1],
                    'name': cli_args.model,
                    'max_tokens': 500
                }]

            # Override trials per condition
            if cli_args.trials:
                for exp_name in self.config['experiments']:
                    self.config['experiments'][exp_name]['trials_per_condition'] = cli_args.trials

            # Filter scenarios
            if cli_args.scenarios:
                for exp_name, exp_config in self.config['experiments'].items():
                    if 'scenarios' in exp_config:
                        exp_config['scenarios'] = [
                            s for s in exp_config['scenarios']
                            if s['id'] in cli_args.scenarios
                        ]

            # Filter stake levels
            if cli_args.stakes:
                for exp_name, exp_config in self.config['experiments'].items():
                    if 'stake_levels' in exp_config:
                        exp_config['stake_levels'] = [
                            s for s in exp_config['stake_levels']
                            if s['id'] in cli_args.stakes
                        ]
                    if 'total_amounts' in exp_config:  # Ultimatum uses total_amounts
                        exp_config['total_amounts'] = [
                            s for s in exp_config['total_amounts']
                            if s['id'] in cli_args.stakes
                        ]

            # Filter opponent types (handled in _run_pd_ablations)
        else:
            db_path = f"results/{self.config['output']['database']}"

        self.db = DatabaseV2(db_path, log_raw_responses=True)

        self.db.update_run_metadata(
            config_snapshot=self.config,
            experiments_run=[exp for exp in self.config['experiments'].keys()]
        )

        self.scenarios = self._load_scenarios()

        self.total_calls = 0
        self.successful_calls = 0
        self.failed_parses = []

        logger.info(f"Initialized ablation runner with config: {config_path}")
        if cli_args and cli_args.opponent_type:
            logger.info(f"Filtering to opponent type: {cli_args.opponent_type}")
        if cli_args and cli_args.model:
            logger.info(f"Using model: {cli_args.model}")
        
    def _load_scenarios(self) -> Dict[str, str]:
        """Load all scenario template files."""
        scenarios = {}
        
        for exp_name, exp_config in self.config['experiments'].items():
            if 'scenarios' not in exp_config:
                continue
            
            for scenario in exp_config['scenarios']:
                scenario_id = f"{exp_name}_{scenario['id']}"
                file_path = Path(scenario['file'])
                
                if file_path.exists():
                    with open(file_path) as f:
                        scenarios[scenario_id] = f.read()
                else:
                    logger.warning(f"Scenario file not found: {file_path}")
        
        logger.info(f"Loaded {len(scenarios)} scenario templates")
        return scenarios
    
    def _fill_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Fill template with variables."""
        result = template
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            result = result.replace(placeholder, str(value))
        return result
    
    async def run_all(self, api_key: str):
        """Run all ablation conditions."""
        existing_stats = self.db.get_stats()
        if existing_stats['total_trials'] > 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"RESUME MODE: Found {existing_stats['total_trials']} existing trials")
            logger.info(f"Parse success: {existing_stats['successful_parses']}/{existing_stats['total_trials']}")
            logger.info(f"{'='*60}\n")

        # Concurrency limit for API calls
        concurrent_limit = self.config.get('settings', {}).get('concurrent_calls', 10)
        self.semaphore = asyncio.Semaphore(concurrent_limit)
        logger.info(f"Using {concurrent_limit} concurrent API calls")

        async with OpenRouterClient(api_key) as client:
            for model in self.config['models']:
                logger.info(f"\n{'='*80}")
                logger.info(f"STARTING MODEL: {model['name']}")
                logger.info(f"{'='*80}\n")
                
                for exp_name, exp_config in self.config['experiments'].items():
                    if not exp_config.get('enabled'):
                        continue
                    
                    logger.info(f"\n--- Experiment: {exp_name} ---")
                    
                    if exp_name == 'prisoner_dilemma':
                        await self._run_pd_ablations(client, model, exp_config)
                    elif exp_name == 'public_goods':
                        await self._run_pg_ablations(client, model, exp_config)
                    elif exp_name == 'ultimatum':
                        await self._run_ultimatum_ablations(client, model, exp_config)
                    elif exp_name == 'allais':
                        await self._run_allais_ablations(client, model, exp_config)
        
        self.db.update_run_metadata(status='completed')
        
        final_stats = self.db.get_stats()
        logger.info(f"\n{'='*80}")
        logger.info(f"ABLATION STUDY COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total trials: {final_stats['total_trials']}")
        logger.info(f"Parse success: {final_stats['successful_parses']}/{final_stats['total_trials']}")
        logger.info(f"{'='*80}\n")
    
    async def _run_pd_ablations(self, client, model, config):
        """Run PD with all ablation conditions."""

        # Filter opponent types if CLI arg provided
        opponent_types = config['opponent_types']
        if self.cli_args and self.cli_args.opponent_type:
            opponent_types = [
                ot for ot in opponent_types
                if ot['id'] == self.cli_args.opponent_type
            ]
            if not opponent_types:
                logger.warning(f"No matching opponent type found for: {self.cli_args.opponent_type}")
                return

        for group_size in config['group_sizes']:
            for stake in config['stake_levels']:
                for incentive in config['incentive_structures']:
                    for scenario in config['scenarios']:
                        for instruction_var in config['instruction_variants']:
                            for opponent_type in opponent_types:
                                
                                logger.info(f"PD: model={model['id']}, N={group_size}, "
                                          f"instruction={instruction_var['id']}, "
                                          f"opponent={opponent_type['id']}")
                                
                                await self._run_pd_condition(
                                    client, model, group_size, stake, incentive,
                                    scenario, instruction_var, opponent_type,
                                    config['trials_per_condition'], config
                                )
    
    async def _run_pd_condition(self, client, model, group_size, stake, incentive,
                                scenario, instruction_var, opponent_type, n_trials, exp_config):
        """Run N trials of a single PD condition, skipping already completed trials."""

        # Check existing trials for this condition
        existing_count = self.db.count_condition_trials(
            experiment='prisoner_dilemma',
            model=model['name'],
            condition_filters={
                'group_size': group_size,
                'stake_level': stake['id'],
                'incentive_structure': incentive['id'],
                'scenario_id': scenario['id'],
                'instruction_variant': instruction_var['id'],
                'opponent_type': opponent_type['id']
            }
        )

        remaining = n_trials - existing_count
        if remaining <= 0:
            logger.debug(f"Skipping PD condition (already have {existing_count}/{n_trials} trials)")
            return

        if existing_count > 0:
            logger.info(f"Resuming PD condition: {existing_count}/{n_trials} complete, running {remaining} more")

        cc = int(100 * stake['multiplier'])
        dd = int(50 * stake['multiplier'])
        cd_defector = int(150 * stake['multiplier'])
        cd_sucker = int(0 * stake['multiplier'])

        temptation_ratio = cd_defector / cc
        punishment_ratio = dd / cc
        sucker_ratio = cd_sucker / cc

        scenario_key = f"prisoner_dilemma_{scenario['id']}"
        template = self.scenarios[scenario_key]

        # Get opponent text (e.g., "Other players: humans.")
        opponent_text = opponent_type.get('opponent_text', '')

        prompt_base = self._fill_template(template, {
            'n': group_size,
            'cc': cc,
            'dd': dd,
            'cd_defector': cd_defector,
            'cd_sucker': cd_sucker,
            'opponent_text': opponent_text
        })
        
        prompt = instruction_var['text'] + prompt_base

        # Helper for single trial with semaphore
        async def run_single_trial(trial_idx):
            async with self.semaphore:
                try:
                    result = await client.query(
                        model=model['name'],
                        prompt=prompt,
                        temperature=self.config['settings']['temperature'],
                        max_tokens=model.get('max_tokens', 500)
                    )
                    latency_ms = result.get('latency', 0)
                    response = result.get('raw', '')

                    parser = create_parser('prisoner_dilemma', exp_config)
                    parsed = parser.parse(response)

                    self.db.insert_trial({
                        'experiment': 'prisoner_dilemma',
                        'condition': f"N{group_size}_{stake['id']}_{incentive['id']}_{scenario['id']}",
                        'model': model['name'],
                        'group_size': group_size,
                        'agent_id': 0,
                        'stake_level': stake['id'],
                        'stake_multiplier': stake['multiplier'],
                        'incentive_structure': incentive['id'],
                        'scenario_id': scenario['id'],
                        'temptation_ratio': temptation_ratio,
                        'punishment_ratio': punishment_ratio,
                        'sucker_ratio': sucker_ratio,
                        'payoffs': {
                            'cc': cc,
                            'dd': dd,
                            'cd_defector': cd_defector,
                            'cd_sucker': cd_sucker
                        },
                        'prompt': prompt,
                        'response_raw': response,
                        'choice': parsed.get('choice'),
                        'confidence': parsed.get('confidence'),
                        'reasoning': parsed.get('reasoning'),
                        'parse_success': parsed.get('parse_success'),
                        'extraction_method': parsed.get('extraction_method'),
                        'total_choice_occurrences': parsed.get('total_choice_occurrences'),
                        'instruction_variant': instruction_var['id'],
                        'opponent_type': opponent_type['id'],
                        'temperature': self.config['settings']['temperature'],
                        'latency_ms': latency_ms
                    })

                    self.total_calls += 1
                    if parsed.get('parse_success'):
                        self.successful_calls += 1

                    return parsed.get('choice'), parsed.get('extraction_method')

                except Exception as e:
                    logger.error(f"Error in PD trial: {e}")
                    self.db.insert_trial({
                        'experiment': 'prisoner_dilemma',
                        'model': model['name'],
                        'error': str(e),
                        'instruction_variant': instruction_var['id'],
                        'opponent_type': opponent_type['id']
                    })
                    return None, None

        # Run all trials concurrently
        tasks = [run_single_trial(i) for i in range(remaining)]
        results = await asyncio.gather(*tasks)

        # Log summary
        choices = [r[0] for r in results if r[0]]
        coop = sum(1 for c in choices if c == 'COOPERATE')
        defect = sum(1 for c in choices if c == 'DEFECT')
        logger.info(f"  Completed {len(results)} trials: {coop} COOPERATE, {defect} DEFECT")

    async def _run_pg_ablations(self, client, model, config):
        """Run Public Goods with all ablation conditions."""

        # Filter opponent types if CLI arg provided
        opponent_types = config.get('opponent_types', [{'id': 'unspecified', 'opponent_text': ''}])
        if self.cli_args and self.cli_args.opponent_type:
            opponent_types = [
                ot for ot in opponent_types
                if ot['id'] == self.cli_args.opponent_type
            ]
            if not opponent_types:
                logger.warning(f"No matching opponent type found for: {self.cli_args.opponent_type}")
                return

        for group_size in config['group_sizes']:
            for endowment in config['endowment_levels']:
                for multiplier in config['multiplier_levels']:
                    for scenario in config['scenarios']:
                        for instruction_var in config['instruction_variants']:
                            for opponent_type in opponent_types:

                                logger.info(f"PG: model={model['id']}, N={group_size}, "
                                          f"endowment=${endowment['amount']}, "
                                          f"multiplier={multiplier['value']}x, "
                                          f"instruction={instruction_var['id']}, "
                                          f"opponent={opponent_type['id']}")

                                await self._run_pg_condition(
                                    client, model, group_size, endowment, multiplier,
                                    scenario, instruction_var, opponent_type,
                                    config['trials_per_condition'], config
                                )

    async def _run_pg_condition(self, client, model, group_size, endowment, multiplier,
                                scenario, instruction_var, opponent_type, n_trials, exp_config):
        """Run N trials of a single PG condition, skipping already completed trials."""

        # Check existing trials for this condition
        existing_count = self.db.count_condition_trials(
            experiment='public_goods',
            model=model['name'],
            condition_filters={
                'group_size': group_size,
                'stake_level': endowment['id'],
                'stake_multiplier': multiplier['value'],
                'scenario_id': scenario['id'],
                'instruction_variant': instruction_var['id'],
                'opponent_type': opponent_type['id']
            }
        )

        remaining = n_trials - existing_count
        if remaining <= 0:
            logger.debug(f"Skipping PG condition (already have {existing_count}/{n_trials} trials)")
            return

        if existing_count > 0:
            logger.info(f"Resuming PG condition: {existing_count}/{n_trials} complete, running {remaining} more")

        endowment_amt = endowment['amount']
        multiplier_val = multiplier['value']
        mpcr = multiplier_val / group_size  # Marginal per-capita return

        scenario_key = f"public_goods_{scenario['id']}"
        template = self.scenarios.get(scenario_key)

        if not template:
            logger.warning(f"Scenario template not found: {scenario_key}")
            return

        opponent_text = opponent_type.get('opponent_text', '')

        prompt_base = self._fill_template(template, {
            'n': group_size,
            'endowment': endowment_amt,
            'multiplier': multiplier_val,
            'opponent_text': opponent_text
        })

        prompt = instruction_var['text'] + prompt_base

        for trial_idx in range(remaining):
            try:
                result = await client.query(
                    model=model['name'],
                    prompt=prompt,
                    temperature=self.config['settings']['temperature'],
                    max_tokens=model.get('max_tokens', 500)
                )
                latency_ms = result.get('latency', 0)
                response = result.get('raw', '')

                parser = create_parser('public_goods', exp_config)
                parsed = parser.parse(response)

                # Extract numeric contribution
                contribution = parsed.get('numeric_value')
                if contribution is None and parsed.get('choice'):
                    # Try to parse from choice string
                    try:
                        contribution = float(parsed['choice'])
                    except:
                        pass

                self.db.insert_trial({
                    'experiment': 'public_goods',
                    'condition': f"N{group_size}_E{endowment['id']}_{multiplier['id']}_{scenario['id']}",
                    'model': model['name'],
                    'group_size': group_size,
                    'agent_id': 0,
                    'stake_level': endowment['id'],
                    'stake_multiplier': multiplier_val,
                    'mpcr': mpcr,
                    'scenario_id': scenario['id'],
                    'payoffs': {
                        'endowment': endowment_amt,
                        'multiplier': multiplier_val,
                        'mpcr': mpcr
                    },
                    'prompt': prompt,
                    'response_raw': response,
                    'choice': parsed.get('choice'),
                    'numeric_value': contribution,
                    'confidence': parsed.get('confidence'),
                    'reasoning': parsed.get('reasoning'),
                    'parse_success': parsed.get('parse_success'),
                    'extraction_method': parsed.get('extraction_method'),
                    'total_choice_occurrences': parsed.get('total_choice_occurrences'),
                    'instruction_variant': instruction_var['id'],
                    'opponent_type': opponent_type['id'],
                    'temperature': self.config['settings']['temperature'],
                    'latency_ms': latency_ms
                })

                self.total_calls += 1
                if parsed.get('parse_success'):
                    self.successful_calls += 1

                if trial_idx % 10 == 0:
                    logger.info(f"  Trial {trial_idx+1}/{n_trials}: ${contribution} "
                              f"(parse: {parsed.get('extraction_method')})")

            except Exception as e:
                logger.error(f"Error in PG trial: {e}")
                self.db.insert_trial({
                    'experiment': 'public_goods',
                    'model': model['name'],
                    'error': str(e),
                    'instruction_variant': instruction_var['id'],
                    'opponent_type': opponent_type['id']
                })

            await asyncio.sleep(0.1)

    async def _run_ultimatum_ablations(self, client, model, config):
        """Run Ultimatum with all ablation conditions."""

        for total_amount in config['total_amounts']:
            for role in config['roles']:
                for scenario in config['scenarios']:
                    for instruction_var in config['instruction_variants']:

                        logger.info(f"Ultimatum: model={model['id']}, "
                                  f"amount=${total_amount['amount']}, "
                                  f"role={role['id']}, "
                                  f"instruction={instruction_var['id']}")

                        await self._run_ultimatum_condition(
                            client, model, total_amount, role,
                            scenario, instruction_var,
                            config['trials_per_condition'], config
                        )

    async def _run_ultimatum_condition(self, client, model, total_amount, role,
                                       scenario, instruction_var, n_trials, exp_config):
        """Run N trials of a single Ultimatum condition, skipping already completed trials."""

        # Check existing trials for this condition
        existing_count = self.db.count_condition_trials(
            experiment='ultimatum',
            model=model['name'],
            condition_filters={
                'stake_level': total_amount['id'],
                'scenario_id': scenario['id'],
                'instruction_variant': instruction_var['id']
            }
        )

        # Also filter by role (stored in condition string)
        # We need to be more specific - check the condition field
        role_id = role['id']

        remaining = n_trials - existing_count
        if remaining <= 0:
            logger.debug(f"Skipping Ultimatum condition (already have {existing_count}/{n_trials} trials)")
            return

        if existing_count > 0:
            logger.info(f"Resuming Ultimatum condition: {existing_count}/{n_trials} complete, running {remaining} more")

        amount = total_amount['amount']

        scenario_key = f"ultimatum_{scenario['id']}"
        template = self.scenarios.get(scenario_key)

        if not template:
            logger.warning(f"Scenario template not found: {scenario_key}")
            return

        # Role-specific text
        if role_id == 'proposer':
            role_text = "You must propose how to split this amount."
            task_text = "How much do you offer to the other person?"
            choice_format = "X"
            example_text = "I offer $50 because...\n[CHOICE: 50]\n[CONFIDENCE: 80]"
        else:  # responder
            # For responder, we simulate an offer
            offer_pct = 30  # Standard low-ball offer
            offer_amt = int(amount * offer_pct / 100)
            role_text = f"The other person offers you ${offer_amt} (keeping ${amount - offer_amt})."
            task_text = "Do you ACCEPT or REJECT this offer?"
            choice_format = "ACCEPT or REJECT"
            example_text = "I accept because...\n[CHOICE: ACCEPT]\n[CONFIDENCE: 80]"

        prompt_base = self._fill_template(template, {
            'role': role_id,
            'total_amount': amount,
            'role_text': role_text,
            'task_text': task_text,
            'choice_format': choice_format,
            'example_text': example_text,
            'opponent_text': ''  # No opponent identity for ultimatum
        })

        prompt = instruction_var['text'] + prompt_base

        for trial_idx in range(remaining):
            try:
                result = await client.query(
                    model=model['name'],
                    prompt=prompt,
                    temperature=self.config['settings']['temperature'],
                    max_tokens=model.get('max_tokens', 500)
                )
                latency_ms = result.get('latency', 0)
                response = result.get('raw', '')

                parser = create_parser('ultimatum', exp_config)
                parsed = parser.parse(response)

                # Extract numeric value for proposer
                offer_amount = None
                if role_id == 'proposer' and parsed.get('choice'):
                    try:
                        offer_amount = float(parsed['choice'])
                    except:
                        pass

                self.db.insert_trial({
                    'experiment': 'ultimatum',
                    'condition': f"{total_amount['id']}_{role_id}_{scenario['id']}",
                    'model': model['name'],
                    'group_size': 2,  # Always 2 players
                    'agent_id': 0,
                    'stake_level': total_amount['id'],
                    'stake_multiplier': 1.0,
                    'scenario_id': scenario['id'],
                    'payoffs': {
                        'total_amount': amount,
                        'role': role_id
                    },
                    'prompt': prompt,
                    'response_raw': response,
                    'choice': parsed.get('choice'),
                    'numeric_value': offer_amount,
                    'confidence': parsed.get('confidence'),
                    'reasoning': parsed.get('reasoning'),
                    'parse_success': parsed.get('parse_success'),
                    'extraction_method': parsed.get('extraction_method'),
                    'total_choice_occurrences': parsed.get('total_choice_occurrences'),
                    'instruction_variant': instruction_var['id'],
                    'opponent_type': 'unspecified',
                    'temperature': self.config['settings']['temperature'],
                    'latency_ms': latency_ms
                })

                self.total_calls += 1
                if parsed.get('parse_success'):
                    self.successful_calls += 1

                if trial_idx % 10 == 0:
                    logger.info(f"  Trial {trial_idx+1}/{n_trials}: {parsed.get('choice')} "
                              f"(parse: {parsed.get('extraction_method')})")

            except Exception as e:
                logger.error(f"Error in Ultimatum trial: {e}")
                self.db.insert_trial({
                    'experiment': 'ultimatum',
                    'model': model['name'],
                    'error': str(e),
                    'instruction_variant': instruction_var['id'],
                    'opponent_type': 'unspecified'
                })

            await asyncio.sleep(0.1)

    async def _run_allais_ablations(self, client, model, config):
        """Run Allais with all ablation conditions."""
        
        models_to_test = config.get('models_to_test', [])
        if models_to_test and model['id'] not in models_to_test:
            logger.info(f"Skipping Allais for {model['id']} (not in models_to_test)")
            return
        
        for stake in config['stake_levels']:
            for scenario in config['scenarios']:
                for instruction_var in config['instruction_variants']:
                    
                    logger.info(f"Allais: model={model['id']}, stake={stake['id']}, "
                              f"instruction={instruction_var['id']}")
                    
                    await self._run_allais_condition(
                        client, model, stake, scenario, instruction_var,
                        config['trials_per_condition'], config
                    )
    
    async def _run_allais_condition(self, client, model, stake, scenario,
                                   instruction_var, n_trials, exp_config):
        """Run N trials of Allais (each trial = paired questions), skipping already completed."""

        # Check existing pairs for this condition
        existing_count = self.db.count_allais_pairs(
            model=model['name'],
            condition_filters={
                'stake_level': stake['id'],
                'scenario_id': scenario['id'],
                'variant': instruction_var['id']
            }
        )

        remaining = n_trials - existing_count
        if remaining <= 0:
            logger.debug(f"Skipping Allais condition (already have {existing_count}/{n_trials} pairs)")
            return

        if existing_count > 0:
            logger.info(f"Resuming Allais condition: {existing_count}/{n_trials} complete, running {remaining} more")

        scenario_key = f"allais_{scenario['id']}"
        template_q1 = self.scenarios[scenario_key]
        
        q2_template = """Now consider a second choice:

CHOICE_C: {prob_c_high}% chance of ${choice_c_high}, {prob_c_low}% chance of ${choice_c_low}
CHOICE_D: {prob_d_high}% chance of ${choice_d_high}, {prob_d_low}% chance of ${choice_d_low}

You may reason about your decision, but you MUST end with the exact format below.

CRITICAL: Your response MUST include these lines using square brackets:
[CHOICE: CHOICE_C] or [CHOICE: CHOICE_D]
[CONFIDENCE: 0-100]

Example response:
I choose C because...
[CHOICE: CHOICE_C]
[CONFIDENCE: 85]
"""

        for trial_idx in range(remaining):
            try:
                prompt_q1 = self._fill_template(template_q1, {
                    'instruction_text': instruction_var['text'],
                    'choice_a_certain': stake['choice_a_certain'],
                    'choice_b_high': stake['choice_b_high'],
                    'choice_b_mid': stake['choice_b_mid'],
                    'choice_b_low': stake['choice_b_low'],
                    'prob_high_pct': stake['prob_high'],
                    'prob_mid_pct': stake['prob_mid'],
                    'prob_low_pct': stake['prob_low']
                })
                
                start_time = datetime.now()
                result_q1 = await client.query(
                    model=model['name'],
                    prompt=prompt_q1,
                    temperature=self.config['settings']['temperature'],
                    max_tokens=model.get('max_tokens', 500)
                )
                response_q1 = result_q1.get('raw', '')
                
                parser = create_parser('allais', exp_config)
                parsed_q1 = parser.parse(response_q1)
                choice_1 = parsed_q1.get('choice')
                
                prob_c_high = stake['prob_high']
                choice_c_high = stake['choice_b_high']
                prob_c_low = 100 - prob_c_high
                choice_c_low = stake['choice_b_low']
                
                prob_d_high = stake['prob_high'] + 1
                choice_d_high = stake['choice_a_certain']
                prob_d_low = 100 - prob_d_high
                choice_d_low = stake['choice_b_low']
                
                prompt_q2 = instruction_var['text'] + "\n" + self._fill_template(q2_template, {
                    'prob_c_high': prob_c_high,
                    'choice_c_high': choice_c_high,
                    'prob_c_low': prob_c_low,
                    'choice_c_low': choice_c_low,
                    'prob_d_high': prob_d_high,
                    'choice_d_high': choice_d_high,
                    'prob_d_low': prob_d_low,
                    'choice_d_low': choice_d_low
                })
                
                result_q2 = await client.query(
                    model=model['name'],
                    prompt=prompt_q2,
                    temperature=self.config['settings']['temperature'],
                    max_tokens=model.get('max_tokens', 500)
                )
                response_q2 = result_q2.get('raw', '')
                latency_ms = result_q1.get('latency', 0) + result_q2.get('latency', 0)
                
                parsed_q2 = parser.parse(response_q2)
                choice_2 = parsed_q2.get('choice')
                
                self.db.insert_allais_pair({
                    'model': model['name'],
                    'variant': 'no_history',
                    'stake_level': stake['id'],
                    'scenario_id': scenario['id'],
                    'choice_1': choice_1,
                    'choice_2': choice_2,
                    'instruction_variant': instruction_var['id'],
                    'opponent_type': 'unspecified',
                    'response_q1': response_q1,
                    'response_q2': response_q2,
                    'metadata': {
                        'q1_prompt': prompt_q1,
                        'q2_prompt': prompt_q2,
                        'latency_ms': latency_ms
                    }
                })
                
                self.total_calls += 2
                if parsed_q1.get('parse_success'):
                    self.successful_calls += 1
                if parsed_q2.get('parse_success'):
                    self.successful_calls += 1
                
                if trial_idx % 10 == 0:
                    logger.info(f"  Trial {trial_idx+1}/{n_trials}: {choice_1} → {choice_2}")
                
            except Exception as e:
                logger.error(f"Error in Allais trial: {e}")
            
            await asyncio.sleep(0.1)


def parse_args():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Run Wave 1 ablation experiments')
    parser.add_argument('--config', default='configs/config_wave1.yaml',
                        help='Path to config file')
    parser.add_argument('--model',
                        help='Override model (e.g., deepseek/deepseek-v3.2)')
    parser.add_argument('--opponent-type',
                        choices=['control', 'vs_human', 'vs_ai_generic', 'vs_self',
                                 'vs_gpt', 'vs_claude', 'vs_gemini'],
                        help='Filter to single opponent type')
    parser.add_argument('--output-db',
                        help='Override output database path')
    parser.add_argument('--scenarios', nargs='+',
                        help='Filter to specific scenarios (e.g., business environment)')
    parser.add_argument('--stakes', nargs='+',
                        help='Filter to specific stake levels (e.g., medium large huge)')
    parser.add_argument('--trials', type=int,
                        help='Override trials per condition')
    return parser.parse_args()


async def main():
    import os

    args = parse_args()

    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    runner = AblationRunner(args.config, cli_args=args)
    await runner.run_all(api_key)


if __name__ == "__main__":
    asyncio.run(main())
