"""
Experiment runner with systematic parameter variation.

Features:
- Balanced sampling (not random)
- Full parameter tracking
- Raw response logging
- Square bracket parsing
"""

import asyncio
import yaml
import random
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path
from datetime import datetime

from openrouter_client import OpenRouterClient
from database_v2 import DatabaseV2
from parser import create_parser

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('results/logs/experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ExperimentRunnerV2:
    """Run systematic behavioral economics experiments."""
    
    def __init__(self, config_path: str = "config_v2.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Initialize database
        db_path = f"results/{self.config['output']['database']}"
        self.db = DatabaseV2(db_path, log_raw_responses=True)
        
        # Save config snapshot
        self.db.update_run_metadata(
            config_snapshot=self.config,
            experiments_run=[exp for exp, cfg in self.config['experiments'].items() if cfg.get('enabled')]
        )
        
        # Load scenario templates
        self.scenarios = self._load_scenarios()
        
        # Statistics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_parses = []
        
        logger.info(f"Initialized runner with config: {config_path}")
        
    def _load_scenarios(self) -> Dict[str, str]:
        """Load all scenario template files."""
        scenarios = {}
        
        for exp_name, exp_config in self.config['experiments'].items():
            if not exp_config.get('enabled'):
                continue
            
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
        """Run all enabled experiments."""
        existing_stats = self.db.get_stats()
        if existing_stats['total_trials'] > 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"RESUME MODE: Found {existing_stats['total_trials']} existing trials")
            logger.info(f"Parse success: {existing_stats['successful_parses']}/{existing_stats['total_trials']}")
            logger.info(f"{'='*60}\n")
        
        async with OpenRouterClient(
            api_key=api_key,
            base_url=self.config['api']['base_url'],
            max_retries=self.config['execution']['max_retries']
        ) as client:
            
            models = {m['id'].split('/')[-1]: m['id'] 
                     for m in self.config['models']['test_models']}
            
            self.db.update_run_metadata(models_used=list(models.values()))
            
            logger.info(f"Testing models: {list(models.values())}")
            
            tasks = []
            
            # Create tasks for each experiment
            for exp_name, exp_config in self.config['experiments'].items():
                if not exp_config.get('enabled'):
                    continue
                
                logger.info(f"Setting up {exp_name}...")
                
                if exp_name == 'prisoner_dilemma':
                    tasks.extend(self._create_pd_tasks(client, models, exp_config))
                elif exp_name == 'public_goods':
                    tasks.extend(self._create_pg_tasks(client, models, exp_config))
                elif exp_name == 'allais':
                    tasks.extend(self._create_allais_tasks(client, models, exp_config))
                elif exp_name == 'ultimatum':
                    tasks.extend(self._create_ultimatum_tasks(client, models, exp_config))
                elif exp_name == 'framing':
                    tasks.extend(self._create_framing_tasks(client, models, exp_config))
                elif exp_name == 'stag_hunt':
                    tasks.extend(self._create_stag_hunt_tasks(client, models, exp_config))
            
            logger.info(f"\nTotal tasks: {len(tasks)}")
            logger.info(f"Estimated API calls: {self.total_calls}")
            self.db.update_run_metadata(total_trials_planned=self.total_calls)
            
            # Run with concurrency limit
            semaphore = asyncio.Semaphore(
                self.config['api']['rate_limit']['concurrent_requests']
            )
            
            async def limited_task(task):
                async with semaphore:
                    return await task
            
            logger.info("Starting execution...\n")
            results = await asyncio.gather(
                *[limited_task(t) for t in tasks],
                return_exceptions=True
            )
            
            # Log errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Task {i} failed: {result}")
            
            # Final stats
            logger.info(f"\n{'='*60}")
            logger.info(f"Completed {self.successful_calls}/{self.total_calls} trials")
            logger.info(f"Parse success rate: {self.successful_calls/self.total_calls*100:.1f}%")
            
            if self.failed_parses:
                logger.warning(f"Failed parses: {len(self.failed_parses)}")
                logger.warning(f"See: results/logs/failed_parses.log")
                
                with open('results/logs/failed_parses.log', 'w') as f:
                    for fp in self.failed_parses:
                        f.write(f"\n{'='*60}\n")
                        f.write(f"Trial: {fp['trial_id']}\n")
                        f.write(f"Model: {fp['model']}\n")
                        f.write(f"Response: {fp['response'][:200]}...\n")
            
            self.db.update_run_metadata(status='completed')
            logger.info(f"Run ID: {self.db.run_id}")
            logger.info(f"Database: {self.db.db_path}")
    
    def _create_pd_tasks(self, client, models, config) -> List:
        """Create Prisoner's Dilemma tasks with balanced sampling."""
        tasks = []
        
        stake_levels = config['stake_levels']
        incentive_structures = config['incentive_structures']
        scenarios = config['scenarios']
        group_sizes = config['group_sizes']
        
        trials_per_group = config['trials_per_condition']
        trials_per_stake = trials_per_group // len(stake_levels)
        
        for model_name, model_id in models.items():
            for group_size in group_sizes:
                trial_num = 0
                
                # Balanced sampling: 20 per stake level
                for stake in stake_levels:
                    for i in range(trials_per_stake):
                        # Cycle through incentive structures
                        incentive = incentive_structures[i % len(incentive_structures)]
                        
                        # Random scenario
                        scenario = random.choice(scenarios)
                        
                        # Calculate payoffs
                        payoffs = self._calculate_pd_payoffs(stake, incentive)
                        
                        # Create task
                        if group_size == 1:
                            tasks.append(self._run_pd_single(
                                client, model_id, payoffs, scenario,
                                stake, incentive, trial_num
                            ))
                            self.total_calls += 1
                        else:
                            tasks.append(self._run_pd_multi(
                                client, model_id, group_size, payoffs, scenario,
                                stake, incentive, trial_num
                            ))
                            self.total_calls += group_size
                        
                        trial_num += 1
        
        return tasks
    
    def _calculate_pd_payoffs(self, stake: Dict, incentive: Dict) -> Dict:
        """Calculate PD payoffs from stake × incentive structure."""
        base_cc = stake['base_cc']
        
        return {
            'cc': int(base_cc),
            'dd': int(base_cc * incentive['punishment_ratio']),
            'cd_sucker': int(base_cc * incentive['sucker_ratio']),
            'cd_defector': int(base_cc * incentive['temptation_ratio']),
            'stake_level': stake['id'],
            'stake_multiplier': stake['multiplier'],
            'incentive_structure': incentive['id'],
            'temptation_ratio': incentive['temptation_ratio'],
            'punishment_ratio': incentive['punishment_ratio'],
            'sucker_ratio': incentive['sucker_ratio']
        }
    
    async def _run_pd_single(self, client, model, payoffs, scenario, 
                            stake, incentive, trial):
        """Run single-agent PD (predicting others)."""
        if self.db.check_trial_exists(
            experiment='prisoner_dilemma',
            model=model,
            condition='single',
            group_size=1,
            stake_level=stake['id'],
            incentive_structure=incentive['id'],
            scenario_id=scenario['id'],
            agent_id=0
        ):
            logger.info(f"⊙ PD/single/{model}/N=1/trial_{trial}: SKIP (exists)")
            self.successful_calls += 1
            return
        
        scenario_id = f"prisoner_dilemma_{scenario['id']}"
        template = self.scenarios.get(scenario_id)
        
        if not template:
            logger.error(f"Missing template: {scenario_id}")
            return
        
        prompt = self._fill_template(template, {
            'n': 1,
            **payoffs
        })
        
        result = await client.query(
            model=model,
            prompt=prompt,
            temperature=self.config['execution']['temperature'],
            max_tokens=self.config['execution']['max_tokens']
        )
        
        # Parse response
        parser = create_parser('prisoner_dilemma', 
                              self.config['experiments']['prisoner_dilemma'])
        parsed = parser.parse(result['raw'])
        
        if parsed['parse_success']:
            self.successful_calls += 1
        else:
            self.failed_parses.append({
                'trial_id': trial,
                'model': model,
                'response': result['raw']
            })
        
        # Save to database
        self.db.insert_trial({
            'experiment': 'prisoner_dilemma',
            'condition': 'single',
            'model': model,
            'group_size': 1,
            'stake_level': stake['id'],
            'stake_multiplier': stake['multiplier'],
            'incentive_structure': incentive['id'],
            'scenario_id': scenario['id'],
            'temptation_ratio': payoffs['temptation_ratio'],
            'punishment_ratio': payoffs['punishment_ratio'],
            'sucker_ratio': payoffs['sucker_ratio'],
            'payoffs': payoffs,
            'prompt': prompt,
            'response_raw': result['raw'],
            'temperature': self.config['execution']['temperature'],
            'latency_ms': result['latency'],
            **parsed
        })
        
        logger.info(f"✓ PD/single/{model}/N=1/trial_{trial}: {parsed.get('choice', 'FAIL')}")
        return parsed
    
    async def _run_pd_multi(self, client, model, n_agents, payoffs, scenario,
                           stake, incentive, trial):
        """Run multi-agent PD."""
        all_exist = all(
            self.db.check_trial_exists(
                experiment='prisoner_dilemma',
                model=model,
                condition='multi',
                group_size=n_agents,
                stake_level=stake['id'],
                incentive_structure=incentive['id'],
                scenario_id=scenario['id'],
                agent_id=i
            )
            for i in range(n_agents)
        )
        
        if all_exist:
            logger.info(f"⊙ PD/multi/{model}/N={n_agents}/trial_{trial}: SKIP (exists)")
            self.successful_calls += n_agents
            return
        
        scenario_id = f"prisoner_dilemma_{scenario['id']}"
        template = self.scenarios.get(scenario_id)
        
        if not template:
            logger.error(f"Missing template: {scenario_id}")
            return
        
        prompt = self._fill_template(template, {
            'n': n_agents,
            **payoffs
        })
        
        # Query all agents in parallel
        agent_tasks = [
            client.query(
                model=model,
                prompt=prompt,
                temperature=self.config['execution']['temperature'],
                max_tokens=self.config['execution']['max_tokens']
            )
            for _ in range(n_agents)
        ]
        
        results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # Parse all responses
        parser = create_parser('prisoner_dilemma',
                              self.config['experiments']['prisoner_dilemma'])
        
        decisions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Agent {i} error: {result}")
                decisions.append('DEFECT')
                continue
            
            parsed = parser.parse(result['raw'])
            
            if parsed['parse_success']:
                self.successful_calls += 1
                decisions.append(parsed['choice'])
            else:
                self.failed_parses.append({
                    'trial_id': f"{trial}_agent{i}",
                    'model': model,
                    'response': result['raw']
                })
                decisions.append('DEFECT')
        
        # Save aggregated game
        cooperation_rate = sum(1 for d in decisions if d == 'COOPERATE') / len(decisions)
        
        self.db.insert_multi_agent_game({
            'experiment': 'prisoner_dilemma',
            'model': model,
            'group_size': n_agents,
            'stake_level': stake['id'],
            'incentive_structure': incentive['id'],
            'scenario_id': scenario['id'],
            'payoffs': payoffs,
            'agent_decisions': decisions,
            'metadata': {'trial': trial}
        })
        
        logger.info(f"✓ PD/multi/{model}/N={n_agents}/trial_{trial}: {cooperation_rate*100:.0f}% coop")
        return decisions
    
    def _create_pg_tasks(self, client, models, config) -> List:
        """Create Public Goods tasks."""
        tasks = []
        
        endowment_levels = config['endowment_levels']
        multiplier_levels = config['multiplier_levels']
        scenarios = config['scenarios']
        group_sizes = config['group_sizes']
        
        trials_per_group = config['trials_per_condition']
        trials_per_endowment = trials_per_group // len(endowment_levels)
        
        for model_name, model_id in models.items():
            for group_size in group_sizes:
                trial_num = 0
                
                for endowment in endowment_levels:
                    for i in range(trials_per_endowment):
                        # Cycle through multipliers
                        multiplier = multiplier_levels[i % len(multiplier_levels)]
                        
                        # Random scenario
                        scenario = random.choice(scenarios)
                        
                        params = {
                            'endowment': endowment['amount'],
                            'multiplier': multiplier['value'],
                            'endowment_level': endowment['id'],
                            'multiplier_level': multiplier['id'],
                            'mpcr': multiplier['value'] / group_size
                        }
                        
                        if group_size == 1:
                            tasks.append(self._run_pg_single(
                                client, model_id, params, scenario, trial_num
                            ))
                            self.total_calls += 1
                        else:
                            tasks.append(self._run_pg_multi(
                                client, model_id, group_size, params, scenario, trial_num
                            ))
                            self.total_calls += group_size
                        
                        trial_num += 1
        
        return tasks
    
    async def _run_pg_single(self, client, model, params, scenario, trial):
        """Run single-agent public goods."""
        if self.db.check_trial_exists(
            experiment='public_goods',
            model=model,
            condition='single',
            group_size=1,
            stake_level=params['endowment_level'],
            incentive_structure=params['multiplier_level'],
            scenario_id=scenario['id'],
            agent_id=0
        ):
            logger.info(f"⊙ PG/single/{model}/N=1/trial_{trial}: SKIP (exists)")
            self.successful_calls += 1
            return
        
        scenario_id = f"public_goods_{scenario['id']}"
        template = self.scenarios.get(scenario_id)
        
        if not template:
            logger.error(f"Missing template: {scenario_id}")
            return
        
        # Calculate example
        example_contrib = params['endowment'] // 2
        example_pool = example_contrib * 1
        example_multiplied = example_pool * params['multiplier']
        example_share = example_multiplied / 1
        example_total = (params['endowment'] - example_contrib) + example_share
        
        prompt = self._fill_template(template, {
            'n': 1,
            'endowment': params['endowment'],
            'multiplier': params['multiplier'],
            'example_contrib': example_contrib,
            'example_pool': example_pool,
            'example_multiplied': int(example_multiplied),
            'example_share': f"{example_share:.1f}",
            'example_total': f"{example_total:.1f}"
        })
        
        result = await client.query(
            model=model,
            prompt=prompt,
            temperature=self.config['execution']['temperature'],
            max_tokens=self.config['execution']['max_tokens']
        )
        
        parser = create_parser('public_goods',
                              self.config['experiments']['public_goods'])
        parsed = parser.parse(result['raw'])
        
        if parsed['parse_success']:
            self.successful_calls += 1
        else:
            self.failed_parses.append({
                'trial_id': trial,
                'model': model,
                'response': result['raw']
            })
        
        self.db.insert_trial({
            'experiment': 'public_goods',
            'condition': 'single',
            'model': model,
            'group_size': 1,
            'stake_level': params['endowment_level'],
            'stake_multiplier': params['endowment'],
            'incentive_structure': params['multiplier_level'],
            'scenario_id': scenario['id'],
            'mpcr': params['mpcr'],
            'payoffs': params,
            'prompt': prompt,
            'response_raw': result['raw'],
            'temperature': self.config['execution']['temperature'],
            'latency_ms': result['latency'],
            **parsed
        })
        
        contrib = parsed.get('numeric_value', 0)
        logger.info(f"✓ PG/single/{model}/N=1/trial_{trial}: ${contrib}")
        return parsed
    
    async def _run_pg_multi(self, client, model, n_agents, params, scenario, trial):
        """Run multi-agent public goods."""
        all_exist = all(
            self.db.check_trial_exists(
                experiment='public_goods',
                model=model,
                condition='multi',
                group_size=n_agents,
                stake_level=params['endowment_level'],
                incentive_structure=params['multiplier_level'],
                scenario_id=scenario['id'],
                agent_id=i
            )
            for i in range(n_agents)
        )
        
        if all_exist:
            logger.info(f"⊙ PG/multi/{model}/N={n_agents}/trial_{trial}: SKIP (exists)")
            self.successful_calls += n_agents
            return
        
        scenario_id = f"public_goods_{scenario['id']}"
        template = self.scenarios.get(scenario_id)
        
        if not template:
            logger.error(f"Missing template: {scenario_id}")
            return
        
        # Calculate example with all contributing half
        example_contrib = params['endowment'] // 2
        example_pool = example_contrib * n_agents
        example_multiplied = example_pool * params['multiplier']
        example_share = example_multiplied / n_agents
        example_total = (params['endowment'] - example_contrib) + example_share
        
        prompt = self._fill_template(template, {
            'n': n_agents,
            'endowment': params['endowment'],
            'multiplier': params['multiplier'],
            'example_contrib': example_contrib,
            'example_pool': example_pool,
            'example_multiplied': int(example_multiplied),
            'example_share': f"{example_share:.1f}",
            'example_total': f"{example_total:.1f}"
        })
        
        agent_tasks = [
            client.query(
                model=model,
                prompt=prompt,
                temperature=self.config['execution']['temperature'],
                max_tokens=self.config['execution']['max_tokens']
            )
            for _ in range(n_agents)
        ]
        
        results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        parser = create_parser('public_goods',
                              self.config['experiments']['public_goods'])
        
        contributions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Agent {i} error: {result}")
                contributions.append(0)
                continue
            
            parsed = parser.parse(result['raw'])
            
            if parsed['parse_success']:
                self.successful_calls += 1
                contrib = parsed.get('numeric_value', 0)
                contributions.append(min(params['endowment'], max(0, contrib)))
            else:
                self.failed_parses.append({
                    'trial_id': f"{trial}_agent{i}",
                    'model': model,
                    'response': result['raw']
                })
                contributions.append(0)
        
        avg_contrib = sum(contributions) / len(contributions)
        
        self.db.insert_multi_agent_game({
            'experiment': 'public_goods',
            'model': model,
            'group_size': n_agents,
            'stake_level': params['endowment_level'],
            'incentive_structure': params['multiplier_level'],
            'scenario_id': scenario['id'],
            'payoffs': params,
            'agent_decisions': contributions,
            'metadata': {'trial': trial}
        })
        
        logger.info(f"✓ PG/multi/{model}/N={n_agents}/trial_{trial}: ${avg_contrib:.1f} avg")
        return contributions
    
    async def _run_allais(self, client, model, stake, scenario, trial):
        """Run Allais Paradox - two paired choices."""
        # Check if exists
        if self.db.check_allais_pair_exists(
            model=model,
            stake_level=stake['id'],
            scenario_id=scenario['id']
        ):
            logger.info(f"⊙ Allais/{model}/stake={stake['id']}/trial_{trial}: SKIP (exists)")
            self.successful_calls += 2
            return
        
        scenario_id = f"allais_{scenario['id']}"
        template = self.scenarios.get(scenario_id)
        
        if not template:
            logger.error(f"Missing template: {scenario_id}")
            return
        
        # Question 1: A vs B (common consequence effect)
        prompt_q1 = self._fill_template(template, {
            'choice_a_certain': f"{stake['certain']:,}",
            'choice_b_high': f"{stake['risky_high']:,}",
            'choice_b_mid': f"{stake['risky_mid']:,}",
            'choice_b_low': f"{stake['risky_low']:,}",
            'prob_high_pct': int(stake['prob_high'] * 100),
            'prob_mid_pct': int(stake['prob_mid'] * 100),
            'prob_low_pct': int(stake['prob_low'] * 100)
        })
        
        result_q1 = await client.query(
            model=model,
            prompt=prompt_q1,
            temperature=self.config['execution']['temperature'],
            max_tokens=self.config['execution']['max_tokens']
        )
        
        # Parse Q1
        parser = create_parser('allais', self.config['experiments']['allais'])
        parsed_q1 = parser.parse(result_q1['raw'])
        choice_1 = parsed_q1.get('choice', None)
        
        # Question 2: C vs D (shifted probabilities)
        # In classic Allais: remove common 89% outcome
        # C: 10% chance of risky_high, 90% of 0
        # D: 11% chance of certain, 89% of 0
        prompt_q2 = f"""Now consider a DIFFERENT choice:

CHOICE_C: {int(stake['prob_high'] * 100)}% chance of ${stake['risky_high']:,}, {int((1 - stake['prob_high']) * 100)}% chance of $0
CHOICE_D: {int((stake['prob_high'] + stake['prob_low']) * 100)}% chance of ${stake['certain']:,}, {int(stake['prob_mid'] * 100)}% chance of $0

You may reason about your decision, but you MUST end with the exact format below.

CRITICAL: Your response MUST include these lines using square brackets:
[CHOICE: CHOICE_C] or [CHOICE: CHOICE_D]
[CONFIDENCE: 0-100]"""
        
        result_q2 = await client.query(
            model=model,
            prompt=prompt_q2,
            temperature=self.config['execution']['temperature'],
            max_tokens=self.config['execution']['max_tokens']
        )
        
        parsed_q2 = parser.parse(result_q2['raw'])
        choice_2 = parsed_q2.get('choice', None)
        
        # Determine violation pattern
        violated = False
        pattern = None
        if choice_1 and choice_2:
            # Independence axiom violation: A+C or B+D
            # A+C = chose certainty in Q1, then higher-risk EV in Q2 (inconsistent)
            # B+D = chose risky EV in Q1, then certainty in Q2 (inconsistent)
            if choice_1 == 'CHOICE_A' and choice_2 == 'CHOICE_C':
                violated = True
                pattern = 'A+C'  # Certainty effect violation
            elif choice_1 == 'CHOICE_B' and choice_2 == 'CHOICE_D':
                violated = True
                pattern = 'B+D'  # Reverse certainty effect
            elif choice_1 == 'CHOICE_A' and choice_2 == 'CHOICE_D':
                pattern = 'A+D'  # Consistent risk-averse
            elif choice_1 == 'CHOICE_B' and choice_2 == 'CHOICE_C':
                pattern = 'B+C'  # Consistent EV-maximizing
        
        if parsed_q1['parse_success'] and parsed_q2['parse_success']:
            self.successful_calls += 2
        
        # Save to allais_pairs table
        self.db.insert_allais_pair({
            'model': model,
            'variant': 'standard',
            'stake_level': stake['id'],
            'scenario_id': scenario['id'],
            'choice_1': choice_1,
            'choice_2': choice_2,
            'violated': violated,
            'pattern': pattern,
            'metadata': {
                'trial': trial,
                'stake_values': stake,
                'q1_confidence': parsed_q1.get('confidence'),
                'q2_confidence': parsed_q2.get('confidence')
            }
        })
        
        logger.info(f"✓ Allais/{model}/stake={stake['id']}/trial_{trial}: {pattern or 'PARSE_FAIL'}")
        return {choice_1, choice_2, violated, pattern}
    
    async def _run_ultimatum_proposer(self, client, model, amount, scenario, trial):
        """Run Ultimatum Game as proposer."""
        # Check if exists
        if self.db.check_trial_exists(
            experiment='ultimatum',
            model=model,
            condition='proposer',
            group_size=2,
            stake_level=amount['id'],
            incentive_structure='proposer',
            scenario_id=scenario['id'],
            agent_id=0
        ):
            logger.info(f"⊙ Ultimatum/{model}/proposer/amt={amount['id']}/trial_{trial}: SKIP (exists)")
            self.successful_calls += 1
            return
        
        scenario_id = f"ultimatum_{scenario['id']}"
        template = self.scenarios.get(scenario_id)
        
        if not template:
            logger.error(f"Missing template: {scenario_id}")
            return
        
        # Extract proposer section from template
        total = amount['value']
        max_offer = total // 2  # Can't offer more than half to each person (N-1 = 1)
        fair_each = total // 2
        
        prompt = self._fill_template(template, {
            'n': 2,
            'n_others': 1,
            'total': total,
            'max_offer': total,  # For single responder, can offer 0 to total
            'fair_each': fair_each
        })
        
        # Extract only PROPOSER MODE section
        if "PROPOSER MODE:" in prompt:
            prompt = prompt.split("RESPONDER MODE:")[0]
        
        result = await client.query(
            model=model,
            prompt=prompt,
            temperature=self.config['execution']['temperature'],
            max_tokens=self.config['execution']['max_tokens']
        )
        
        # Parse response - extract OFFER_X
        parser = create_parser('ultimatum_proposer', self.config['experiments']['ultimatum'])
        parsed = parser.parse(result['raw'])
        
        if parsed['parse_success']:
            self.successful_calls += 1
        else:
            self.failed_parses.append({
                'trial_id': trial,
                'model': model,
                'response': result['raw']
            })
        
        # Calculate offer percentage
        offer_amount = parsed.get('numeric_value', 0)
        offer_pct = (offer_amount / total * 100) if total > 0 else 0
        
        # Save to trials table
        self.db.insert_trial({
            'experiment': 'ultimatum',
            'condition': 'proposer',
            'model': model,
            'group_size': 2,
            'agent_id': 0,
            'stake_level': amount['id'],
            'stake_multiplier': amount['value'],
            'incentive_structure': 'proposer',
            'scenario_id': scenario['id'],
            'payoffs': {'total': total, 'offer': offer_amount, 'offer_pct': offer_pct},
            'prompt': prompt,
            'response_raw': result['raw'],
            'temperature': self.config['execution']['temperature'],
            'latency_ms': result['latency'],
            **parsed
        })
        
        logger.info(f"✓ Ultimatum/{model}/proposer/amt={amount['id']}/trial_{trial}: ${offer_amount} ({offer_pct:.0f}%)")
        return parsed
    
    async def _run_ultimatum_responder(self, client, model, amount, offer_pct, scenario, trial):
        """Run Ultimatum Game as responder to fixed offer."""
        # Check if exists
        if self.db.check_trial_exists(
            experiment='ultimatum',
            model=model,
            condition='responder',
            group_size=2,
            stake_level=amount['id'],
            scenario_id=scenario['id'],
            agent_id=1,
            incentive_structure=f"offer_{offer_pct}pct"  # Use this field to track offer percentage
        ):
            logger.info(f"⊙ Ultimatum/{model}/responder/amt={amount['id']}/offer={offer_pct}%/trial_{trial}: SKIP (exists)")
            self.successful_calls += 1
            return
        
        scenario_id = f"ultimatum_{scenario['id']}"
        template = self.scenarios.get(scenario_id)
        
        if not template:
            logger.error(f"Missing template: {scenario_id}")
            return
        
        total = amount['value']
        offer_amount = int(total * offer_pct / 100)
        proposer_keeps = total - offer_amount
        
        # Create responder prompt
        prompt = f"""You are one of 2 business partners. Another partner has proposed how to split ${total} in profits.

RESPONDER MODE:
They propose to give you ${offer_amount} and keep ${proposer_keeps} for themselves.

Do you ACCEPT this split (everyone gets their share) or REJECT it (both get nothing)?

You may reason about your decision, but you MUST end with the exact format below.

CRITICAL: Your response MUST include these lines using square brackets:
[CHOICE: ACCEPT] or [CHOICE: REJECT]
[CONFIDENCE: 0-100]

Example response:
This offer is fair because...
[CHOICE: ACCEPT]
[CONFIDENCE: 85]"""
        
        result = await client.query(
            model=model,
            prompt=prompt,
            temperature=self.config['execution']['temperature'],
            max_tokens=self.config['execution']['max_tokens']
        )
        
        # Parse response
        parser = create_parser('ultimatum_responder', self.config['experiments']['ultimatum'])
        parsed = parser.parse(result['raw'])
        
        if parsed['parse_success']:
            self.successful_calls += 1
        else:
            self.failed_parses.append({
                'trial_id': trial,
                'model': model,
                'response': result['raw']
            })
        
        # Save to trials table
        self.db.insert_trial({
            'experiment': 'ultimatum',
            'condition': 'responder',
            'model': model,
            'group_size': 2,
            'agent_id': 1,
            'stake_level': amount['id'],
            'stake_multiplier': amount['value'],
            'incentive_structure': f"offer_{offer_pct}pct",
            'scenario_id': scenario['id'],
            'payoffs': {'total': total, 'offer': offer_amount, 'offer_pct': offer_pct, 'proposer_keeps': proposer_keeps},
            'prompt': prompt,
            'response_raw': result['raw'],
            'temperature': self.config['execution']['temperature'],
            'latency_ms': result['latency'],
            **parsed
        })
        
        logger.info(f"✓ Ultimatum/{model}/responder/amt={amount['id']}/offer={offer_pct}%/trial_{trial}: {parsed.get('choice', 'FAIL')}")
        return parsed
    
    def _create_allais_tasks(self, client, models, config) -> List:
        """Create Allais Paradox tasks - paired choices testing independence axiom."""
        tasks = []
        
        stake_levels = config['stake_levels']
        scenarios = config['scenarios']
        trials_total = config['trials_per_condition']
        trials_per_stake = trials_total // len(stake_levels)
        
        for model_name, model_id in models.items():
            trial_num = 0
            
            # Cycle through stake levels
            for stake in stake_levels:
                for i in range(trials_per_stake):
                    # Random scenario
                    scenario = random.choice(scenarios)
                    
                    tasks.append(self._run_allais(
                        client, model_id, stake, scenario, trial_num
                    ))
                    self.total_calls += 2  # Two questions per trial
                    trial_num += 1
        
        return tasks
    
    def _create_ultimatum_tasks(self, client, models, config) -> List:
        """Create Ultimatum Game tasks - proposer and responder modes."""
        tasks = []
        
        total_amounts = config['total_amounts']
        scenarios = config['scenarios']
        test_offers = config['test_offer_percentages']
        proposer_trials = config['sampling']['proposer_trials']
        responder_trials = config['sampling']['responder_trials']
        
        for model_name, model_id in models.items():
            trial_num = 0
            
            # PROPOSER MODE: Cycle through amounts
            proposer_per_amount = proposer_trials // len(total_amounts)
            for amount in total_amounts:
                for i in range(proposer_per_amount):
                    scenario = random.choice(scenarios)
                    
                    tasks.append(self._run_ultimatum_proposer(
                        client, model_id, amount, scenario, trial_num
                    ))
                    self.total_calls += 1
                    trial_num += 1
            
            # RESPONDER MODE: Test different offer percentages
            # Cycle through amounts × offer percentages
            for amount in total_amounts:
                for offer_pct in test_offers:
                    scenario = random.choice(scenarios)
                    
                    tasks.append(self._run_ultimatum_responder(
                        client, model_id, amount, offer_pct, scenario, trial_num
                    ))
                    self.total_calls += 1
                    trial_num += 1
        
        return tasks
    
    def _create_stag_hunt_tasks(self, client, models, config) -> List:
        """Create Stag Hunt tasks with balanced sampling."""
        tasks = []

        stake_levels = config['stake_levels']
        incentive_structures = config['incentive_structures']
        group_sizes = config['group_sizes']
        scenarios_dir = config.get('scenarios_dir', 'scenarios_stag_hunt')
        scenario_files = config['scenarios']

        trials_per_group = config['trials_per_condition']
        trials_per_stake = trials_per_group // len(stake_levels)

        # Load scenario templates
        scenarios = []
        for filename in scenario_files:
            filepath = Path(scenarios_dir) / filename
            if filepath.exists():
                with open(filepath) as f:
                    scenarios.append({
                        'id': filename.replace('.txt', ''),
                        'template': f.read()
                    })
            else:
                logger.warning(f"Stag Hunt scenario not found: {filepath}")

        if not scenarios:
            logger.error("No Stag Hunt scenarios loaded!")
            return tasks

        for model_name, model_id in models.items():
            for group_size in group_sizes:
                trial_num = 0

                # Balanced sampling: cycle through stake levels
                for stake_name, stake in stake_levels.items():
                    for i in range(trials_per_stake):
                        # Cycle through incentive structures
                        incentive_names = list(incentive_structures.keys())
                        incentive_name = incentive_names[i % len(incentive_names)]
                        incentive = incentive_structures[incentive_name]

                        # Cycle through scenarios
                        scenario = scenarios[i % len(scenarios)]

                        # Calculate payoffs
                        payoffs = self._calculate_sh_payoffs(stake, incentive, stake_name, incentive_name)

                        # Create task
                        if group_size == 1:
                            tasks.append(self._run_sh_single(
                                client, model_id, payoffs, scenario,
                                stake_name, incentive_name, trial_num
                            ))
                            self.total_calls += 1
                        else:
                            tasks.append(self._run_sh_multi(
                                client, model_id, group_size, payoffs, scenario,
                                stake_name, incentive_name, trial_num
                            ))
                            self.total_calls += group_size

                        trial_num += 1

        return tasks

    def _calculate_sh_payoffs(self, stake: Dict, incentive: Dict, stake_name: str, incentive_name: str) -> Dict:
        """Calculate Stag Hunt payoffs from stake × incentive structure.

        Stag Hunt structure: R > T >= P > S
        - R = mutual coordination reward (cc)
        - T = temptation to defect (but T < R in Stag Hunt!)
        - P = mutual defection punishment (dd)
        - S = sucker's payoff when you coordinated alone
        """
        base = stake['base']
        multiplier = stake['multiplier']
        base_cc = base * multiplier

        return {
            'cc': int(base_cc),  # Mutual coordination (Stag, Stag) - highest
            'dd': int(base_cc * incentive['punishment_ratio']),  # Mutual defection (Hare, Hare)
            'cd_sucker': int(base_cc * incentive['sucker_ratio']),  # You coordinated, they didn't
            'cd_defector': int(base_cc * incentive['temptation_ratio']),  # You defected, they coordinated (T < R in Stag Hunt)
            'stake_level': stake_name,
            'stake_multiplier': multiplier,
            'incentive_structure': incentive_name,
            'temptation_ratio': incentive['temptation_ratio'],
            'punishment_ratio': incentive['punishment_ratio'],
            'sucker_ratio': incentive['sucker_ratio']
        }

    async def _run_sh_single(self, client, model, payoffs, scenario,
                             stake_name, incentive_name, trial):
        """Run single-agent Stag Hunt (predicting others)."""
        if self.db.check_trial_exists(
            experiment='stag_hunt',
            model=model,
            condition='single',
            group_size=1,
            stake_level=stake_name,
            incentive_structure=incentive_name,
            scenario_id=scenario['id'],
            agent_id=0
        ):
            logger.info(f"⊙ SH/single/{model}/N=1/trial_{trial}: SKIP (exists)")
            self.successful_calls += 1
            return

        template = scenario['template']

        # Generate opponent text for single agent
        opponent_text = "Imagine other participants are making their choices simultaneously."

        prompt = self._fill_template(template, {
            'n': 1,
            'opponent_text': opponent_text,
            'cc': payoffs['cc'],
            'dd': payoffs['dd'],
            'cd': payoffs['cd_sucker'],
            'dc': payoffs['cd_defector']
        })

        result = await client.query(
            model=model,
            prompt=prompt,
            temperature=self.config['execution']['temperature'],
            max_tokens=self.config['execution']['max_tokens']
        )

        # Parse response
        parser = create_parser('stag_hunt', {})
        parsed = parser.parse(result['raw'])

        if parsed['parse_success']:
            self.successful_calls += 1
        else:
            self.failed_parses.append({
                'trial_id': trial,
                'model': model,
                'response': result['raw']
            })

        # Save to database
        self.db.insert_trial({
            'experiment': 'stag_hunt',
            'condition': 'single',
            'model': model,
            'group_size': 1,
            'stake_level': stake_name,
            'stake_multiplier': payoffs['stake_multiplier'],
            'incentive_structure': incentive_name,
            'scenario_id': scenario['id'],
            'temptation_ratio': payoffs['temptation_ratio'],
            'punishment_ratio': payoffs['punishment_ratio'],
            'sucker_ratio': payoffs['sucker_ratio'],
            'payoffs': payoffs,
            'prompt': prompt,
            'response_raw': result['raw'],
            'temperature': self.config['execution']['temperature'],
            'latency_ms': result['latency'],
            **parsed
        })

        logger.info(f"✓ SH/single/{model}/N=1/trial_{trial}: {parsed.get('choice', 'FAIL')}")
        return parsed

    async def _run_sh_multi(self, client, model, n_agents, payoffs, scenario,
                            stake_name, incentive_name, trial):
        """Run multi-agent Stag Hunt."""
        all_exist = all(
            self.db.check_trial_exists(
                experiment='stag_hunt',
                model=model,
                condition='multi',
                group_size=n_agents,
                stake_level=stake_name,
                incentive_structure=incentive_name,
                scenario_id=scenario['id'],
                agent_id=i
            )
            for i in range(n_agents)
        )

        if all_exist:
            logger.info(f"⊙ SH/multi/{model}/N={n_agents}/trial_{trial}: SKIP (exists)")
            self.successful_calls += n_agents
            return

        template = scenario['template']

        # Generate opponent text
        if n_agents == 2:
            opponent_text = "You are making this decision with 1 other participant."
        else:
            opponent_text = f"You are making this decision with {n_agents - 1} other participants."

        prompt = self._fill_template(template, {
            'n': n_agents,
            'opponent_text': opponent_text,
            'cc': payoffs['cc'],
            'dd': payoffs['dd'],
            'cd': payoffs['cd_sucker'],
            'dc': payoffs['cd_defector']
        })

        # Query all agents in parallel
        agent_tasks = [
            client.query(
                model=model,
                prompt=prompt,
                temperature=self.config['execution']['temperature'],
                max_tokens=self.config['execution']['max_tokens']
            )
            for _ in range(n_agents)
        ]

        results = await asyncio.gather(*agent_tasks, return_exceptions=True)

        # Parse all responses
        parser = create_parser('stag_hunt', {})

        decisions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Agent {i} error: {result}")
                decisions.append('SOLO')  # Default to safe choice on error
                continue

            parsed = parser.parse(result['raw'])

            if parsed['parse_success']:
                self.successful_calls += 1
                decisions.append(parsed['choice'])
            else:
                self.failed_parses.append({
                    'trial_id': f"{trial}_agent{i}",
                    'model': model,
                    'response': result['raw']
                })
                decisions.append('SOLO')

        # Save aggregated game
        coordination_rate = sum(1 for d in decisions if d == 'COORDINATE') / len(decisions)

        self.db.insert_multi_agent_game({
            'experiment': 'stag_hunt',
            'model': model,
            'group_size': n_agents,
            'stake_level': stake_name,
            'incentive_structure': incentive_name,
            'scenario_id': scenario['id'],
            'payoffs': payoffs,
            'agent_decisions': decisions,
            'metadata': {'trial': trial}
        })

        logger.info(f"✓ SH/multi/{model}/N={n_agents}/trial_{trial}: {coordination_rate*100:.0f}% coordinate")
        return decisions

    def _create_framing_tasks(self, client, models, config) -> List:
        """Create Framing tasks."""
        tasks = []
        # Implement similar pattern...
        return tasks


async def main():
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")
    
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config_v2.yaml"
    
    runner = ExperimentRunnerV2(config_file)
    await runner.run_all(api_key)


if __name__ == "__main__":
    asyncio.run(main())
