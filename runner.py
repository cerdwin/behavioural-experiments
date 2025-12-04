import asyncio
import yaml
import json
import os
import uuid
import random
from typing import List, Dict, Any
from pathlib import Path
from openrouter_client import OpenRouterClient
from database import Database
from dotenv import load_dotenv

load_dotenv()

class ExperimentRunner:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.db = Database(self.config['output']['database'])
        self.prompts = self._load_prompts()
        self.total_calls = 0
        self.successful_calls = 0
        
    def _load_prompts(self) -> Dict:
        base_prompts = {}
        with open('prompts.json') as f:
            base_prompts.update(json.load(f))
        
        if Path('prompts_variations.json').exists():
            with open('prompts_variations.json') as f:
                base_prompts.update(json.load(f))
        
        return base_prompts
    
    def _get_prompt(self, prompt_name: str) -> str:
        prompt_data = self.prompts[prompt_name]
        if isinstance(prompt_data, dict):
            return prompt_data.get('prompt', '')
        return prompt_data
    
    def _select_variation(self, base_name: str) -> tuple[str, int]:
        """Select a random variation of an experiment prompt.
        Returns: (prompt_key, variation_number)
        """
        variations = [k for k in self.prompts.keys() if k.startswith(base_name)]
        if not variations:
            return (base_name, 0)
        
        selected = random.choice(variations)
        
        if '_v' in selected:
            var_num = int(selected.split('_v')[1].split('_')[0])
        else:
            var_num = 0
            
        return (selected, var_num)
            
    async def run_all(self):
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")
            
        async with OpenRouterClient(
            api_key=api_key,
            base_url=self.config['api']['base_url'],
            max_retries=self.config['execution']['max_retries']
        ) as client:
            if self.config['models']['discover']:
                print("Discovering models...")
                discovered = await client.discover_models(self.config['models']['families'])
                models = discovered if discovered else {
                    m.split('/')[-1]: m for m in self.config['models']['fallback']
                }
                print(f"Using models: {list(models.values())}")
            else:
                models = {m.split('/')[-1]: m for m in self.config['models']['fallback']}
                
            tasks = []
            
            for exp_name, exp_config in self.config['experiments'].items():
                if not exp_config.get('enabled', True):
                    continue
                    
                print(f"Setting up {exp_name}...")
                
                if exp_name == 'allais':
                    tasks.extend(self._create_allais_tasks(client, models, exp_config))
                elif exp_name == 'prisoner_dilemma':
                    tasks.extend(self._create_pd_tasks(client, models, exp_config))
                elif exp_name == 'public_goods':
                    tasks.extend(self._create_pg_tasks(client, models, exp_config))
                elif exp_name == 'ultimatum':
                    tasks.extend(self._create_ultimatum_tasks(client, models, exp_config))
                elif exp_name == 'framing':
                    tasks.extend(self._create_framing_tasks(client, models, exp_config))
                elif exp_name == 'iterated_pd':
                    tasks.extend(self._create_iterated_pd_tasks(client, models, exp_config))
                    
            print(f"\nTotal tasks: {len(tasks)}")
            print(f"Estimated API calls: {self.total_calls}")
            print("Starting execution...\n")
            
            semaphore = asyncio.Semaphore(
                self.config['api']['rate_limit']['concurrent_requests']
            )
            
            async def limited_task(task):
                async with semaphore:
                    return await task
                    
            results = await asyncio.gather(*[limited_task(t) for t in tasks], return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    print(f"Task failed: {result}")
                    import traceback
                    traceback.print_exception(type(result), result, result.__traceback__)
                    
            print(f"\nCompleted {self.successful_calls}/{self.total_calls} trials successfully")
            
    def _create_allais_tasks(self, client, models, config):
        tasks = []
        trials = self.config['execution']['trials_per_condition']
        
        for model_name, model_id in models.items():
            for variant in config['variants']:
                for trial in range(trials):
                    tasks.append(
                        self._run_allais_pair(
                            client, model_id, variant, trial
                        )
                    )
                    self.total_calls += 2
                    
        return tasks
        
    def _create_pd_tasks(self, client, models, config):
        tasks = []
        trials = self.config['execution']['trials_per_condition']
        
        for model_name, model_id in models.items():
            for n_agents in config['agent_counts']:
                if n_agents == 1:
                    for trial in range(trials):
                        prompt_key, var_num = self._select_variation('pd_single')
                        tasks.append(
                            self._run_single_agent(
                                client, model_id, 'prisoner_dilemma', 'single',
                                self._get_prompt(prompt_key), trial, var_num
                            )
                        )
                        self.total_calls += 1
                else:
                    for trial in range(trials):
                        tasks.append(
                            self._run_pd_multi(
                                client, model_id, n_agents, trial
                            )
                        )
                        self.total_calls += n_agents
                        
        return tasks
        
    def _create_pg_tasks(self, client, models, config):
        tasks = []
        trials = self.config['execution']['trials_per_condition']
        
        for model_name, model_id in models.items():
            for n_agents in config['agent_counts']:
                if n_agents == 1:
                    for trial in range(trials):
                        prompt_key, var_num = self._select_variation('pg_single')
                        tasks.append(
                            self._run_single_agent(
                                client, model_id, 'public_goods', 'single',
                                self._get_prompt(prompt_key), trial, var_num
                            )
                        )
                        self.total_calls += 1
                else:
                    for trial in range(trials):
                        tasks.append(
                            self._run_pg_multi(
                                client, model_id, n_agents, trial,
                                config['hyperparameters']
                            )
                        )
                        self.total_calls += n_agents
                        
        return tasks
        
    def _create_ultimatum_tasks(self, client, models, config):
        tasks = []
        trials = self.config['execution']['trials_per_condition']
        
        for model_name, model_id in models.items():
            if 'proposer' in config['modes']:
                for trial in range(trials):
                    prompt_key, var_num = self._select_variation('ultimatum_proposer')
                    tasks.append(
                        self._run_single_agent(
                            client, model_id, 'ultimatum', 'proposer',
                            self._get_prompt(prompt_key), trial, var_num
                        )
                    )
                    self.total_calls += 1
                    
            if 'responder' in config['modes']:
                for offer in config['test_offers']:
                    for trial in range(trials):
                        tasks.append(
                            self._run_ultimatum_responder(
                                client, model_id, offer, trial
                            )
                        )
                        self.total_calls += 1
                        
        return tasks
        
    def _create_framing_tasks(self, client, models, config):
        tasks = []
        trials = self.config['execution']['trials_per_condition']
        
        for model_name, model_id in models.items():
            for variant in config['variants']:
                for trial in range(trials):
                    prompt_key, var_num = self._select_variation(f'framing_{variant}')
                    tasks.append(
                        self._run_single_agent(
                            client, model_id, 'framing', variant,
                            self._get_prompt(prompt_key), trial, var_num
                        )
                    )
                    self.total_calls += 1
                    
        return tasks
        
    async def _run_single_agent(self, client, model, experiment, condition, prompt, trial, variation=0):
        result = await client.query(
            model=model,
            prompt=prompt,
            temperature=self.config['execution']['temperature'],
            max_tokens=self.config['execution']['max_tokens']
        )
        
        if result['success']:
            self.successful_calls += 1
            
        trial_data = {
            'experiment': experiment,
            'condition': condition,
            'model': model,
            'trial': trial,
            'n_agents': 1,
            'prompt': prompt,
            **result,
            'metadata': {'variation': variation}
        }
        
        self.db.insert_trial(trial_data)
        print(f"✓ {experiment}/{condition}/{model}/trial_{trial}")
        return trial_data
        
    async def _run_allais_pair(self, client, model, variant, trial):
        base_key = self._select_variation('allais_v')[0].replace('_1', '').replace('_2', '')
        
        if base_key not in self.prompts:
            base_key = 'allais_v1'
        
        if base_key.endswith('_v1') or base_key.endswith('_v2') or base_key.endswith('_v3') or base_key.endswith('_v4'):
            prompt_key_1 = base_key + '_1'
            prompt_key_2 = base_key + '_2'
            var_num = int(base_key.split('_v')[1]) if '_v' in base_key else 1
        else:
            prompt_key_1 = 'allais_1_no_history'
            prompt_key_2 = 'allais_2_no_history'
            var_num = 0
        
        result1 = await client.query(
            model=model,
            prompt=self._get_prompt(prompt_key_1),
            temperature=self.config['execution']['temperature']
        )
        
        result2 = await client.query(
            model=model,
            prompt=self._get_prompt(prompt_key_2),
            temperature=self.config['execution']['temperature']
        )
            
        if result1['success'] and result2['success']:
            self.successful_calls += 2
            
        choice1 = result1['parsed'].get('choice', '') if result1['parsed'] else ''
        choice2 = result2['parsed'].get('choice', '') if result2['parsed'] else ''
        
        pair_data = {
            'model': model,
            'variant': variant,
            'choice_1': choice1,
            'choice_2': choice2,
            'metadata': {
                'trial': trial,
                'variation': var_num,
                'both_successful': result1['success'] and result2['success']
            }
        }
        
        self.db.insert_allais_pair(pair_data)
        print(f"✓ allais/{variant}/{model}/trial_{trial}: {choice1}, {choice2}")
        return pair_data
        
    async def _run_pd_multi(self, client, model, n_agents, trial):
        prompt_key, var_num = self._select_variation('pd_multi')
        prompt = self._fill_template(prompt_key, {
            'n': n_agents,
            'example_n': n_agents - 1,
            'coop_total': 200 + (n_agents - 1) * 50,
            'defect_total': 350 + (n_agents - 1) * 50
        })
        
        agent_tasks = [
            client.query(
                model=model,
                prompt=prompt,
                temperature=self.config['execution']['temperature']
            )
            for _ in range(n_agents)
        ]
        
        results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        decisions = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"Agent {i} error: {r}")
                decisions.append('DEFECT')
            elif r.get('parsed') and isinstance(r['parsed'], dict):
                decisions.append(r['parsed'].get('choice', 'DEFECT'))
            else:
                print(f"Agent {i} parse failed. Raw: {r.get('raw', '')[:100]}")
                decisions.append('DEFECT')
        
        payoffs = self._calculate_pd_payoffs(decisions)
        
        self.successful_calls += sum(1 for r in results if not isinstance(r, Exception) and r.get('success'))
        
        game_data = {
            'experiment': 'prisoner_dilemma',
            'condition': 'multi',
            'model': model,
            'n_agents': n_agents,
            'agent_decisions': decisions,
            'payoffs': payoffs,
            'metadata': {
                'trial': trial,
                'variation': var_num,
                'success_count': sum(1 for r in results if not isinstance(r, Exception) and r.get('success')),
                'total_agents': len(results)
            }
        }
        
        self.db.insert_multi_agent_game(game_data)
        print(f"✓ prisoner_dilemma/multi/{model}/n={n_agents}/trial_{trial}: {decisions}")
        return game_data
        
    async def _run_pg_multi(self, client, model, n_agents, trial, hyperparams):
        endowment = hyperparams['endowment']
        multiplier = hyperparams['multiplier']
        
        pool_example = n_agents * 10
        multiplied = pool_example * multiplier
        share = multiplied / n_agents
        total = 10 + share
        
        prompt_key, var_num = self._select_variation('pg_multi')
        prompt = self._fill_template(prompt_key, {
            'n': n_agents,
            'pool': pool_example,
            'multiplied': int(multiplied),
            'share': f"{share:.1f}",
            'total': f"{total:.1f}"
        })
        
        agent_tasks = [
            client.query(
                model=model,
                prompt=prompt,
                temperature=self.config['execution']['temperature']
            )
            for _ in range(n_agents)
        ]
        
        results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        contributions = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"Agent {i} error: {r}")
                contributions.append(0)
            elif r.get('parsed') and isinstance(r['parsed'], dict) and 'contribution' in r['parsed']:
                contrib = r['parsed']['contribution']
                contributions.append(max(0, min(endowment, contrib)))
            else:
                print(f"Agent {i} parse failed. Raw: {r.get('raw', '')[:100]}")
                contributions.append(0)
                
        payoffs = self._calculate_pg_payoffs(contributions, n_agents, endowment, multiplier)
        
        self.successful_calls += sum(1 for r in results if not isinstance(r, Exception) and r.get('success'))
        
        game_data = {
            'experiment': 'public_goods',
            'condition': 'multi',
            'model': model,
            'n_agents': n_agents,
            'agent_decisions': contributions,
            'payoffs': payoffs,
            'metadata': {
                'trial': trial,
                'variation': var_num,
                'hyperparams': hyperparams,
                'success_count': sum(1 for r in results if not isinstance(r, Exception) and r.get('success')),
                'total_agents': len(results)
            }
        }
        
        self.db.insert_multi_agent_game(game_data)
        print(f"✓ public_goods/multi/{model}/n={n_agents}/trial_{trial}: {contributions}")
        return game_data
        
    async def _run_ultimatum_responder(self, client, model, offer, trial):
        keep = 100 - offer
        prompt_key, var_num = self._select_variation('ultimatum_responder')
        prompt = self._fill_template(prompt_key, {
            'offer': offer,
            'keep': keep
        })
        
        result = await client.query(
            model=model,
            prompt=prompt,
            temperature=self.config['execution']['temperature']
        )
        
        if result['success']:
            self.successful_calls += 1
            
        trial_data = {
            'experiment': 'ultimatum',
            'condition': 'responder',
            'variant': f'offer_{offer}',
            'model': model,
            'trial': trial,
            'n_agents': 1,
            'prompt': prompt,
            **result,
            'metadata': {'offer': offer, 'keep': keep, 'variation': var_num}
        }
        
        self.db.insert_trial(trial_data)
        choice = result['parsed'].get('choice', 'N/A') if result['parsed'] else 'N/A'
        print(f"✓ ultimatum/responder/{model}/offer_{offer}/trial_{trial}: {choice}")
        return trial_data
        
    def _fill_template(self, template_name: str, values: Dict) -> str:
        template = self._get_prompt(template_name)
        for key, value in values.items():
            template = template.replace(f'{{{key}}}', str(value))
        return template
        
    def _calculate_pd_payoffs(self, decisions: List[str]) -> List[int]:
        n = len(decisions)
        payoffs = []
        
        for decision in decisions:
            if decision == 'COOPERATE':
                base = 200
            else:
                base = 350
                
            cooperators = sum(1 for d in decisions if d == 'COOPERATE')
            if decision == 'COOPERATE':
                cooperators -= 1
                
            payoff = base + (cooperators * 50)
            payoffs.append(payoff)
            
        return payoffs
        
    def _calculate_pg_payoffs(self, contributions: List[float], n: int, 
                             endowment: float, multiplier: float) -> List[float]:
        total_pool = sum(contributions)
        multiplied_pool = total_pool * multiplier
        share = multiplied_pool / n
        
        payoffs = [endowment - c + share for c in contributions]
        return payoffs
    
    async def _run_iterated_pd(self, client, model: str, n_rounds: int = 5, trial: int = 1):
        history = []
        scores = [0, 0]
        choices = [[], []]
        
        for round_num in range(1, n_rounds + 1):
            if round_num == 1:
                prompt1 = self._get_prompt('pd_iterated_round_1')
                prompt2 = self._get_prompt('pd_iterated_round_1')
            else:
                history_str_1 = "\n".join([
                    f"Round {r}: You: {choices[0][r-1]}, Them: {choices[1][r-1]}"
                    for r in range(1, round_num)
                ])
                history_str_2 = "\n".join([
                    f"Round {r}: You: {choices[1][r-1]}, Them: {choices[0][r-1]}"
                    for r in range(1, round_num)
                ])
                prompt1 = self._fill_template('pd_iterated_round_n', {
                    'round': round_num,
                    'history': history_str_1,
                    'your_score': scores[0],
                    'opp_score': scores[1]
                })
                prompt2 = self._fill_template('pd_iterated_round_n', {
                    'round': round_num,
                    'history': history_str_2,
                    'your_score': scores[1],
                    'opp_score': scores[0]
                })
            
            result1, result2 = await asyncio.gather(
                client.query(model=model, prompt=prompt1, temperature=self.config['execution']['temperature']),
                client.query(model=model, prompt=prompt2, temperature=self.config['execution']['temperature'])
            )
            
            choice1 = result1['parsed'].get('choice', 'DEFECT') if result1['success'] and result1['parsed'] else 'DEFECT'
            choice2 = result2['parsed'].get('choice', 'DEFECT') if result2['success'] and result2['parsed'] else 'DEFECT'
            
            choices[0].append(choice1)
            choices[1].append(choice2)
            
            if choice1 == 'COOPERATE' and choice2 == 'COOPERATE':
                scores[0] += 3
                scores[1] += 3
            elif choice1 == 'DEFECT' and choice2 == 'DEFECT':
                scores[0] += 1
                scores[1] += 1
            elif choice1 == 'COOPERATE':
                scores[1] += 5
            else:
                scores[0] += 5
        
        strategy = self._detect_strategy(choices[0])
        
        self.db.insert_iterated_game({
            'game_id': str(uuid.uuid4()),
            'experiment': 'iterated_pd',
            'model': model,
            'n_rounds': n_rounds,
            'temperature': self.config['execution']['temperature'],
            'agent_choices': choices,
            'agent_scores': scores,
            'final_score': scores[0],
            'strategy_detected': strategy,
            'metadata': {'trial': trial}
        })
        
        print(f"✓ iterated_pd/{model}/trial_{trial}: Agent1={scores[0]}, Agent2={scores[1]}, Strategy={strategy}")
        self.successful_calls += 2 * n_rounds
        return {'scores': scores, 'choices': choices, 'strategy': strategy}
    
    def _detect_strategy(self, choices: List[str]) -> str:
        if all(c == 'COOPERATE' for c in choices):
            return 'always-cooperate'
        if all(c == 'DEFECT' for c in choices):
            return 'always-defect'
        if len(choices) > 1 and choices[0] == 'COOPERATE':
            return 'cooperative'
        return 'mixed'
    
    def _create_iterated_pd_tasks(self, client, models, config):
        tasks = []
        trials = config.get('trials', self.config['execution']['trials_per_condition'])
        n_rounds = config.get('n_rounds', 5)
        
        for model_name, model_id in models.items():
            for trial in range(trials):
                tasks.append(
                    self._run_iterated_pd(client, model_id, n_rounds, trial)
                )
                self.total_calls += 2 * n_rounds
        
        return tasks

if __name__ == "__main__":
    import sys
    
    config_path = "config.yaml"
    if len(sys.argv) > 1:
        if sys.argv[1] == "--config" and len(sys.argv) > 2:
            config_path = sys.argv[2]
        else:
            config_path = sys.argv[1]
    
    runner = ExperimentRunner(config_path)
    asyncio.run(runner.run_all())
