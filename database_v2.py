import sqlite3
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class DatabaseV2:
    """Enhanced database with parameter tracking for systematic analysis."""
    
    def __init__(self, db_path: str = "results_v2.db", log_raw_responses: bool = True):
        self.db_path = db_path
        self.log_raw_responses = log_raw_responses
        self.logger = logging.getLogger(__name__)
        
        # Setup results directory structure
        self.results_dir = Path("results")
        self.raw_dir = self.results_dir / "raw_responses"
        self.logs_dir = self.results_dir / "logs"
        
        if log_raw_responses:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        # Enable WAL mode for better concurrent write performance
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self._create_tables()
        self._create_run_metadata()
        
    def _create_tables(self):
        # Main trials table with full parameter tracking
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                trial_id TEXT PRIMARY KEY,
                timestamp TEXT,
                
                -- Experiment info
                experiment TEXT,
                condition TEXT,
                model TEXT,
                
                -- Group dynamics
                group_size INTEGER,
                agent_id INTEGER,
                
                -- Numerical parameters (what we vary)
                stake_level TEXT,
                stake_multiplier REAL,
                incentive_structure TEXT,
                scenario_id TEXT,
                
                -- Game-theoretic properties
                temptation_ratio REAL,
                punishment_ratio REAL,
                sucker_ratio REAL,
                mpcr REAL,
                
                -- Actual payoffs used
                payoffs_json TEXT,
                
                -- Prompt and response
                prompt TEXT,
                response_raw TEXT,
                
                -- Parsed output
                choice TEXT,
                numeric_value REAL,
                confidence INTEGER,
                reasoning TEXT,
                
                -- Parsing metadata
                parse_success INTEGER,
                extraction_method TEXT,
                total_choice_occurrences INTEGER,
                
                -- Ablation parameters
                instruction_variant TEXT,
                opponent_type TEXT,
                
                -- Execution metadata
                temperature REAL,
                latency_ms INTEGER,
                error TEXT
            )
        """)
        
        # Multi-agent games (aggregated results)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS multi_agent_games (
                game_id TEXT PRIMARY KEY,
                timestamp TEXT,
                experiment TEXT,
                model TEXT,
                
                -- Parameters
                group_size INTEGER,
                stake_level TEXT,
                incentive_structure TEXT,
                scenario_id TEXT,
                
                -- Payoffs used
                payoffs_json TEXT,
                
                -- Aggregate results
                agent_decisions_json TEXT,
                agent_payoffs_json TEXT,
                
                -- Summary stats
                cooperation_rate REAL,
                avg_contribution REAL,
                total_pool REAL,
                
                -- Metadata
                metadata_json TEXT
            )
        """)
        
        # Allais pairs
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS allais_pairs (
                pair_id TEXT PRIMARY KEY,
                timestamp TEXT,
                model TEXT,
                variant TEXT,
                
                -- Parameters
                stake_level TEXT,
                scenario_id TEXT,
                
                -- Choices
                choice_1 TEXT,
                choice_2 TEXT,
                violated INTEGER,
                
                -- Pattern
                pattern TEXT,
                
                -- Ablation parameters
                instruction_variant TEXT,
                opponent_type TEXT,
                
                -- Metadata
                metadata_json TEXT
            )
        """)
        
        # Run metadata table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS run_metadata (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT,
                config_snapshot TEXT,
                models_used TEXT,
                git_commit TEXT,
                experiments_run TEXT,
                total_trials_planned INTEGER,
                status TEXT,
                notes TEXT
            )
        """)

        # Questionnaire responses table (stated preferences)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS questionnaire_responses (
                response_id TEXT PRIMARY KEY,
                timestamp TEXT,
                model TEXT,
                question_id TEXT,
                trial_num INTEGER,

                -- Raw response
                raw_response TEXT,

                -- Parsed output
                parsed_choice TEXT,
                numeric_value REAL,
                explanation TEXT,

                -- Parse metadata
                parse_method TEXT,
                confidence TEXT,

                -- Execution metadata
                temperature REAL,
                latency_ms INTEGER,
                error TEXT
            )
        """)
        
        # Create indexes for common queries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_trials_exp ON trials(experiment)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_trials_model ON trials(model)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_trials_group_size ON trials(group_size)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_trials_stake ON trials(stake_level)")
        
        self.conn.commit()
    
    def _create_run_metadata(self):
        """Create metadata entry for this run."""
        import subprocess
        
        # Get git commit if available
        try:
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except:
            git_commit = 'unknown'
        
        import uuid
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        self.conn.execute("""
            INSERT INTO run_metadata (run_id, timestamp, git_commit, status)
            VALUES (?, ?, ?, ?)
        """, (self.run_id, datetime.now().isoformat(), git_commit, 'running'))
        self.conn.commit()
        
        self.logger.info(f"Started run: {self.run_id} (git: {git_commit})")
    
    def update_run_metadata(self, **kwargs):
        """Update run metadata fields."""
        for key, value in kwargs.items():
            if key in ['config_snapshot', 'models_used', 'experiments_run']:
                value = json.dumps(value) if not isinstance(value, str) else value
            
            self.conn.execute(f"""
                UPDATE run_metadata SET {key} = ? WHERE run_id = ?
            """, (value, self.run_id))
        
        self.conn.commit()
        
    def insert_trial(self, data: Dict[str, Any]) -> str:
        """Insert single trial with all tracking parameters."""
        trial_id = str(uuid.uuid4())
        
        # Save raw response to file if enabled
        if self.log_raw_responses and data.get('response_raw'):
            exp = data.get('experiment', 'unknown')
            model = data.get('model', 'unknown').split('/')[-1]
            
            raw_file = self.raw_dir / f"{exp}_{model}_{trial_id[:8]}.txt"
            with open(raw_file, 'w') as f:
                f.write(f"Trial ID: {trial_id}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Model: {data.get('model')}\n")
                f.write(f"Experiment: {exp}\n")
                f.write(f"Group Size: {data.get('group_size')}\n")
                f.write(f"Stake: {data.get('stake_level')}\n")
                f.write(f"Scenario: {data.get('scenario_id')}\n")
                f.write(f"\n{'='*60}\nPROMPT:\n{'='*60}\n")
                f.write(data.get('prompt', ''))
                f.write(f"\n\n{'='*60}\nRESPONSE:\n{'='*60}\n")
                f.write(data.get('response_raw'))
                f.write(f"\n\n{'='*60}\nPARSED:\n{'='*60}\n")
                f.write(f"Choice: {data.get('choice')}\n")
                f.write(f"Numeric: {data.get('numeric_value')}\n")
                f.write(f"Confidence: {data.get('confidence')}\n")
                f.write(f"Parse Success: {data.get('parse_success')}\n")
                f.write(f"Method: {data.get('extraction_method')}\n")
        
        self.conn.execute("""
            INSERT INTO trials VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            trial_id,
            datetime.now().isoformat(),
            
            # Experiment
            data.get('experiment'),
            data.get('condition'),
            data.get('model'),
            
            # Group
            data.get('group_size', 1),
            data.get('agent_id', 0),
            
            # Parameters
            data.get('stake_level'),
            data.get('stake_multiplier'),
            data.get('incentive_structure'),
            data.get('scenario_id'),
            
            # Game properties
            data.get('temptation_ratio'),
            data.get('punishment_ratio'),
            data.get('sucker_ratio'),
            data.get('mpcr'),
            
            # Payoffs
            json.dumps(data.get('payoffs', {})),
            
            # Prompt/response
            data.get('prompt'),
            data.get('response_raw'),
            
            # Parsed
            data.get('choice'),
            data.get('numeric_value'),
            data.get('confidence'),
            data.get('reasoning'),
            
            # Parse metadata
            1 if data.get('parse_success') else 0,
            data.get('extraction_method'),
            data.get('total_choice_occurrences', 0),
            
            # Ablation
            data.get('instruction_variant'),
            data.get('opponent_type'),
            
            # Execution
            data.get('temperature', 0.7),
            data.get('latency_ms', 0),
            data.get('error')
        ))
        
        self.conn.commit()
        return trial_id
        
    def insert_multi_agent_game(self, data: Dict[str, Any]) -> str:
        """Insert multi-agent game results."""
        game_id = str(uuid.uuid4())
        
        decisions = data.get('agent_decisions', [])
        
        # Calculate summary stats
        if data['experiment'] == 'prisoner_dilemma':
            cooperation_rate = sum(1 for d in decisions if d == 'COOPERATE') / len(decisions) if decisions else 0
            avg_contribution = None
            total_pool = None
        elif data['experiment'] == 'public_goods':
            cooperation_rate = None
            avg_contribution = sum(decisions) / len(decisions) if decisions else 0
            total_pool = sum(decisions)
        elif data['experiment'] == 'chicken':
            cooperation_rate = sum(1 for d in decisions if d == 'CONCEDE') / len(decisions) if decisions else 0
            avg_contribution = None
            total_pool = None
        else:
            cooperation_rate = None
            avg_contribution = None
            total_pool = None
        
        self.conn.execute("""
            INSERT INTO multi_agent_games VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_id,
            datetime.now().isoformat(),
            data.get('experiment'),
            data.get('model'),
            
            data.get('group_size'),
            data.get('stake_level'),
            data.get('incentive_structure'),
            data.get('scenario_id'),
            
            json.dumps(data.get('payoffs', {})),
            json.dumps(decisions),
            json.dumps(data.get('agent_payoffs', [])),
            
            cooperation_rate,
            avg_contribution,
            total_pool,
            
            json.dumps(data.get('metadata', {}))
        ))
        
        self.conn.commit()
        return game_id
        
    def insert_allais_pair(self, data: Dict[str, Any]) -> str:
        """Insert Allais paradox pair."""
        pair_id = str(uuid.uuid4())

        choice1 = data.get('choice_1', '')
        choice2 = data.get('choice_2', '')

        # Determine violation - handle both A/B and CHOICE_A/CHOICE_B formats
        violated = 0
        pattern = ''
        if choice1 and choice2:
            pattern = f"{choice1}-{choice2}"
            # Normalize choices for comparison
            c1_norm = choice1.upper().replace('CHOICE_', '').replace('OPTION_', '')
            c2_norm = choice2.upper().replace('CHOICE_', '').replace('OPTION_', '')
            # A-C or B-D violates independence axiom (Allais paradox)
            # A-C: chose certainty in Q1, but higher EV gamble in Q2
            # B-D: chose higher EV gamble in Q1, but lower EV in Q2
            if (c1_norm == 'A' and c2_norm == 'C') or \
               (c1_norm == 'B' and c2_norm == 'D'):
                violated = 1

        # Save raw responses to file if enabled
        if self.log_raw_responses and (data.get('response_q1') or data.get('response_q2')):
            model = data.get('model', 'unknown').split('/')[-1]
            raw_file = self.raw_dir / f"allais_{model}_{pair_id[:8]}.txt"
            with open(raw_file, 'w') as f:
                f.write(f"Pair ID: {pair_id}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Model: {data.get('model')}\n")
                f.write(f"Stake: {data.get('stake_level')}\n")
                f.write(f"Scenario: {data.get('scenario_id')}\n")
                f.write(f"Instruction: {data.get('instruction_variant')}\n")
                f.write(f"\n{'='*60}\nQ1 PROMPT:\n{'='*60}\n")
                f.write(data.get('metadata', {}).get('q1_prompt', ''))
                f.write(f"\n\n{'='*60}\nQ1 RESPONSE:\n{'='*60}\n")
                f.write(data.get('response_q1', ''))
                f.write(f"\n\n{'='*60}\nQ2 PROMPT:\n{'='*60}\n")
                f.write(data.get('metadata', {}).get('q2_prompt', ''))
                f.write(f"\n\n{'='*60}\nQ2 RESPONSE:\n{'='*60}\n")
                f.write(data.get('response_q2', ''))
                f.write(f"\n\n{'='*60}\nPARSED:\n{'='*60}\n")
                f.write(f"Choice 1: {choice1}\n")
                f.write(f"Choice 2: {choice2}\n")
                f.write(f"Pattern: {pattern}\n")
                f.write(f"Violated: {violated}\n")

        self.conn.execute("""
            INSERT INTO allais_pairs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pair_id,
            datetime.now().isoformat(),
            data.get('model'),
            data.get('variant'),

            data.get('stake_level'),
            data.get('scenario_id'),

            choice1,
            choice2,
            violated,
            pattern,

            data.get('instruction_variant'),
            data.get('opponent_type'),

            json.dumps(data.get('metadata', {}))
        ))

        self.conn.commit()
        return pair_id
        
    def get_trials(self, experiment: Optional[str] = None, 
                   model: Optional[str] = None) -> List[Dict]:
        """Get trials with optional filtering."""
        query = "SELECT * FROM trials WHERE 1=1"
        params = []
        
        if experiment:
            query += " AND experiment = ?"
            params.append(experiment)
        if model:
            query += " AND model = ?"
            params.append(model)
        
        cursor = self.conn.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
        
    def get_multi_agent_games(self, experiment: Optional[str] = None) -> List[Dict]:
        """Get multi-agent games."""
        query = "SELECT * FROM multi_agent_games"
        if experiment:
            query += " WHERE experiment = ?"
            cursor = self.conn.execute(query, [experiment])
        else:
            cursor = self.conn.execute(query)
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
        
    def get_allais_pairs(self, model: Optional[str] = None) -> List[Dict]:
        """Get Allais pairs."""
        query = "SELECT * FROM allais_pairs"
        if model:
            query += " WHERE model = ?"
            cursor = self.conn.execute(query, [model])
        else:
            cursor = self.conn.execute(query)
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def check_trial_exists(self, experiment: str, model: str, condition: str,
                          group_size: int, stake_level: str,
                          incentive_structure: str, scenario_id: str,
                          agent_id: int = 0) -> bool:
        """Check if a specific trial already exists in database (any trial, not just successful)."""
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM trials
            WHERE experiment = ?
            AND model = ?
            AND condition = ?
            AND group_size = ?
            AND stake_level = ?
            AND incentive_structure = ?
            AND scenario_id = ?
            AND agent_id = ?
        """, (experiment, model, condition, group_size, stake_level,
              incentive_structure, scenario_id, agent_id))

        return cursor.fetchone()[0] > 0

    def count_trials_for_scenario(self, experiment: str, model: str, condition: str,
                                   group_size: int, scenario_id: str) -> int:
        """Count trials for a specific model/condition/scenario (ignoring stake/incentive since we cycle)."""
        # Count unique trials (by agent_id=0 to avoid double-counting multi-agent)
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM trials
            WHERE experiment = ?
            AND model = ?
            AND condition = ?
            AND group_size = ?
            AND scenario_id = ?
            AND agent_id = 0
        """, (experiment, model, condition, group_size, scenario_id))

        return cursor.fetchone()[0]
    
    def check_allais_pair_exists(self, model: str, stake_level: str, scenario_id: str) -> bool:
        """Check if an Allais paired choice already exists."""
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM allais_pairs
            WHERE model = ? 
            AND stake_level = ?
            AND scenario_id = ?
        """, (model, stake_level, scenario_id))
        
        return cursor.fetchone()[0] > 0
    
    def get_completed_trial_count(self, experiment: str, model: str, 
                                  group_size: int) -> int:
        """Get count of completed trials for a specific configuration."""
        cursor = self.conn.execute("""
            SELECT COUNT(DISTINCT scenario_id || stake_level || incentive_structure) 
            FROM trials 
            WHERE experiment = ? 
            AND model = ? 
            AND group_size = ?
            AND parse_success = 1
        """, (experiment, model, group_size))
        
        return cursor.fetchone()[0]

    def count_condition_trials(self, experiment: str, model: str,
                               condition_filters: Dict[str, Any]) -> int:
        """
        Count existing trials for a specific condition.

        Args:
            experiment: e.g., 'prisoner_dilemma', 'public_goods', 'ultimatum', 'allais'
            model: model name
            condition_filters: dict of column->value pairs to filter by
                e.g., {'scenario_id': 'business', 'stake_level': 'base', 'group_size': 2}

        Returns:
            Number of existing trials matching the condition
        """
        query = "SELECT COUNT(*) FROM trials WHERE experiment = ? AND model = ?"
        params = [experiment, model]

        for col, val in condition_filters.items():
            query += f" AND {col} = ?"
            params.append(val)

        cursor = self.conn.execute(query, params)
        return cursor.fetchone()[0]

    def count_allais_pairs(self, model: str, condition_filters: Dict[str, Any]) -> int:
        """
        Count existing Allais pairs for a specific condition.

        Args:
            model: model name
            condition_filters: dict of column->value pairs
                e.g., {'scenario_id': 'business', 'stake_level': 'base'}

        Returns:
            Number of existing pairs matching the condition
        """
        query = "SELECT COUNT(*) FROM allais_pairs WHERE model = ?"
        params = [model]

        for col, val in condition_filters.items():
            query += f" AND {col} = ?"
            params.append(val)

        cursor = self.conn.execute(query, params)
        return cursor.fetchone()[0]

    def insert_questionnaire_response(self, data: Dict[str, Any]) -> str:
        """Insert questionnaire response (stated preference)."""
        response_id = str(uuid.uuid4())

        # Save raw response to file if enabled
        if self.log_raw_responses and data.get('raw_response'):
            model = data.get('model', 'unknown').split('/')[-1]
            q_id = data.get('question_id', 'unknown')
            raw_file = self.raw_dir / f"questionnaire_{model}_{q_id}_{response_id[:8]}.txt"
            with open(raw_file, 'w') as f:
                f.write(f"Response ID: {response_id}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Model: {data.get('model')}\n")
                f.write(f"Question: {q_id}\n")
                f.write(f"Trial: {data.get('trial_num')}\n")
                f.write(f"\n{'='*60}\nRESPONSE:\n{'='*60}\n")
                f.write(data.get('raw_response', ''))
                f.write(f"\n\n{'='*60}\nPARSED:\n{'='*60}\n")
                f.write(f"Choice: {data.get('parsed_choice')}\n")
                f.write(f"Numeric: {data.get('numeric_value')}\n")
                f.write(f"Method: {data.get('parse_method')}\n")
                f.write(f"Confidence: {data.get('confidence')}\n")

        self.conn.execute("""
            INSERT INTO questionnaire_responses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            response_id,
            datetime.now().isoformat(),
            data.get('model'),
            data.get('question_id'),
            data.get('trial_num', 0),
            data.get('raw_response'),
            data.get('parsed_choice'),
            data.get('numeric_value'),
            data.get('explanation'),
            data.get('parse_method'),
            data.get('confidence'),
            data.get('temperature', 0.7),
            data.get('latency_ms', 0),
            data.get('error')
        ))

        self.conn.commit()
        return response_id

    def count_questionnaire_responses(self, model: str, question_id: str) -> int:
        """Count existing questionnaire responses for a model/question combo."""
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM questionnaire_responses
            WHERE model = ? AND question_id = ?
        """, (model, question_id))
        return cursor.fetchone()[0]

    def get_questionnaire_responses(self, model: Optional[str] = None,
                                     question_id: Optional[str] = None) -> List[Dict]:
        """Get questionnaire responses with optional filtering."""
        query = "SELECT * FROM questionnaire_responses WHERE 1=1"
        params = []

        if model:
            query += " AND model = ?"
            params.append(model)
        if question_id:
            query += " AND question_id = ?"
            params.append(question_id)

        cursor = self.conn.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def export_to_csv(self, experiment: str, output_file: str):
        """Export experiment data to CSV."""
        import csv
        
        trials = self.get_trials(experiment=experiment)
        
        if not trials:
            print(f"No data for {experiment}")
            return
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trials[0].keys())
            writer.writeheader()
            writer.writerows(trials)
        
        print(f"Exported {len(trials)} trials to {output_file}")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}
        
        cursor = self.conn.execute("SELECT COUNT(*) FROM trials")
        stats['total_trials'] = cursor.fetchone()[0]
        
        cursor = self.conn.execute("SELECT COUNT(*) FROM trials WHERE parse_success = 1")
        stats['successful_parses'] = cursor.fetchone()[0]
        
        cursor = self.conn.execute("SELECT experiment, COUNT(*) FROM trials GROUP BY experiment")
        stats['by_experiment'] = dict(cursor.fetchall())
        
        cursor = self.conn.execute("SELECT model, COUNT(*) FROM trials GROUP BY model")
        stats['by_model'] = dict(cursor.fetchall())
        
        if stats['total_trials'] > 0:
            stats['parse_success_rate'] = stats['successful_parses'] / stats['total_trials']
        else:
            stats['parse_success_rate'] = 0.0
        
        return stats
        
    def close(self):
        """Close database connection."""
        self.conn.close()


if __name__ == "__main__":
    # Test database
    db = DatabaseV2("test.db")
    
    # Insert test trial
    db.insert_trial({
        'experiment': 'prisoner_dilemma',
        'condition': 'multi',
        'model': 'anthropic/claude-3.7-sonnet',
        'group_size': 4,
        'stake_level': 'medium',
        'stake_multiplier': 2.0,
        'incentive_structure': 'standard',
        'scenario_id': 'business',
        'temptation_ratio': 1.67,
        'payoffs': {'cc': 600, 'dd': 200},
        'prompt': 'test prompt',
        'response_raw': '[CHOICE: COOPERATE]\n[CONFIDENCE: 75]',
        'choice': 'COOPERATE',
        'confidence': 75,
        'parse_success': True,
        'extraction_method': 'square_brackets'
    })
    
    stats = db.get_stats()
    print("Database stats:")
    for key, val in stats.items():
        print(f"  {key}: {val}")
    
    db.close()
    
    import os
    os.remove("test.db")
    print("\nTest passed ✓")
