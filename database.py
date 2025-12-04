import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

class Database:
    def __init__(self, db_path: str = "results.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        
    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                trial_id TEXT PRIMARY KEY,
                timestamp TEXT,
                experiment TEXT,
                condition TEXT,
                variant TEXT,
                model TEXT,
                n_agents INTEGER,
                temperature REAL,
                prompt TEXT,
                response_raw TEXT,
                response_parsed TEXT,
                parse_success INTEGER,
                latency_ms INTEGER,
                error TEXT,
                metadata TEXT
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS multi_agent_games (
                game_id TEXT PRIMARY KEY,
                timestamp TEXT,
                experiment TEXT,
                condition TEXT,
                model TEXT,
                n_agents INTEGER,
                agent_decisions TEXT,
                payoffs TEXT,
                metadata TEXT
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS allais_pairs (
                pair_id TEXT PRIMARY KEY,
                timestamp TEXT,
                model TEXT,
                variant TEXT,
                choice_1 TEXT,
                choice_2 TEXT,
                violated INTEGER,
                metadata TEXT
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS iterated_games (
                game_id TEXT PRIMARY KEY,
                timestamp TEXT,
                experiment TEXT,
                model TEXT,
                n_rounds INTEGER,
                temperature REAL,
                agent_choices TEXT,
                agent_scores TEXT,
                final_score INTEGER,
                strategy_detected TEXT,
                metadata TEXT
            )
        """)
        
        self.conn.commit()
        
    def insert_trial(self, data: Dict[str, Any]) -> str:
        trial_id = str(uuid.uuid4())
        
        self.conn.execute("""
            INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trial_id,
            datetime.now().isoformat(),
            data.get('experiment', ''),
            data.get('condition', ''),
            data.get('variant', ''),
            data.get('model', ''),
            data.get('n_agents', 1),
            data.get('temperature', 0.7),
            data.get('prompt', ''),
            data.get('raw', ''),
            json.dumps(data.get('parsed')) if data.get('parsed') else None,
            1 if data.get('success') else 0,
            data.get('latency', 0),
            data.get('error', ''),
            json.dumps(data.get('metadata', {}))
        ))
        
        self.conn.commit()
        return trial_id
        
    def insert_multi_agent_game(self, data: Dict[str, Any]) -> str:
        game_id = str(uuid.uuid4())
        
        self.conn.execute("""
            INSERT INTO multi_agent_games VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_id,
            datetime.now().isoformat(),
            data.get('experiment', ''),
            data.get('condition', ''),
            data.get('model', ''),
            data.get('n_agents', 0),
            json.dumps(data.get('agent_decisions', [])),
            json.dumps(data.get('payoffs', [])),
            json.dumps(data.get('metadata', {}))
        ))
        
        self.conn.commit()
        return game_id
        
    def insert_allais_pair(self, data: Dict[str, Any]) -> str:
        pair_id = str(uuid.uuid4())
        
        violated = 0
        choice1 = data.get('choice_1', '')
        choice2 = data.get('choice_2', '')
        
        if choice1 and choice2:
            if (choice1 == 'A' and choice2 == 'D') or (choice1 == 'B' and choice2 == 'C'):
                violated = 1
        
        self.conn.execute("""
            INSERT INTO allais_pairs VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pair_id,
            datetime.now().isoformat(),
            data.get('model', ''),
            data.get('variant', ''),
            choice1,
            choice2,
            violated,
            json.dumps(data.get('metadata', {}))
        ))
        
        self.conn.commit()
        return pair_id
        
    def get_all_trials(self) -> List[Dict]:
        cursor = self.conn.execute("SELECT * FROM trials")
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]
        
    def get_multi_agent_games(self) -> List[Dict]:
        cursor = self.conn.execute("SELECT * FROM multi_agent_games")
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]
        
    def get_allais_pairs(self) -> List[Dict]:
        cursor = self.conn.execute("SELECT * FROM allais_pairs")
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]
    
    def insert_iterated_game(self, data: Dict[str, Any]) -> str:
        game_id = data.get('game_id', str(uuid.uuid4()))
        
        self.conn.execute("""
            INSERT INTO iterated_games VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_id,
            datetime.now().isoformat(),
            data.get('experiment', ''),
            data.get('model', ''),
            data.get('n_rounds', 0),
            data.get('temperature', 0.7),
            json.dumps(data.get('agent_choices', [])),
            json.dumps(data.get('agent_scores', [])),
            data.get('final_score', 0),
            data.get('strategy_detected', ''),
            json.dumps(data.get('metadata', {}))
        ))
        
        self.conn.commit()
        return game_id
    
    def get_iterated_games(self) -> List[Dict]:
        cursor = self.conn.execute("SELECT * FROM iterated_games")
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]
        
    def close(self):
        self.conn.close()
