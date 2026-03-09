import pandas as pd
import sqlite3
import json
from pathlib import Path


def _safe_json_loads(x):
    """Safely parse JSON from a database column that may contain NaN/None/non-string values.
    Always returns a dict."""
    if not isinstance(x, str):
        return {}
    try:
        result = json.loads(x)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}

def analyze(db_path: str = "results.db"):
    conn = sqlite3.connect(db_path)
    
    print("=" * 80)
    print("BEHAVIORAL EXPERIMENTS ANALYSIS")
    print("=" * 80)
    
    df_trials = pd.read_sql_query("SELECT * FROM trials", conn)
    df_games = pd.read_sql_query("SELECT * FROM multi_agent_games", conn)
    df_allais = pd.read_sql_query("SELECT * FROM allais_pairs", conn)
    
    if len(df_trials) == 0 and len(df_games) == 0 and len(df_allais) == 0:
        print("\nNo data found. Run experiments first.")
        return
    
    print(f"\nTotal single-agent trials: {len(df_trials)}")
    print(f"Total multi-agent games: {len(df_games)}")
    print(f"Total Allais pairs: {len(df_allais)}")
    
    if len(df_trials) > 0:
        parse_success = df_trials['parse_success'].mean()
        print(f"\nParse success rate: {parse_success:.1%}")
        
        avg_latency = df_trials['latency_ms'].mean()
        print(f"Average latency: {avg_latency:.0f}ms")
    
    print("\n" + "=" * 80)
    print("ALLAIS PARADOX")
    print("=" * 80)
    
    if len(df_allais) > 0:
        for model in df_allais['model'].unique():
            model_data = df_allais[df_allais['model'] == model]
            
            for variant in model_data['variant'].unique():
                variant_data = model_data[model_data['variant'] == variant]
                violation_rate = variant_data['violated'].mean()
                n = len(variant_data)
                
                print(f"\n{model} ({variant}):")
                print(f"  Violation rate: {violation_rate:.1%} (n={n})")
                
                choice1_counts = variant_data['choice_1'].value_counts()
                choice2_counts = variant_data['choice_2'].value_counts()
                print(f"  Choice 1: {choice1_counts.to_dict()}")
                print(f"  Choice 2: {choice2_counts.to_dict()}")
    
    print("\n" + "=" * 80)
    print("PRISONER'S DILEMMA")
    print("=" * 80)
    
    pd_trials = df_trials[df_trials['experiment'] == 'prisoner_dilemma']
    if len(pd_trials) > 0:
        pd_trials['parsed_data'] = pd_trials['response_parsed'].apply(
            _safe_json_loads
        )
        pd_trials['cooperated'] = pd_trials['parsed_data'].apply(
            lambda x: x.get('choice') == 'COOPERATE'
        )
        
        print("\nSingle-agent cooperation:")
        for model in pd_trials['model'].unique():
            model_data = pd_trials[pd_trials['model'] == model]
            coop_rate = model_data['cooperated'].mean()
            n = len(model_data)
            print(f"  {model}: {coop_rate:.1%} (n={n})")
    
    pd_games = df_games[df_games['experiment'] == 'prisoner_dilemma']
    if len(pd_games) > 0:
        print("\nMulti-agent cooperation:")
        for model in pd_games['model'].unique():
            model_data = pd_games[pd_games['model'] == model]
            
            for n_agents in sorted(model_data['n_agents'].unique()):
                n_data = model_data[model_data['n_agents'] == n_agents]
                
                all_decisions = []
                for decisions_json in n_data['agent_decisions']:
                    decisions = json.loads(decisions_json)
                    all_decisions.extend(decisions)
                
                coop_count = sum(1 for d in all_decisions if d == 'COOPERATE')
                coop_rate = coop_count / len(all_decisions) if all_decisions else 0
                n_games = len(n_data)
                
                print(f"  {model} (n={n_agents}): {coop_rate:.1%} (games={n_games})")
    
    print("\n" + "=" * 80)
    print("PUBLIC GOODS GAME")
    print("=" * 80)
    
    pg_trials = df_trials[df_trials['experiment'] == 'public_goods']
    if len(pg_trials) > 0:
        pg_trials['parsed_data'] = pg_trials['response_parsed'].apply(
            _safe_json_loads
        )
        pg_trials['contribution'] = pg_trials['parsed_data'].apply(
            lambda x: x.get('contribution', 0)
        )
        
        print("\nSingle-agent contributions:")
        for model in pg_trials['model'].unique():
            model_data = pg_trials[pg_trials['model'] == model]
            avg_contrib = model_data['contribution'].mean()
            n = len(model_data)
            print(f"  {model}: ${avg_contrib:.1f} / $20 (n={n})")
    
    pg_games = df_games[df_games['experiment'] == 'public_goods']
    if len(pg_games) > 0:
        print("\nMulti-agent contributions:")
        for model in pg_games['model'].unique():
            model_data = pg_games[pg_games['model'] == model]
            
            for n_agents in sorted(model_data['n_agents'].unique()):
                n_data = model_data[model_data['n_agents'] == n_agents]
                
                all_contributions = []
                for decisions_json in n_data['agent_decisions']:
                    contributions = json.loads(decisions_json)
                    all_contributions.extend(contributions)
                
                avg_contrib = sum(all_contributions) / len(all_contributions) if all_contributions else 0
                n_games = len(n_data)
                
                print(f"  {model} (n={n_agents}): ${avg_contrib:.1f} / $20 (games={n_games})")
    
    print("\n" + "=" * 80)
    print("ULTIMATUM GAME")
    print("=" * 80)
    
    ult_proposer = df_trials[
        (df_trials['experiment'] == 'ultimatum') & 
        (df_trials['condition'] == 'proposer')
    ]
    if len(ult_proposer) > 0:
        ult_proposer['parsed_data'] = ult_proposer['response_parsed'].apply(
            _safe_json_loads
        )
        ult_proposer['offer'] = ult_proposer['parsed_data'].apply(
            lambda x: x.get('offer', 0)
        )
        
        print("\nProposer offers:")
        for model in ult_proposer['model'].unique():
            model_data = ult_proposer[ult_proposer['model'] == model]
            avg_offer = model_data['offer'].mean()
            n = len(model_data)
            print(f"  {model}: ${avg_offer:.1f} / $100 (n={n})")
    
    ult_responder = df_trials[
        (df_trials['experiment'] == 'ultimatum') & 
        (df_trials['condition'] == 'responder')
    ]
    if len(ult_responder) > 0:
        ult_responder['parsed_data'] = ult_responder['response_parsed'].apply(
            _safe_json_loads
        )
        ult_responder['accepted'] = ult_responder['parsed_data'].apply(
            lambda x: x.get('choice') == 'ACCEPT'
        )
        
        print("\nResponder acceptance rates:")
        for model in ult_responder['model'].unique():
            model_data = ult_responder[ult_responder['model'] == model]
            
            for variant in sorted(model_data['variant'].unique()):
                variant_data = model_data[model_data['variant'] == variant]
                accept_rate = variant_data['accepted'].mean()
                n = len(variant_data)
                offer_amount = variant.split('_')[1]
                print(f"  {model} (offer=${offer_amount}): {accept_rate:.1%} (n={n})")
    
    print("\n" + "=" * 80)
    print("FRAMING EFFECT")
    print("=" * 80)
    
    framing = df_trials[df_trials['experiment'] == 'framing']
    if len(framing) > 0:
        framing['parsed_data'] = framing['response_parsed'].apply(
            _safe_json_loads
        )
        framing['choice'] = framing['parsed_data'].apply(
            lambda x: x.get('choice', '')
        )
        
        for model in framing['model'].unique():
            model_data = framing[framing['model'] == model]
            
            print(f"\n{model}:")
            for variant in ['gain', 'loss']:
                variant_data = model_data[model_data['condition'] == variant]
                if len(variant_data) > 0:
                    choice_counts = variant_data['choice'].value_counts()
                    n = len(variant_data)
                    print(f"  {variant}: {choice_counts.to_dict()} (n={n})")
    
    print("\n" + "=" * 80)
    print("EXPORTING DATA")
    print("=" * 80)
    
    df_trials.to_csv('results_trials.csv', index=False)
    print("✓ Exported results_trials.csv")
    
    if len(df_games) > 0:
        df_games.to_csv('results_games.csv', index=False)
        print("✓ Exported results_games.csv")
    
    if len(df_allais) > 0:
        df_allais.to_csv('results_allais.csv', index=False)
        print("✓ Exported results_allais.csv")
    
    conn.close()
    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze()
