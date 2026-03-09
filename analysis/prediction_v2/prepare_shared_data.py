#!/usr/bin/env python3
"""
Prepare shared data for all prediction approaches.
Runs PCA once and saves the resulting DataFrame, ensuring all scripts
use identical features. Eliminates PCA non-determinism across runs.
"""

import sys
sys.stdout = sys.stderr = open(sys.stdout.fileno(), mode='w', buffering=1)

import pandas as pd
import numpy as np
from pathlib import Path
import json

from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'analysis' / 'lomo' / 'results_test2' / 'prepared_data.csv'
EMBEDDING_CACHE = PROJECT_ROOT / 'analysis' / 'embeddings' / 'results' / 'openrouter_embeddings_cache.json'
OUTPUT_FILE = Path(__file__).parent / 'shared_prepared_data.csv'

MODEL_MAP = {
    'anthropic/claude-3.7-sonnet': 'Claude_3.7',
    'anthropic/claude-haiku-4.5': 'Haiku_4.5',
    'deepseek/deepseek-v3.2': 'DeepSeek',
    'openai/gpt-5.2': 'GPT_5.2',
    'openai/o4-mini': 'O4_mini',
    'google/gemini-2.5-pro': 'Gemini_2.5_Pro',
    'google/gemini-3-pro-preview': 'Gemini_3_Pro'
}

MODEL_TO_FAMILY = {
    'Claude_3.7': 'Anthropic', 'Haiku_4.5': 'Anthropic',
    'GPT_5.2': 'OpenAI', 'O4_mini': 'OpenAI',
    'Gemini_2.5_Pro': 'Google', 'Gemini_3_Pro': 'Google',
    'DeepSeek': 'DeepSeek'
}


def main():
    print("Preparing shared data for prediction_v2 scripts...")

    df = pd.read_csv(DATA_FILE)
    df['model_name'] = df['model'].map(MODEL_MAP)
    df['coop_pct'] = df['coop_rate']
    df['scenario_key'] = df['scenario'].str.lower().str.replace(' ', '_')
    df['family'] = df['model_name'].map(MODEL_TO_FAMILY)

    with open(EMBEDDING_CACHE, 'r') as f:
        emb_cache = json.load(f)

    # Filter to rows with embeddings
    valid_idx = [i for i in range(len(df)) if df.iloc[i]['scenario_key'] in emb_cache]
    df = df.iloc[valid_idx].reset_index(drop=True)

    # Filter to common scenarios (present in all models)
    n_models = df['model_name'].nunique()
    scenario_counts = df.groupby('scenario_key')['model_name'].nunique()
    common_scenarios = scenario_counts[scenario_counts == n_models].index
    df = df[df['scenario_key'].isin(common_scenarios)].reset_index(drop=True)

    # PCA on embeddings
    embeddings = np.array([emb_cache[df.iloc[i]['scenario_key']] for i in range(len(df))])
    unique_scenarios = df['scenario_key'].unique()
    unique_emb = np.array([emb_cache[s] for s in unique_scenarios])

    n_pca = 30
    n_pca_actual = min(n_pca, unique_emb.shape[1], len(unique_scenarios) - 1)
    pca = PCA(n_components=n_pca_actual, random_state=42)
    pca.fit(unique_emb)
    emb_pcs = pca.transform(embeddings)

    pc_cols = []
    for i in range(n_pca_actual):
        col = f'pc{i+1}'
        df[col] = emb_pcs[:, i]
        pc_cols.append(col)

    # Save columns needed by all scripts
    keep_cols = ['scenario', 'scenario_key', 'model', 'model_name', 'family',
                 'coop_rate', 'coop_pct', 'n_trials', 'R', 'S', 'C',
                 'quadrant', 'scenario_id'] + pc_cols
    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_csv(OUTPUT_FILE, index=False)

    print(f"  Saved: {OUTPUT_FILE}")
    print(f"  Shape: {df.shape[0]} rows, {len(keep_cols)} columns")
    print(f"  Models: {df['model_name'].nunique()}")
    print(f"  Scenarios: {df['scenario_key'].nunique()}")
    print(f"  PCs: {n_pca_actual}")
    print(f"  PCA explained variance (total): {pca.explained_variance_ratio_.sum():.4f}")


if __name__ == '__main__':
    main()
