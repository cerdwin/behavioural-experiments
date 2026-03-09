#!/usr/bin/env python3
"""Run variance explained progression analysis"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("VARIANCE EXPLAINED PROGRESSION (LOMO Analysis)")
print("=" * 100)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'analysis' / 'lomo' / 'results_test2' / 'prepared_data.csv'
EMBEDDING_CACHE = PROJECT_ROOT / 'analysis' / 'embeddings' / 'results' / 'openrouter_embeddings_cache.json'
OUTPUT_DIR = Path(__file__).parent / 'results'

MODEL_MAP = {
    'anthropic/claude-3.7-sonnet': 'Claude_3.7',
    'anthropic/claude-haiku-4.5': 'Haiku_4.5',
    'deepseek/deepseek-v3.2': 'DeepSeek',
    'openai/gpt-5.2': 'GPT_5.2',
    'openai/o4-mini': 'O4_mini',
    'google/gemini-2.5-pro': 'Gemini_2.5_Pro',
    'google/gemini-3-pro-preview': 'Gemini_3_Pro'
}

# Load data
df = pd.read_csv(DATA_FILE)
df['model_name'] = df['model'].map(MODEL_MAP)
df['coop_pct'] = df['coop_rate']
df['scenario_key'] = df['scenario'].str.lower().str.replace(' ', '_')

with open(EMBEDDING_CACHE, 'r') as f:
    emb_cache = json.load(f)

# Create scenario-level dataframe
scenarios = df['scenario_key'].unique()
valid_scenarios = [s for s in scenarios if s in emb_cache]
scenario_df = df.groupby('scenario_key').agg({
    'R': 'first', 'S': 'first', 'C': 'first',
    'coop_pct': 'mean', 'scenario_name': 'first'
}).reset_index()
scenario_df.columns = ['scenario_key', 'R', 'S', 'C', 'mean_coop', 'scenario_name']
scenario_df = scenario_df[scenario_df['scenario_key'].isin(valid_scenarios)]

# Get embeddings and compute PCA
embeddings = np.array([emb_cache[s] for s in scenario_df['scenario_key']])
pca = PCA(n_components=15)
pc_scores = pca.fit_transform(embeddings)
var_explained = pca.explained_variance_ratio_

for i in range(15):
    scenario_df[f'PC{i+1}'] = pc_scores[:, i]

pc_cols = [f'PC{i+1}' for i in range(15)]

# Define feature sets
feature_sets = {
    'R/S/C only': ['R', 'S', 'C'],
    'PC1-3': pc_cols[:3],
    'PC1-5': pc_cols[:5],
    'All PCs (15)': pc_cols,
    'R/S/C + PC1-3': ['R', 'S', 'C'] + pc_cols[:3],
    'R/S/C + All PCs': ['R', 'S', 'C'] + pc_cols
}

models = sorted(df['model_name'].dropna().unique())
n_cal = 10
n_sim = 30

print("\nRunning LOMO for each feature set...")

results = []
for name, feat_cols in feature_sets.items():
    print(f"  Processing: {name}...")
    r2_scores = []

    for held_out in models:
        train_df = df[df['model_name'] != held_out].copy()
        test_df = df[df['model_name'] == held_out].copy()

        train_merged = train_df.merge(scenario_df[['scenario_key'] + feat_cols],
                                       on='scenario_key', how='inner', suffixes=('', '_sc'))
        test_merged = test_df.merge(scenario_df[['scenario_key'] + feat_cols],
                                     on='scenario_key', how='inner', suffixes=('', '_sc'))

        if len(test_merged) < 15 or len(train_merged) < 50:
            continue

        actual_cols = [c if c in train_merged.columns else c + '_sc' for c in feat_cols]
        actual_cols = [c for c in actual_cols if c in train_merged.columns]

        X_train = train_merged[actual_cols].values
        y_train = train_merged['coop_pct'].values
        X_test = test_merged[actual_cols].values
        y_test = test_merged['coop_pct'].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        ridge = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
        ridge.fit(X_train_s, y_train)

        for sim in range(n_sim):
            np.random.seed(sim)
            indices = np.random.permutation(len(test_merged))
            cal_idx = indices[:n_cal]
            eval_idx = indices[n_cal:]

            if len(eval_idx) < 5:
                continue

            y_pred_cal = ridge.predict(X_test_s[cal_idx])
            adj = np.mean(y_test[cal_idx] - y_pred_cal)
            y_pred_eval = ridge.predict(X_test_s[eval_idx]) + adj

            r2_scores.append(r2_score(y_test[eval_idx], y_pred_eval))

    if r2_scores:
        results.append({
            'Feature Set': name,
            'R2': np.mean(r2_scores),
            'R2_std': np.std(r2_scores),
            'n_features': len(feat_cols)
        })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / 'variance_explained.csv', index=False)

print("\n" + "=" * 100)
print("VARIANCE EXPLAINED PROGRESSION")
print("=" * 100)
print(f"\n{'Feature Set':<20} {'R2':>10} {'Std':>10} {'n_feat':>10}")
print("-" * 55)
for _, row in results_df.iterrows():
    print(f"{row['Feature Set']:<20} {row['R2']:>10.3f} {row['R2_std']:>10.3f} {row['n_features']:>10}")
print("-" * 55)

# Key findings
rsc_r2 = results_df[results_df['Feature Set'] == 'R/S/C only']['R2'].values[0]
pc13_r2 = results_df[results_df['Feature Set'] == 'PC1-3']['R2'].values[0]
combined_r2 = results_df[results_df['Feature Set'] == 'R/S/C + All PCs']['R2'].values[0]

print(f"\nKey Findings:")
print(f"  - R/S/C only: R² = {rsc_r2:.3f}")
print(f"  - PC1-3 interpretable: R² = {pc13_r2:.3f}")
print(f"  - Combined (R/S/C + All PCs): R² = {combined_r2:.3f}")
print(f"  - Improvement from embeddings: +{combined_r2 - rsc_r2:.3f}")
print(f"\nSaved: variance_explained.csv")
