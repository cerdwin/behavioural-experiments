#!/usr/bin/env python3
"""
Test if scenario text embeddings improve prediction beyond R/S/C features.

Since sentence-transformers isn't available, we use TF-IDF as a simpler
but reasonable alternative for capturing semantic content.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SCENARIOS_DIR = PROJECT_ROOT / 'scenarios_v2'
DATA_FILE = PROJECT_ROOT / 'analysis' / 'lomo' / 'results_test2' / 'prepared_data.csv'
OUTPUT_DIR = PROJECT_ROOT / 'analysis' / 'embeddings' / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_MAP = {
    'anthropic/claude-3.7-sonnet': 'Claude_3.7',
    'anthropic/claude-haiku-4.5': 'Haiku_4.5',
    'deepseek/deepseek-v3.2': 'DeepSeek',
    'openai/gpt-5.2': 'GPT_5.2',
    'openai/o4-mini': 'O4_mini',
    'google/gemini-2.5-pro': 'Gemini_2.5_Pro',
    'google/gemini-3-pro-preview': 'Gemini_3_Pro'
}

def load_scenario_texts():
    """Load all scenario texts from pd_v2_*.txt files"""
    scenarios = {}
    for f in SCENARIOS_DIR.glob('pd_v2_*.txt'):
        # Extract scenario name: pd_v2_arms_control.txt -> arms_control
        name = f.stem.replace('pd_v2_', '')
        with open(f, 'r') as file:
            scenarios[name] = file.read()
    return scenarios

def load_cooperation_data():
    """Load cooperation rates and R/S/C features"""
    df = pd.read_csv(DATA_FILE)
    df['model_name'] = df['model'].map(MODEL_MAP)
    df['coop_pct'] = df['coop_rate']
    return df

def create_embeddings(scenario_texts, n_components=20):
    """Create TF-IDF embeddings and reduce with SVD"""
    # TF-IDF on scenario texts
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1
    )

    scenario_names = list(scenario_texts.keys())
    texts = [scenario_texts[name] for name in scenario_names]

    tfidf_matrix = vectorizer.fit_transform(texts)

    # Reduce dimensions with SVD (like PCA for sparse matrices)
    n_components = min(n_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1])
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(tfidf_matrix)

    # Create DataFrame with scenario names as index
    emb_df = pd.DataFrame(
        embeddings,
        index=scenario_names,
        columns=[f'emb_{i}' for i in range(n_components)]
    )

    print(f"Created embeddings: {tfidf_matrix.shape[0]} scenarios → {n_components} dimensions")
    print(f"Variance explained by SVD: {svd.explained_variance_ratio_.sum():.1%}")

    return emb_df, vectorizer, svd

def evaluate_model(X, y, cv=10, alphas=[0.1, 1, 10, 100, 1000]):
    """Evaluate Ridge regression with cross-validation"""
    if len(X) < cv:
        cv = len(X) // 2

    model = RidgeCV(alphas=alphas, cv=cv)
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    return scores.mean(), scores.std()

def compute_metrics(y_true, y_pred):
    """Compute R², correlation, MAE, accuracy"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    corr = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_true) > 0 and np.std(y_pred) > 0 else 0
    mae = np.mean(np.abs(y_true - y_pred))
    acc = np.mean((y_true > 50) == (y_pred > 50)) * 100

    return {'R2': r2, 'corr': corr, 'MAE': mae, 'acc': acc}

def phase1_within_model(df, emb_df, n_components_list=[5, 10, 15, 20]):
    """Phase 1: Within-model comparison of R/S/C vs embeddings vs combined"""

    print("\n" + "=" * 100)
    print("PHASE 1: WITHIN-MODEL COMPARISON")
    print("=" * 100)
    print("Question: Do embeddings capture something R/S/C misses for each model?")

    results = []
    models = df['model_name'].unique()

    # Try different numbers of embedding components
    best_n_comp = 10  # Default

    for n_comp in n_components_list:
        emb_cols = [f'emb_{i}' for i in range(min(n_comp, emb_df.shape[1]))]

        print(f"\n--- Testing with {n_comp} embedding components ---")
        print(f"{'Model':<16} {'R/S/C':>10} {'Emb':>10} {'Combined':>10} {'Δ':>10}")
        print("-" * 60)

        for model in sorted(models):
            model_df = df[df['model_name'] == model].copy()

            # Match scenarios
            model_df['scenario_key'] = model_df['scenario'].str.lower().str.replace(' ', '_')
            matched = model_df[model_df['scenario_key'].isin(emb_df.index)]

            if len(matched) < 20:
                continue

            y = matched['coop_pct'].values
            X_rsc = matched[['R', 'S', 'C']].values
            X_emb = emb_df.loc[matched['scenario_key'], emb_cols].values
            X_combined = np.hstack([X_rsc, X_emb])

            # Standardize
            scaler = StandardScaler()
            X_rsc_scaled = scaler.fit_transform(X_rsc)
            X_emb_scaled = scaler.fit_transform(X_emb)
            X_combined_scaled = scaler.fit_transform(X_combined)

            # Evaluate
            r2_rsc, _ = evaluate_model(X_rsc_scaled, y, cv=5)
            r2_emb, _ = evaluate_model(X_emb_scaled, y, cv=5)
            r2_comb, _ = evaluate_model(X_combined_scaled, y, cv=5)

            delta = r2_comb - r2_rsc

            results.append({
                'model': model,
                'n_components': n_comp,
                'R2_rsc': r2_rsc,
                'R2_emb': r2_emb,
                'R2_combined': r2_comb,
                'delta': delta
            })

            if n_comp == 10:  # Print for default
                print(f"{model:<16} {r2_rsc:>10.3f} {r2_emb:>10.3f} {r2_comb:>10.3f} {delta:>+10.3f}")

    results_df = pd.DataFrame(results)

    # Summary by n_components
    print("\n" + "-" * 60)
    print("Summary by number of embedding components:")
    summary = results_df.groupby('n_components').agg({
        'R2_rsc': 'mean',
        'R2_emb': 'mean',
        'R2_combined': 'mean',
        'delta': 'mean'
    }).round(3)
    print(summary)

    return results_df

def phase2_lomo(df, emb_df, n_cal=10):
    """Phase 2: LOMO test - do embeddings help cross-model transfer?"""

    print("\n" + "=" * 100)
    print("PHASE 2: LOMO TEST (CROSS-MODEL TRANSFER)")
    print("=" * 100)
    print("Question: Do embeddings help predict a NEW model after calibration?")

    emb_cols = [f'emb_{i}' for i in range(min(10, emb_df.shape[1]))]
    models = sorted(df['model_name'].unique())

    results = []
    n_sim = 20

    print(f"\n{'Held-out':<16} {'R/S/C R²':>10} {'Emb R²':>10} {'Comb R²':>10} {'R/S/C MAE':>10} {'Comb MAE':>10}")
    print("-" * 75)

    for held_out in models:
        # Prepare data
        train_df = df[df['model_name'] != held_out].copy()
        test_df = df[df['model_name'] == held_out].copy()

        train_df['scenario_key'] = train_df['scenario'].str.lower().str.replace(' ', '_')
        test_df['scenario_key'] = test_df['scenario'].str.lower().str.replace(' ', '_')

        train_df = train_df[train_df['scenario_key'].isin(emb_df.index)]
        test_df = test_df[test_df['scenario_key'].isin(emb_df.index)]

        if len(test_df) < 20:
            continue

        # Features
        X_train_rsc = train_df[['R', 'S', 'C']].values
        X_train_emb = emb_df.loc[train_df['scenario_key'], emb_cols].values
        X_train_comb = np.hstack([X_train_rsc, X_train_emb])
        y_train = train_df['coop_pct'].values

        X_test_rsc = test_df[['R', 'S', 'C']].values
        X_test_emb = emb_df.loc[test_df['scenario_key'], emb_cols].values
        X_test_comb = np.hstack([X_test_rsc, X_test_emb])
        y_test = test_df['coop_pct'].values

        # Train models
        model_rsc = RidgeCV(alphas=[1, 10, 100, 1000])
        model_emb = RidgeCV(alphas=[1, 10, 100, 1000])
        model_comb = RidgeCV(alphas=[1, 10, 100, 1000])

        model_rsc.fit(X_train_rsc, y_train)
        model_emb.fit(X_train_emb, y_train)
        model_comb.fit(X_train_comb, y_train)

        # Calibrated prediction (average over simulations)
        metrics_rsc = []
        metrics_emb = []
        metrics_comb = []

        for sim in range(n_sim):
            np.random.seed(sim)
            indices = np.random.permutation(len(test_df))
            cal_idx = indices[:n_cal]
            eval_idx = indices[n_cal:]

            # R/S/C calibration
            y_pred_cal = model_rsc.predict(X_test_rsc[cal_idx])
            intercept_adj = np.mean(y_test[cal_idx] - y_pred_cal)
            y_pred_eval = model_rsc.predict(X_test_rsc[eval_idx]) + intercept_adj
            metrics_rsc.append(compute_metrics(y_test[eval_idx], y_pred_eval))

            # Emb calibration
            y_pred_cal = model_emb.predict(X_test_emb[cal_idx])
            intercept_adj = np.mean(y_test[cal_idx] - y_pred_cal)
            y_pred_eval = model_emb.predict(X_test_emb[eval_idx]) + intercept_adj
            metrics_emb.append(compute_metrics(y_test[eval_idx], y_pred_eval))

            # Combined calibration
            y_pred_cal = model_comb.predict(X_test_comb[cal_idx])
            intercept_adj = np.mean(y_test[cal_idx] - y_pred_cal)
            y_pred_eval = model_comb.predict(X_test_comb[eval_idx]) + intercept_adj
            metrics_comb.append(compute_metrics(y_test[eval_idx], y_pred_eval))

        # Average metrics
        avg_rsc = {k: np.mean([m[k] for m in metrics_rsc]) for k in metrics_rsc[0]}
        avg_emb = {k: np.mean([m[k] for m in metrics_emb]) for k in metrics_emb[0]}
        avg_comb = {k: np.mean([m[k] for m in metrics_comb]) for k in metrics_comb[0]}

        results.append({
            'model': held_out,
            'R2_rsc': avg_rsc['R2'],
            'R2_emb': avg_emb['R2'],
            'R2_combined': avg_comb['R2'],
            'MAE_rsc': avg_rsc['MAE'],
            'MAE_emb': avg_emb['MAE'],
            'MAE_combined': avg_comb['MAE'],
            'acc_rsc': avg_rsc['acc'],
            'acc_combined': avg_comb['acc']
        })

        print(f"{held_out:<16} {avg_rsc['R2']:>10.3f} {avg_emb['R2']:>10.3f} {avg_comb['R2']:>10.3f} {avg_rsc['MAE']:>9.1f}pp {avg_comb['MAE']:>9.1f}pp")

    results_df = pd.DataFrame(results)

    print("-" * 75)
    print(f"{'MEAN':<16} {results_df['R2_rsc'].mean():>10.3f} {results_df['R2_emb'].mean():>10.3f} {results_df['R2_combined'].mean():>10.3f} {results_df['MAE_rsc'].mean():>9.1f}pp {results_df['MAE_combined'].mean():>9.1f}pp")

    return results_df

def phase3_clustering(emb_df, scenario_texts, n_clusters=6):
    """Phase 3: Cluster scenarios based on embeddings"""

    print("\n" + "=" * 100)
    print("PHASE 3: SCENARIO CLUSTERING")
    print("=" * 100)

    emb_cols = [c for c in emb_df.columns if c.startswith('emb_')]
    X = emb_df[emb_cols].values

    # Cluster
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(X)

    # Assign labels to scenarios
    emb_df['cluster'] = labels

    print(f"\nScenarios grouped into {n_clusters} clusters:")
    print("-" * 80)

    for cluster in range(n_clusters):
        scenarios = emb_df[emb_df['cluster'] == cluster].index.tolist()
        print(f"\nCluster {cluster} ({len(scenarios)} scenarios):")
        for s in scenarios[:8]:  # Show first 8
            print(f"  - {s}")
        if len(scenarios) > 8:
            print(f"  ... and {len(scenarios) - 8} more")

    return emb_df

def main():
    print("=" * 100)
    print("EMBEDDING EXPERIMENT: Do scenario texts improve prediction beyond R/S/C?")
    print("=" * 100)

    # Load data
    print("\n[1] Loading data...")
    scenario_texts = load_scenario_texts()
    print(f"    Loaded {len(scenario_texts)} scenario texts")

    df = load_cooperation_data()
    print(f"    Loaded {len(df)} cooperation observations")

    # Create embeddings
    print("\n[2] Creating TF-IDF embeddings...")
    emb_df, vectorizer, svd = create_embeddings(scenario_texts, n_components=20)

    # Save embeddings
    emb_df.to_csv(OUTPUT_DIR / 'scenario_embeddings.csv')
    print(f"    Saved to {OUTPUT_DIR / 'scenario_embeddings.csv'}")

    # Phase 1: Within-model
    print("\n[3] Running Phase 1: Within-model comparison...")
    phase1_results = phase1_within_model(df, emb_df)
    phase1_results.to_csv(OUTPUT_DIR / 'phase1_within_model.csv', index=False)

    # Phase 2: LOMO
    print("\n[4] Running Phase 2: LOMO cross-model test...")
    phase2_results = phase2_lomo(df, emb_df)
    phase2_results.to_csv(OUTPUT_DIR / 'phase2_lomo.csv', index=False)

    # Phase 3: Clustering
    print("\n[5] Running Phase 3: Scenario clustering...")
    emb_df = phase3_clustering(emb_df, scenario_texts)
    emb_df.to_csv(OUTPUT_DIR / 'scenario_clusters.csv')

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    p1_mean = phase1_results[phase1_results['n_components'] == 10].mean()
    p2_mean = phase2_results.mean()

    print(f"""
┌────────────────────────────────────────────────────────────────────────────────────┐
│ WITHIN-MODEL (Phase 1)                                                             │
├────────────────────────────────────────────────────────────────────────────────────┤
│   R/S/C only:      R² = {p1_mean['R2_rsc']:.3f}                                                  │
│   Embeddings only: R² = {p1_mean['R2_emb']:.3f}                                                  │
│   Combined:        R² = {p1_mean['R2_combined']:.3f}                                                  │
│   Δ (Comb - R/S/C):    {p1_mean['delta']:+.3f}                                                  │
├────────────────────────────────────────────────────────────────────────────────────┤
│ LOMO CROSS-MODEL (Phase 2, with 10-scenario calibration)                           │
├────────────────────────────────────────────────────────────────────────────────────┤
│   R/S/C only:      R² = {p2_mean['R2_rsc']:.3f}, MAE = {p2_mean['MAE_rsc']:.1f}pp                          │
│   Embeddings only: R² = {p2_mean['R2_emb']:.3f}, MAE = {p2_mean['MAE_emb']:.1f}pp                          │
│   Combined:        R² = {p2_mean['R2_combined']:.3f}, MAE = {p2_mean['MAE_combined']:.1f}pp                          │
└────────────────────────────────────────────────────────────────────────────────────┘
""")

    # Interpretation
    delta_within = p1_mean['delta']
    delta_lomo = p2_mean['R2_combined'] - p2_mean['R2_rsc']

    if delta_within > 0.05 and delta_lomo > 0.02:
        interpretation = "Embeddings help both within-model and cross-model → Universal semantic effects"
    elif delta_within > 0.05 and delta_lomo <= 0.02:
        interpretation = "Embeddings help within but not cross-model → Model-specific semantic associations"
    elif delta_within <= 0.05 and delta_lomo <= 0.02:
        interpretation = "Embeddings don't help → R/S/C is sufficient (validates manual ratings)"
    else:
        interpretation = "Mixed results - needs more investigation"

    print(f"INTERPRETATION: {interpretation}")

    print(f"\nResults saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
