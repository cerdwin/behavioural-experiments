#!/usr/bin/env python3
"""
Approach 3: Meta-Learning Evaluation Protocol

Recasts LOMO + calibration as a few-shot learning problem:
  - Each "task" = predicting a new model's behavior
  - "Support set" = calibration scenarios from the held-out model
  - "Query set" = remaining scenarios to predict

Three analysis components:
  1. Prototypical Network analogue: predict via nearest training-model prototypes
  2. MAML-inspired evaluation: shared initialization + per-model adaptation
  3. Formal sample efficiency analysis with bootstrap CIs

Key output: calibration curves showing R² vs k (number of calibration scenarios)
with confidence intervals, plus task similarity analysis.
"""

import sys
sys.stdout = sys.stderr = open(sys.stdout.fileno(), mode='w', buffering=1)

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SHARED_DATA = Path(__file__).parent.parent / 'shared_prepared_data.csv'
DATA_FILE = PROJECT_ROOT / 'analysis' / 'lomo' / 'results_test2' / 'prepared_data.csv'
EMBEDDING_CACHE = PROJECT_ROOT / 'analysis' / 'embeddings' / 'results' / 'openrouter_embeddings_cache.json'
OUTPUT_DIR = Path(__file__).parent / 'results'
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


def load_data(n_pca=30):
    """Load cooperation data + embedding PCA features"""
    # Use shared pre-computed data if available (ensures identical PCA across scripts)
    if SHARED_DATA.exists():
        df = pd.read_csv(SHARED_DATA)
        pc_cols = [c for c in df.columns if c.startswith('pc')]
        print(f"  Data: {len(df)} obs, {df['model_name'].nunique()} models, "
              f"{df['scenario_key'].nunique()} scenarios, {len(pc_cols)} PCs")
        print(f"  (loaded from shared prepared data)")
        return df, pc_cols

    df = pd.read_csv(DATA_FILE)
    df['model_name'] = df['model'].map(MODEL_MAP)
    df['coop_pct'] = df['coop_rate']
    df['scenario_key'] = df['scenario'].str.lower().str.replace(' ', '_')

    with open(EMBEDDING_CACHE, 'r') as f:
        emb_cache = json.load(f)

    embeddings = []
    valid_idx = []
    for i in range(len(df)):
        key = df.iloc[i]['scenario_key']
        if key in emb_cache:
            embeddings.append(emb_cache[key])
            valid_idx.append(i)

    df = df.iloc[valid_idx].reset_index(drop=True)

    # Filter to common scenarios
    n_models = df['model_name'].nunique()
    scenario_counts = df.groupby('scenario_key')['model_name'].nunique()
    common_scenarios = scenario_counts[scenario_counts == n_models].index
    df = df[df['scenario_key'].isin(common_scenarios)].reset_index(drop=True)

    # PCA
    embeddings_filtered = np.array([emb_cache[df.iloc[i]['scenario_key']] for i in range(len(df))])
    unique_scenarios = df['scenario_key'].unique()
    unique_emb = np.array([emb_cache[s] for s in unique_scenarios])
    n_pca_actual = min(n_pca, unique_emb.shape[1], len(unique_scenarios) - 1)
    pca = PCA(n_components=n_pca_actual, random_state=42)
    pca.fit(unique_emb)
    emb_pcs = pca.transform(embeddings_filtered)

    pc_cols = []
    for i in range(n_pca_actual):
        col = f'pc{i+1}'
        df[col] = emb_pcs[:, i]
        pc_cols.append(col)

    print(f"  Data: {len(df)} obs, {df['model_name'].nunique()} models, "
          f"{df['scenario_key'].nunique()} scenarios, {len(pc_cols)} PCs")

    return df, pc_cols


def phase1_prototypical_prediction(df, pc_cols, n_cal=10, n_sim=20):
    """
    Prototypical Network analogue:
    - For each training model, compute a "prototype" = mean behavior vector
    - For held-out model, use calibration scenarios to compute a prototype
    - Predict as a weighted combination of nearest training-model prototypes
    """
    print("\n" + "=" * 80)
    print("PHASE 1: PROTOTYPICAL NETWORK ANALOGUE")
    print("=" * 80)
    print("Predict held-out model via nearest training-model prototypes.")

    models = sorted(df['model_name'].unique())
    common_scenarios = sorted(df['scenario_key'].unique())
    feature_cols = ['R', 'S', 'C'] + pc_cols

    # Build behavior profiles: model → scenario → coop_pct
    behavior = df.pivot_table(index='model_name', columns='scenario_key',
                               values='coop_pct', aggfunc='first')
    behavior = behavior[common_scenarios]  # Ensure consistent ordering

    results = []

    print(f"\n{'Held-out':<16} {'Proto R²':>10} {'Ridge R²':>10} {'Δ':>10}")
    print("-" * 50)

    for held_out in models:
        train_models = [m for m in models if m != held_out]
        test_behavior = behavior.loc[held_out].values  # 52-dim vector (sorted scenario order)

        # Sort test_df by scenario_key to match behavior pivot table column order
        test_df = df[df['model_name'] == held_out].sort_values('scenario_key').reset_index(drop=True)
        train_df = df[df['model_name'] != held_out]

        X_train = train_df[feature_cols].values
        y_train = train_df['coop_pct'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['coop_pct'].values  # Now in same sorted order as behavior pivot

        # Fit Ridge once outside the sim loop (training data doesn't change)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        ridge = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
        ridge.fit(X_train_s, y_train)

        sim_r2_proto = []
        sim_r2_ridge = []

        for sim in range(n_sim):
            np.random.seed(sim)
            indices = np.random.permutation(len(common_scenarios))
            cal_idx = indices[:n_cal]
            eval_idx = indices[n_cal:]

            # Held-out model's calibration behavior
            cal_behavior = test_behavior[cal_idx]

            # Compute similarity to each training model (using calibration scenarios)
            similarities = []
            for train_model in train_models:
                train_cal = behavior.loc[train_model].values[cal_idx]
                # Pearson correlation on calibration scenarios
                if np.std(cal_behavior) > 0 and np.std(train_cal) > 0:
                    sim_val = np.corrcoef(cal_behavior, train_cal)[0, 1]
                else:
                    sim_val = 0
                similarities.append(max(sim_val, 0))  # Clamp negative

            # Normalized weights (linear, not softmax)
            sims = np.array(similarities)
            if sims.sum() > 0:
                weights = sims / sims.sum()
            else:
                weights = np.ones(len(train_models)) / len(train_models)

            # Prototype prediction on calibration set (for offset correction)
            y_pred_proto_cal = np.zeros(len(cal_idx))
            for i, train_model in enumerate(train_models):
                y_pred_proto_cal += weights[i] * behavior.loc[train_model].values[cal_idx]

            # Offset correction: match mean on calibration set (same as Ridge)
            proto_adj = np.mean(cal_behavior - y_pred_proto_cal)

            # Prototype prediction on eval set + offset
            y_pred_proto = np.zeros(len(eval_idx))
            for i, train_model in enumerate(train_models):
                train_eval = behavior.loc[train_model].values[eval_idx]
                y_pred_proto += weights[i] * train_eval
            y_pred_proto += proto_adj

            y_true = test_behavior[eval_idx]
            if np.var(y_true) > 0:
                sim_r2_proto.append(r2_score(y_true, y_pred_proto))

            # Ridge baseline — same cal/eval split, same row order as behavior pivot
            adj = np.mean(y_test[cal_idx] - ridge.predict(X_test_s[cal_idx]))
            y_pred_ridge = ridge.predict(X_test_s[eval_idx]) + adj
            if np.var(y_test[eval_idx]) > 0:
                sim_r2_ridge.append(r2_score(y_test[eval_idx], y_pred_ridge))

        if sim_r2_proto and sim_r2_ridge:
            r2_proto = np.mean(sim_r2_proto)
            r2_ridge = np.mean(sim_r2_ridge)
            results.append({
                'model': held_out,
                'R2_proto': r2_proto,
                'R2_proto_std': np.std(sim_r2_proto),
                'R2_ridge': r2_ridge,
                'R2_ridge_std': np.std(sim_r2_ridge),
                'delta': r2_proto - r2_ridge
            })
            print(f"{held_out:<16} {r2_proto:>10.3f} {r2_ridge:>10.3f} {r2_proto - r2_ridge:>+10.3f}")

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        print("-" * 50)
        print(f"{'MEAN':<16} {results_df['R2_proto'].mean():>10.3f} "
              f"{results_df['R2_ridge'].mean():>10.3f} {results_df['delta'].mean():>+10.3f}")

    results_df.to_csv(OUTPUT_DIR / 'prototypical_results.csv', index=False)
    return results_df


def phase2_calibration_curves(df, pc_cols, n_sim=50):
    """
    MAML-inspired evaluation: measure sample efficiency.
    The shared ridge weights are the "initialization" and the intercept
    offset is the "adaptation step."

    Key output: R² vs k (calibration size) with bootstrap CIs.
    """
    print("\n" + "=" * 80)
    print("PHASE 2: CALIBRATION CURVES (SAMPLE EFFICIENCY)")
    print("=" * 80)
    print("Measuring adaptation rate: how quickly does R² improve with k scenarios?")

    models = sorted(df['model_name'].unique())
    feature_cols = ['R', 'S', 'C'] + pc_cols
    k_values = [0, 2, 5, 10, 15, 20, 25, 30]

    all_curves = []
    max_k = max(k_values)

    print(f"\n{'Held-out':<16}", end='')
    for k in k_values:
        print(f" {'k='+str(k):>8}", end='')
    print()
    print("-" * (16 + 9 * len(k_values)))

    for held_out in models:
        train_df = df[df['model_name'] != held_out]
        test_df = df[df['model_name'] == held_out]

        if len(test_df) < max_k + 5:
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df['coop_pct'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['coop_pct'].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        ridge = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
        ridge.fit(X_train_s, y_train)

        # Collect R² per (k, sim) with FIXED eval set across all k values
        # For each sim, eval_idx is always the last (N - max_k) scenarios
        # from the permutation, so all k values are evaluated on the same set
        k_sim_r2s = {k: [] for k in k_values}

        for sim in range(n_sim):
            np.random.seed(sim)
            indices = np.random.permutation(len(test_df))
            eval_idx = indices[max_k:]  # Fixed eval set (N - max_k scenarios)

            if len(eval_idx) < 5 or np.var(y_test[eval_idx]) == 0:
                continue

            y_pred_base = ridge.predict(X_test_s)

            for k in k_values:
                if k == 0:
                    # Zero-shot: no calibration offset
                    y_pred_eval = y_pred_base[eval_idx]
                else:
                    cal_idx = indices[:k]
                    adj = np.mean(y_test[cal_idx] - y_pred_base[cal_idx])
                    y_pred_eval = y_pred_base[eval_idx] + adj

                k_sim_r2s[k].append(r2_score(y_test[eval_idx], y_pred_eval))

        print(f"{held_out:<16}", end='')

        for k in k_values:
            sim_r2s = k_sim_r2s[k]
            if sim_r2s:
                mean_r2 = np.mean(sim_r2s)
                print(f" {mean_r2:>8.3f}", end='')

                # Bootstrap CI
                if len(sim_r2s) > 1:
                    bootstrap_means = []
                    for b in range(200):
                        np.random.seed(10000 + b)
                        boot_sample = np.random.choice(sim_r2s, size=len(sim_r2s), replace=True)
                        bootstrap_means.append(np.mean(boot_sample))
                    ci_low = np.percentile(bootstrap_means, 2.5)
                    ci_high = np.percentile(bootstrap_means, 97.5)
                else:
                    ci_low = mean_r2
                    ci_high = mean_r2

                all_curves.append({
                    'model': held_out,
                    'k': k,
                    'R2_mean': mean_r2,
                    'R2_std': np.std(sim_r2s),
                    'R2_ci_low': ci_low,
                    'R2_ci_high': ci_high,
                    'n_sims': len(sim_r2s)
                })
            else:
                print(f" {'N/A':>8}", end='')

        print()

    curves_df = pd.DataFrame(all_curves)

    # Summary: average calibration curve
    # CI computed from distribution of per-model means (not averaging per-model CIs)
    print("\n" + "-" * 80)
    print("AVERAGE CALIBRATION CURVE")
    print("-" * 80)
    print(f"{'k':>5} {'Mean R²':>10} {'95% CI':>20} {'Δ vs k=0':>12}")

    r2_at_0 = curves_df[curves_df['k'] == 0]['R2_mean'].mean() if 0 in curves_df['k'].values else 0

    for k in k_values:
        k_data = curves_df[curves_df['k'] == k]
        if len(k_data) > 0:
            per_model_means = k_data['R2_mean'].values
            mean_r2 = np.mean(per_model_means)
            # Bootstrap CI from per-model means (accounts for between-model variability)
            if len(per_model_means) > 1:
                boot_means = []
                for b in range(1000):
                    np.random.seed(20000 + b)
                    boot_sample = np.random.choice(per_model_means, size=len(per_model_means), replace=True)
                    boot_means.append(np.mean(boot_sample))
                ci_low = np.percentile(boot_means, 2.5)
                ci_high = np.percentile(boot_means, 97.5)
            else:
                ci_low = mean_r2
                ci_high = mean_r2
            delta = mean_r2 - r2_at_0
            print(f"{k:>5} {mean_r2:>10.3f} [{ci_low:>8.3f}, {ci_high:>8.3f}] {delta:>+12.3f}")

    # Compute AUC of calibration curve (area under R² vs k)
    auc_results = []
    for model in models:
        model_curves = curves_df[curves_df['model'] == model].sort_values('k')
        if len(model_curves) >= 3:
            ks = model_curves['k'].values
            r2s = model_curves['R2_mean'].values
            # Trapezoidal integration, normalized by k range
            auc = np.trapz(r2s, ks) / (ks[-1] - ks[0]) if ks[-1] > ks[0] else 0
            auc_results.append({
                'model': model,
                'AUC': auc,
                'R2_at_0': model_curves[model_curves['k'] == 0]['R2_mean'].values[0] if 0 in model_curves['k'].values else np.nan,
                'R2_at_10': model_curves[model_curves['k'] == 10]['R2_mean'].values[0] if 10 in model_curves['k'].values else np.nan,
                'R2_at_30': model_curves[model_curves['k'] == 30]['R2_mean'].values[0] if 30 in model_curves['k'].values else np.nan
            })

    auc_df = pd.DataFrame(auc_results)

    print("\n  ADAPTATION EFFICIENCY (AUC of calibration curve)")
    print(f"  {'Model':<16} {'AUC':>8} {'R²@0':>8} {'R²@10':>8} {'R²@30':>8} {'Ease':>8}")
    print("  " + "-" * 55)
    for _, row in auc_df.iterrows():
        ease = 'easy' if row['AUC'] > 0.4 else ('moderate' if row['AUC'] > 0.2 else 'hard')
        print(f"  {row['model']:<16} {row['AUC']:>8.3f} {row['R2_at_0']:>8.3f} "
              f"{row['R2_at_10']:>8.3f} {row['R2_at_30']:>8.3f} {ease:>8}")

    curves_df.to_csv(OUTPUT_DIR / 'calibration_curves.csv', index=False)
    auc_df.to_csv(OUTPUT_DIR / 'adaptation_efficiency.csv', index=False)

    return curves_df, auc_df


def phase3_task_similarity(df, pc_cols):
    """
    Task similarity analysis: show that calibration success correlates
    with behavioral similarity to training models.
    """
    print("\n" + "=" * 80)
    print("PHASE 3: TASK SIMILARITY ANALYSIS")
    print("=" * 80)
    print("Does behavioral similarity to training models predict calibration success?")

    models = sorted(df['model_name'].unique())
    common_scenarios = sorted(df['scenario_key'].unique())
    feature_cols = ['R', 'S', 'C'] + pc_cols

    # Build behavior profiles
    behavior = df.pivot_table(index='model_name', columns='scenario_key',
                               values='coop_pct', aggfunc='first')
    behavior = behavior[common_scenarios]

    # Compute pairwise distances
    model_distances = {}
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i < j:
                v1 = behavior.loc[m1].values
                v2 = behavior.loc[m2].values
                corr = np.corrcoef(v1, v2)[0, 1]
                mae = mean_absolute_error(v1, v2)
                model_distances[(m1, m2)] = {'corr': corr, 'mae': mae}

    # For each held-out model, compute average similarity to training models
    # and actual LOMO R² (at k=10)
    lomo_r2s = {}
    avg_similarities = {}

    for held_out in models:
        train_models = [m for m in models if m != held_out]
        train_df = df[df['model_name'] != held_out]
        test_df = df[df['model_name'] == held_out]

        # Compute avg similarity to training models
        sims = []
        for tm in train_models:
            key = tuple(sorted([held_out, tm]))
            sims.append(model_distances[key]['corr'])
        avg_similarities[held_out] = np.mean(sims)

        # Quick LOMO R² at k=10
        X_train = train_df[feature_cols].values
        y_train = train_df['coop_pct'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['coop_pct'].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        ridge = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
        ridge.fit(X_train_s, y_train)

        sim_r2s = []
        for sim in range(20):
            np.random.seed(sim)
            indices = np.random.permutation(len(test_df))
            cal_idx, eval_idx = indices[:10], indices[10:]
            if len(eval_idx) < 5:
                continue
            adj = np.mean(y_test[cal_idx] - ridge.predict(X_test_s[cal_idx]))
            y_pred = ridge.predict(X_test_s[eval_idx]) + adj
            sim_r2s.append(r2_score(y_test[eval_idx], y_pred))

        lomo_r2s[held_out] = np.mean(sim_r2s) if sim_r2s else np.nan

    # Analysis: similarity vs LOMO R²
    sim_values = [avg_similarities[m] for m in models]
    r2_values = [lomo_r2s[m] for m in models]

    spearman_r, spearman_p = spearmanr(sim_values, r2_values)

    print(f"\n{'Model':<16} {'Avg Sim':>10} {'LOMO R²':>10}")
    print("-" * 40)
    for m in models:
        print(f"{m:<16} {avg_similarities[m]:>10.3f} {lomo_r2s[m]:>10.3f}")

    print(f"\n  Spearman correlation (similarity → R²): ρ = {spearman_r:.3f} (p = {spearman_p:.3f})")

    if spearman_r > 0.3:
        print("  → Positive correlation: models similar to training set are easier to predict.")
        print("    This supports the meta-learning framing: adaptation depends on task similarity.")
    else:
        print("  → Weak correlation: prediction difficulty is not simply explained by similarity.")

    # Save
    similarity_df = pd.DataFrame([
        {'model': m, 'avg_similarity': avg_similarities[m], 'lomo_r2': lomo_r2s[m]}
        for m in models
    ])
    similarity_df.to_csv(OUTPUT_DIR / 'task_similarity_vs_performance.csv', index=False)

    # Full distance matrix
    dist_rows = []
    for (m1, m2), vals in model_distances.items():
        dist_rows.append({
            'model_1': m1, 'model_2': m2,
            'correlation': vals['corr'], 'mae': vals['mae']
        })
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(OUTPUT_DIR / 'model_distance_matrix.csv', index=False)

    return similarity_df, spearman_r


def main():
    print("=" * 80)
    print("APPROACH 3: META-LEARNING EVALUATION PROTOCOL")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data...")
    df, pc_cols = load_data(n_pca=30)

    # Phase 1: Prototypical prediction
    print("\n[2] Prototypical network analogue...")
    proto_df = phase1_prototypical_prediction(df, pc_cols, n_cal=10, n_sim=20)

    # Phase 2: Calibration curves
    print("\n[3] Calibration curves with bootstrap CIs...")
    curves_df, auc_df = phase2_calibration_curves(df, pc_cols, n_sim=50)

    # Phase 3: Task similarity
    print("\n[4] Task similarity analysis...")
    sim_df, spearman_r = phase3_task_similarity(df, pc_cols)

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if proto_df is not None and len(proto_df) > 0:
        print(f"\n  Prototypical Network Analogue:")
        print(f"    Proto LOMO R²: {proto_df['R2_proto'].mean():.3f}")
        print(f"    Ridge LOMO R²: {proto_df['R2_ridge'].mean():.3f}")
        delta = proto_df['delta'].mean()
        print(f"    Delta: {delta:+.3f}")

    if auc_df is not None and len(auc_df) > 0:
        print(f"\n  Adaptation Efficiency:")
        print(f"    Mean AUC: {auc_df['AUC'].mean():.3f}")
        print(f"    Easiest model: {auc_df.loc[auc_df['AUC'].idxmax(), 'model']}"
              f" (AUC={auc_df['AUC'].max():.3f})")
        print(f"    Hardest model: {auc_df.loc[auc_df['AUC'].idxmin(), 'model']}"
              f" (AUC={auc_df['AUC'].min():.3f})")

    print(f"\n  Task Similarity → Performance:")
    print(f"    Spearman ρ = {spearman_r:.3f}")

    print(f"\n  Results saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
