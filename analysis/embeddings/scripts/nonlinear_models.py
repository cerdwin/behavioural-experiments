#!/usr/bin/env python3
"""
Compare linear vs non-linear models for cooperation prediction.
Tests: Ridge, Random Forest, Gradient Boosting, MLP, Kernel Ridge (RBF).
"""

import sys
# Force unbuffered output
sys.stdout = sys.stderr = open(sys.stdout.fileno(), mode='w', buffering=1)

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'analysis' / 'lomo' / 'results_test2' / 'prepared_data.csv'
EMBEDDING_CACHE = PROJECT_ROOT / 'analysis' / 'embeddings' / 'results' / 'openrouter_embeddings_cache.json'
OUTPUT_DIR = PROJECT_ROOT / 'analysis' / 'embeddings' / 'results'

MODEL_MAP = {
    'anthropic/claude-3.7-sonnet': 'Claude_3.7',
    'anthropic/claude-haiku-4.5': 'Haiku_4.5',
    'deepseek/deepseek-v3.2': 'DeepSeek',
    'openai/gpt-5.2': 'GPT_5.2',
    'openai/o4-mini': 'O4_mini',
    'google/gemini-2.5-pro': 'Gemini_2.5_Pro',
    'google/gemini-3-pro-preview': 'Gemini_3_Pro'
}


def load_data():
    """Load cooperation data and embeddings"""
    # Load cooperation rates
    df = pd.read_csv(DATA_FILE)
    df['model_name'] = df['model'].map(MODEL_MAP)
    df['coop_pct'] = df['coop_rate']
    df['scenario_key'] = df['scenario'].str.lower().str.replace(' ', '_')

    # Load embeddings
    with open(EMBEDDING_CACHE, 'r') as f:
        emb_cache = json.load(f)

    return df, emb_cache


def get_features(df, emb_cache, use_embeddings=True, n_pca=15):
    """Extract features for modeling"""
    from sklearn.decomposition import PCA

    # Reset index to ensure positional indexing works
    df = df.reset_index(drop=True)

    # R/S/C features
    X_rsc = df[['R', 'S', 'C']].values

    if not use_embeddings:
        return X_rsc, df['coop_pct'].values

    # Get embeddings for each row (using positional index)
    embeddings = []
    valid_idx = []
    for i in range(len(df)):
        key = df.iloc[i]['scenario_key']
        if key in emb_cache:
            embeddings.append(emb_cache[key])
            valid_idx.append(i)

    if len(embeddings) == 0:
        return X_rsc, df['coop_pct'].values

    embeddings = np.array(embeddings)

    # PCA reduction
    pca = PCA(n_components=min(n_pca, embeddings.shape[1]))
    emb_reduced = pca.fit_transform(embeddings)

    # Combine features
    X_rsc_valid = X_rsc[valid_idx]
    X_combined = np.hstack([X_rsc_valid, emb_reduced])
    y = df.iloc[valid_idx]['coop_pct'].values

    return X_combined, y


def get_models():
    """Define models to compare"""
    models = {
        'Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
        ]),

        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ),

        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=3,
            random_state=42
        ),

        'MLP': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                alpha=0.01,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42
            ))
        ]),

        'Kernel Ridge (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1))
        ]),

        'SVR (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf', C=10, gamma='scale'))
        ])
    }
    return models


def evaluate_within_model(df, emb_cache, models):
    """Phase 1: Within-model cross-validation"""
    print("\n" + "=" * 100)
    print("PHASE 1: WITHIN-MODEL COMPARISON (5-fold CV)")
    print("=" * 100)

    model_names = sorted(df['model_name'].dropna().unique())
    results = {name: [] for name in models.keys()}

    # Header
    header = f"{'LLM':<16}"
    for name in models.keys():
        header += f" {name[:12]:>12}"
    print(f"\n{header}")
    print("-" * (16 + 13 * len(models)))

    for llm in model_names:
        llm_df = df[df['model_name'] == llm].copy()

        # Get features with embeddings
        X, y = get_features(llm_df, emb_cache, use_embeddings=True)

        if len(X) < 20:
            print(f"{llm:<16} (skipped, n={len(X)})")
            continue

        row = f"{llm:<16}"
        for model_name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                r2 = np.mean(scores)
                results[model_name].append(r2)
                row += f" {r2:>12.3f}"
            except Exception as e:
                row += f" {'ERR':>12}"

        print(row)

    # Mean row
    print("-" * (16 + 13 * len(models)))
    mean_row = f"{'MEAN':<16}"
    for model_name in models.keys():
        if results[model_name]:
            mean_row += f" {np.mean(results[model_name]):>12.3f}"
        else:
            mean_row += f" {'N/A':>12}"
    print(mean_row)

    return results


def evaluate_lomo(df, emb_cache, models, n_cal=10, n_sim=20):
    """Phase 2: Leave-One-Model-Out with calibration"""
    print("\n" + "=" * 100)
    print(f"PHASE 2: LOMO CROSS-MODEL TRANSFER ({n_cal}-scenario calibration)")
    print("=" * 100)

    model_names = sorted(df['model_name'].dropna().unique())

    # Results storage
    r2_results = {name: [] for name in models.keys()}
    mae_results = {name: [] for name in models.keys()}

    # Header
    print(f"\n{'Held-out':<16}", end="")
    for name in models.keys():
        print(f" {name[:10]:>10}", end="")
    print()
    print("-" * (16 + 11 * len(models)))

    for held_out in model_names:
        train_df = df[df['model_name'] != held_out].copy()
        test_df = df[df['model_name'] == held_out].copy()

        # Get training features
        X_train, y_train = get_features(train_df, emb_cache, use_embeddings=True)
        X_test, y_test = get_features(test_df, emb_cache, use_embeddings=True)

        if len(X_test) < 15 or len(X_train) < 50:
            print(f"{held_out:<16} (skipped)")
            continue

        row = f"{held_out:<16}"

        for model_name, model in models.items():
            sim_r2s = []
            sim_maes = []

            for sim in range(n_sim):
                np.random.seed(sim)
                indices = np.random.permutation(len(X_test))
                cal_idx = indices[:n_cal]
                eval_idx = indices[n_cal:]

                if len(eval_idx) < 5:
                    continue

                try:
                    # Train model
                    model_copy = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
                    model_copy.fit(X_train, y_train)

                    # Predict and calibrate
                    y_pred_cal = model_copy.predict(X_test[cal_idx])
                    adj = np.mean(y_test[cal_idx] - y_pred_cal)

                    y_pred_eval = model_copy.predict(X_test[eval_idx]) + adj

                    r2 = r2_score(y_test[eval_idx], y_pred_eval)
                    mae = mean_absolute_error(y_test[eval_idx], y_pred_eval)

                    sim_r2s.append(r2)
                    sim_maes.append(mae)
                except Exception as e:
                    pass

            if sim_r2s:
                avg_r2 = np.mean(sim_r2s)
                avg_mae = np.mean(sim_maes)
                r2_results[model_name].append(avg_r2)
                mae_results[model_name].append(avg_mae)
                row += f" {avg_r2:>10.3f}"
            else:
                row += f" {'ERR':>10}"

        print(row)

    # Mean row
    print("-" * (16 + 11 * len(models)))
    mean_row = f"{'MEAN R²':<16}"
    for model_name in models.keys():
        if r2_results[model_name]:
            mean_row += f" {np.mean(r2_results[model_name]):>10.3f}"
        else:
            mean_row += f" {'N/A':>10}"
    print(mean_row)

    mae_row = f"{'MEAN MAE':<16}"
    for model_name in models.keys():
        if mae_results[model_name]:
            mae_row += f" {np.mean(mae_results[model_name]):>9.1f}p"
        else:
            mae_row += f" {'N/A':>10}"
    print(mae_row)

    return r2_results, mae_results


def compare_feature_sets(df, emb_cache, n_sim=20, n_cal=10):
    """Compare R/S/C only vs R/S/C + embeddings for best model"""
    print("\n" + "=" * 100)
    print("FEATURE ABLATION: R/S/C vs R/S/C + Embeddings (using Random Forest)")
    print("=" * 100)

    model_names = sorted(df['model_name'].dropna().unique())

    results_rsc = []
    results_combined = []

    print(f"\n{'Held-out':<16} {'R/S/C only':>12} {'+ Embeddings':>14} {'Δ':>10}")
    print("-" * 55)

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=3, random_state=42)

    for held_out in model_names:
        train_df = df[df['model_name'] != held_out].copy()
        test_df = df[df['model_name'] == held_out].copy()

        # R/S/C only
        X_train_rsc = train_df[['R', 'S', 'C']].values
        X_test_rsc = test_df[['R', 'S', 'C']].values
        y_train = train_df['coop_pct'].values
        y_test = test_df['coop_pct'].values

        # Combined
        X_train_comb, _ = get_features(train_df, emb_cache, use_embeddings=True)
        X_test_comb, y_test_comb = get_features(test_df, emb_cache, use_embeddings=True)

        if len(X_test_rsc) < 15:
            continue

        r2_rsc_sims = []
        r2_comb_sims = []

        for sim in range(n_sim):
            np.random.seed(sim)

            # R/S/C only
            indices = np.random.permutation(len(X_test_rsc))
            cal_idx, eval_idx = indices[:n_cal], indices[n_cal:]
            if len(eval_idx) < 5:
                continue

            rf_rsc = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=3, random_state=42)
            rf_rsc.fit(X_train_rsc, y_train)
            y_pred = rf_rsc.predict(X_test_rsc[eval_idx]) + np.mean(y_test[cal_idx] - rf_rsc.predict(X_test_rsc[cal_idx]))
            r2_rsc_sims.append(r2_score(y_test[eval_idx], y_pred))

            # Combined
            rf_comb = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=3, random_state=42)
            rf_comb.fit(X_train_comb, y_train[:len(X_train_comb)])
            y_pred = rf_comb.predict(X_test_comb[eval_idx]) + np.mean(y_test_comb[cal_idx] - rf_comb.predict(X_test_comb[cal_idx]))
            r2_comb_sims.append(r2_score(y_test_comb[eval_idx], y_pred))

        if r2_rsc_sims:
            r2_rsc = np.mean(r2_rsc_sims)
            r2_comb = np.mean(r2_comb_sims)
            results_rsc.append(r2_rsc)
            results_combined.append(r2_comb)
            delta = r2_comb - r2_rsc
            print(f"{held_out:<16} {r2_rsc:>12.3f} {r2_comb:>14.3f} {delta:>+10.3f}")

    print("-" * 55)
    if results_rsc:
        mean_rsc = np.mean(results_rsc)
        mean_comb = np.mean(results_combined)
        print(f"{'MEAN':<16} {mean_rsc:>12.3f} {mean_comb:>14.3f} {mean_comb - mean_rsc:>+10.3f}")


def main():
    print("=" * 100)
    print("NON-LINEAR MODEL COMPARISON: Ridge vs RF vs GB vs MLP vs Kernel")
    print("=" * 100)

    # Load data
    print("\n[1] Loading data...")
    df, emb_cache = load_data()
    print(f"    Observations: {len(df)}")
    print(f"    Models: {df['model_name'].nunique()}")
    print(f"    Cached embeddings: {len(emb_cache)}")

    # Get models
    models = get_models()
    print(f"\n[2] Models to compare: {list(models.keys())}")

    # Phase 1: Within-model
    print("\n[3] Running Phase 1: Within-model CV...")
    within_results = evaluate_within_model(df, emb_cache, models)

    # Phase 2: LOMO
    print("\n[4] Running Phase 2: LOMO cross-model transfer...")
    lomo_r2, lomo_mae = evaluate_lomo(df, emb_cache, models)

    # Feature ablation
    print("\n[5] Running feature ablation...")
    compare_feature_sets(df, emb_cache)

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY: BEST MODEL FOR LOMO CROSS-MODEL TRANSFER")
    print("=" * 100)

    summary = []
    for name in models.keys():
        if lomo_r2[name]:
            summary.append({
                'Model': name,
                'LOMO R²': np.mean(lomo_r2[name]),
                'LOMO MAE': np.mean(lomo_mae[name]),
                'Within R²': np.mean(within_results[name]) if within_results[name] else np.nan
            })

    summary_df = pd.DataFrame(summary).sort_values('LOMO R²', ascending=False)
    print(f"\n{summary_df.to_string(index=False)}")

    # Save results
    summary_df.to_csv(OUTPUT_DIR / 'nonlinear_model_comparison.csv', index=False)
    print(f"\nResults saved to: {OUTPUT_DIR / 'nonlinear_model_comparison.csv'}")

    # Interpretation
    best = summary_df.iloc[0]
    ridge_r2 = summary_df[summary_df['Model'] == 'Ridge']['LOMO R²'].values[0]

    print(f"\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    if best['LOMO R²'] > ridge_r2 + 0.02:
        print(f"""
Best model: {best['Model']} (R² = {best['LOMO R²']:.3f})
Ridge baseline: R² = {ridge_r2:.3f}
Improvement: +{best['LOMO R²'] - ridge_r2:.3f}

Non-linear relationships exist between features and cooperation rate.
The {best['Model']} captures interactions/non-linearities that Ridge misses.
""")
    else:
        print(f"""
Best model: {best['Model']} (R² = {best['LOMO R²']:.3f})
Ridge baseline: R² = {ridge_r2:.3f}
Difference: {best['LOMO R²'] - ridge_r2:+.3f}

Linear Ridge performs comparably to non-linear models.
The relationship between embeddings and cooperation is approximately linear.
Parsimony favors the simpler Ridge model.
""")


if __name__ == '__main__':
    main()
