#!/usr/bin/env python3
"""
Approach 1: Quantal Response Equilibrium (QRE) + Social Utility Model

Theoretical basis: McKelvey & Palfrey (1995) QRE framework.
Agents choose probabilistically based on utility differences:

  P(cooperate | scenario s, model m) = sigmoid(lambda_m * DeltaU(s) + beta_m)

where:
  DeltaU(s) = social_utility_weight(s) - 0  [utility surplus from cooperation]
  social_utility_weight(s) = w^T @ features(s)  [learned from scenario features]
  lambda_m = rationality parameter (model-specific)
  beta_m = cooperation bias (model-specific)

LOMO protocol: Train social utility weights on 6 models, then estimate
lambda_m and beta_m for the held-out model from calibration scenarios.
"""

import sys
sys.stdout = sys.stderr = open(sys.stdout.fileno(), mode='w', buffering=1)

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import minimize
from scipy.special import expit  # sigmoid
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import RidgeCV

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
    embeddings = np.array(embeddings)

    # Filter to common scenarios (present in all models)
    n_models = df['model_name'].nunique()
    scenario_counts = df.groupby('scenario_key')['model_name'].nunique()
    common_scenarios = scenario_counts[scenario_counts == n_models].index
    common_mask = df['scenario_key'].isin(common_scenarios)
    df = df[common_mask].reset_index(drop=True)

    # Recompute embeddings for filtered rows
    embeddings_filtered = []
    for i in range(len(df)):
        key = df.iloc[i]['scenario_key']
        embeddings_filtered.append(emb_cache[key])
    embeddings = np.array(embeddings_filtered)

    # PCA reduction
    n_pca_actual = min(n_pca, embeddings.shape[1], len(np.unique(df['scenario_key'])) - 1)
    pca = PCA(n_components=n_pca_actual, random_state=42)

    # Fit PCA on unique scenarios only to avoid data leakage from repeated scenarios
    unique_scenarios = df['scenario_key'].unique()
    unique_emb = np.array([emb_cache[s] for s in unique_scenarios])
    pca.fit(unique_emb)

    emb_pcs = pca.transform(embeddings)

    pc_cols = []
    for i in range(n_pca_actual):
        col = f'pc{i+1}'
        df[col] = emb_pcs[:, i]
        pc_cols.append(col)

    print(f"  Data: {len(df)} obs, {df['model_name'].nunique()} models, "
          f"{df['scenario_key'].nunique()} scenarios, {len(pc_cols)} PCs")

    return df, pc_cols


def sigmoid(x):
    """Numerically stable sigmoid"""
    return expit(x)


def qre_loss(params, X, y_coop_frac, model_indices, n_models, n_features):
    """
    MSE loss for QRE model (treats cooperation rates as continuous targets).

    params layout:
      [0:n_features]            = w (social utility weights for scenario features)
      [n_features:n_features+n_models]     = lambda_m (rationality per model)
      [n_features+n_models:n_features+2*n_models] = beta_m (bias per model)

    y_coop_frac: cooperation rate as fraction [0, 1]
    """
    w = params[:n_features]
    lambdas = params[n_features:n_features + n_models]
    betas = params[n_features + n_models:n_features + 2 * n_models]

    # Social utility surplus per scenario
    delta_u = X @ w  # (N,)

    # QRE probability per observation
    lam = lambdas[model_indices]
    beta = betas[model_indices]
    logit_p = lam * delta_u + beta
    p = sigmoid(logit_p)

    # Clamp for numerical stability
    p = np.clip(p, 1e-6, 1 - 1e-6)

    # Treat cooperation rate as a continuous Beta-like likelihood
    # Use MSE loss (equivalent to Gaussian likelihood on cooperation rate)
    mse = np.mean((p - y_coop_frac) ** 2)

    # L2 regularization on weights
    reg = 0.01 * np.sum(w ** 2)

    return mse + reg


def fit_qre_model(df, feature_cols, models_to_fit):
    """
    Fit QRE model on given data.
    Returns fitted parameters and model info.
    """
    model_list = sorted(models_to_fit)
    model_to_idx = {m: i for i, m in enumerate(model_list)}
    n_models = len(model_list)

    mask = df['model_name'].isin(model_list)
    fit_df = df[mask].copy()

    X = fit_df[feature_cols].values
    y = fit_df['coop_pct'].values / 100.0  # Convert to [0, 1]
    model_indices = fit_df['model_name'].map(model_to_idx).values.astype(int)

    n_features = X.shape[1]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize: w=0, lambda=1, beta from mean coop rate per model
    w0 = np.zeros(n_features)
    lam0 = np.ones(n_models)
    beta0 = np.zeros(n_models)
    for i, m in enumerate(model_list):
        mean_coop = fit_df[fit_df['model_name'] == m]['coop_pct'].mean() / 100.0
        # Inverse sigmoid of mean cooperation rate
        mean_coop_clipped = np.clip(mean_coop, 0.01, 0.99)
        beta0[i] = np.log(mean_coop_clipped / (1 - mean_coop_clipped))

    x0 = np.concatenate([w0, lam0, beta0])

    # Bounds: w unconstrained, lambda >= 0 (rationality must be non-negative),
    # beta unconstrained
    bounds = (
        [(None, None)] * n_features +        # w: unconstrained
        [(0, 50)] * n_models +               # lambda: [0, 50] (>50 is degenerate step fn)
        [(None, None)] * n_models             # beta: unconstrained
    )

    result = minimize(
        qre_loss,
        x0,
        args=(X_scaled, y, model_indices, n_models, n_features),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 10000, 'maxfun': 50000, 'ftol': 1e-9}
    )

    if not result.success:
        print(f"  WARNING: QRE optimizer did not converge: {result.message}")
        print(f"    Final loss: {result.fun:.6f}, nit: {result.nit}, nfev: {result.nfev}")

    # Extract parameters
    w_fit = result.x[:n_features]
    lambdas_fit = result.x[n_features:n_features + n_models]
    betas_fit = result.x[n_features + n_models:]

    return {
        'w': w_fit,
        'lambdas': {m: lambdas_fit[i] for i, m in enumerate(model_list)},
        'betas': {m: betas_fit[i] for i, m in enumerate(model_list)},
        'scaler': scaler,
        'model_list': model_list,
        'loss': result.fun,
        'success': result.success
    }


def predict_qre(fit_result, X_raw, model_name):
    """Predict cooperation rate using fitted QRE model"""
    X_scaled = fit_result['scaler'].transform(X_raw)
    delta_u = X_scaled @ fit_result['w']
    lam = fit_result['lambdas'].get(model_name, 1.0)
    beta = fit_result['betas'].get(model_name, 0.0)
    p = sigmoid(lam * delta_u + beta)
    return p * 100.0  # Back to percentage


def calibrate_qre_for_new_model(fit_result, X_cal, y_cal, X_pred):
    """
    Calibrate QRE model for a new (held-out) model.
    Estimate lambda and beta for the new model using calibration data,
    keeping the social utility weights (w) fixed.
    """
    X_cal_scaled = fit_result['scaler'].transform(X_cal)
    X_pred_scaled = fit_result['scaler'].transform(X_pred)

    w = fit_result['w']
    delta_u_cal = X_cal_scaled @ w
    delta_u_pred = X_pred_scaled @ w

    y_frac = y_cal / 100.0

    # Optimize lambda and beta for this model
    # Use L-BFGS-B to enforce lambda >= 0 (rationality must be non-negative)
    def obj(params):
        lam, beta = params
        p = sigmoid(lam * delta_u_cal + beta)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.mean((p - y_frac) ** 2)

    # Initialize from mean coop rate
    mean_coop = np.clip(np.mean(y_frac), 0.01, 0.99)
    beta_init = np.log(mean_coop / (1 - mean_coop))

    result = minimize(obj, [1.0, beta_init], method='L-BFGS-B',
                      bounds=[(0, 50), (None, None)],
                      options={'maxiter': 2000, 'ftol': 1e-9})

    if not result.success:
        print(f"    WARNING: Calibration optimizer did not converge: {result.message}")

    lam_hat, beta_hat = result.x

    # Predict
    p_pred = sigmoid(lam_hat * delta_u_pred + beta_hat)
    return p_pred * 100.0, lam_hat, beta_hat


def run_lomo_qre(df, feature_cols, n_cal=10, n_sim=20):
    """
    Leave-One-Model-Out evaluation of QRE model.
    For each held-out model:
      1. Fit QRE on remaining 6 models (learn w, lambda_m, beta_m)
      2. Calibrate lambda and beta for held-out model using n_cal scenarios
      3. Predict remaining scenarios
    """
    models = sorted(df['model_name'].unique())
    results = []

    print(f"\n{'Held-out':<16} {'QRE R²':>10} {'QRE MAE':>10} {'λ_hat':>10} {'β_hat':>10}")
    print("-" * 60)

    for held_out in models:
        train_models = [m for m in models if m != held_out]
        test_df = df[df['model_name'] == held_out]

        if len(test_df) < 15:
            continue

        # Fit QRE on training models
        fit_result = fit_qre_model(df, feature_cols, train_models)

        X_test = test_df[feature_cols].values
        y_test = test_df['coop_pct'].values

        sim_r2s = []
        sim_maes = []
        sim_lams = []
        sim_betas = []

        for sim in range(n_sim):
            np.random.seed(sim)
            indices = np.random.permutation(len(test_df))
            cal_idx = indices[:n_cal]
            eval_idx = indices[n_cal:]

            if len(eval_idx) < 5:
                continue

            X_cal = X_test[cal_idx]
            y_cal = y_test[cal_idx]
            X_eval = X_test[eval_idx]
            y_eval = y_test[eval_idx]

            y_pred, lam_hat, beta_hat = calibrate_qre_for_new_model(
                fit_result, X_cal, y_cal, X_eval
            )

            sim_r2s.append(r2_score(y_eval, y_pred))
            sim_maes.append(mean_absolute_error(y_eval, y_pred))
            sim_lams.append(lam_hat)
            sim_betas.append(beta_hat)

        if sim_r2s:
            results.append({
                'model': held_out,
                'R2': np.mean(sim_r2s),
                'R2_std': np.std(sim_r2s),
                'MAE': np.mean(sim_maes),
                'MAE_std': np.std(sim_maes),
                'lambda_hat': np.mean(sim_lams),
                'beta_hat': np.mean(sim_betas)
            })
            print(f"{held_out:<16} {np.mean(sim_r2s):>10.3f} {np.mean(sim_maes):>9.1f}pp "
                  f"{np.mean(sim_lams):>10.3f} {np.mean(sim_betas):>10.3f}")

    return pd.DataFrame(results)


def run_lomo_ridge_baseline(df, feature_cols, n_cal=10, n_sim=20):
    """Ridge baseline for direct comparison"""
    models = sorted(df['model_name'].unique())
    results = []

    for held_out in models:
        train_df = df[df['model_name'] != held_out]
        test_df = df[df['model_name'] == held_out]

        if len(test_df) < 15:
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

        sim_r2s = []
        sim_maes = []

        for sim in range(n_sim):
            np.random.seed(sim)
            indices = np.random.permutation(len(test_df))
            cal_idx = indices[:n_cal]
            eval_idx = indices[n_cal:]

            if len(eval_idx) < 5:
                continue

            y_pred_cal = ridge.predict(X_test_s[cal_idx])
            adj = np.mean(y_test[cal_idx] - y_pred_cal)
            y_pred_eval = ridge.predict(X_test_s[eval_idx]) + adj

            sim_r2s.append(r2_score(y_test[eval_idx], y_pred_eval))
            sim_maes.append(mean_absolute_error(y_test[eval_idx], y_pred_eval))

        if sim_r2s:
            results.append({
                'model': held_out,
                'R2': np.mean(sim_r2s),
                'R2_std': np.std(sim_r2s),
                'MAE': np.mean(sim_maes),
                'MAE_std': np.std(sim_maes)
            })

    return pd.DataFrame(results)


def extract_model_parameters(df, feature_cols):
    """
    Fit QRE on ALL data and extract interpretable per-model parameters.
    These are the key scientific outputs.
    """
    models = sorted(df['model_name'].unique())
    fit_result = fit_qre_model(df, feature_cols, models)

    param_rows = []
    for m in models:
        param_rows.append({
            'model': m,
            'lambda': fit_result['lambdas'][m],
            'beta': fit_result['betas'][m],
            'mean_coop_pct': df[df['model_name'] == m]['coop_pct'].mean(),
            'implied_baseline_coop': sigmoid(fit_result['betas'][m]) * 100
        })

    params_df = pd.DataFrame(param_rows)

    # Social utility weights
    w = fit_result['w']
    weight_names = feature_cols
    weight_rows = [{'feature': name, 'weight': w[i]} for i, name in enumerate(weight_names)]
    weights_df = pd.DataFrame(weight_rows)

    return params_df, weights_df, fit_result


def main():
    print("=" * 80)
    print("APPROACH 1: QRE + SOCIAL UTILITY MODEL")
    print("Theoretical basis: McKelvey & Palfrey (1995)")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data...")
    df, pc_cols = load_data(n_pca=30)
    feature_cols = ['R', 'S', 'C'] + pc_cols

    # Phase 1: Extract interpretable parameters from full model
    print("\n[2] Fitting full QRE model for parameter extraction...")
    params_df, weights_df, full_fit = extract_model_parameters(df, feature_cols)

    print("\n  MODEL-SPECIFIC QRE PARAMETERS")
    print("  " + "-" * 65)
    print(f"  {'Model':<16} {'λ (rational.)':>14} {'β (bias)':>10} {'Mean Coop%':>12} {'Implied Base%':>14}")
    print("  " + "-" * 65)
    for _, row in params_df.iterrows():
        print(f"  {row['model']:<16} {row['lambda']:>14.3f} {row['beta']:>10.3f} "
              f"{row['mean_coop_pct']:>11.1f}% {row['implied_baseline_coop']:>13.1f}%")

    print("\n  TOP SOCIAL UTILITY WEIGHTS")
    print("  " + "-" * 40)
    top_weights = weights_df.reindex(weights_df['weight'].abs().sort_values(ascending=False).index).head(10)
    for _, row in top_weights.iterrows():
        print(f"  {row['feature']:<20} {row['weight']:>+10.4f}")

    params_df.to_csv(OUTPUT_DIR / 'qre_model_parameters.csv', index=False)
    weights_df.to_csv(OUTPUT_DIR / 'qre_social_utility_weights.csv', index=False)

    # Phase 2: LOMO evaluation
    print("\n[3] Running LOMO evaluation (QRE)...")
    qre_results = run_lomo_qre(df, feature_cols, n_cal=10, n_sim=20)

    print("\n[4] Running LOMO evaluation (Ridge baseline)...")
    print(f"\n{'Held-out':<16} {'Ridge R²':>10} {'Ridge MAE':>10}")
    print("-" * 40)
    ridge_results = run_lomo_ridge_baseline(df, feature_cols, n_cal=10, n_sim=20)
    for _, row in ridge_results.iterrows():
        print(f"{row['model']:<16} {row['R2']:>10.3f} {row['MAE']:>9.1f}pp")

    # Merge results
    comparison = qre_results[['model', 'R2', 'MAE']].rename(
        columns={'R2': 'QRE_R2', 'MAE': 'QRE_MAE'})
    comparison = comparison.merge(
        ridge_results[['model', 'R2', 'MAE']].rename(
            columns={'R2': 'Ridge_R2', 'MAE': 'Ridge_MAE'}),
        on='model', how='outer'
    )
    comparison['Delta_R2'] = comparison['QRE_R2'] - comparison['Ridge_R2']

    print("\n" + "=" * 80)
    print("COMPARISON: QRE vs Ridge (LOMO)")
    print("=" * 80)
    print(f"\n{'Model':<16} {'QRE R²':>10} {'Ridge R²':>10} {'Δ R²':>10}")
    print("-" * 50)
    for _, row in comparison.iterrows():
        print(f"{row['model']:<16} {row['QRE_R2']:>10.3f} {row['Ridge_R2']:>10.3f} {row['Delta_R2']:>+10.3f}")
    print("-" * 50)
    print(f"{'MEAN':<16} {comparison['QRE_R2'].mean():>10.3f} {comparison['Ridge_R2'].mean():>10.3f} "
          f"{comparison['Delta_R2'].mean():>+10.3f}")

    # Save results
    qre_results.to_csv(OUTPUT_DIR / 'qre_lomo_results.csv', index=False)
    ridge_results.to_csv(OUTPUT_DIR / 'ridge_baseline_results.csv', index=False)
    comparison.to_csv(OUTPUT_DIR / 'qre_vs_ridge_comparison.csv', index=False)

    # Final summary
    qre_mean = comparison['QRE_R2'].mean()
    ridge_mean = comparison['Ridge_R2'].mean()
    delta = qre_mean - ridge_mean

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  QRE LOMO R²:   {qre_mean:.3f}")
    print(f"  Ridge LOMO R²: {ridge_mean:.3f}")
    print(f"  Delta:         {delta:+.3f}")
    print(f"\n  Interpretable parameters saved to: {OUTPUT_DIR}")

    if qre_mean > ridge_mean:
        print("  -> QRE improves over Ridge: theoretical structure helps!")
    else:
        print("  -> QRE does not improve over Ridge.")
        print("     However, QRE provides interpretable parameters (lambda, beta)")
        print("     that connect to behavioral game theory literature.")

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
