#!/usr/bin/env python3
"""
Mixed-Effects Model Experiment for Cooperation Prediction

Tests if hierarchical modeling improves prediction by accounting for
model-specific intercepts and slopes.

Target: Push LOMO R² from 0.44 toward 0.50+
"""

import sys
sys.stdout = sys.stderr = open(sys.stdout.fileno(), mode='w', buffering=1)

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'analysis' / 'lomo' / 'results_test2' / 'prepared_data.csv'
EMBEDDING_CACHE = PROJECT_ROOT / 'analysis' / 'embeddings' / 'results' / 'openrouter_embeddings_cache.json'
OUTPUT_DIR = PROJECT_ROOT / 'analysis' / 'embeddings' / 'results' / 'mixed_effects'
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


def load_data(n_pca=15):
    """Load and prepare data with embeddings"""
    # Load cooperation rates
    df = pd.read_csv(DATA_FILE)
    df['model_name'] = df['model'].map(MODEL_MAP)
    df['coop_pct'] = df['coop_rate']
    df['scenario_key'] = df['scenario'].str.lower().str.replace(' ', '_')

    # Load embeddings
    with open(EMBEDDING_CACHE, 'r') as f:
        emb_cache = json.load(f)

    # Get embeddings for each row
    embeddings = []
    valid_idx = []
    for i in range(len(df)):
        key = df.iloc[i]['scenario_key']
        if key in emb_cache:
            embeddings.append(emb_cache[key])
            valid_idx.append(i)

    df = df.iloc[valid_idx].reset_index(drop=True)
    embeddings = np.array(embeddings)

    # PCA reduction
    pca = PCA(n_components=n_pca)
    emb_pcs = pca.fit_transform(embeddings)
    var_explained = sum(pca.explained_variance_ratio_)

    # Add PCs to dataframe
    for i in range(n_pca):
        df[f'pc{i+1}'] = emb_pcs[:, i]

    return df, var_explained, pca


def compute_r2_metrics(y_true, y_pred):
    """Compute R², correlation, MAE"""
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    mae = mean_absolute_error(y_true, y_pred)
    return {'R2': r2, 'corr': corr, 'MAE': mae}


def marginal_r2(model, y):
    """Compute marginal R² (fixed effects only)"""
    y_pred_fixed = model.fittedvalues - model.random_effects_cov
    # Approximate: use variance decomposition
    var_fixed = np.var(model.fittedvalues)
    var_total = np.var(y)
    return var_fixed / var_total if var_total > 0 else 0


def conditional_r2(model, y):
    """Compute conditional R² (fixed + random effects)"""
    y_pred = model.fittedvalues
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0


def phase1_model_comparison(df):
    """Phase 1: Compare mixed-effects model specifications"""
    print("\n" + "=" * 100)
    print("PHASE 1: MIXED-EFFECTS MODEL COMPARISON (Within-Sample)")
    print("=" * 100)

    # Standardize features for stability
    for col in ['R', 'S', 'C'] + [f'pc{i}' for i in range(1, 16)]:
        if col in df.columns:
            df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std()

    results = []
    models_fitted = {}

    # M0: Random intercept only
    print("\n[M0] Random intercept only: coop_pct ~ R + S + C + (1|model)")
    try:
        m0 = smf.mixedlm("coop_pct ~ R_z + S_z + C_z", df, groups=df["model_name"])
        m0_fit = m0.fit(reml=True)
        cond_r2 = conditional_r2(m0_fit, df['coop_pct'])
        results.append({
            'Model': 'M0: Random Intercept',
            'Fixed': 'R + S + C',
            'Random': '(1|model)',
            'AIC': m0_fit.aic,
            'BIC': m0_fit.bic,
            'LogLik': m0_fit.llf,
            'Cond_R2': cond_r2,
            'RE_Var_Intercept': m0_fit.cov_re.iloc[0, 0] if hasattr(m0_fit.cov_re, 'iloc') else float(m0_fit.cov_re)
        })
        models_fitted['M0'] = m0_fit
        print(f"    Conditional R²: {cond_r2:.3f}")
        print(f"    AIC: {m0_fit.aic:.1f}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # M1: Random intercept + random R slope
    print("\n[M1] Random intercept + R slope: coop_pct ~ R + S + C + (1 + R|model)")
    try:
        m1 = smf.mixedlm("coop_pct ~ R_z + S_z + C_z", df, groups=df["model_name"],
                         re_formula="~R_z")
        m1_fit = m1.fit(reml=True)
        cond_r2 = conditional_r2(m1_fit, df['coop_pct'])
        results.append({
            'Model': 'M1: Random Intercept + R Slope',
            'Fixed': 'R + S + C',
            'Random': '(1 + R|model)',
            'AIC': m1_fit.aic,
            'BIC': m1_fit.bic,
            'LogLik': m1_fit.llf,
            'Cond_R2': cond_r2,
            'RE_Var_Intercept': m1_fit.cov_re.iloc[0, 0] if hasattr(m1_fit.cov_re, 'iloc') else np.nan
        })
        models_fitted['M1'] = m1_fit
        print(f"    Conditional R²: {cond_r2:.3f}")
        print(f"    AIC: {m1_fit.aic:.1f}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # M2: Embeddings + random intercept
    print("\n[M2] Embeddings + random intercept: coop_pct ~ R + S + C + PCs + (1|model)")
    pc_terms = ' + '.join([f'pc{i}_z' for i in range(1, 6)])  # Use top 5 PCs for stability
    try:
        formula = f"coop_pct ~ R_z + S_z + C_z + {pc_terms}"
        m2 = smf.mixedlm(formula, df, groups=df["model_name"])
        m2_fit = m2.fit(reml=True)
        cond_r2 = conditional_r2(m2_fit, df['coop_pct'])
        results.append({
            'Model': 'M2: Embeddings + Random Intercept',
            'Fixed': 'R + S + C + PC1-5',
            'Random': '(1|model)',
            'AIC': m2_fit.aic,
            'BIC': m2_fit.bic,
            'LogLik': m2_fit.llf,
            'Cond_R2': cond_r2,
            'RE_Var_Intercept': m2_fit.cov_re.iloc[0, 0] if hasattr(m2_fit.cov_re, 'iloc') else float(m2_fit.cov_re)
        })
        models_fitted['M2'] = m2_fit
        print(f"    Conditional R²: {cond_r2:.3f}")
        print(f"    AIC: {m2_fit.aic:.1f}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # M3: Full model - embeddings + random intercept + R slope
    print("\n[M3] Full: coop_pct ~ R + S + C + PCs + (1 + R|model)")
    try:
        formula = f"coop_pct ~ R_z + S_z + C_z + {pc_terms}"
        m3 = smf.mixedlm(formula, df, groups=df["model_name"], re_formula="~R_z")
        m3_fit = m3.fit(reml=True)
        cond_r2 = conditional_r2(m3_fit, df['coop_pct'])
        results.append({
            'Model': 'M3: Full (Emb + Random R)',
            'Fixed': 'R + S + C + PC1-5',
            'Random': '(1 + R|model)',
            'AIC': m3_fit.aic,
            'BIC': m3_fit.bic,
            'LogLik': m3_fit.llf,
            'Cond_R2': cond_r2,
            'RE_Var_Intercept': m3_fit.cov_re.iloc[0, 0] if hasattr(m3_fit.cov_re, 'iloc') else np.nan
        })
        models_fitted['M3'] = m3_fit
        print(f"    Conditional R²: {cond_r2:.3f}")
        print(f"    AIC: {m3_fit.aic:.1f}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Summary table
    if results:
        results_df = pd.DataFrame(results)
        print("\n" + "-" * 100)
        print("MODEL COMPARISON SUMMARY")
        print("-" * 100)
        print(results_df[['Model', 'Cond_R2', 'AIC', 'BIC']].to_string(index=False))
        results_df.to_csv(OUTPUT_DIR / 'model_comparison.csv', index=False)

    # Extract random effects
    print("\n" + "-" * 100)
    print("RANDOM EFFECTS (Model-Specific Intercepts)")
    print("-" * 100)

    if 'M1' in models_fitted:
        re_df = []
        for model_name, effects in models_fitted['M1'].random_effects.items():
            re_df.append({
                'model': model_name,
                'intercept': effects.iloc[0] if hasattr(effects, 'iloc') else effects[0],
                'R_slope': effects.iloc[1] if len(effects) > 1 else 0
            })
        re_df = pd.DataFrame(re_df)
        print(re_df.to_string(index=False))
        re_df.to_csv(OUTPUT_DIR / 'random_effects.csv', index=False)

    return models_fitted, results_df if results else None


def phase2_lomo_comparison(df):
    """Phase 2: LOMO comparison - Mixed-Effects vs Ridge"""
    print("\n" + "=" * 100)
    print("PHASE 2: LOMO CROSS-MODEL TRANSFER")
    print("=" * 100)

    models = sorted(df['model_name'].unique())
    n_cal = 10
    n_sim = 20

    # Standardize features
    for col in ['R', 'S', 'C'] + [f'pc{i}' for i in range(1, 16)]:
        if col in df.columns:
            df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std()

    pc_cols = [f'pc{i}' for i in range(1, 16)]
    pc_cols_z = [f'pc{i}_z' for i in range(1, 6)]

    results = []

    print(f"\n{'Held-out':<16} {'Ridge':>10} {'MixedEff':>10} {'Δ':>10}")
    print("-" * 50)

    for held_out in models:
        train_df = df[df['model_name'] != held_out].copy()
        test_df = df[df['model_name'] == held_out].copy()

        if len(test_df) < 15:
            continue

        ridge_r2s = []
        mixed_r2s = []

        for sim in range(n_sim):
            np.random.seed(sim)
            indices = np.random.permutation(len(test_df))
            cal_idx = indices[:n_cal]
            eval_idx = indices[n_cal:]

            if len(eval_idx) < 5:
                continue

            # Ridge baseline
            X_train = train_df[['R', 'S', 'C'] + pc_cols].values
            y_train = train_df['coop_pct'].values
            X_test = test_df[['R', 'S', 'C'] + pc_cols].values
            y_test = test_df['coop_pct'].values

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            ridge = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
            ridge.fit(X_train_s, y_train)

            y_pred_cal = ridge.predict(X_test_s[cal_idx])
            adj = np.mean(y_test[cal_idx] - y_pred_cal)
            y_pred_eval = ridge.predict(X_test_s[eval_idx]) + adj

            ridge_r2s.append(r2_score(y_test[eval_idx], y_pred_eval))

            # Mixed-effects approach: fit on training, extract fixed effects, calibrate
            try:
                pc_terms = ' + '.join(pc_cols_z)
                formula = f"coop_pct ~ R_z + S_z + C_z + {pc_terms}"
                mixed = smf.mixedlm(formula, train_df, groups=train_df["model_name"])
                mixed_fit = mixed.fit(reml=True)

                # For held-out model: use population-level (fixed effects) predictions
                # Create design matrix for test data
                X_test_design = test_df[['R_z', 'S_z', 'C_z'] + pc_cols_z].values
                X_test_design = np.column_stack([np.ones(len(X_test_design)), X_test_design])

                # Fixed effects coefficients
                fe_params = mixed_fit.fe_params.values

                # Population prediction
                y_pred_pop = X_test_design @ fe_params

                # Calibrate with held-out model's random intercept
                adj_mixed = np.mean(y_test[cal_idx] - y_pred_pop[cal_idx])
                y_pred_mixed_eval = y_pred_pop[eval_idx] + adj_mixed

                mixed_r2s.append(r2_score(y_test[eval_idx], y_pred_mixed_eval))
            except Exception as e:
                pass

        if ridge_r2s and mixed_r2s:
            ridge_mean = np.mean(ridge_r2s)
            mixed_mean = np.mean(mixed_r2s)
            delta = mixed_mean - ridge_mean
            results.append({
                'held_out': held_out,
                'ridge_r2': ridge_mean,
                'mixed_r2': mixed_mean,
                'delta': delta
            })
            print(f"{held_out:<16} {ridge_mean:>10.3f} {mixed_mean:>10.3f} {delta:>+10.3f}")

    if results:
        results_df = pd.DataFrame(results)
        print("-" * 50)
        print(f"{'MEAN':<16} {results_df['ridge_r2'].mean():>10.3f} {results_df['mixed_r2'].mean():>10.3f} {results_df['delta'].mean():>+10.3f}")
        results_df.to_csv(OUTPUT_DIR / 'lomo_comparison.csv', index=False)
        return results_df

    return None


def phase3_pca_sweep(df_orig):
    """Phase 3: Test different PCA dimensions"""
    print("\n" + "=" * 100)
    print("PHASE 3: PCA DIMENSION SWEEP")
    print("=" * 100)

    # Reload embeddings for each PCA dim
    with open(EMBEDDING_CACHE, 'r') as f:
        emb_cache = json.load(f)

    df_base = pd.read_csv(DATA_FILE)
    df_base['model_name'] = df_base['model'].map(MODEL_MAP)
    df_base['coop_pct'] = df_base['coop_rate']
    df_base['scenario_key'] = df_base['scenario'].str.lower().str.replace(' ', '_')

    # Get embeddings
    embeddings = []
    valid_idx = []
    for i in range(len(df_base)):
        key = df_base.iloc[i]['scenario_key']
        if key in emb_cache:
            embeddings.append(emb_cache[key])
            valid_idx.append(i)

    df_base = df_base.iloc[valid_idx].reset_index(drop=True)
    embeddings = np.array(embeddings)

    pca_dims = [5, 10, 15, 20, 30]
    results = []

    print(f"\n{'PCA Dims':<12} {'Var Expl':>12} {'Within R²':>12} {'LOMO R²':>12}")
    print("-" * 52)

    for n_pca in pca_dims:
        n_pca_actual = min(n_pca, embeddings.shape[1], len(df_base) - 1)

        # PCA reduction
        pca = PCA(n_components=n_pca_actual)
        emb_pcs = pca.fit_transform(embeddings)
        var_explained = sum(pca.explained_variance_ratio_)

        # Create dataframe with PCs
        df = df_base.copy()
        for i in range(n_pca_actual):
            df[f'pc{i+1}'] = emb_pcs[:, i]

        # Standardize
        for col in ['R', 'S', 'C'] + [f'pc{i}' for i in range(1, n_pca_actual + 1)]:
            df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std()

        # Within-sample R² (M2 model)
        pc_terms = ' + '.join([f'pc{i}_z' for i in range(1, min(n_pca_actual, 10) + 1)])
        try:
            formula = f"coop_pct ~ R_z + S_z + C_z + {pc_terms}"
            m2 = smf.mixedlm(formula, df, groups=df["model_name"])
            m2_fit = m2.fit(reml=True)
            within_r2 = conditional_r2(m2_fit, df['coop_pct'])
        except:
            within_r2 = np.nan

        # LOMO R² (simplified - one simulation)
        models = sorted(df['model_name'].unique())
        lomo_r2s = []
        pc_cols = [f'pc{i}' for i in range(1, n_pca_actual + 1)]

        for held_out in models:
            train_df = df[df['model_name'] != held_out]
            test_df = df[df['model_name'] == held_out]

            if len(test_df) < 15:
                continue

            X_train = train_df[['R', 'S', 'C'] + pc_cols].values
            y_train = train_df['coop_pct'].values
            X_test = test_df[['R', 'S', 'C'] + pc_cols].values
            y_test = test_df['coop_pct'].values

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            ridge = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
            ridge.fit(X_train_s, y_train)

            # 10-cal
            np.random.seed(42)
            idx = np.random.permutation(len(test_df))
            cal_idx, eval_idx = idx[:10], idx[10:]

            y_pred_cal = ridge.predict(X_test_s[cal_idx])
            adj = np.mean(y_test[cal_idx] - y_pred_cal)
            y_pred_eval = ridge.predict(X_test_s[eval_idx]) + adj

            lomo_r2s.append(r2_score(y_test[eval_idx], y_pred_eval))

        lomo_r2 = np.mean(lomo_r2s) if lomo_r2s else np.nan

        results.append({
            'pca_dims': n_pca,
            'var_explained': var_explained,
            'within_r2': within_r2,
            'lomo_r2': lomo_r2
        })

        print(f"{n_pca:<12} {var_explained:>12.1%} {within_r2:>12.3f} {lomo_r2:>12.3f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'pca_sweep.csv', index=False)

    # Find optimal
    best_idx = results_df['lomo_r2'].idxmax()
    best = results_df.iloc[best_idx]
    print(f"\nOptimal: {int(best['pca_dims'])} PCs (LOMO R² = {best['lomo_r2']:.3f})")

    return results_df


def phase4_interactions(df):
    """Phase 4: Test R × embedding interactions"""
    print("\n" + "=" * 100)
    print("PHASE 4: INTERACTION TERMS (R × Embedding PCs)")
    print("=" * 100)

    # Standardize
    for col in ['R', 'S', 'C'] + [f'pc{i}' for i in range(1, 16)]:
        if col in df.columns:
            df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std()

    # Create interactions
    for i in range(1, 4):
        df[f'R_x_pc{i}'] = df['R_z'] * df[f'pc{i}_z']

    results = []

    # Base model (no interactions)
    print("\n[Base] R + S + C + PC1-5 + (1|model)")
    pc_terms = ' + '.join([f'pc{i}_z' for i in range(1, 6)])
    try:
        formula = f"coop_pct ~ R_z + S_z + C_z + {pc_terms}"
        m_base = smf.mixedlm(formula, df, groups=df["model_name"])
        m_base_fit = m_base.fit(reml=True)
        base_r2 = conditional_r2(m_base_fit, df['coop_pct'])
        results.append({'Model': 'Base', 'Formula': 'R+S+C+PC1-5', 'Cond_R2': base_r2, 'AIC': m_base_fit.aic})
        print(f"    Conditional R²: {base_r2:.3f}, AIC: {m_base_fit.aic:.1f}")
    except Exception as e:
        print(f"    ERROR: {e}")
        base_r2 = np.nan

    # With R × PC interactions
    print("\n[+Interactions] R + S + C + PC1-5 + R×PC1-3 + (1|model)")
    int_terms = ' + '.join([f'R_x_pc{i}' for i in range(1, 4)])
    try:
        formula = f"coop_pct ~ R_z + S_z + C_z + {pc_terms} + {int_terms}"
        m_int = smf.mixedlm(formula, df, groups=df["model_name"])
        m_int_fit = m_int.fit(reml=True)
        int_r2 = conditional_r2(m_int_fit, df['coop_pct'])
        results.append({'Model': '+Interactions', 'Formula': 'R+S+C+PC1-5+R×PC1-3', 'Cond_R2': int_r2, 'AIC': m_int_fit.aic})
        print(f"    Conditional R²: {int_r2:.3f}, AIC: {m_int_fit.aic:.1f}")

        # Check interaction coefficients
        print("\n    Interaction coefficients:")
        for i in range(1, 4):
            coef = m_int_fit.fe_params.get(f'R_x_pc{i}', np.nan)
            print(f"      R × PC{i}: {coef:.3f}")
    except Exception as e:
        print(f"    ERROR: {e}")
        int_r2 = np.nan

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_DIR / 'interactions.csv', index=False)

        if not np.isnan(base_r2) and not np.isnan(int_r2):
            delta = int_r2 - base_r2
            print(f"\n    Δ R² from interactions: {delta:+.3f}")

    return results_df if results else None


def main():
    print("=" * 100)
    print("MIXED-EFFECTS MODEL EXPERIMENT")
    print("Goal: Improve LOMO R² from 0.44 toward 0.50+")
    print("=" * 100)

    # Load data
    print("\n[1] Loading data...")
    df, var_explained, pca = load_data(n_pca=15)
    print(f"    Observations: {len(df)}")
    print(f"    Models: {df['model_name'].nunique()}")
    print(f"    PCA variance explained: {var_explained:.1%}")

    # Phase 1: Model comparison
    print("\n[2] Phase 1: Mixed-effects model comparison...")
    models_fitted, comparison_df = phase1_model_comparison(df.copy())

    # Phase 2: LOMO comparison
    print("\n[3] Phase 2: LOMO comparison (Ridge vs Mixed-Effects)...")
    lomo_df = phase2_lomo_comparison(df.copy())

    # Phase 3: PCA sweep
    print("\n[4] Phase 3: PCA dimension sweep...")
    pca_df = phase3_pca_sweep(df.copy())

    # Phase 4: Interactions
    print("\n[5] Phase 4: Interaction terms...")
    int_df = phase4_interactions(df.copy())

    # Final summary
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    if lomo_df is not None:
        ridge_mean = lomo_df['ridge_r2'].mean()
        mixed_mean = lomo_df['mixed_r2'].mean()
        delta = mixed_mean - ridge_mean

        print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│ LOMO CROSS-MODEL TRANSFER COMPARISON                                        │
├────────────────────────────────────────────────────────────────────────────┤
│   Ridge (baseline):        R² = {ridge_mean:.3f}                                        │
│   Mixed-Effects:           R² = {mixed_mean:.3f}                                        │
│   Δ (Mixed - Ridge):           {delta:+.3f}                                        │
├────────────────────────────────────────────────────────────────────────────┤
│ INTERPRETATION                                                              │
├────────────────────────────────────────────────────────────────────────────┤
""")

        if mixed_mean > 0.50:
            print("│   SUCCESS: LOMO R² > 0.50 - Strong result, report as main method         │")
        elif mixed_mean > ridge_mean + 0.01:
            print("│   MODEST IMPROVEMENT: Mixed-effects helps cross-model transfer           │")
        else:
            print("│   NO IMPROVEMENT: Mixed-effects doesn't help for new models              │")
            print("│   Model differences are idiosyncratic, not systematic                    │")

        print("└────────────────────────────────────────────────────────────────────────────┘")

    # Save summary
    with open(OUTPUT_DIR / 'summary.md', 'w') as f:
        f.write("# Mixed-Effects Model Experiment Summary\n\n")
        if comparison_df is not None:
            f.write("## Phase 1: Model Comparison\n")
            f.write(comparison_df.to_markdown(index=False))
            f.write("\n\n")
        if lomo_df is not None:
            f.write("## Phase 2: LOMO Comparison\n")
            f.write(lomo_df.to_markdown(index=False))
            f.write(f"\n\nMean Ridge R²: {lomo_df['ridge_r2'].mean():.3f}\n")
            f.write(f"Mean Mixed R²: {lomo_df['mixed_r2'].mean():.3f}\n")
            f.write("\n")
        if pca_df is not None:
            f.write("## Phase 3: PCA Sweep\n")
            f.write(pca_df.to_markdown(index=False))
            f.write("\n\n")

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
