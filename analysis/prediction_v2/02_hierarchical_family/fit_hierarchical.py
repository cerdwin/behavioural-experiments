#!/usr/bin/env python3
"""
Approach 2: Hierarchical Model with Family Effects

Groups models into families (Anthropic, OpenAI, Google, DeepSeek) and tests
whether family-level priors improve cross-model prediction.

Three-level hierarchy:
  Level 1 (observation): coop_rate_ms ~ Normal(mu_ms, sigma^2)
  Level 2 (model):       mu_ms = X_s @ beta + gamma_family(m) + delta_m
  Level 3 (family):      gamma_family is a fixed family effect
                         delta_m ~ Normal(0, tau^2) is a model-level random effect

Evaluation:
  - LOMO: Hold out one model, use family effect + calibration
  - LOFO: Hold out entire family (harder test of family structure)
  - Variance decomposition: how much is family vs model vs scenario?
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

FAMILIES = {
    'Anthropic': ['Claude_3.7', 'Haiku_4.5'],
    'OpenAI': ['GPT_5.2', 'O4_mini'],
    'Google': ['Gemini_2.5_Pro', 'Gemini_3_Pro'],
    'DeepSeek': ['DeepSeek']
}

MODEL_TO_FAMILY = {}
for fam, models in FAMILIES.items():
    for m in models:
        MODEL_TO_FAMILY[m] = fam


def load_data(n_pca=30):
    """Load cooperation data + embedding PCA features"""
    # Use shared pre-computed data if available (ensures identical PCA across scripts)
    if SHARED_DATA.exists():
        df = pd.read_csv(SHARED_DATA)
        if 'family' not in df.columns:
            df['family'] = df['model_name'].map(MODEL_TO_FAMILY)
        pc_cols = [c for c in df.columns if c.startswith('pc')]
        print(f"  Data: {len(df)} obs, {df['model_name'].nunique()} models, "
              f"{df['family'].nunique()} families, {df['scenario_key'].nunique()} scenarios")
        print(f"  (loaded from shared prepared data)")
        return df, pc_cols

    df = pd.read_csv(DATA_FILE)
    df['model_name'] = df['model'].map(MODEL_MAP)
    df['coop_pct'] = df['coop_rate']
    df['scenario_key'] = df['scenario'].str.lower().str.replace(' ', '_')
    df['family'] = df['model_name'].map(MODEL_TO_FAMILY)

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

    # Filter to common scenarios
    n_models = df['model_name'].nunique()
    scenario_counts = df.groupby('scenario_key')['model_name'].nunique()
    common_scenarios = scenario_counts[scenario_counts == n_models].index
    df = df[df['scenario_key'].isin(common_scenarios)].reset_index(drop=True)

    # Recompute embeddings for filtered rows
    embeddings_filtered = np.array([emb_cache[df.iloc[i]['scenario_key']] for i in range(len(df))])

    # PCA reduction
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
          f"{df['family'].nunique()} families, {df['scenario_key'].nunique()} scenarios")

    return df, pc_cols


def phase1_variance_decomposition(df, pc_cols):
    """
    Decompose variance into family, model-within-family, and scenario components.
    """
    print("\n" + "=" * 80)
    print("PHASE 1: VARIANCE DECOMPOSITION")
    print("=" * 80)

    # Standardize features
    for col in ['R', 'S', 'C'] + pc_cols[:10]:
        df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std()

    pc_terms = ' + '.join([f'pc{i}_z' for i in range(1, min(11, len(pc_cols) + 1))])

    results = []

    # Create family dummies upfront so M2 and M3 can both use them
    family_dummies = pd.get_dummies(df['family'], prefix='fam', drop_first=True)
    fam_cols = list(family_dummies.columns)
    for col in fam_cols:
        df[col] = family_dummies[col].values
    fam_terms = ' + '.join(fam_cols)

    # M0: No grouping (fixed effects only)
    print("\n[M0] Fixed effects only: coop_pct ~ R + S + C + PCs")
    try:
        from sklearn.linear_model import LinearRegression
        X = df[['R_z', 'S_z', 'C_z'] + [f'pc{i}_z' for i in range(1, min(11, len(pc_cols)+1))]].values
        y = df['coop_pct'].values
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        r2_fixed = r2_score(y, y_pred)
        results.append({'Model': 'M0_Fixed_Only', 'R2': r2_fixed, 'Random': 'None'})
        print(f"  R² = {r2_fixed:.3f}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # M1: Random model intercept (no family)
    print("\n[M1] Random model intercept: + (1|model)")
    try:
        formula = f"coop_pct ~ R_z + S_z + C_z + {pc_terms}"
        m1 = smf.mixedlm(formula, df, groups=df["model_name"])
        m1_fit = m1.fit(reml=True)
        y_pred = m1_fit.fittedvalues
        r2_m1 = r2_score(df['coop_pct'], y_pred)
        re_var_model = float(m1_fit.cov_re.iloc[0, 0]) if hasattr(m1_fit.cov_re, 'iloc') else float(m1_fit.cov_re)
        residual_var = m1_fit.scale
        results.append({
            'Model': 'M1_Random_Model', 'R2': r2_m1,
            'Random': '(1|model)',
            'RE_Var_Model': re_var_model,
            'Residual_Var': residual_var
        })
        print(f"  R² = {r2_m1:.3f} (model RE var = {re_var_model:.1f}, residual var = {residual_var:.1f})")
    except Exception as e:
        print(f"  ERROR: {e}")

    # M2: Family as fixed effect + random model intercept
    print("\n[M2] Family fixed effect + random model: + family + (1|model)")
    try:
        formula = f"coop_pct ~ R_z + S_z + C_z + {pc_terms} + {fam_terms}"
        m2 = smf.mixedlm(formula, df, groups=df["model_name"])
        m2_fit = m2.fit(reml=True)
        y_pred = m2_fit.fittedvalues
        r2_m2 = r2_score(df['coop_pct'], y_pred)
        re_var_model_m2 = float(m2_fit.cov_re.iloc[0, 0]) if hasattr(m2_fit.cov_re, 'iloc') else float(m2_fit.cov_re)
        residual_var_m2 = m2_fit.scale
        results.append({
            'Model': 'M2_Family_Fixed_Model_Random', 'R2': r2_m2,
            'Random': 'family(fixed) + (1|model)',
            'RE_Var_Model': re_var_model_m2,
            'Residual_Var': residual_var_m2
        })
        print(f"  R² = {r2_m2:.3f} (model RE var = {re_var_model_m2:.1f}, residual var = {residual_var_m2:.1f})")

        # Family effects
        print("\n  Family effects (relative to reference):")
        for col in fam_cols:
            coef = m2_fit.fe_params.get(col, 0)
            print(f"    {col}: {coef:+.2f}pp")
    except Exception as e:
        print(f"  ERROR: {e}")

    # M3: Family fixed + random model intercept + R slope
    print("\n[M3] Family fixed + random model + R slope: + family + (1 + R|model)")
    try:
        formula = f"coop_pct ~ R_z + S_z + C_z + {pc_terms} + {fam_terms}"
        m3 = smf.mixedlm(formula, df, groups=df["model_name"], re_formula="~R_z")
        m3_fit = m3.fit(reml=True)
        y_pred = m3_fit.fittedvalues
        r2_m3 = r2_score(df['coop_pct'], y_pred)
        results.append({
            'Model': 'M3_Family_Fixed_Model_Random_RSlope', 'R2': r2_m3,
            'Random': 'family(fixed) + (1+R|model)'
        })
        print(f"  R² = {r2_m3:.3f}")

        # Random effects per model
        print("\n  Random effects per model:")
        print(f"  {'Model':<16} {'Intercept':>12} {'R Slope':>12}")
        print("  " + "-" * 42)
        re_data = []
        for model_name, effects in m3_fit.random_effects.items():
            intercept = effects.iloc[0] if hasattr(effects, 'iloc') else effects[0]
            r_slope = effects.iloc[1] if len(effects) > 1 else 0
            print(f"  {model_name:<16} {intercept:>+12.2f} {r_slope:>+12.3f}")
            re_data.append({
                'model': model_name,
                'family': MODEL_TO_FAMILY.get(model_name, 'Unknown'),
                'intercept': intercept,
                'R_slope': r_slope
            })
        re_df = pd.DataFrame(re_data)
        re_df.to_csv(OUTPUT_DIR / 'random_effects_by_model.csv', index=False)
    except Exception as e:
        print(f"  ERROR: {e}")

    # Summary
    results_df = pd.DataFrame(results)
    print("\n" + "-" * 80)
    print("VARIANCE DECOMPOSITION SUMMARY")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(f"  {row['Model']:<45} R² = {row['R2']:.3f}")

    results_df.to_csv(OUTPUT_DIR / 'variance_decomposition.csv', index=False)
    return results_df


def phase2_lomo_with_family(df, pc_cols, n_cal=10, n_sim=20):
    """
    LOMO evaluation: Does family structure help cross-model prediction?

    Three methods compared:
    1. Ridge (no family info) + intercept calibration
    2. Ridge with family dummies + intercept calibration
    3. Family-aware: use same-family model average as calibration prior
    """
    print("\n" + "=" * 80)
    print("PHASE 2: LOMO - DOES FAMILY STRUCTURE HELP?")
    print("=" * 80)

    models = sorted(df['model_name'].unique())
    feature_cols = ['R', 'S', 'C'] + pc_cols

    results = []

    print(f"\n{'Held-out':<16} {'No Family':>12} {'+ Family':>12} {'Fam-Aware':>12} {'Family':>10}")
    print("-" * 65)

    for held_out in models:
        train_df = df[df['model_name'] != held_out].copy()
        test_df = df[df['model_name'] == held_out].copy()
        held_out_family = MODEL_TO_FAMILY[held_out]

        if len(test_df) < 15:
            continue

        # Method 1: Plain Ridge (no family)
        X_train = train_df[feature_cols].values
        y_train = train_df['coop_pct'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['coop_pct'].values

        scaler1 = StandardScaler()
        X_train_s = scaler1.fit_transform(X_train)
        X_test_s = scaler1.transform(X_test)

        ridge1 = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
        ridge1.fit(X_train_s, y_train)

        # Method 2: Ridge with family dummies
        # Scale continuous features only; leave binary dummies unscaled
        # No drop_first: Ridge regularization handles multicollinearity,
        # and drop_first on single-family test data would drop the only column
        family_dummies_train = pd.get_dummies(train_df['family'], prefix='fam')
        family_dummies_test = pd.get_dummies(test_df['family'], prefix='fam')
        # Ensure same columns
        for col in family_dummies_train.columns:
            if col not in family_dummies_test.columns:
                family_dummies_test[col] = 0
        for col in family_dummies_test.columns:
            if col not in family_dummies_train.columns:
                family_dummies_train[col] = 0
        fam_col_names = sorted(family_dummies_train.columns)
        family_dummies_train = family_dummies_train[fam_col_names]
        family_dummies_test = family_dummies_test[fam_col_names]

        X_train_fam = np.hstack([X_train_s, family_dummies_train.values])
        X_test_fam = np.hstack([X_test_s, family_dummies_test.values])

        ridge2 = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
        ridge2.fit(X_train_fam, y_train)

        # Method 3: Family-aware calibration
        # If another model from the same family is in training, use its mean
        # as a prior for the calibration offset
        same_family_models = [m for m in FAMILIES[held_out_family] if m != held_out]
        has_family_prior = any(m in train_df['model_name'].values for m in same_family_models)

        if has_family_prior:
            family_train_data = train_df[train_df['family'] == held_out_family]
            family_mean = family_train_data['coop_pct'].mean()
        else:
            family_mean = None

        sim_r2_plain = []
        sim_r2_fam = []
        sim_r2_fam_aware = []

        for sim in range(n_sim):
            np.random.seed(sim)
            indices = np.random.permutation(len(test_df))
            cal_idx = indices[:n_cal]
            eval_idx = indices[n_cal:]

            if len(eval_idx) < 5:
                continue

            # Method 1: Plain Ridge
            adj1 = np.mean(y_test[cal_idx] - ridge1.predict(X_test_s[cal_idx]))
            y_pred1 = ridge1.predict(X_test_s[eval_idx]) + adj1
            sim_r2_plain.append(r2_score(y_test[eval_idx], y_pred1))

            # Method 2: Ridge + family dummies
            adj2 = np.mean(y_test[cal_idx] - ridge2.predict(X_test_fam[cal_idx]))
            y_pred2 = ridge2.predict(X_test_fam[eval_idx]) + adj2
            sim_r2_fam.append(r2_score(y_test[eval_idx], y_pred2))

            # Method 3: Family-aware calibration
            # Use a shrinkage estimator: weighted average of calibration offset
            # and family prior
            y_pred_base = ridge1.predict(X_test_s)
            cal_offset = np.mean(y_test[cal_idx] - y_pred_base[cal_idx])

            if family_mean is not None:
                # Family prior for the offset
                family_offset = family_mean - np.mean(y_pred_base)
                # Shrinkage: weight calibration data vs family prior
                # More calibration data → more weight on calibration
                shrinkage = n_cal / (n_cal + 5)  # 5 is a pseudo-count
                adj3 = shrinkage * cal_offset + (1 - shrinkage) * family_offset
            else:
                adj3 = cal_offset  # No family info, fall back to standard calibration

            y_pred3 = y_pred_base[eval_idx] + adj3
            sim_r2_fam_aware.append(r2_score(y_test[eval_idx], y_pred3))

        if sim_r2_plain:
            results.append({
                'model': held_out,
                'family': held_out_family,
                'has_family_sibling': has_family_prior,
                'R2_plain': np.mean(sim_r2_plain),
                'R2_family_dummies': np.mean(sim_r2_fam),
                'R2_family_aware': np.mean(sim_r2_fam_aware),
                'R2_plain_std': np.std(sim_r2_plain),
                'R2_family_dummies_std': np.std(sim_r2_fam),
                'R2_family_aware_std': np.std(sim_r2_fam_aware)
            })
            print(f"{held_out:<16} {np.mean(sim_r2_plain):>12.3f} {np.mean(sim_r2_fam):>12.3f} "
                  f"{np.mean(sim_r2_fam_aware):>12.3f} {held_out_family:>10}")

    results_df = pd.DataFrame(results)
    print("-" * 65)
    print(f"{'MEAN':<16} {results_df['R2_plain'].mean():>12.3f} {results_df['R2_family_dummies'].mean():>12.3f} "
          f"{results_df['R2_family_aware'].mean():>12.3f}")

    # Models with family siblings vs singletons
    siblings = results_df[results_df['has_family_sibling']]
    singletons = results_df[~results_df['has_family_sibling']]

    if len(siblings) > 0:
        print(f"\n  Models with family siblings (n={len(siblings)}):")
        print(f"    Plain: {siblings['R2_plain'].mean():.3f}, "
              f"Family-aware: {siblings['R2_family_aware'].mean():.3f}, "
              f"Δ: {(siblings['R2_family_aware'] - siblings['R2_plain']).mean():+.3f}")
    if len(singletons) > 0:
        print(f"  Singleton models (n={len(singletons)}):")
        print(f"    Plain: {singletons['R2_plain'].mean():.3f}, "
              f"Family-aware: {singletons['R2_family_aware'].mean():.3f}")

    results_df.to_csv(OUTPUT_DIR / 'lomo_family_comparison.csv', index=False)
    return results_df


def phase3_lofo(df, pc_cols, n_cal=10, n_sim=20):
    """
    Leave-One-Family-Out: Hold out ALL models from a family.
    Harder test of cross-family generalization.
    """
    print("\n" + "=" * 80)
    print("PHASE 3: LEAVE-ONE-FAMILY-OUT (LOFO)")
    print("=" * 80)

    families = sorted(FAMILIES.keys())
    feature_cols = ['R', 'S', 'C'] + pc_cols

    results = []

    print(f"\n{'Family':<12} {'Models':<30} {'LOFO R²':>10} {'LOFO MAE':>10}")
    print("-" * 65)

    for held_out_family in families:
        held_out_models = FAMILIES[held_out_family]
        train_df = df[~df['model_name'].isin(held_out_models)]
        test_df = df[df['model_name'].isin(held_out_models)]

        if len(test_df) < 15 or len(train_df) < 15:
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

        # For LOFO, we can still calibrate per-model within the held-out family
        per_model_results = []

        for model_name in held_out_models:
            model_df = test_df[test_df['model_name'] == model_name]
            if len(model_df) < 15:
                continue

            model_X = scaler.transform(model_df[feature_cols].values)
            model_y = model_df['coop_pct'].values

            sim_r2s = []
            sim_maes = []

            for sim in range(n_sim):
                np.random.seed(sim)
                indices = np.random.permutation(len(model_df))
                cal_idx = indices[:n_cal]
                eval_idx = indices[n_cal:]

                if len(eval_idx) < 5:
                    continue

                adj = np.mean(model_y[cal_idx] - ridge.predict(model_X[cal_idx]))
                y_pred = ridge.predict(model_X[eval_idx]) + adj

                sim_r2s.append(r2_score(model_y[eval_idx], y_pred))
                sim_maes.append(mean_absolute_error(model_y[eval_idx], y_pred))

            if sim_r2s:
                per_model_results.append({
                    'family': held_out_family,
                    'model': model_name,
                    'R2': np.mean(sim_r2s),
                    'MAE': np.mean(sim_maes)
                })

        if per_model_results:
            family_r2 = np.mean([r['R2'] for r in per_model_results])
            family_mae = np.mean([r['MAE'] for r in per_model_results])
            models_str = ', '.join(held_out_models)
            print(f"{held_out_family:<12} {models_str:<30} {family_r2:>10.3f} {family_mae:>9.1f}pp")

            results.append({
                'family': held_out_family,
                'n_models': len(held_out_models),
                'R2': family_r2,
                'MAE': family_mae
            })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        print("-" * 65)
        print(f"{'MEAN':<12} {'':30} {results_df['R2'].mean():>10.3f} {results_df['MAE'].mean():>9.1f}pp")

    results_df.to_csv(OUTPUT_DIR / 'lofo_results.csv', index=False)
    return results_df


def phase4_family_similarity(df):
    """
    Analyze behavioral similarity between models within vs across families.
    """
    print("\n" + "=" * 80)
    print("PHASE 4: FAMILY BEHAVIORAL SIMILARITY")
    print("=" * 80)

    models = sorted(df['model_name'].unique())
    common_scenarios = sorted(df['scenario_key'].unique())

    # Build behavior matrix: models × scenarios → cooperation rate
    behavior_matrix = df.pivot_table(index='model_name', columns='scenario_key',
                                      values='coop_pct', aggfunc='first')
    behavior_matrix = behavior_matrix.dropna(axis=1)

    # Pairwise correlation matrix
    print("\n  Pairwise behavioral correlation:")
    print(f"  {'':16}", end='')
    for m in models:
        print(f" {m[:8]:>9}", end='')
    print()
    print("  " + "-" * (16 + 10 * len(models)))

    similarity_rows = []
    for i, m1 in enumerate(models):
        print(f"  {m1:<16}", end='')
        for j, m2 in enumerate(models):
            v1 = behavior_matrix.loc[m1].values
            v2 = behavior_matrix.loc[m2].values
            corr = np.corrcoef(v1, v2)[0, 1]
            print(f" {corr:>9.3f}", end='')
            if i < j:
                similarity_rows.append({
                    'model_1': m1, 'model_2': m2,
                    'family_1': MODEL_TO_FAMILY[m1],
                    'family_2': MODEL_TO_FAMILY[m2],
                    'same_family': MODEL_TO_FAMILY[m1] == MODEL_TO_FAMILY[m2],
                    'correlation': corr,
                    'mae': mean_absolute_error(v1, v2)
                })
        print()

    sim_df = pd.DataFrame(similarity_rows)

    # Within-family vs across-family
    within = sim_df[sim_df['same_family']]
    across = sim_df[~sim_df['same_family']]

    print(f"\n  Within-family correlation:  {within['correlation'].mean():.3f} "
          f"(n={len(within)}, MAE={within['mae'].mean():.1f}pp)")
    print(f"  Across-family correlation: {across['correlation'].mean():.3f} "
          f"(n={len(across)}, MAE={across['mae'].mean():.1f}pp)")
    print(f"  Difference: {within['correlation'].mean() - across['correlation'].mean():+.3f}")

    sim_df.to_csv(OUTPUT_DIR / 'family_similarity.csv', index=False)
    return sim_df


def main():
    print("=" * 80)
    print("APPROACH 2: HIERARCHICAL MODEL WITH FAMILY EFFECTS")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data...")
    df, pc_cols = load_data(n_pca=30)

    # Phase 1: Variance decomposition
    print("\n[2] Variance decomposition...")
    var_df = phase1_variance_decomposition(df.copy(), pc_cols)

    # Phase 2: LOMO with family structure
    print("\n[3] LOMO with family structure...")
    lomo_df = phase2_lomo_with_family(df.copy(), pc_cols, n_cal=10, n_sim=20)

    # Phase 3: LOFO
    print("\n[4] Leave-One-Family-Out...")
    lofo_df = phase3_lofo(df.copy(), pc_cols, n_cal=10, n_sim=20)

    # Phase 4: Family similarity
    print("\n[5] Family behavioral similarity...")
    sim_df = phase4_family_similarity(df.copy())

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if lomo_df is not None and len(lomo_df) > 0:
        plain_mean = lomo_df['R2_plain'].mean()
        fam_dum_mean = lomo_df['R2_family_dummies'].mean()
        fam_aware_mean = lomo_df['R2_family_aware'].mean()

        print(f"\n  LOMO Results:")
        print(f"    Plain Ridge:           R² = {plain_mean:.3f}")
        print(f"    + Family Dummies:      R² = {fam_dum_mean:.3f} (Δ={fam_dum_mean - plain_mean:+.3f})")
        print(f"    Family-Aware Calib.:   R² = {fam_aware_mean:.3f} (Δ={fam_aware_mean - plain_mean:+.3f})")

    if lofo_df is not None and len(lofo_df) > 0:
        print(f"\n  LOFO Results (harder test):")
        print(f"    Mean LOFO R²:          {lofo_df['R2'].mean():.3f}")

    print(f"\n  Results saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
