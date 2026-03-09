#!/usr/bin/env python3
"""
Component Analysis: Interpret what each PCA dimension captures semantically.

Addresses the gap: "embeddings help predict cooperation, but we don't know why."

Outputs:
- PC correlations with R/S/C and cooperation rates
- Model-specific PC correlations (which PCs predict which models?)
- Extreme scenarios per PC (high/low)
- Variance explained progression (R/S/C -> PC1-3 -> All PCs)
- Figures: correlation heatmap, PC scatter, variance explained bar
"""

import sys
sys.stdout = sys.stderr = open(sys.stdout.fileno(), mode='w', buffering=1)

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Try to import matplotlib (optional for figure generation)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib/seaborn not available. Figures will not be generated.")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'analysis' / 'lomo' / 'results_test2' / 'prepared_data.csv'
EMBEDDING_CACHE = PROJECT_ROOT / 'analysis' / 'embeddings' / 'results' / 'openrouter_embeddings_cache.json'
EMBEDDING_PCA = PROJECT_ROOT / 'analysis' / 'embeddings' / 'results' / 'scenario_embeddings_openai.csv'
OUTPUT_DIR = Path(__file__).parent / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR = OUTPUT_DIR / 'figures'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

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
    """Load cooperation data and embeddings, compute PCA"""
    # Load cooperation data
    df = pd.read_csv(DATA_FILE)
    df['model_name'] = df['model'].map(MODEL_MAP)
    df['coop_pct'] = df['coop_rate']
    df['scenario_key'] = df['scenario'].str.lower().str.replace(' ', '_')

    # Load raw embeddings
    with open(EMBEDDING_CACHE, 'r') as f:
        emb_cache = json.load(f)

    # Get unique scenarios with embeddings
    scenarios = df['scenario_key'].unique()
    valid_scenarios = [s for s in scenarios if s in emb_cache]

    # Create scenario-level dataframe with R/S/C and mean cooperation
    scenario_df = df.groupby('scenario_key').agg({
        'R': 'first',
        'S': 'first',
        'C': 'first',
        'quadrant': 'first',
        'coop_pct': 'mean',
        'scenario_name': 'first'
    }).reset_index()
    scenario_df.columns = ['scenario_key', 'R', 'S', 'C', 'quadrant', 'mean_coop', 'scenario_name']
    scenario_df = scenario_df[scenario_df['scenario_key'].isin(valid_scenarios)]

    # Get embeddings in same order as scenario_df
    embeddings = np.array([emb_cache[s] for s in scenario_df['scenario_key']])

    # PCA
    n_pca_actual = min(n_pca, embeddings.shape[1], len(scenario_df) - 1)
    pca = PCA(n_components=n_pca_actual)
    pc_scores = pca.fit_transform(embeddings)
    var_explained = pca.explained_variance_ratio_

    # Add PCs to scenario_df
    pc_cols = []
    for i in range(n_pca_actual):
        col = f'PC{i+1}'
        scenario_df[col] = pc_scores[:, i]
        pc_cols.append(col)

    # Also get model-specific cooperation rates
    model_coop = df.pivot_table(
        index='scenario_key',
        columns='model_name',
        values='coop_pct'
    ).reset_index()
    scenario_df = scenario_df.merge(model_coop, on='scenario_key', how='left')

    return df, scenario_df, pc_cols, var_explained, pca


def compute_pc_correlations(scenario_df, pc_cols):
    """Compute correlations between PCs and features"""
    features = ['R', 'S', 'C', 'mean_coop']
    models = [m for m in MODEL_MAP.values() if m in scenario_df.columns]

    correlations = []
    for pc in pc_cols:
        row = {'PC': pc}
        for feat in features:
            if feat in scenario_df.columns:
                r = np.corrcoef(scenario_df[pc], scenario_df[feat])[0, 1]
                row[feat] = r
        for model in models:
            if model in scenario_df.columns and not scenario_df[model].isna().all():
                r = np.corrcoef(scenario_df[pc].values,
                               scenario_df[model].fillna(scenario_df[model].mean()).values)[0, 1]
                row[model] = r
        correlations.append(row)

    return pd.DataFrame(correlations)


def find_extreme_scenarios(scenario_df, pc_cols, n_extreme=5):
    """Find highest and lowest scenarios for each PC"""
    extremes = []
    for pc in pc_cols:
        sorted_df = scenario_df.sort_values(pc)

        # Lowest
        for idx, row in sorted_df.head(n_extreme).iterrows():
            extremes.append({
                'PC': pc,
                'direction': 'low',
                'scenario': row['scenario_name'],
                'scenario_key': row['scenario_key'],
                'pc_score': row[pc],
                'mean_coop': row['mean_coop'],
                'R': row['R'],
                'S': row['S'],
                'C': row['C']
            })

        # Highest
        for idx, row in sorted_df.tail(n_extreme).iterrows():
            extremes.append({
                'PC': pc,
                'direction': 'high',
                'scenario': row['scenario_name'],
                'scenario_key': row['scenario_key'],
                'pc_score': row[pc],
                'mean_coop': row['mean_coop'],
                'R': row['R'],
                'S': row['S'],
                'C': row['C']
            })

    return pd.DataFrame(extremes)


def compute_variance_explained_progression(df, scenario_df, pc_cols, n_sim=50, n_cal=10):
    """Compare R² from different feature sets using LOMO"""
    models = sorted(df['model_name'].dropna().unique())

    feature_sets = {
        'R/S/C only': ['R', 'S', 'C'],
        'PC1-3': pc_cols[:3],
        'PC1-5': pc_cols[:5],
        'All PCs (15)': pc_cols,
        'R/S/C + PC1-3': ['R', 'S', 'C'] + pc_cols[:3],
        'R/S/C + All PCs': ['R', 'S', 'C'] + pc_cols
    }

    results = []

    for name, feat_cols in feature_sets.items():
        # Check all features exist
        missing = [c for c in feat_cols if c not in scenario_df.columns]
        if missing:
            print(f"  Warning: Missing columns for {name}: {missing}")
            continue

        r2_scores = []

        for held_out in models:
            train_df = df[df['model_name'] != held_out].copy()
            test_df = df[df['model_name'] == held_out].copy()

            # Merge with scenario features
            train_merged = train_df.merge(
                scenario_df[['scenario_key'] + feat_cols],
                on='scenario_key',
                how='inner',
                suffixes=('', '_scenario')
            )
            test_merged = test_df.merge(
                scenario_df[['scenario_key'] + feat_cols],
                on='scenario_key',
                how='inner',
                suffixes=('', '_scenario')
            )

            if len(test_merged) < 15 or len(train_merged) < 50:
                continue

            # Get feature columns (handle potential suffix from merge)
            actual_feat_cols = []
            for col in feat_cols:
                if col in train_merged.columns:
                    actual_feat_cols.append(col)
                elif col + '_scenario' in train_merged.columns:
                    actual_feat_cols.append(col + '_scenario')

            X_train = train_merged[actual_feat_cols].values
            y_train = train_merged['coop_pct'].values
            X_test = test_merged[actual_feat_cols].values
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

    return pd.DataFrame(results)


def compute_model_divergence(scenario_df, pc_cols):
    """Check which PCs predict model-specific variance (divergent rationality)"""
    models = [m for m in MODEL_MAP.values() if m in scenario_df.columns]

    # Compute model heterogeneity (std across models) per scenario
    scenario_df['model_heterogeneity'] = scenario_df[models].std(axis=1)

    # Correlate each PC with model heterogeneity
    divergence_corrs = []
    for pc in pc_cols:
        r = np.corrcoef(scenario_df[pc].values,
                       scenario_df['model_heterogeneity'].fillna(0).values)[0, 1]
        divergence_corrs.append({'PC': pc, 'heterogeneity_corr': r})

    return pd.DataFrame(divergence_corrs)


def generate_figures(scenario_df, pc_cols, corr_df, var_explained, var_progression_df):
    """Generate publication-quality figures"""
    if not HAS_MATPLOTLIB:
        print("  Skipping figure generation (matplotlib not available)")
        return

    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. PC Correlation Heatmap
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for heatmap
    features = ['R', 'S', 'C', 'mean_coop']
    models = [m for m in MODEL_MAP.values() if m in corr_df.columns]
    cols = features + models
    cols = [c for c in cols if c in corr_df.columns]

    heatmap_data = corr_df.set_index('PC')[cols].iloc[:5]  # Top 5 PCs

    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Pearson r'})
    ax.set_title('PC Correlations with Features and Model-Specific Cooperation', fontsize=14)
    ax.set_xlabel('Feature / Model')
    ax.set_ylabel('Principal Component')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'pc_correlation_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURE_DIR / 'pc_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. PC1 vs PC2 Scatter with cooperation coloring
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(scenario_df['PC1'], scenario_df['PC2'],
                        c=scenario_df['mean_coop'], cmap='RdYlGn',
                        s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Label extreme scenarios
    for pc in ['PC1', 'PC2']:
        sorted_df = scenario_df.sort_values(pc)
        for idx, row in sorted_df.head(2).iterrows():
            ax.annotate(row['scenario_name'], (row['PC1'], row['PC2']),
                       fontsize=8, alpha=0.8, ha='center')
        for idx, row in sorted_df.tail(2).iterrows():
            ax.annotate(row['scenario_name'], (row['PC1'], row['PC2']),
                       fontsize=8, alpha=0.8, ha='center')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mean Cooperation Rate (%)', fontsize=12)
    ax.set_xlabel(f'PC1 ({var_explained[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1%} variance)', fontsize=12)
    ax.set_title('Scenario Embedding Space (PC1 vs PC2)', fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'scenario_pc_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURE_DIR / 'scenario_pc_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Variance Explained Progression Bar Chart
    if len(var_progression_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = sns.color_palette('viridis', len(var_progression_df))
        bars = ax.bar(range(len(var_progression_df)), var_progression_df['R2'],
                     yerr=var_progression_df['R2_std'], capsize=5, color=colors)

        ax.set_xticks(range(len(var_progression_df)))
        ax.set_xticklabels(var_progression_df['Feature Set'], rotation=45, ha='right')
        ax.set_ylabel('LOMO R² (with 10-scenario calibration)', fontsize=12)
        ax.set_title('Variance Explained by Different Feature Sets', fontsize=14)
        ax.set_ylim(0, 0.6)

        # Add value labels on bars
        for i, (bar, r2) in enumerate(zip(bars, var_progression_df['R2'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{r2:.2f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(FIGURE_DIR / 'variance_explained_bar.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(FIGURE_DIR / 'variance_explained_bar.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Model-specific heatmap (only top 5 PCs)
    fig, ax = plt.subplots(figsize=(10, 5))

    models = [m for m in MODEL_MAP.values() if m in corr_df.columns]
    model_data = corr_df.set_index('PC')[models].iloc[:5]

    sns.heatmap(model_data, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-0.8, vmax=0.8, ax=ax, cbar_kws={'label': 'Pearson r'})
    ax.set_title('PC-Model Cooperation Correlations: Identifying Family-Specific Patterns', fontsize=14)
    ax.set_xlabel('Model')
    ax.set_ylabel('Principal Component')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'model_specific_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURE_DIR / 'model_specific_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Figures saved to {FIGURE_DIR}")


def generate_interpretation_report(corr_df, extremes_df, var_progression_df,
                                   divergence_df, var_explained, pc_cols):
    """Generate human-readable interpretation document"""

    report = []
    report.append("# Principal Component Interpretation Report\n")
    report.append("This report interprets what each embedding dimension captures semantically.\n\n")

    # Variance explained
    report.append("## 1. Variance Explained by PCs\n")
    cumvar = np.cumsum(var_explained) * 100
    for i, (v, c) in enumerate(zip(var_explained[:5], cumvar[:5])):
        report.append(f"- **PC{i+1}**: {v*100:.1f}% (cumulative: {c:.1f}%)\n")
    report.append(f"\nTop 5 PCs explain {cumvar[4]:.1f}% of embedding variance.\n\n")

    # Per-PC interpretation
    report.append("## 2. PC Interpretations\n")

    features = ['R', 'S', 'C', 'mean_coop']
    for i, pc in enumerate(pc_cols[:5]):
        report.append(f"### {pc} ({var_explained[i]*100:.1f}% variance)\n\n")

        # Key correlations
        row = corr_df[corr_df['PC'] == pc].iloc[0]
        report.append("**Feature Correlations:**\n")
        for feat in features:
            if feat in row:
                r = row[feat]
                strength = "strong" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak"
                direction = "positive" if r > 0 else "negative"
                report.append(f"- {feat}: r = {r:.3f} ({strength} {direction})\n")

        # Extreme scenarios
        pc_extremes = extremes_df[extremes_df['PC'] == pc]

        report.append("\n**High-scoring scenarios:**\n")
        for _, e in pc_extremes[pc_extremes['direction'] == 'high'].head(3).iterrows():
            report.append(f"- {e['scenario']} (score: {e['pc_score']:.2f}, coop: {e['mean_coop']:.0f}%)\n")

        report.append("\n**Low-scoring scenarios:**\n")
        for _, e in pc_extremes[pc_extremes['direction'] == 'low'].head(3).iterrows():
            report.append(f"- {e['scenario']} (score: {e['pc_score']:.2f}, coop: {e['mean_coop']:.0f}%)\n")

        # Interpretation attempt
        report.append("\n**Tentative Interpretation:**\n")
        # Auto-generate based on correlations
        if abs(row.get('mean_coop', 0)) > 0.4:
            report.append(f"- Strongly predicts cooperation (r = {row['mean_coop']:.2f})\n")
        if abs(row.get('R', 0)) > 0.3:
            report.append(f"- Related to Reward structure (r = {row['R']:.2f})\n")
        if abs(row.get('S', 0)) > 0.3:
            report.append(f"- Related to Sucker payoff (r = {row['S']:.2f})\n")

        report.append("\n")

    # Variance progression
    report.append("## 3. Variance Explained Progression (LOMO R²)\n\n")
    report.append("| Feature Set | R² | Std | n_features |\n")
    report.append("|------------|-----|-----|------------|\n")
    for _, row in var_progression_df.iterrows():
        report.append(f"| {row['Feature Set']} | {row['R2']:.3f} | {row['R2_std']:.3f} | {row['n_features']} |\n")

    report.append("\n**Key Finding:** ")
    if len(var_progression_df) >= 2:
        rsc_r2 = var_progression_df[var_progression_df['Feature Set'] == 'R/S/C only']['R2'].values
        combined_r2 = var_progression_df[var_progression_df['Feature Set'] == 'R/S/C + All PCs']['R2'].values
        if len(rsc_r2) > 0 and len(combined_r2) > 0:
            improvement = combined_r2[0] - rsc_r2[0]
            report.append(f"Adding embeddings improves R² by {improvement:.3f} ({improvement/rsc_r2[0]*100:.0f}% relative improvement).\n")

    # Model divergence
    report.append("\n## 4. Model Divergence Analysis\n\n")
    report.append("Which PCs predict where models disagree (high heterogeneity)?\n\n")
    report.append("| PC | Heterogeneity Correlation |\n")
    report.append("|----|---------------------------|\n")
    for _, row in divergence_df.iterrows():
        report.append(f"| {row['PC']} | {row['heterogeneity_corr']:.3f} |\n")

    # Find strongest divergence predictor
    if len(divergence_df) > 0:
        max_div = divergence_df.loc[divergence_df['heterogeneity_corr'].abs().idxmax()]
        report.append(f"\n**Key Finding:** {max_div['PC']} most strongly predicts model disagreement ")
        report.append(f"(r = {max_div['heterogeneity_corr']:.3f}).\n")

    # Save report
    with open(OUTPUT_DIR / 'pc_interpretation.md', 'w') as f:
        f.write(''.join(report))

    print(f"  Interpretation report saved to {OUTPUT_DIR / 'pc_interpretation.md'}")


def main():
    print("=" * 100)
    print("COMPONENT ANALYSIS: Interpreting Embedding Dimensions")
    print("=" * 100)

    # Load data
    print("\n[1] Loading data and computing PCA...")
    df, scenario_df, pc_cols, var_explained, pca = load_data(n_pca=15)
    print(f"    Scenarios: {len(scenario_df)}")
    print(f"    Observations: {len(df)}")
    print(f"    Models: {df['model_name'].nunique()}")
    print(f"    PCs computed: {len(pc_cols)}")
    print(f"    Cumulative variance (top 5): {sum(var_explained[:5])*100:.1f}%")

    # Compute PC correlations
    print("\n[2] Computing PC correlations with features and cooperation...")
    corr_df = compute_pc_correlations(scenario_df, pc_cols)
    corr_df.to_csv(OUTPUT_DIR / 'pc_correlations.csv', index=False)
    print(f"    Saved: pc_correlations.csv")

    # Print top correlations
    print("\n    Top PC-Feature Correlations:")
    for pc in pc_cols[:5]:
        row = corr_df[corr_df['PC'] == pc].iloc[0]
        mean_coop_r = row.get('mean_coop', 0)
        print(f"    {pc}: mean_coop r={mean_coop_r:.3f}, R r={row.get('R', 0):.3f}, S r={row.get('S', 0):.3f}")

    # Find extreme scenarios
    print("\n[3] Finding extreme scenarios per PC...")
    extremes_df = find_extreme_scenarios(scenario_df, pc_cols[:5], n_extreme=5)
    extremes_df.to_csv(OUTPUT_DIR / 'extreme_scenarios.csv', index=False)
    print(f"    Saved: extreme_scenarios.csv")

    # Variance explained progression
    print("\n[4] Computing variance explained progression (LOMO)...")
    var_progression_df = compute_variance_explained_progression(df, scenario_df, pc_cols)
    var_progression_df.to_csv(OUTPUT_DIR / 'variance_explained.csv', index=False)
    print(f"    Saved: variance_explained.csv")

    print("\n    Variance Explained Progression:")
    for _, row in var_progression_df.iterrows():
        print(f"    {row['Feature Set']:<20}: R² = {row['R2']:.3f} ± {row['R2_std']:.3f}")

    # Model divergence analysis
    print("\n[5] Computing model divergence analysis...")
    divergence_df = compute_model_divergence(scenario_df, pc_cols[:5])
    divergence_df.to_csv(OUTPUT_DIR / 'pc_mechanism_correlations.csv', index=False)
    print(f"    Saved: pc_mechanism_correlations.csv")

    # Generate figures
    print("\n[6] Generating figures...")
    generate_figures(scenario_df, pc_cols, corr_df, var_explained, var_progression_df)

    # Generate interpretation report
    print("\n[7] Generating interpretation report...")
    generate_interpretation_report(corr_df, extremes_df, var_progression_df,
                                   divergence_df, var_explained, pc_cols)

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(f"""
Results saved to: {OUTPUT_DIR}

Key Outputs:
- pc_correlations.csv: PC × feature correlations
- extreme_scenarios.csv: High/low scenarios per PC
- variance_explained.csv: LOMO R² for different feature sets
- pc_mechanism_correlations.csv: PC × model heterogeneity
- pc_interpretation.md: Human-readable interpretation
- figures/: Publication-quality figures

Next Steps:
1. Review pc_interpretation.md for semantic interpretations
2. Check variance_explained.csv for R² progression
3. Use figures for paper Section 6
""")

    # Final table
    print("\nPC Interpretation Summary (Top 5):")
    print("-" * 80)
    print(f"{'PC':<6} {'Var%':<8} {'mean_coop r':<12} {'R r':<10} {'S r':<10} {'Interpretation':<30}")
    print("-" * 80)

    for i, pc in enumerate(pc_cols[:5]):
        row = corr_df[corr_df['PC'] == pc].iloc[0]
        var_pct = var_explained[i] * 100

        # Auto interpretation
        interp = []
        if abs(row.get('mean_coop', 0)) > 0.3:
            interp.append("coop-predictive")
        if abs(row.get('R', 0)) > 0.3:
            interp.append("R-related")
        if abs(row.get('S', 0)) > 0.3:
            interp.append("S-related")

        interp_str = ", ".join(interp) if interp else "subtle semantic"

        print(f"{pc:<6} {var_pct:<8.1f} {row.get('mean_coop', 0):<12.3f} {row.get('R', 0):<10.3f} {row.get('S', 0):<10.3f} {interp_str:<30}")

    print("-" * 80)


if __name__ == '__main__':
    main()
