#!/usr/bin/env python3
"""
Head-to-Head Comparison of All Prediction Approaches

Loads results from all three approaches and the Ridge baseline,
then generates:
  1. Comparison table (per-model and mean R², MAE)
  2. Paired statistical tests (t-test on R² across models)
  3. LaTeX table for paper
  4. Summary of key findings
"""

import sys
sys.stdout = sys.stderr = open(sys.stdout.fileno(), mode='w', buffering=1)

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = Path(__file__).parent
QRE_DIR = RESULTS_DIR / '01_qre_social_utility' / 'results'
HIER_DIR = RESULTS_DIR / '02_hierarchical_family' / 'results'
META_DIR = RESULTS_DIR / '03_meta_learning_eval' / 'results'
OUTPUT_DIR = RESULTS_DIR / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    """Load results from all approaches"""
    results = {}

    # QRE results
    qre_path = QRE_DIR / 'qre_lomo_results.csv'
    ridge_path = QRE_DIR / 'ridge_baseline_results.csv'
    if qre_path.exists():
        results['QRE'] = pd.read_csv(qre_path)
        print(f"  Loaded QRE results: {len(results['QRE'])} models")
    if ridge_path.exists():
        results['Ridge'] = pd.read_csv(ridge_path)
        print(f"  Loaded Ridge baseline: {len(results['Ridge'])} models")

    # Hierarchical results
    hier_path = HIER_DIR / 'lomo_family_comparison.csv'
    if hier_path.exists():
        hier_df = pd.read_csv(hier_path)
        # Extract the three variants
        results['Hier_Plain'] = hier_df[['model', 'R2_plain', 'R2_plain_std']].rename(
            columns={'R2_plain': 'R2', 'R2_plain_std': 'R2_std'})
        results['Hier_FamDummy'] = hier_df[['model', 'R2_family_dummies', 'R2_family_dummies_std']].rename(
            columns={'R2_family_dummies': 'R2', 'R2_family_dummies_std': 'R2_std'})
        results['Hier_FamAware'] = hier_df[['model', 'R2_family_aware', 'R2_family_aware_std']].rename(
            columns={'R2_family_aware': 'R2', 'R2_family_aware_std': 'R2_std'})
        print(f"  Loaded Hierarchical results: {len(hier_df)} models × 3 variants")

    # Meta-learning results
    proto_path = META_DIR / 'prototypical_results.csv'
    if proto_path.exists():
        proto_df = pd.read_csv(proto_path)
        results['Proto'] = proto_df[['model', 'R2_proto', 'R2_proto_std']].rename(
            columns={'R2_proto': 'R2', 'R2_proto_std': 'R2_std'})
        print(f"  Loaded Prototypical results: {len(proto_df)} models")

    # LOFO
    lofo_path = HIER_DIR / 'lofo_results.csv'
    if lofo_path.exists():
        results['LOFO'] = pd.read_csv(lofo_path)
        print(f"  Loaded LOFO results: {len(results['LOFO'])} families")

    return results


def build_comparison_table(results):
    """Build per-model comparison table"""
    print("\n" + "=" * 80)
    print("PER-MODEL COMPARISON (LOMO R²)")
    print("=" * 80)

    # Get list of models from any result
    models = None
    for key in ['Ridge', 'QRE', 'Hier_Plain']:
        if key in results:
            models = sorted(results[key]['model'].unique())
            break

    if models is None:
        print("  No results available!")
        return None

    # Build comparison DataFrame
    approach_names = ['Ridge', 'QRE', 'Hier_FamDummy', 'Hier_FamAware', 'Proto']
    display_names = ['Ridge\n(baseline)', 'QRE\n(Approach 1)', 'Hier+Family\n(Approach 2a)',
                     'Fam-Aware Cal.\n(Approach 2b)', 'Prototypical\n(Approach 3)']

    rows = []
    for model in models:
        row = {'Model': model}
        for approach, display in zip(approach_names, display_names):
            if approach in results:
                model_data = results[approach][results[approach]['model'] == model]
                if len(model_data) > 0:
                    row[display] = model_data['R2'].values[0]
        rows.append(row)

    # Add mean row
    mean_row = {'Model': 'MEAN'}
    for display in display_names:
        vals = [r.get(display) for r in rows if display in r and r.get(display) is not None]
        if vals:
            mean_row[display] = np.mean(vals)
    rows.append(mean_row)

    comp_df = pd.DataFrame(rows)

    # Print table
    print()
    header = f"{'Model':<16}"
    for display in display_names:
        short = display.split('\n')[0]
        header += f" {short:>14}"
    print(header)
    print("-" * (16 + 15 * len(display_names)))

    for _, row in comp_df.iterrows():
        line = f"{row['Model']:<16}"
        for display in display_names:
            if display in row and pd.notna(row[display]):
                line += f" {row[display]:>14.3f}"
            else:
                line += f" {'N/A':>14}"
        print(line)

    return comp_df


def run_statistical_tests(results):
    """Paired t-tests comparing approaches"""
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS (Paired t-tests on per-model R²)")
    print("=" * 80)

    comparisons = [
        ('Ridge', 'QRE', 'Ridge vs QRE'),
        ('Ridge', 'Hier_FamDummy', 'Ridge vs Hierarchical+Family'),
        ('Ridge', 'Hier_FamAware', 'Ridge vs Family-Aware Calibration'),
        ('Ridge', 'Proto', 'Ridge vs Prototypical'),
        ('QRE', 'Hier_FamAware', 'QRE vs Family-Aware'),
        ('QRE', 'Proto', 'QRE vs Prototypical'),
    ]

    test_results = []

    print(f"\n{'Comparison':<40} {'Δ R²':>8} {'t-stat':>8} {'p-value':>10} {'Sig':>5}")
    print("-" * 75)

    for approach1, approach2, label in comparisons:
        if approach1 not in results or approach2 not in results:
            continue

        df1 = results[approach1].set_index('model')['R2']
        df2 = results[approach2].set_index('model')['R2']

        common_models = df1.index.intersection(df2.index)
        if len(common_models) < 3:
            continue

        vals1 = df1.loc[common_models].values
        vals2 = df2.loc[common_models].values

        t_stat, p_value = stats.ttest_rel(vals2, vals1)
        delta = np.mean(vals2) - np.mean(vals1)
        sig = '*' if p_value < 0.05 else ('†' if p_value < 0.10 else '')

        test_results.append({
            'comparison': label,
            'approach_1': approach1,
            'approach_2': approach2,
            'delta_R2': delta,
            't_stat': t_stat,
            'p_value': p_value,
            'n_models': len(common_models),
            'significant': p_value < 0.05
        })

        print(f"{label:<40} {delta:>+8.3f} {t_stat:>8.2f} {p_value:>10.4f} {sig:>5}")

    print("\n  * = p < 0.05, † = p < 0.10")
    print(f"  Note: With only 7 models, statistical power is limited.")

    tests_df = pd.DataFrame(test_results)
    tests_df.to_csv(OUTPUT_DIR / 'statistical_tests.csv', index=False)
    return tests_df


def generate_latex_table(results):
    """Generate LaTeX table for paper"""
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)

    models = None
    for key in ['Ridge', 'QRE']:
        if key in results:
            models = sorted(results[key]['model'].unique())
            break

    if models is None:
        return

    approaches = [
        ('Ridge', 'Ridge (baseline)'),
        ('QRE', 'QRE + Social Utility'),
        ('Hier_FamAware', 'Family-Aware Calib.'),
        ('Proto', 'Prototypical')
    ]

    # Get means for bold formatting (best per model)
    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Leave-One-Model-Out R$^2$ across prediction approaches. "
                       r"Bold indicates best per model.}")
    latex_lines.append(r"\label{tab:prediction_comparison}")

    n_cols = len([a for a, _ in approaches if a in results])
    latex_lines.append(r"\begin{tabular}{l" + "c" * n_cols + "}")
    latex_lines.append(r"\toprule")

    # Header
    header_parts = ["Model"]
    for approach, display in approaches:
        if approach in results:
            header_parts.append(display)
    latex_lines.append(" & ".join(header_parts) + r" \\")
    latex_lines.append(r"\midrule")

    # Data rows
    means = {a: [] for a, _ in approaches}

    for model in models:
        row_vals = {}
        for approach, _ in approaches:
            if approach in results:
                model_data = results[approach][results[approach]['model'] == model]
                if len(model_data) > 0:
                    row_vals[approach] = model_data['R2'].values[0]
                    means[approach].append(row_vals[approach])

        best_val = max(row_vals.values()) if row_vals else -999

        parts = [model.replace('_', r'\_')]
        for approach, _ in approaches:
            if approach in results:
                if approach in row_vals:
                    val = row_vals[approach]
                    if abs(val - best_val) < 0.001:
                        parts.append(rf"\textbf{{{val:.3f}}}")
                    else:
                        parts.append(f"{val:.3f}")
                else:
                    parts.append("--")
        latex_lines.append(" & ".join(parts) + r" \\")

    # Mean row
    latex_lines.append(r"\midrule")
    mean_parts = [r"\textbf{Mean}"]
    mean_vals = {}
    for approach, _ in approaches:
        if approach in results and means[approach]:
            mean_vals[approach] = np.mean(means[approach])

    best_mean = max(mean_vals.values()) if mean_vals else -999

    for approach, _ in approaches:
        if approach in results:
            if approach in mean_vals:
                val = mean_vals[approach]
                if abs(val - best_mean) < 0.001:
                    mean_parts.append(rf"\textbf{{{val:.3f}}}")
                else:
                    mean_parts.append(f"{val:.3f}")
            else:
                mean_parts.append("--")
    latex_lines.append(" & ".join(mean_parts) + r" \\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    latex_str = "\n".join(latex_lines)
    print()
    print(latex_str)

    with open(OUTPUT_DIR / 'comparison_table.tex', 'w') as f:
        f.write(latex_str)

    return latex_str


def summarize_findings(results):
    """Generate key findings summary"""
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    findings = []

    # Best approach
    approach_means = {}
    for name in ['Ridge', 'QRE', 'Hier_FamDummy', 'Hier_FamAware', 'Proto']:
        if name in results:
            approach_means[name] = results[name]['R2'].mean()

    if approach_means:
        best = max(approach_means, key=approach_means.get)
        display_map = {
            'Ridge': 'Ridge regression (baseline)',
            'QRE': 'QRE + Social Utility',
            'Hier_FamDummy': 'Hierarchical with family dummies',
            'Hier_FamAware': 'Family-aware calibration',
            'Proto': 'Prototypical network analogue'
        }
        print(f"\n  1. BEST APPROACH: {display_map.get(best, best)} (R² = {approach_means[best]:.3f})")
        findings.append(f"Best approach: {display_map.get(best, best)} (R² = {approach_means[best]:.3f})")

        if 'Ridge' in approach_means:
            ridge_r2 = approach_means['Ridge']
            print(f"     Ridge baseline: R² = {ridge_r2:.3f}")
            for name, r2 in sorted(approach_means.items(), key=lambda x: -x[1]):
                if name != 'Ridge':
                    delta = r2 - ridge_r2
                    print(f"     {display_map.get(name, name)}: R² = {r2:.3f} (Δ = {delta:+.3f})")

    # QRE interpretability
    qre_params_path = QRE_DIR / 'qre_model_parameters.csv'
    if qre_params_path.exists():
        params = pd.read_csv(qre_params_path)
        print(f"\n  2. QRE MODEL INTERPRETABILITY:")
        print(f"     Rationality (λ) range: [{params['lambda'].min():.2f}, {params['lambda'].max():.2f}]")
        print(f"     Cooperation bias (β) range: [{params['beta'].min():.2f}, {params['beta'].max():.2f}]")
        most_rational = params.loc[params['lambda'].idxmax(), 'model']
        most_cooperative = params.loc[params['beta'].idxmax(), 'model']
        print(f"     Most 'rational' (highest λ): {most_rational}")
        print(f"     Most cooperative (highest β): {most_cooperative}")
        findings.append(f"QRE: λ range [{params['lambda'].min():.2f}, {params['lambda'].max():.2f}], "
                       f"most rational = {most_rational}")

    # Family effects
    if 'Hier_FamAware' in results and 'Ridge' in results:
        fam_delta = approach_means.get('Hier_FamAware', 0) - approach_means.get('Ridge', 0)
        print(f"\n  3. FAMILY STRUCTURE:")
        print(f"     Family-aware vs plain: Δ R² = {fam_delta:+.3f}")
        if fam_delta > 0.01:
            print(f"     → Family structure provides modest improvement")
        else:
            print(f"     → Family structure does not meaningfully help")
        findings.append(f"Family structure delta: {fam_delta:+.3f}")

    # LOFO results
    if 'LOFO' in results:
        lofo = results['LOFO']
        print(f"\n  4. CROSS-FAMILY TRANSFER (LOFO):")
        print(f"     Mean LOFO R² = {lofo['R2'].mean():.3f}")
        for _, row in lofo.iterrows():
            print(f"     {row['family']}: R² = {row['R2']:.3f}")

    # Calibration efficiency
    auc_path = META_DIR / 'adaptation_efficiency.csv'
    if auc_path.exists():
        auc_df = pd.read_csv(auc_path)
        print(f"\n  5. ADAPTATION EFFICIENCY:")
        print(f"     Mean AUC: {auc_df['AUC'].mean():.3f}")
        easiest = auc_df.loc[auc_df['AUC'].idxmax()]
        hardest = auc_df.loc[auc_df['AUC'].idxmin()]
        print(f"     Easiest to adapt: {easiest['model']} (AUC={easiest['AUC']:.3f})")
        print(f"     Hardest to adapt: {hardest['model']} (AUC={hardest['AUC']:.3f})")

    # Save findings
    with open(OUTPUT_DIR / 'key_findings.txt', 'w') as f:
        for finding in findings:
            f.write(finding + '\n')

    return findings


def main():
    print("=" * 80)
    print("PREDICTION FRAMEWORK V2: HEAD-TO-HEAD COMPARISON")
    print("=" * 80)

    # Load all results
    print("\n[1] Loading results from all approaches...")
    results = load_results()

    if not results:
        print("\n  ERROR: No results found. Run individual scripts first:")
        print("    python 01_qre_social_utility/fit_qre_model.py")
        print("    python 02_hierarchical_family/fit_hierarchical.py")
        print("    python 03_meta_learning_eval/meta_learning_protocol.py")
        return

    # Build comparison table
    print("\n[2] Building comparison table...")
    comp_df = build_comparison_table(results)
    if comp_df is not None:
        comp_df.to_csv(OUTPUT_DIR / 'comparison_table.csv', index=False)

    # Statistical tests
    print("\n[3] Running statistical tests...")
    tests_df = run_statistical_tests(results)

    # LaTeX table
    print("\n[4] Generating LaTeX table...")
    latex = generate_latex_table(results)

    # Key findings
    print("\n[5] Summarizing findings...")
    findings = summarize_findings(results)

    print(f"\n{'='*80}")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
