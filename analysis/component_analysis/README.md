# Component Analysis: Interpreting Embedding Dimensions

This analysis addresses a key gap in the paper: "embeddings help predict cooperation, but we don't know why."

## Purpose

Principal Component Analysis (PCA) of scenario embeddings captures latent semantic dimensions. This analysis interprets what each PC represents by:

1. **Correlating PCs with known features** (R, S, C, cooperation rates)
2. **Examining extreme scenarios** (highest/lowest per PC)
3. **Testing model-specific patterns** (which PCs predict which models?)
4. **Connecting to mechanisms** (do PCs predict model divergence?)

## Running the Analysis

```bash
cd analysis/component_analysis
python component_analysis.py
```

## Outputs

### Data Files (`results/`)

| File | Description |
|------|-------------|
| `pc_correlations.csv` | Correlation matrix: PC × {R, S, C, mean_coop, model-specific rates} |
| `extreme_scenarios.csv` | Top 5 highest/lowest scenarios per PC |
| `variance_explained.csv` | LOMO R² for different feature sets (R/S/C → PC1-3 → All PCs) |
| `pc_mechanism_correlations.csv` | PC × model heterogeneity (divergent rationality proxy) |
| `pc_interpretation.md` | Human-readable interpretation report |

### Figures (`results/figures/`)

| Figure | Description |
|--------|-------------|
| `pc_correlation_heatmap.pdf` | Heatmap of PC × feature correlations |
| `scenario_pc_scatter.pdf` | PC1 vs PC2 scatter, colored by cooperation |
| `variance_explained_bar.pdf` | Bar chart of R² progression |
| `model_specific_heatmap.pdf` | PC × model-specific cooperation correlations |

## Key Questions Addressed

1. **What does PC1 capture?**
   - Check correlations with R/S/C, cooperation
   - Examine extreme scenarios (e.g., all environmental vs all business?)

2. **How much variance do interpretable PCs explain?**
   - Compare R/S/C only → PC1-3 → All 15 PCs
   - If PC1-3 ≈ All PCs, we have parsimony

3. **Are PCs universal or model-specific?**
   - If PC3 predicts Claude (r=0.6) but not GPT (r=0.1), it's family-specific
   - This explains why embeddings help differently per model

4. **Do PCs connect to main findings?**
   - High heterogeneity correlation → PC predicts where models diverge
   - Links embedding space to "divergent rationality" finding

## Integration with Paper

### Methods (1 paragraph)

> "To interpret embedding dimensions, we applied PCA to scenario embeddings and correlated each component with manual annotations (R/S/C scores) and model-specific cooperation rates. We examined extreme-scoring scenarios to derive semantic interpretations and tested whether components connect to our discovered mechanisms."

### Results (Section 6 expansion)

1. **Table: PC Interpretations** - Copy from `pc_interpretation.md`
2. **Figure: PC1 vs PC2 Scatter** - Use `scenario_pc_scatter.pdf`
3. **Variance Explained** - Reference `variance_explained.csv`

### Appendix

- Full correlation matrix (15 PCs × all features)
- Extended scenario lists per PC

## Dependencies

- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

## Data Dependencies

- `analysis/lomo/results_test2/prepared_data.csv` - Cooperation rates
- `analysis/embeddings/results/openrouter_embeddings_cache.json` - Raw 1536-dim embeddings

## Expected Results

Based on typical embedding behavior:

| PC | Likely Interpretation | Rationale |
|----|----------------------|-----------|
| PC1 | General cooperation norm | Dominant axis; expect correlation with C-score |
| PC2 | Relationship/social distance | Often separates stranger vs partner scenarios |
| PC3 | Domain framing | May separate business vs environmental |
| PC4-5 | Subtler features | Linguistic nuance; harder to interpret |

Target: PC1-3 explain ~40-50% of variance and achieve R² ≈ 0.35-0.40.
