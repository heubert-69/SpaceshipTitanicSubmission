### Epsilon-Correlation Penalized Ensemble for Tabular Prediction

A research-inspired machine learning system that combines heterogeneous models using a diversity-aware, calibration-penalized ensemble weighting scheme.

Instead of simple averaging or stacking, this project introduces an epsilon-controlled correlation penalty to reduce redundancy between models and improve generalization.

---

### Problem Statement

Tabular datasets often benefit from ensemble learning, but traditional methods suffer from:

Over-reliance on similar models (low diversity)
Poor probability calibration
Arbitrary or heuristic weight averaging
Lack of correlation awareness between models

This project addresses these issues using:

A probabilistic ensemble weighting system with correlation penalization

---

### Key Idea

Each model contributes to the final prediction based on:

1. Performance quality
- F1 Score (classification strength)
- Brier Score (probability calibration)
2. Redundancy penalty
- Pairwise correlation between model predictions
- Controlled using an epsilon exponent

---

### Model Architecture

The ensemble consists of heterogeneous learners:
```
| Model                 | Type                       |
| --------------------- | -------------------------- |
| Logistic Regression   | Linear baseline            |
| Random Forest         | Bagging                    |
| Gradient Boosting     | Boosting                   |
| LightGBM              | Histogram-based boosting   |
| CatBoost              | Categorical-aware boosting |
| XGBoost Random Forest | Hybrid ensemble            |
```
---

### Evaluation Metrics

Two complementary metrics are used:

- F1 Score
    - Measures classification performance under imbalance.

- Brier Score
    - Measures probability calibration quality:

        python```BS= 1/N∑(pi−yi)2```

Lower is better.

---

### Ensemble Strategy: 

Step 1 — Train base models

- Each model is trained independently on the same feature space.

Step 2 — Generate validation predictions

- Collect probability outputs:
```python
proba_registry = {
    "model_name": predictions
}
Step 3 — Compute correlation matrix
corr_matrix = proba_df.corr()
```

This measures redundancy between models.

Step 4 — Compute weights

Weights combine:

- performance
- calibration
- diversity penalty

Step 5 — Normalize weights

- Ensures probabilistic interpretation.

Step 6 — Final prediction
```python
ensemble_proba = Σ (w_i × p_i)
ensemble_pred = (ensemble_proba > 0.5)
```
---

### Key Results
Model Performance Snapshot: 
- Best Model: CatBoost
- F1 Score: ~0.816
- Brier Score: ~0.119
```bash
| Model               | Weight |
| ------------------- | ------ |
| CatBoost            | ~0.195 |
| Gradient Boosting   | ~0.185 |
| Random Forest       | ~0.178 |
| LightGBM            | ~0.173 |
| Logistic Regression | ~0.172 |
| XGBoost RF          | ~0.097 |
```

---

### Insights:
- CatBoost dominates due to:
  - strong categorical handling
  - low calibration error
  - high stability
- XGBoost RF is downweighted because:
    - high correlation with tree-based models
    - redundant feature space behavior
- Ensemble improves robustness by:
    - reducing variance
    - improving probability calibration
    - stabilizing predictions under noise

---

### Limitations
- Label encoding introduces ordinal bias
- Correlation assumes linear dependency
- No learned meta-model (pure weighting system)
- Epsilon is manually tuned

---

### Summary

This project implements a diversity-aware ensemble learning system that goes beyond standard stacking by explicitly penalizing model correlation using an epsilon-controlled mechanism.

It demonstrates:

- ensemble theory application
- probabilistic reasoning
- calibration-aware ML
- redundancy-aware weighting