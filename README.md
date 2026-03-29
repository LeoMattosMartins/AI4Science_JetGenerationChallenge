# Conditional Jet Generation with Flow Matching

This repository contains the implementation of a conditional flow-matching (CFM) model designed to generate high-fidelity **JetNet particle clouds**. The project explores advanced architecture and sampling techniques to improve physical realism in jet substructure generation.

## Project Overview
The model generates jets conditioned on five specific types: **gluon (g), quark (q), top (t), W boson (W), and Z boson (Z)**. It treats the task as a flow-matching problem on masked particle clouds, supporting up to 30 particles per jet.

## Key Features & Architecture
To improve upon standard MLP baselines, several sophisticated components were integrated:
* **Set-Aware Architecture:** Uses a masked self-attention velocity network with residual feed-forward blocks to capture permutation-invariant structures and multi-prong substructure.
* **Stronger Conditioning:** Employs sinusoidal time embeddings and learned jet-type embeddings, injected via FiLM-style modulation for precise control over trajectory dynamics.
* **Feature Handling:** Predicts per-particle velocities while enforcing zero output on padded slots to handle variable-length jets.
* **Advanced Sampling:** Replaced standard Euler integration with RK2 (Heun) to reduce discretization error and improve the final Wasserstein-1 (W1) metric.
* **Training Optimizations:** Includes a U-shaped Beta-biased time schedule, gradient clipping, AdamW with warmup/cosine decay, and EMA (Exponential Moving Average) parameters for stability.

## Performance
The final model achieved a composite **Wasserstein-1 (W1) score of 0.03820** on the validation set. 

### Validation W1 Scores by Type and Observable
| Observable | Gluon | Quark | Top | W | Z |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mass** | 0.00092 | 0.00604 | 0.00175 | 0.00368 | 0.00364 |
| **n_rel** | 0.00136 | 0.00152 | 0.00115 | 0.00119 | 0.00050 |
| **phi_rel** | 0.00162 | 0.00181 | 0.00147 | 0.00075 | 0.00046 |
| **pT_rel** | 0.00231 | 0.00231 | 0.00205 | 0.00177 | 0.00189 |
| **Per-type Total** | 0.00621 | 0.01169 | 0.00641 | 0.00739 | 0.00649 |

**Analysis:**
* **Strengths:** The model performs best on gluon, top, and Z jets. Qualitative plots show it successfully recovers realistic prong-like structures in the spatial plane, particularly for heavy-object classes.
* **Challenges:** Quark jets proved to be the most difficult to model. Errors are primarily dominated by Mass and pT distributions rather than angular coordinates.

## Development & Collaboration
This project was developed by **Leonardo Mattos Martins** at Boston University. 
* **AI Collaboration:** Gemini was used as a "critic" to provide ratings and explanations on the strategy, helping to eliminate bias and identify design flaws.
* **Workflow:** The Windsurf IDE managed file synchronization between the local repository and the Google Colab environment.