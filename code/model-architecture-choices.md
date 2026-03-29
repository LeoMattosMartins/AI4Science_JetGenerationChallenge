# JetNet Model Architecture Choices

This document explains the design choices behind the `ParticleSetVelocity` model used in `assignment-03-starter.ipynb` and mirrored in `generate.py`.

## Goal and constraints
- We need conditional jet generation with low composite W1 score.
- Data are variable-size particle sets (max 30 particles) with masks.
- The generation script must be self-contained and exactly match notebook architecture for autograding.

## Why this architecture

### 1) Set-aware representation (not flat MLP)
- Jets are particle sets, so order should not matter.
- We use masked self-attention blocks over particles to let each particle interact with all others.
- This captures multi-prong substructure (e.g., top: 3-prong, W/Z: 2-prong) better than a flat MLP.

### 2) Physics-informed particle features
Input particle features are `(eta_rel, phi_rel, pt_rel)`, and we augment with:
- `log1p(pt_rel)`: stabilizes dynamic range of momentum-like feature.
- radial feature `r = sqrt(eta_rel^2 + phi_rel^2)`: helps encode geometric structure in the eta-phi plane.

These are concatenated and projected before attention blocks.

### 3) Strong conditioning with FiLM
Conditioning signal is time embedding + jet-type embedding.
- Time embedding: sinusoidal + MLP projection.
- Jet type embedding: learned embedding table.
- Combined conditioning is applied via FiLM (`gamma`, `beta`) at multiple stages.

Why: better conditional control over both jet type and diffusion/flow time than simple concatenation.

### 4) Mask-safe processing throughout
- Masks are applied after feature transforms and after each residual update.
- Attention uses a mask to avoid padded particle slots.
- Pooled global features divide by masked counts.

Why: prevents padded tokens from contaminating learned interactions and pooled statistics.

### 5) Local + global fusion
After block updates:
- Compute masked global summary per jet.
- Broadcast global summary and conditioning back to particles.
- Decode velocity from concatenated local/global/conditioning tensors.

Why: combines fine-grained particle information with jet-level context important for mass and shape observables.

## Training choices (paired with architecture)
- `adamw` + warmup cosine schedule.
- EMA of parameters for more stable sampling quality.
- Global norm clipping for stability.
- Weighted loss that emphasizes higher-`pt` particles.
- Mixed time sampling (beta + uniform) to learn dynamics across trajectory regions.

## Sampling choices
- Heun/RK2 integration with 64 steps.
- Reuse real mask multiplicity distribution by sampling masks from training set.
- Clip generated `pt` to nonnegative values at the end.

## Reproducibility and autograder alignment
- `generate.py` mirrors notebook model class and sampler settings.
- `generate.py` now uses `save_submission(...)` from `utils.py`, matching notebook output format.
- This reduces risk of notebook/script mismatch at submission time.
