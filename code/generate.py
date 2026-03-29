"""
Generation script for Assignment 3: Jet Generation Challenge.

This script must be self-contained: it defines the model architecture,
loads trained parameters, generates jets, and saves submission.npz.

Usage:
    python generate.py                  # Full submission: 2000 jets/type
    python generate.py --n-samples 10   # Quick test: 10 jets/type

Update the model/sampler to match the architecture trained in the notebook.
"""

import argparse
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from utils import JET_TYPES, N_FEATURES, N_PARTICLES, N_TYPES, load_model, save_submission, sinusoidal_embedding

N_SAMPLES = 2000  # per jet type
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "model_params.pkl"
TRAIN_DATA_PATH = SCRIPT_DIR.parent / "data" / "train.npz"
SUBMISSION_PATH = SCRIPT_DIR / "submission.npz"


def resolve_train_data_path() -> Path:
    candidates = [
        TRAIN_DATA_PATH,
        SCRIPT_DIR / "data" / "train.npz",
        Path.cwd() / "data" / "train.npz",
        Path.cwd() / "code" / "data" / "train.npz",
    ]

    for path in candidates:
        if path.exists():
            return path

    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not find train.npz. Checked:\n"
        f"{searched}\n"
        "Place data at either data/train.npz or code/data/train.npz."
    )


# ── Model definition ──────────────────────────────────────────
# This model definition must match the architecture used to train
# `model_params.pkl` in the notebook.


class ParticleSetVelocity(nn.Module):
    """JetNet-tailored conditional velocity model (set attention + FiLM conditioning)."""

    hidden_dim: int = 256
    n_blocks: int = 4
    n_heads: int = 4
    ff_mult: int = 2
    n_types: int = N_TYPES
    time_dim: int = 96

    @nn.compact
    def __call__(self, x_t, t, y, mask, train=False):
        B, N, _ = x_t.shape
        m = mask[:, :, None]

        te = sinusoidal_embedding(t, self.time_dim)
        te = nn.silu(nn.Dense(self.hidden_dim)(te))
        y_emb = nn.Embed(num_embeddings=self.n_types, features=self.hidden_dim)(y)
        cond = nn.LayerNorm()(te + y_emb)

        pt = x_t[:, :, 2:3]
        log_pt = jnp.log1p(jnp.clip(pt, a_min=0.0))
        r = jnp.sqrt(jnp.sum(x_t[:, :, :2] ** 2, axis=-1, keepdims=True) + 1e-8)
        h = jnp.concatenate([x_t, log_pt, r], axis=-1)
        h = nn.silu(nn.Dense(self.hidden_dim)(h))

        film0 = nn.Dense(2 * self.hidden_dim)(cond)
        gamma0, beta0 = jnp.split(film0, 2, axis=-1)
        h = (1.0 + gamma0[:, None, :]) * h + beta0[:, None, :]
        h = h * m

        attn_mask = (mask > 0.5)[:, None, None, :]

        for _ in range(self.n_blocks):
            h_in = h
            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.hidden_dim,
                out_features=self.hidden_dim,
                dropout_rate=0.0,
            )(h, h, mask=attn_mask, deterministic=not train)
            h = nn.LayerNorm()(h_in + attn_out)

            ff = nn.Dense(self.hidden_dim * self.ff_mult)(h)
            ff = nn.gelu(ff)
            ff = nn.Dense(self.hidden_dim)(ff)

            film = nn.Dense(2 * self.hidden_dim)(cond)
            gamma, beta = jnp.split(film, 2, axis=-1)
            h = nn.LayerNorm()(h + (1.0 + gamma[:, None, :]) * ff + beta[:, None, :])
            h = h * m

            denom = jnp.clip(mask.sum(axis=1, keepdims=True), a_min=1.0)
            g = h.sum(axis=1) / denom
            g = nn.silu(nn.Dense(self.hidden_dim)(jnp.concatenate([g, cond], axis=-1)))
            h = (h + g[:, None, :]) * m

        denom = jnp.clip(mask.sum(axis=1, keepdims=True), a_min=1.0)
        global_feat = h.sum(axis=1) / denom

        globalN = jnp.broadcast_to(global_feat[:, None, :], (B, N, self.hidden_dim))
        condN = jnp.broadcast_to(cond[:, None, :], (B, N, self.hidden_dim))

        out_in = jnp.concatenate([h, globalN, condN], axis=-1)
        out = nn.silu(nn.Dense(2 * self.hidden_dim)(out_in))
        out = nn.silu(nn.Dense(self.hidden_dim)(out))
        v = nn.Dense(N_FEATURES)(out)
        return v * m


# ── Generation ────────────────────────────────────────────────


def sample_jets(model, params, jet_type_idx, masks_ref, key, n_samples=N_SAMPLES, steps=64):
    """Generate jets via 2nd-order (Heun / RK2) integration.

    Feel free to modify, but don't add additional dependencies, keep the function signature the same for grading.
    """
    dt = 1.0 / steps
    k1, k2 = jr.split(key)
    mask_idx = jr.randint(k1, (n_samples,), 0, masks_ref.shape[0])
    masks = jnp.array(masks_ref[np.array(mask_idx)], dtype=jnp.float32)
    x = jr.normal(k2, (n_samples, N_PARTICLES, N_FEATURES), dtype=jnp.float32) * masks[:, :, None]
    y = jnp.full((n_samples,), jet_type_idx, dtype=jnp.int32)

    @jax.jit
    def rk2_step(x, i):
        t = jnp.full((n_samples,), i * dt, dtype=jnp.float32)
        v1 = model.apply(params, x, t, y, masks, train=False)

        x_pred = x + dt * v1
        t2 = jnp.full((n_samples,), (i + 1) * dt, dtype=jnp.float32)
        v2 = model.apply(params, x_pred, t2, y, masks, train=False)

        return x + dt * 0.5 * (v1 + v2)

    for i in range(steps):
        x = rk2_step(x, i)

    x = x * masks[:, :, None]
    x = x.at[:, :, 2].set(jnp.clip(x[:, :, 2], a_min=0.0))
    return np.array(x), np.array(masks)


# ── Main ─────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES, help="Jets per type (default: 2000)")
    args = parser.parse_args()
    n = args.n_samples

    # Load model
    model = ParticleSetVelocity()
    params = load_model(str(MODEL_PATH))

    # Load training masks (needed to sample realistic particle multiplicities)
    train_npz = np.load(str(resolve_train_data_path()))
    train_masks = {jt: train_npz[f"{jt}_masks"] for jt in JET_TYPES}

    # Generate
    gen_jets, gen_masks = {}, {}
    for i, jt in enumerate(JET_TYPES):
        print(f"Generating {n} {jt} jets...")
        gen_jets[jt], gen_masks[jt] = sample_jets(model, params, i, train_masks[jt], jr.PRNGKey(i), n_samples=n)

    # Save (same helper used in notebook)
    save_submission(gen_jets, gen_masks, str(SUBMISSION_PATH))
    for jt in JET_TYPES:
        print(f"  {jt}: {gen_jets[jt].shape}")


if __name__ == "__main__":
    main()
