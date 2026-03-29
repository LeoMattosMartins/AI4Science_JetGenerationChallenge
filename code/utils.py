"""Shared utilities for Assignment 3: Jet Generation Challenge."""

import pickle

import jax
import jax.numpy as jnp
import numpy as np

JET_TYPES = ["g", "q", "t", "w", "z"]
JET_NAMES = {"g": "Gluon", "q": "Quark", "t": "Top", "w": "W boson", "z": "Z boson"}
N_PARTICLES = 30
N_FEATURES = 3
N_TYPES = len(JET_TYPES)


def sinusoidal_embedding(t, dim=128):
    """Map scalar t in [0,1] to a vector of sines and cosines."""
    half = dim // 2
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / half)
    args = t[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


def save_submission(gen_jets_dict, gen_masks_dict, path="submission.npz"):
    """Save generated jets for evaluation.

    Expected format: .npz file with keys '{type}_jets' and '{type}_masks'
    for each jet type in ['g', 'q', 't', 'w', 'z'].

    Each jets array should be shape (N, 30, 3) and each masks array (N, 30).
    """
    save_dict = {}
    for jt in JET_TYPES:
        save_dict[f"{jt}_jets"] = np.array(gen_jets_dict[jt])
        save_dict[f"{jt}_masks"] = np.array(gen_masks_dict[jt])
    np.savez_compressed(path, **save_dict)
    print(f"Saved submission to {path}")
    for jt in JET_TYPES:
        print(f"  {jt}: {save_dict[f'{jt}_jets'].shape[0]} jets")


def load_submission(path="submission.npz"):
    """Load a submission file."""
    data = np.load(path)
    jets = {jt: data[f"{jt}_jets"] for jt in JET_TYPES}
    masks = {jt: data[f"{jt}_masks"] for jt in JET_TYPES}
    return jets, masks


def save_model(params, path="model_params.pkl"):
    """Save model parameters."""
    params_np = jax.tree.map(np.asarray, params)
    with open(path, "wb") as f:
        pickle.dump(params_np, f)
    print(f"Saved model to {path}")


def load_model(path="model_params.pkl"):
    """Load model parameters."""
    with open(path, "rb") as f:
        params_np = pickle.load(f)
    return jax.tree.map(jnp.asarray, params_np)


def compute_jet_mass(jets, masks):
    """Compute invariant mass from particle (eta, phi, pt_rel).

    Treats each particle as massless: E = pT * cosh(eta), px = pT * cos(phi), etc.
    Jet 4-momentum is the sum of particle 4-momenta.
    """
    eta, phi, pt = jets[:, :, 0], jets[:, :, 1], jets[:, :, 2]
    pt = np.clip(pt, 0, None)
    m = masks

    px = pt * np.cos(phi) * m
    py = pt * np.sin(phi) * m
    pz = pt * np.sinh(eta) * m
    E = pt * np.cosh(eta) * m

    Px, Py, Pz, En = px.sum(1), py.sum(1), pz.sum(1), E.sum(1)
    mass_sq = En**2 - Px**2 - Py**2 - Pz**2
    return np.sqrt(np.clip(mass_sq, 0, None))
