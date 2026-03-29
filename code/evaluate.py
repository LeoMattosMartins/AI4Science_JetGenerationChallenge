"""
Evaluation script for Assignment 3: Jet Generation Challenge.

Computes the composite W1 score (leaderboard metric) for a submission file.

Usage:
    python evaluate.py submission.npz --reference data/val.npz
"""

import argparse

import numpy as np
from scipy.stats import wasserstein_distance

from utils import JET_TYPES, compute_jet_mass, load_submission


def compute_w1_score(gen_jets, gen_masks, real_jets, real_masks):
    """Compute composite W1 score across all jet types and observables."""
    results = {}
    total = 0.0

    for jt in JET_TYPES:
        gen, gen_m = gen_jets[jt], gen_masks[jt]
        real, real_m = real_jets[jt], real_masks[jt]
        scores = {}
        scores["mass"] = wasserstein_distance(compute_jet_mass(gen, gen_m), compute_jet_mass(real, real_m))
        for f_idx, fname in enumerate(["eta", "phi", "pt"]):
            gen_vals = gen[:, :, f_idx][gen_m > 0.5]
            real_vals = real[:, :, f_idx][real_m > 0.5]
            scores[fname] = wasserstein_distance(gen_vals, real_vals)
        results[jt] = scores
        total += sum(scores.values())

    return total, results


def validate_submission(jets, masks):
    """Check that the submission has the right format."""
    for jt in JET_TYPES:
        assert jt in jets, f"Missing jet type: {jt}"
        assert jt in masks, f"Missing mask for jet type: {jt}"
        assert jets[jt].ndim == 3 and jets[jt].shape[1:] == (30, 3), (
            f"Jets[{jt}] shape should be (N, 30, 3), got {jets[jt].shape}"
        )
        assert masks[jt].ndim == 2 and masks[jt].shape[1] == 30, (
            f"Masks[{jt}] shape should be (N, 30), got {masks[jt].shape}"
        )
        assert np.all(np.isfinite(jets[jt])), f"Jets[{jt}] contains NaN or Inf"
    print("Submission format: OK")


def main():
    parser = argparse.ArgumentParser(description="Evaluate jet generation submission")
    parser.add_argument("submission", help="Path to submission .npz file")
    parser.add_argument("--reference", required=True, help="Path to reference .npz file")
    args = parser.parse_args()

    print(f"Loading submission: {args.submission}")
    gen_jets, gen_masks = load_submission(args.submission)
    validate_submission(gen_jets, gen_masks)

    print(f"Loading reference: {args.reference}")
    real_jets, real_masks = load_submission(args.reference)

    print("\nComputing W1 scores...\n")
    total, results = compute_w1_score(gen_jets, gen_masks, real_jets, real_masks)

    print(f"{'Type':>6} | {'Mass':>8} | {'eta':>8} | {'phi':>8} | {'pt':>8} | {'Total':>8}")
    print("-" * 60)
    for jt in JET_TYPES:
        s = results[jt]
        row_total = sum(s.values())
        print(f"{jt:>6} | {s['mass']:8.5f} | {s['eta']:8.5f} | {s['phi']:8.5f} | {s['pt']:8.5f} | {row_total:8.5f}")
    print("-" * 60)
    print(f"{'TOTAL':>6} | {'':>8} | {'':>8} | {'':>8} | {'':>8} | {total:8.5f}")
    print(f"\nLeaderboard score: {total:.5f}")


if __name__ == "__main__":
    main()
