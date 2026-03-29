"""
Autograding test for Assignment 3: Jet Generation Challenge.

Runs generate.py with a small number of samples and validates the output.

Setup command:  pip install -r requirements.txt
Run command:    pytest test_generate.py -v
"""

import os
import subprocess
import sys

import numpy as np

JET_TYPES = ["g", "q", "t", "w", "z"]
N_PARTICLES = 30
N_FEATURES = 3
TEST_N_SAMPLES = 10  # Small number for fast CI validation


def test_generate_and_validate():
    """generate.py runs and produces a valid submission.npz."""

    # Check required files exist
    assert os.path.exists("generate.py"), "generate.py not found"
    assert os.path.exists("model_params.pkl"), "model_params.pkl not found"

    # Remove existing submission so we know generate.py creates a fresh one
    if os.path.exists("submission.npz"):
        os.remove("submission.npz")

    # Run generate.py with small sample count for speed
    result = subprocess.run(
        [sys.executable, "generate.py", "--n-samples", str(TEST_N_SAMPLES)],
        capture_output=True,
        text=True,
        timeout=300,  # 5 min max
    )
    assert result.returncode == 0, (
        f"generate.py failed with exit code {result.returncode}.\n"
        f"stdout: {result.stdout[-500:]}\n"
        f"stderr: {result.stderr[-500:]}"
    )

    # Check submission.npz was created
    assert os.path.exists("submission.npz"), "generate.py did not produce submission.npz"

    # Validate format
    data = np.load("submission.npz")
    for jt in JET_TYPES:
        jets_key = f"{jt}_jets"
        masks_key = f"{jt}_masks"
        assert jets_key in data, f"Missing key: {jets_key}"
        assert masks_key in data, f"Missing key: {masks_key}"

        jets = data[jets_key]
        masks = data[masks_key]

        # Shape: (N, 30, 3) and (N, 30) — N should match TEST_N_SAMPLES
        assert jets.shape == (TEST_N_SAMPLES, N_PARTICLES, N_FEATURES), (
            f"{jets_key} shape should be ({TEST_N_SAMPLES}, {N_PARTICLES}, {N_FEATURES}), got {jets.shape}"
        )
        assert masks.shape == (TEST_N_SAMPLES, N_PARTICLES), (
            f"{masks_key} shape should be ({TEST_N_SAMPLES}, {N_PARTICLES}), got {masks.shape}"
        )

        # Finite
        assert np.all(np.isfinite(jets)), f"{jets_key} contains NaN or Inf"

        # Non-degenerate
        real_particles = jets[masks > 0.5]
        assert real_particles.std() > 0.001, (
            f"{jets_key} has near-zero variance — model may not have trained"
        )

        # Padded particles should be zero
        padded = jets[masks < 0.5]
        if len(padded) > 0:
            assert np.allclose(padded, 0, atol=1e-4), (
                f"{jets_key} has non-zero values for padded particles"
            )
