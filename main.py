import numpy as np
import torch
import time
from cone_collapse import ConeCollapse
from utils import generate_synthetic_data

def run_experiment():
    """Run a complete experiment with the PyTorch Cone Collapsing algorithm."""
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate synthetic data
    m, n, r = 1000, 1000, 5
    X, U_true, V_true = generate_synthetic_data(m, n, r, noise_level=0.01)

    print(f"Generated dataset: X shape = {X.shape}, rank = {r}")

    # Create and fit the PyTorch Cone Collapsing model
    cc = ConeCollapse(
        theta=0.1,
        max_iters=50,
        tol=1e-6,
        batch_size=32,
        verbose=True,
        plot=True,
        adaptive_theta=True,
        early_stopping=True
    )

    # Time the fitting process
    start_time = time.time()
    U_cc, V_cc = cc.fit(X, r)
    end_time = time.time()

    print(f"\nFitting completed in {end_time - start_time:.2f} seconds")

    # Reconstruct the data
    X_reconstructed = cc.reconstruct(U_cc, V_cc)

    # Calculate reconstruction error
    reconstruction_error = np.linalg.norm(X - X_reconstructed) / np.linalg.norm(X)
    print(f"Reconstruction error: {reconstruction_error:.6f}")

    # Verify non-negativity of factors
    print("\nU non-negative:", np.all(U_cc >= -1e-10))
    print("V non-negative:", np.all(V_cc >= -1e-10))

    return U_cc, V_cc, X, reconstruction_error


if __name__ == "__main__":
    run_experiment()