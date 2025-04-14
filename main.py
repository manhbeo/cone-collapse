import numpy as np
import time
from cone_collapse import ConeCollapse
from utils import generate_synthetic_data
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
import torch


def run_experiment(error_metric='frobenius'):
    """
    Run a complete experiment with the PyTorch Cone Collapsing algorithm.

    Args:
        error_metric (str): Error metric to use ('frobenius' or 'kl_divergence')
    """
    # Set random seed for reproducibility
    # np.random.seed(42)
    # torch.manual_seed(42)

    # Generate synthetic data
    m, n, r = 1000, 1000, 5
    X, U_true, V_true = generate_synthetic_data(m, n, r, noise_level=0.01)

    print(f"Generated dataset: X shape = {X.shape}, rank = {r}")

    # Create and fit the PyTorch Cone Collapsing model
    cc = ConeCollapse(
        theta=0.1,
        max_iters=1000,
        tol=1e-8,
        batch_size=1024,
        verbose=True,
        plot=True,
        adaptive_theta=True,
        early_stopping=True,
        error_metric=error_metric
    )

    # Time the fitting process
    start_time = time.time()
    U_cc, V_cc = cc.fit(X, r)
    end_time = time.time()

    print(f"\nFitting completed in {end_time - start_time:.2f} seconds")

    # Reconstruct the data
    X_reconstructed = cc.reconstruct(U_cc, V_cc)

    # Calculate reconstruction error using specified metric
    X_tensor = torch.tensor(X, dtype=torch.float32)
    X_recon_tensor = torch.tensor(X_reconstructed, dtype=torch.float32)

    if error_metric == 'kl_divergence':
        # KL divergence
        epsilon = 1e-10  # Small constant to avoid log(0)
        X_safe = X_tensor + epsilon
        X_recon_safe = X_recon_tensor + epsilon

        # Calculate KL divergence
        kl_div = X_safe * (torch.log(X_safe) - torch.log(X_recon_safe)) - X_safe + X_recon_safe
        reconstruction_error = torch.sum(kl_div)
        print(f"Reconstruction error (KL divergence): {reconstruction_error.item():.6f}")
    else:
        # Frobenius norm
        reconstruction_error = torch.norm(X_tensor - X_recon_tensor, p='fro')
        print(f"Reconstruction error (Frobenius norm): {reconstruction_error.item():.6f}")


    # Verify non-negativity of factors
    print("\nU non-negative:", np.all(U_cc >= -1e-10))
    print("V non-negative:", np.all(V_cc >= -1e-10))

    return U_cc, V_cc, X, reconstruction_error.item() if isinstance(reconstruction_error,
                                                                    torch.Tensor) else reconstruction_error


if __name__ == "__main__":
    # You can choose 'frobenius' or 'kl_divergence' here
    run_experiment(error_metric='frobenius')