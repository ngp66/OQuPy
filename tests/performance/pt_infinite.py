import numpy as np
from oqupy import SimpleProcessTensor
from oqupy import TTInvariantProcessTensor  

hs_dim = 2
dt = 0.1
N_steps = 50  # length of finite PT
rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)

# finite PT for testing
finite_pt = SimpleProcessTensor(hilbert_space_dimension=hs_dim, dt=dt)

# simple MPO tensors: W = identity + small perturbation
for n in range(N_steps):
    W = np.zeros((1, 1, hs_dim**2, hs_dim**2), dtype=complex)
    W[0, 0, :, :] = np.eye(hs_dim**2) + 0.01 * np.random.rand(hs_dim**2, hs_dim**2)
    finite_pt.set_mpo_tensor(n, W)

finite_pt.compute_caps()

# infinite PT
infinite_pt = TTInvariantProcessTensor(finite_pt)

def apply_pt_step(pt, step, rho):
    """Contract rho with MPO and cap for step."""
    W = pt.get_mpo_tensor(step)
    cap = pt.get_cap_tensor(step)

    # reshape rho to vector
    rho_vec = rho.flatten()
    # MPO contraction (past x future x input x output)
    rho_vec = W[0, 0] @ rho_vec
    # apply cap
    if cap is not None and cap.size == 1:
        rho_vec *= cap[0]
    # reshape back to matrix
    return rho_vec.reshape(rho.shape)

# Compare finite and infinite PTs
rho_f = rho0.copy()
rho_i = rho0.copy()
burn_in = 10

for n in range(N_steps):
    rho_f = apply_pt_step(finite_pt, n, rho_f)
    rho_i = apply_pt_step(infinite_pt, n, rho_i)

    if n >= burn_in:
        err = np.linalg.norm(rho_f - rho_i)
        print(f"Step {n}, error = {err:.3e}")
        assert err < 1e-8
