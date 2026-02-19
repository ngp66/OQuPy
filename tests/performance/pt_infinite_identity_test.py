import numpy as np
from oqupy import SimpleProcessTensor
from oqupy.infinite_process_tensor import TTInvariantProcessTensor  

hs_dim = 2
dt = 0.1
N_steps = 50
rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)

# finite PT 
finite_pt = SimpleProcessTensor(hilbert_space_dimension=hs_dim, dt=dt)

# steady-state tensor
W_steady = np.zeros((1, 1, hs_dim**2, hs_dim**2), dtype=complex)
W_steady[0, 0, :, :] = np.eye(hs_dim**2) 

for n in range(N_steps):
    finite_pt.set_mpo_tensor(n, W_steady)

finite_pt.compute_caps()

# convert to infinite PT
infinite_pt = TTInvariantProcessTensor(finite_pt)

def apply_pt_step(pt, step, rho):
    W = pt.get_mpo_tensor(step)
    cap = pt.get_cap_tensor(step)
    
    rho_vec = rho.flatten()
    # contract: bond dims are [0,0] because bond dimension is 1 here
    rho_vec = W[0, 0] @ rho_vec
    
    # cap for bond-dim-1 MPO is [1.0]
    return rho_vec.reshape(rho.shape)

# comparison, finite and inifinite
rho_f = rho0.copy()
rho_i = rho0.copy()

print(f"{'Step':<10} | {'Error':<15}")
print("-" * 30)

for n in range(N_steps):
    rho_f = apply_pt_step(finite_pt, n, rho_f)
    rho_i = apply_pt_step(infinite_pt, n, rho_i)

    err = np.linalg.norm(rho_f - rho_i)
    print(f"{n:<10} | {err:.3e}")
    if n > 0: # step 0 might have slight differences if first_mpo_tensor differs
        assert err < 1e-12
