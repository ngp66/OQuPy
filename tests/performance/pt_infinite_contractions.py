import numpy as np
from scipy.linalg import expm
import tensornetwork as tn

# define a 2 level system
dim = 2
H = np.array([[0.0, 1.0],
              [1.0, 0.0]])  # 2x2 Hamiltonian
dt = 0.1
num_steps = 5

# initial state |0><0|
initial_state = np.zeros((dim, dim), dtype=complex)
initial_state[0, 0] = 1.0

# exact propagator for one step
U = expm(-1j * H * dt)

# define dummy TTInvariantProcessTensor
from oqupy.process_tensor import TTInvariantProcessTensor, BaseProcessTensor

class DummyTTI(TTInvariantProcessTensor):
    def __init__(self, dim, dt):
        # MPO tensor acting as identity
        mpo_tensor = np.zeros((1,1,dim,dim), dtype=complex)
        for i in range(dim):
            mpo_tensor[0,0,i,i] = 1.0
        
        # dummy BaseProcessTensor object for parent constructor
        dummy_pt = BaseProcessTensor.__new__(BaseProcessTensor)
        dummy_pt.hilbert_space_dimension = dim
        dummy_pt.dt = dt
        dummy_pt.max_step = float('inf')
        dummy_pt.get_mpo_tensor = lambda step: mpo_tensor.copy()
        
        super().__init__(dummy_pt)
        self._mpo_tensor = mpo_tensor.copy()
        self._first_mpo_tensor = mpo_tensor.copy()
        self._cap_tensor = np.ones(1)
        self.hilbert_space_dimension = dim
        self.dt = dt
        self.uuid = "dummyPT"

pti = DummyTTI(dim, dt)

# run compute_dynamics
from oqupy.dynamics import compute_dynamics
from oqupy.control import Control

dynamics = compute_dynamics(
    system=type('DummySystem', (), {'dimension': dim, 'get_propagators': lambda self, dt, t0, *_: (lambda step: (U, U))})(),
    initial_state=initial_state,
    dt=dt,
    num_steps=num_steps,
    process_tensor=pti,
    control=Control(dim),
    record_all=True)

# compare with exact evolution
print("Step | Compute Dynamics State | Exact Evolution")
state_exact = initial_state.copy()
for step, state in enumerate(dynamics.states):
    # exact evolution: U^step * rho * (U^dag)^step
    state_exact = U @ state_exact @ U.conj().T
    print(f"{step} |")
    print(np.round(state, 6))
    print("Exact:")
    print(np.round(state_exact, 6))
    # check difference
    assert np.allclose(state, state_exact, atol=1e-12), f"Mismatch at step {step}"

print("matches exact unitary evolution")
