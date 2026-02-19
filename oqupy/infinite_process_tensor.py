from typing import Optional, Text, Union
import uuid
import numpy as np
from numpy import ndarray
from .process_tensor import BaseProcessTensor, NpDtype, create_delta_lastindex

class TTInvariantProcessTensor(BaseProcessTensor):

    def __init__(
            self,
            process_tensor: BaseProcessTensor,
            transform_in: Optional[ndarray] = None,
            transform_out: Optional[ndarray] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            opti: Optional[bool] = False) -> None:

        self.uuid = str(uuid.uuid4())[:14]
        self._initial_tensor = None
        
        hilbert_space_dimension = process_tensor.hilbert_space_dimension
        dt = process_tensor.dt

        max_step = process_tensor.max_step
        if max_step == float('inf'):
            last_idx = 1
        else:
            last_idx = int(max_step - 1) if max_step > 0 else 0

        # Extract tensors
        self._mpo_tensor = process_tensor.get_mpo_tensor(last_idx).copy()
        self._first_mpo_tensor = process_tensor.get_mpo_tensor(0).copy()

        if transform_in is not None:
            for attr in ['_first_mpo_tensor', '_mpo_tensor']:
                t = getattr(self, attr)
                t = np.dot(np.moveaxis(t, -2, -1), transform_in.T)
                setattr(self, attr, np.moveaxis(t, -1, -2))

        if transform_out is not None:
            self._first_mpo_tensor = np.dot(self._first_mpo_tensor, transform_out)
            self._mpo_tensor = np.dot(self._mpo_tensor, transform_out)

        if opti:
            self._mpo_tensor = create_delta_lastindex(self._mpo_tensor)

        super().__init__(
            hilbert_space_dimension,
            dt,
            transform_in,
            transform_out,
            name,
            description)

        self._canonicalise()

    def _canonicalise(self):
        W = self._mpo_tensor
        Dl, Dr = W.shape[0], W.shape[1]
        
        # Contract physical indices to find the transfer matrix E
        # W shape is (left, right, out, in)
        # We contract the last two dimensions (the system physical legs)
        E = np.einsum('abij,cdij->acbd', W, np.conj(W))
        E = E.reshape(Dl * Dl, Dr * Dr)
        
        vals, vecs = np.linalg.eig(E)
        idx = np.argmax(np.abs(vals))
        lam = vals[idx]
        
        # Normalize the repeating tensor so the leading eigenvalue is exactly 1
        # This prevents exponential growth/decay (the 143.1 error)
        self._mpo_tensor = self._mpo_tensor / np.sqrt(np.abs(lam))
        
        # The right eigenvector is the fixed-point 'cap'
        vr = vecs[:, idx].reshape(Dr, Dr)
        # Ensure the cap is normalized such that Tr(rho) is preserved
        self._cap_tensor = vr / np.trace(vr)

    def __len__(self) -> int:
        return 0

    @property
    def max_step(self) -> Union[int, float]:
        return float('inf')

    def set_initial_tensor(
            self,
            initial_tensor: Optional[ndarray] = None) -> None:
        self._initial_tensor = np.array(initial_tensor, dtype=NpDtype) if initial_tensor is not None else None

    def get_initial_tensor(self) -> ndarray:
        return self._initial_tensor

    def get_mpo_tensor(
            self,
            step: int,
            transformed: Optional[bool] = True) -> ndarray:
        if step < 0:
            raise IndexError("Process tensor index out of bound.")
        return self._first_mpo_tensor if step == 0 else self._mpo_tensor

    def get_cap_tensor(self, step: int) -> ndarray:
        # For the infinite PT, we always use the fixed-point cap for t > 0
        if step == 0:
            # Boundary condition for the very first step
            return np.ones(self._first_mpo_tensor.shape[1], dtype=NpDtype)
        return np.diag(self._cap_tensor)

    def get_bond_dimensions(self) -> ndarray:
        return np.array([self._mpo_tensor.shape[0], self._mpo_tensor.shape[1]])
