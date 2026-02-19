class TTInvariantProcessTensor(BaseProcessTensor):

    def __init__(
            self,
            tebd: iTEBD_TEMPO_oqupy,
            transform_in: Optional[ndarray] = None,
            transform_out: Optional[ndarray] = None,
            name: Optional[Text] = None,
            description: Optional[Text] = None,
            opti: Optional[bool] = False) -> None:

        self.uuid = str(uuid.uuid4())[:14]
        self._initial_tensor = None
        hilbert_space_dimension = tebd.s_dim
        dt = tebd.delta
        self._tebd = tebd

        self._mpo_tensor = np.transpose(tebd.f[:, :-1, :], [0, 2, 1])
        self._first_mpo_tensor = ncon(
            [tebd.v_l, self._mpo_tensor], [[1], [1, -1, -2]]
        )
        self._first_mpo_tensor.shape = tuple([1] + list(self._first_mpo_tensor.shape))

        tensor = self._first_mpo_tensor
        if transform_in is not None:
            tensor = np.dot(np.moveaxis(tensor, -2, -1), transform_in.T)
            tensor = np.moveaxis(tensor, -1, -2)
        if transform_out is not None:
            tensor = np.dot(tensor, transform_out)
        self._first_mpo_tensor = tensor

        if not (transform_in is None and transform_out is None and opti):
            tensor = create_delta_lastindex(self._mpo_tensor)
            if transform_in is not None:
                tensor = np.dot(np.moveaxis(tensor, -2, -1), transform_in.T)
                tensor = np.moveaxis(tensor, -1, -2)
            if transform_out is not None:
                tensor = np.dot(tensor, transform_out)
            self._mpo_tensor = tensor

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

        phys_axes = tuple(range(2, W.ndim))

        E = np.tensordot(W, np.conj(W), axes=(phys_axes, phys_axes))
        E = np.transpose(E, (0, 2, 1, 3))
        E = E.reshape(Dl * Dl, Dr * Dr)

        vals, vecs = np.linalg.eig(E)
        idx = np.argmax(np.real(vals))
        lam = np.real(vals[idx])
        vr = vecs[:, idx]

        vr = vr.reshape(Dr, Dr)

        scale = np.sqrt(lam)
        self._mpo_tensor = self._mpo_tensor / scale
        self._first_mpo_tensor = self._first_mpo_tensor / scale

        self._cap_tensor = vr / np.linalg.norm(vr)

    def __len__(self) -> int:
        return 0

    @property
    def max_step(self) -> Union[int, float]:
        return float('inf')

    def set_initial_tensor(
            self,
            initial_tensor: Optional[ndarray] = None) -> None:
        if initial_tensor is None:
            self._initial_tensor = None
        else:
            self._initial_tensor = np.array(initial_tensor, dtype=NpDtype)

    def get_initial_tensor(self) -> ndarray:
        return self._initial_tensor

    def get_mpo_tensor(
            self,
            step: int,
            transformed: Optional[bool] = True) -> ndarray:

        assert transformed

        if step < 0:
            raise IndexError("Process tensor index out of bound.")

        if step == 0:
            return self._first_mpo_tensor
        else:
            return self._mpo_tensor

    def get_cap_tensor(self, step: int) -> ndarray:
        if step == 0:
            return np.array([1.0], dtype=NpDtype)
        else:
            return self._cap_tensor

    def get_bond_dimensions(self) -> ndarray:
        return np.array([self._mpo_tensor.shape[0],
                         self._mpo_tensor.shape[1]])
