from typing import Dict, Optional

import torch
from dotenv import load_dotenv
from torch.optim import Adam


load_dotenv()

if torch.cuda.is_available():
    try:
        # MAGMA is hybrid CPU-GPU. GPU eigh should stay on cuSOLVER.
        torch.backends.cuda.preferred_linalg_library("cusolver")
    except Exception:
        pass

class ProjectedAdam(Adam):
    _PRECOMPUTED_BASIS_KEYS = (
        (
            "soft_q_a",
            "soft_q_b",
            "soft_dual_q_a",
            "soft_dual_q_b",
            "soft_eig_a",
            "soft_eig_b",
        ),
    )

    _FACTOR_KEY_SETS = (
        (("edit_A", "edit_B"), ("cap_A", "cap_B")),
    )

    def __init__(
        self,
        params,
        projection_cache_map: Optional[Dict] = None,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        soft_lambda: float = 1.0,
        factor_damping: float = 1e-5,
        cache_generalized_basis: bool = True,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

        defaults = {
            "projection_cache_map": projection_cache_map or {},
            "soft_lambda": float(soft_lambda),
        }
        for group in self.param_groups:
            group.update(defaults)

        self.factor_damping = max(float(factor_damping), 0.0)
        self.cache_generalized_basis = bool(cache_generalized_basis)
        # Table 3: mean_ij [g_ij / (1+λ ρ_ij)]^2 / mean_ij g_ij^2 in the
        # generalized eigenbasis. 1.0 means the edit gradient is unshrunk.
        self._retained_energy_sum = 0.0
        self._retained_energy_count = 0

    def mean_retained_edit_gradient_energy(self) -> float:
        if self._retained_energy_count <= 0:
            return 1.0
        return self._retained_energy_sum / self._retained_energy_count

    def _record_retained_energy(self, coeffs: torch.Tensor, denom: torch.Tensor) -> None:
        orig = coeffs.square().sum().clamp(min=1e-12)
        kept = (coeffs / denom.clamp(min=1e-12)).square().sum().clamp(min=0.0)
        self._retained_energy_sum += float((kept / orig).item())
        self._retained_energy_count += 1

    def reset_cache(self, new_projection_cache_map):
        # Sequential edits swap K-FAC caches; re-project Adam momentum into the new basis.
        new_projection_cache_map = new_projection_cache_map or {}
        for group in self.param_groups:
            group["projection_cache_map"] = new_projection_cache_map
        self.precompute_bases()
        for group in self.param_groups:
            self._project_momentum(group, group.get("projection_cache_map") or {})

    def precompute_bases(self):
        """Run GPU fp32 generalized eigh on each weight's device, before activations."""
        for group in self.param_groups:
            cache_map = group.get("projection_cache_map") or {}
            for p, cache in cache_map.items():
                if cache is None or getattr(p, "ndim", 0) != 2:
                    continue
                keys = self._PRECOMPUTED_BASIS_KEYS[0]
                if all(key in cache for key in keys):
                    continue
                self._basis_from_factors(
                    p.detach(),
                    cache,
                    torch.float32,
                    linalg_device=p.device,
                    return_basis=False,
                )
                self._release_linalg_workspace(p.device)

    def _project_momentum(self, group, cache_map: Dict):
        for p in group["params"]:
            if p not in self.state or p not in cache_map:
                continue

            state = self.state[p]
            exp_avg = state.get("exp_avg", None)
            if exp_avg is None or exp_avg.ndim != 2:
                continue

            projected = self._soft_kfac_precondition(
                exp_avg,
                cache_map[p],
                soft_lambda=group.get("soft_lambda", 1.0),
            )
            if projected is not None:
                exp_avg.copy_(projected)

    # Same as the wiki run that finished: fp32 Cholesky/eigh on the weight GPU.
    # fp64 on the embed GPU (emptiest-card picker) NaN'd the first Qwen forward.
    _WORK_DTYPE = torch.float32

    @staticmethod
    def _tensor(
        cache: Dict,
        key: str,
        like: torch.Tensor,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
    ):
        value = cache.get(key, None)
        if value is None:
            return None
        return value.to(
            device=like.device if device is None else device,
            dtype=dtype,
            non_blocking=value.is_pinned(),
        )

    @staticmethod
    def _release_linalg_workspace(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    @staticmethod
    def _host_store(tensor: torch.Tensor) -> torch.Tensor:
        host = tensor.detach().to("cpu")
        if torch.cuda.is_available():
            try:
                return host.pin_memory()
            except RuntimeError:
                return host
        return host

    @staticmethod
    def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
        return 0.5 * (matrix + matrix.T)

    @staticmethod
    def _check_square(matrix: torch.Tensor, name: str):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be a square matrix, got {tuple(matrix.shape)}.")

    @staticmethod
    def _check_basis_shape(
        tensor: torch.Tensor,
        q_a: torch.Tensor,
        q_b: torch.Tensor,
        dual_q_a: torch.Tensor,
        dual_q_b: torch.Tensor,
        eig_a: torch.Tensor,
        eig_b: torch.Tensor,
    ):
        out_dim, in_dim = tensor.shape
        if q_a.ndim != 2 or q_b.ndim != 2:
            raise ValueError(
                f"Generalized bases must be matrices, got q_a={tuple(q_a.shape)}, "
                f"q_b={tuple(q_b.shape)}."
            )
        if q_a.shape[0] != in_dim or q_b.shape[0] != out_dim:
            raise ValueError(
                "Generalized basis dimension mismatch: "
                f"tensor={tuple(tensor.shape)}, q_a={tuple(q_a.shape)}, "
                f"q_b={tuple(q_b.shape)}."
            )
        if dual_q_a.shape != q_a.shape or dual_q_b.shape != q_b.shape:
            raise ValueError(
                "Dual basis dimension mismatch: "
                f"q_a={tuple(q_a.shape)}, dual_q_a={tuple(dual_q_a.shape)}, "
                f"q_b={tuple(q_b.shape)}, dual_q_b={tuple(dual_q_b.shape)}."
            )
        if eig_a.numel() != q_a.shape[1] or eig_b.numel() != q_b.shape[1]:
            raise ValueError(
                "Generalized eigenvalue dimension mismatch: "
                f"eig_a={tuple(eig_a.shape)}, q_a={tuple(q_a.shape)}, "
                f"eig_b={tuple(eig_b.shape)}, q_b={tuple(q_b.shape)}."
            )

    def _generalized_basis(self, edit_factor, cap_factor):
        # Solve Q^T A_e Q = I and Q^T A_c Q = diag(a) via damped Cholesky:
        # A_e_reg = L L^T, whitened = L^{-1} A_c L^{-T}, then q = L^{-T} V.
        # fp32 Cholesky/eigh on the weight GPU (same as the successful wiki run).
        self._check_square(edit_factor, "edit_factor")
        self._check_square(cap_factor, "cap_factor")

        work_dtype = self._WORK_DTYPE
        device = edit_factor.device
        edit_factor = self._symmetrize(edit_factor.to(device=device, dtype=work_dtype))
        cap_factor = self._symmetrize(cap_factor.to(device=device, dtype=work_dtype))
        n = edit_factor.shape[0]
        print(
            f"[RhoEdit] {work_dtype} generalized basis n={n} on {device}",
            flush=True,
        )

        trace_scale = edit_factor.diagonal().abs().mean().clamp(min=1e-12)
        damp = max(float(self.factor_damping), 0.0)
        L = None
        info = None
        for _ in range(8):
            eps = (damp if damp > 0.0 else 1e-5) * trace_scale
            edit_factor_reg = edit_factor + eps * torch.eye(
                n, device=device, dtype=work_dtype
            )
            L, info = torch.linalg.cholesky_ex(edit_factor_reg)
            del edit_factor_reg
            info_v = int(info.item()) if torch.is_tensor(info) else int(info)
            if info_v == 0 and torch.isfinite(L).all():
                break
            damp = max(damp * 10.0, 1e-5) if damp > 0.0 else 1e-4
            L = None
        if L is None:
            raise RuntimeError(
                f"Cholesky failed for n={n} on {device}, last info={info}"
            )

        tmp = torch.linalg.solve_triangular(L, cap_factor, upper=False)
        del cap_factor
        whitened_cap = self._symmetrize(
            torch.linalg.solve_triangular(L, tmp.T, upper=False).T
        )
        del tmp
        cap_eigs, cap_vecs = torch.linalg.eigh(whitened_cap)
        del whitened_cap

        q = torch.linalg.solve_triangular(L.mT, cap_vecs, upper=True)
        dual_q = L @ cap_vecs
        del L, cap_vecs
        cap_eigs = torch.clamp(cap_eigs, min=0.0)

        if (
            not torch.isfinite(q).all()
            or not torch.isfinite(dual_q).all()
            or not torch.isfinite(cap_eigs).all()
        ):
            raise RuntimeError(
                "generalized basis produced non-finite values, check factor conditioning"
            )

        return (
            q.contiguous(),
            dual_q.contiguous(),
            cap_eigs.contiguous(),
        )

    def _basis_from_precomputed(
        self,
        tensor: torch.Tensor,
        cache: Dict,
        dtype: torch.dtype,
    ):
        for (
            q_a_key,
            q_b_key,
            dual_q_a_key,
            dual_q_b_key,
            eig_a_key,
            eig_b_key,
        ) in self._PRECOMPUTED_BASIS_KEYS:
            keys = (
                q_a_key,
                q_b_key,
                dual_q_a_key,
                dual_q_b_key,
                eig_a_key,
                eig_b_key,
            )
            if all(key in cache for key in keys):
                q_a = self._tensor(cache, q_a_key, tensor, dtype)
                q_b = self._tensor(cache, q_b_key, tensor, dtype)
                dual_q_a = self._tensor(cache, dual_q_a_key, tensor, dtype)
                dual_q_b = self._tensor(cache, dual_q_b_key, tensor, dtype)
                eig_a = self._tensor(cache, eig_a_key, tensor, dtype)
                eig_b = self._tensor(cache, eig_b_key, tensor, dtype)
                return (
                    q_a,
                    q_b,
                    dual_q_a,
                    dual_q_b,
                    eig_a.flatten(),
                    eig_b.flatten(),
                )
        return None

    def _compute_one_side(
        self,
        tensor: torch.Tensor,
        cache: Dict,
        edit_key: str,
        cap_key: str,
        linalg_device: torch.device,
    ):
        work_dtype = self._WORK_DTYPE

        def _run(device: torch.device):
            edit = self._tensor(cache, edit_key, tensor, work_dtype, device=device)
            cap = self._tensor(cache, cap_key, tensor, work_dtype, device=device)
            try:
                return self._generalized_basis(edit, cap)
            finally:
                del edit, cap

        try:
            return _run(linalg_device)
        except torch.cuda.OutOfMemoryError:
            if linalg_device.type != "cuda":
                raise
            print(
                f"[RhoEdit] fp32 eigh OOM on {linalg_device}, retrying on CPU",
                flush=True,
            )
            self._release_linalg_workspace(linalg_device)
            return _run(torch.device("cpu"))

    def _store_side(self, cache: Dict, side: str, q, dual_q, eig) -> None:
        cache[f"soft_q_{side}"] = self._host_store(q)
        cache[f"soft_dual_q_{side}"] = self._host_store(dual_q)
        cache[f"soft_eig_{side}"] = self._host_store(eig)

    def _basis_from_factors(
        self,
        tensor: torch.Tensor,
        cache: Dict,
        dtype: torch.dtype,
        linalg_device: Optional[torch.device] = None,
        return_basis: bool = True,
    ):
        if linalg_device is None:
            linalg_device = tensor.device
        for edit_keys, cap_keys in self._FACTOR_KEY_SETS:
            if not all(key in cache for key in (*edit_keys, *cap_keys)):
                continue
            # One side at a time so 18944-d A and 3584-d B do not share VRAM.
            q_a, dual_q_a, eig_a = self._compute_one_side(
                tensor, cache, edit_keys[0], cap_keys[0], linalg_device
            )
            if self.cache_generalized_basis:
                self._store_side(cache, "a", q_a, dual_q_a, eig_a)
                del q_a, dual_q_a, eig_a
                self._release_linalg_workspace(linalg_device)

            q_b, dual_q_b, eig_b = self._compute_one_side(
                tensor, cache, edit_keys[1], cap_keys[1], linalg_device
            )
            if self.cache_generalized_basis:
                self._store_side(cache, "b", q_b, dual_q_b, eig_b)
                del q_b, dual_q_b, eig_b
                self._release_linalg_workspace(linalg_device)
                if not return_basis:
                    return None
                return self._basis_from_precomputed(tensor, cache, dtype)

            return (
                q_a.to(device=tensor.device, dtype=dtype),
                q_b.to(device=tensor.device, dtype=dtype),
                dual_q_a.to(device=tensor.device, dtype=dtype),
                dual_q_b.to(device=tensor.device, dtype=dtype),
                eig_a.to(device=tensor.device, dtype=dtype),
                eig_b.to(device=tensor.device, dtype=dtype),
            )
        return None

    def _get_generalized_basis(self, tensor: torch.Tensor, cache: Dict, dtype: torch.dtype):
        precomputed = self._basis_from_precomputed(tensor, cache, dtype)
        if precomputed is not None:
            return precomputed
        return self._basis_from_factors(tensor, cache, dtype)

    def _soft_kfac_precondition(
        self,
        tensor: torch.Tensor,
        cache: Optional[Dict],
        soft_lambda: float,
    ):
        # Spectral filter: 1 / (1 + λ * b_i * a_j) in the generalized basis.
        if cache is None or tensor.ndim != 2:
            return None

        compute_dtype = (
            torch.float32
            if tensor.dtype in (torch.float16, torch.bfloat16)
            else tensor.dtype
        )
        source = tensor.to(dtype=compute_dtype)

        basis = self._get_generalized_basis(source, cache, compute_dtype)
        if basis is None:
            return None

        q_a, q_b, dual_q_a, dual_q_b, eig_a, eig_b = basis
        self._check_basis_shape(
            source,
            q_a,
            q_b,
            dual_q_a,
            dual_q_b,
            eig_a,
            eig_b,
        )

        coeffs = q_b.T @ source @ q_a
        joint_eigs = torch.outer(
            torch.clamp(eig_b.flatten(), min=0.0),
            torch.clamp(eig_a.flatten(), min=0.0),
        ).to(device=source.device, dtype=source.dtype)
        denom = 1.0 + float(soft_lambda) * joint_eigs
        self._record_retained_energy(coeffs, denom)
        filtered = dual_q_b @ (coeffs / denom.clamp(min=1e-12)) @ dual_q_a.T
        return filtered.to(dtype=tensor.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            cache_map = group.get("projection_cache_map", {}) or {}
            soft_lambda = group.get("soft_lambda", 1.0)

            for p in group["params"]:
                if p.grad is None or p.grad.ndim != 2:
                    continue
                if p not in cache_map:
                    continue
                grad_proj = self._soft_kfac_precondition(
                    p.grad,
                    cache_map[p],
                    soft_lambda=soft_lambda,
                )
                if grad_proj is not None:
                    p.grad.copy_(grad_proj)
        return super().step(closure)
