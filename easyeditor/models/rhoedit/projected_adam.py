from typing import Dict, Optional

import torch
from dotenv import load_dotenv
from torch.optim import Adam


load_dotenv()

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

    def reset_cache(self, new_projection_cache_map):
        # Sequential edits swap K-FAC caches; re-project Adam momentum into the new basis.
        new_projection_cache_map = new_projection_cache_map or {}
        for group in self.param_groups:
            group["projection_cache_map"] = new_projection_cache_map
            self._project_momentum(group, new_projection_cache_map)

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

    # Qwen2.5-7B down_proj input dim is 18944; GPU eigh of that fp64
    # factor asks for ~8 GiB extra workspace and OOMs next to the model.
    _CPU_FACTOR_DIM = 8192

    @staticmethod
    def _tensor(cache: Dict, key: str, like: torch.Tensor, dtype: torch.dtype):
        value = cache.get(key, None)
        if value is None:
            return None
        return value.to(device=like.device, dtype=dtype)

    @classmethod
    def _factor_work_device(cls, tensor: torch.Tensor) -> torch.device:
        if max(tensor.shape) >= cls._CPU_FACTOR_DIM:
            return torch.device("cpu")
        return tensor.device

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
        self._check_square(edit_factor, "edit_factor")
        self._check_square(cap_factor, "cap_factor")

        cache_dtype = edit_factor.dtype
        edit_factor = edit_factor.to(dtype=torch.float64)
        cap_factor = cap_factor.to(dtype=torch.float64)

        edit_factor = self._symmetrize(edit_factor)
        cap_factor = self._symmetrize(cap_factor)

        n = edit_factor.shape[0]
        trace_scale = edit_factor.diagonal().abs().mean().clamp(min=1e-12)
        eps = self.factor_damping * trace_scale
        edit_factor_reg = edit_factor + eps * torch.eye(
            n, device=edit_factor.device, dtype=edit_factor.dtype
        )

        L = torch.linalg.cholesky(edit_factor_reg)
        tmp = torch.linalg.solve_triangular(L, cap_factor, upper=False)
        whitened_cap = torch.linalg.solve_triangular(L, tmp.T, upper=False).T
        cap_eigs, cap_vecs = torch.linalg.eigh(whitened_cap)

        q = torch.linalg.solve_triangular(L.transpose(-1, -2), cap_vecs, upper=True)
        # Dual basis R = Q^{-T} = L V, with R^T Q = I.
        dual_q = L @ cap_vecs
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
            q.to(dtype=cache_dtype).contiguous(),
            dual_q.to(dtype=cache_dtype).contiguous(),
            cap_eigs.to(dtype=cache_dtype).contiguous(),
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

    def _factors_from_cache(self, tensor: torch.Tensor, cache: Dict, dtype: torch.dtype):
        for edit_keys, cap_keys in self._FACTOR_KEY_SETS:
            if all(key in cache for key in (*edit_keys, *cap_keys)):
                edit_a = self._tensor(cache, edit_keys[0], tensor, dtype)
                edit_b = self._tensor(cache, edit_keys[1], tensor, dtype)
                cap_a = self._tensor(cache, cap_keys[0], tensor, dtype)
                cap_b = self._tensor(cache, cap_keys[1], tensor, dtype)
                return edit_a, edit_b, cap_a, cap_b
        return None

    def _basis_from_factors(self, tensor: torch.Tensor, cache: Dict, dtype: torch.dtype):
        factors = self._factors_from_cache(tensor, cache, dtype)
        if factors is None:
            return None

        edit_a, edit_b, cap_a, cap_b = factors
        q_a, dual_q_a, eig_a = self._generalized_basis(edit_a, cap_a)
        q_b, dual_q_b, eig_b = self._generalized_basis(edit_b, cap_b)

        if self.cache_generalized_basis:
            cache["soft_q_a"] = q_a.detach().cpu()
            cache["soft_q_b"] = q_b.detach().cpu()
            cache["soft_dual_q_a"] = dual_q_a.detach().cpu()
            cache["soft_dual_q_b"] = dual_q_b.detach().cpu()
            cache["soft_eig_a"] = eig_a.detach().cpu()
            cache["soft_eig_b"] = eig_b.detach().cpu()

        return q_a, q_b, dual_q_a, dual_q_b, eig_a, eig_b

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
