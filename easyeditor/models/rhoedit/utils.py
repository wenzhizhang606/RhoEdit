import gc
import json
import os
import time
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import random
from dotenv import load_dotenv
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from .projected_adam import  ProjectedAdam
from ..rome.layer_stats import (
    calculate_cache_loss,
    calculate_request_loss,
    layer_stats_kfac_one_pass,
    layer_stats_kfac_with_txt_tgt,
)
from .RhoEdit_hparams import AdamHyperParams
from .seq_tracker import CAP_LOSS_KEY, cov_cache_nbytes
from easyeditor.tools import ExperimentTracker

load_dotenv()
STATS_DIR = os.getenv("STATS_DIR")

# Only used when loading K-FAC stats. Edit JSON loading stays in the root utils.py.
KFAC_DATASET_ALIASES = {
    "zsre": "zsre_mend_3k",
    "zsre3k": "zsre_mend_3k",
    "zsre_3k": "zsre_mend_3k",
    "zsre10k": "zsre_mend_10k",
    "zsre163k": "zsre_mend_163k",
    "counterfact": "counterfact-edit_3k",
    "wiki": "wiki_big_edit_3k",
}


def normalize_kfac_dataset(ds_name):
    if not ds_name:
        return ds_name
    return KFAC_DATASET_ALIASES.get(str(ds_name).lower(), ds_name)


def _is_llama_or_phi(model_name: str) -> bool:
    lower = str(model_name).lower()
    return "llama" in lower or "phi" in lower or "qwen" in lower


def _model_device(model) -> torch.device:
    return getattr(model, "device", next(model.parameters()).device)


def _embed_device(model) -> torch.device:
    embed = model.get_input_embeddings()
    weight = getattr(embed, "weight", None)
    if weight is not None:
        return weight.device
    return _model_device(model)


def _lm_head_device(model) -> torch.device:
    lm_head = model.get_output_embeddings()
    weight = getattr(lm_head, "weight", None)
    if weight is not None:
        return weight.device
    return _embed_device(model)


def _to_device_via_host(tensor: torch.Tensor, device) -> torch.Tensor:
    """Move a tensor, routing GPU->GPU moves through host memory.

    Direct peer copies (``cuda:1 -> cuda:0``) return stale/garbage memory on
    hosts where PCIe P2P is broken (AMD IOMMU without ``iommu=pt`` is the
    usual culprit). CUDA still reports ``can_device_access_peer=True`` there,
    so PyTorch happily takes the P2P path. Host-staged copies are always
    correct; the tensors crossing shard boundaries are small (mask, rotary
    cos/sin, one hidden-state block), so the cost is negligible.
    """
    device = torch.device(device)
    if tensor.device == device:
        return tensor
    if tensor.device.type == "cuda" and device.type == "cuda":
        return tensor.cpu().to(device)
    return tensor.to(device)


def _stage_cross_cuda(obj, device: torch.device, skip_keys=()):
    """Recursively pull tensors that would cross GPUs onto the host first."""
    if torch.is_tensor(obj):
        if obj.device.type == "cuda" and obj.device != device:
            return obj.cpu()
        return obj
    if isinstance(obj, (list, tuple)):
        staged = [_stage_cross_cuda(o, device, skip_keys) for o in obj]
        if hasattr(obj, "_fields"):  # namedtuple
            return type(obj)(*staged)
        return type(obj)(staged)
    if isinstance(obj, Mapping):
        return type(obj)(
            {
                k: (v if k in skip_keys else _stage_cross_cuda(v, device, skip_keys))
                for k, v in obj.items()
            }
        )
    return obj


def _patch_accelerate_host_transfers() -> None:
    """Make Accelerate's device_map hooks stage GPU->GPU moves through host."""
    try:
        import accelerate.hooks as accelerate_hooks
    except ImportError:
        return
    orig = accelerate_hooks.send_to_device
    if getattr(orig, "_rhoedit_host_staged", False):
        return

    def send_to_device(tensor, device, non_blocking=False, skip_keys=None):
        if isinstance(skip_keys, str):
            skip = (skip_keys,)
        else:
            skip = tuple(skip_keys or ())
        try:
            target = torch.device(device)
        except (TypeError, RuntimeError):
            target = None
        if target is not None and target.type == "cuda":
            tensor = _stage_cross_cuda(tensor, target, skip)
        return orig(tensor, device, non_blocking=non_blocking, skip_keys=skip_keys)

    send_to_device._rhoedit_host_staged = True
    accelerate_hooks.send_to_device = send_to_device


def _check_gpu_p2p_copy(src: torch.device, dst: torch.device) -> bool:
    """True if a direct ``src -> dst`` GPU copy round-trips bit-exactly."""
    x = torch.randn(1 << 18, device=src)
    torch.cuda.synchronize(src)
    return torch.equal(x.to(dst).cpu(), x.cpu())


def patch_device_map_host_transfers(model) -> bool:
    """Keep multi-GPU ``device_map`` forwards off the GPU P2P copy path.

    * Accelerate's ``send_to_device`` stages cross-GPU tensors via host.
    * The top-level hook stops copying logits back to the embed GPU
      (``io_same_device``), so the loss/backward stay on the lm_head GPU and
      autograd never issues a GPU->GPU copy either.

    Returns True when the model spans more than one CUDA device.
    """
    if getattr(model, "_rhoedit_host_transfers", False):
        return True
    cuda_devices = set()
    for value in (getattr(model, "hf_device_map", {}) or {}).values():
        if isinstance(value, int):
            cuda_devices.add(torch.device("cuda", value))
        elif isinstance(value, str) and value.startswith("cuda"):
            cuda_devices.add(torch.device(value))
    if len(cuda_devices) < 2:
        return False

    _patch_accelerate_host_transfers()
    hook = getattr(model, "_hf_hook", None)
    if hook is not None and getattr(hook, "io_same_device", False):
        hook.io_same_device = False

    devs = sorted(cuda_devices, key=lambda d: d.index)
    p2p_ok = _check_gpu_p2p_copy(devs[1], devs[0]) and _check_gpu_p2p_copy(devs[0], devs[1])
    state = "OK" if p2p_ok else "BROKEN (direct GPU->GPU copies return garbage)"
    print(
        f"[RhoEdit] device_map spans {[str(d) for d in devs]}; GPU P2P copy check: {state}. "
        "Cross-GPU tensors are staged through host; logits stay on the lm_head GPU."
    )
    model._rhoedit_host_transfers = True
    return True


def _align_attn_mask(attention_mask, hidden_states, unmask_fn=None):
    if attention_mask is None or not torch.is_tensor(attention_mask):
        return attention_mask
    if attention_mask.device != hidden_states.device:
        attention_mask = _to_device_via_host(attention_mask, hidden_states.device)
    if attention_mask.is_floating_point() and attention_mask.dtype != hidden_states.dtype:
        attention_mask = attention_mask.to(dtype=hidden_states.dtype)
    if attention_mask.dim() == 4:
        attention_mask = attention_mask.contiguous()
        if unmask_fn is not None:
            min_dtype = torch.finfo(attention_mask.dtype).min
            attention_mask = unmask_fn(attention_mask, min_dtype)
        # Clone: Qwen slices `mask[..., :kv]` and later layers must not see
        # in-place writes into that view (layer 13 was the first all-NaN layer).
        attention_mask = attention_mask.contiguous().clone()
    elif not attention_mask.is_contiguous():
        attention_mask = attention_mask.contiguous()
    return attention_mask


def _align_position_embeddings(position_embeddings, hidden_states):
    if not isinstance(position_embeddings, tuple):
        return position_embeddings
    aligned = []
    for item in position_embeddings:
        if not torch.is_tensor(item):
            aligned.append(item)
            continue
        if item.device != hidden_states.device:
            item = _to_device_via_host(item, hidden_states.device)
        if item.is_floating_point() and item.dtype != hidden_states.dtype:
            item = item.to(dtype=hidden_states.dtype)
        item = item.contiguous().clone()
        aligned.append(item)
    return tuple(aligned)


def _patch_sdpa_attn_mask(unmask_fn) -> None:
    """Contiguous + unmask the custom attn_mask SDPA actually receives.

    Qwen2SdpaAttention slices ``attention_mask[:, :, :, :kv]`` and only
    ``contiguous()``s Q/K/V. A non-contiguous mask hits PyTorch #112577;
    fully masked pad rows hit #110213. HuggingFace unmasks only for SDPA
    and only when the *2D* mask is already on CUDA.
    """
    import torch.nn.functional as F

    orig = F.scaled_dot_product_attention
    if getattr(orig, "_rhoedit_mask_patched", False):
        return

    def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        if attn_mask is not None and torch.is_tensor(attn_mask):
            if attn_mask.device != query.device:
                attn_mask = _to_device_via_host(attn_mask, query.device)
            if attn_mask.is_floating_point() and attn_mask.dtype != query.dtype:
                attn_mask = attn_mask.to(dtype=query.dtype)
            if attn_mask.dim() == 4:
                attn_mask = attn_mask.contiguous()
                if unmask_fn is not None:
                    min_dtype = torch.finfo(attn_mask.dtype).min
                    attn_mask = unmask_fn(attn_mask, min_dtype)
                attn_mask = attn_mask.contiguous().clone()
            elif not attn_mask.is_contiguous():
                attn_mask = attn_mask.contiguous()
        return orig(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    _sdpa._rhoedit_mask_patched = True
    F.scaled_dot_product_attention = _sdpa


def patch_qwen_device_map_attention(model) -> None:
    """Keep Qwen2 4D causal masks valid under Accelerate ``device_map``.

    The 4D mask is built on the embed GPU. Crossing devices can make it a
    non-contiguous view, and SDPA's mem-efficient path then emits NaN (fully
    masked pad rows do the same). Align dtype/device, ``contiguous()``, and
    unmask attended rows before every decoder layer / SDPA call.

    Do not wrap Accelerate ``_old_forward``: that calling convention previously
    broke eager attention.
    """
    # Independent of the mask fix: any multi-GPU device_map must avoid P2P copies.
    patch_device_map_host_transfers(model)
    if getattr(model, "_rhoedit_attn_mask_patched", False):
        return
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "_update_causal_mask"):
        return
    try:
        from transformers.modeling_attn_mask_utils import AttentionMaskConverter
        unmask_fn = AttentionMaskConverter._unmask_unattended
    except ImportError:
        unmask_fn = None

    orig_update = inner._update_causal_mask

    def _update_causal_mask(attention_mask, input_tensor, cache_position=None, *args, **kwargs):
        device = input_tensor.device
        if attention_mask is not None and torch.is_tensor(attention_mask):
            if attention_mask.device != device:
                attention_mask = _to_device_via_host(attention_mask, device)
            attention_mask = attention_mask.contiguous()
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.clone()
        if cache_position is not None and torch.is_tensor(cache_position) and cache_position.device != device:
            cache_position = _to_device_via_host(cache_position, device)
        causal_mask = orig_update(
            attention_mask, input_tensor, cache_position, *args, **kwargs
        )
        if causal_mask is None:
            return None
        return _align_attn_mask(causal_mask, input_tensor, unmask_fn=unmask_fn)

    inner._update_causal_mask = _update_causal_mask
    _patch_sdpa_attn_mask(unmask_fn)

    layers = getattr(inner, "layers", None)
    if layers is None:
        model._rhoedit_attn_mask_patched = True
        return

    def _layer_pre_hook(module, args, kwargs):
        if not args:
            return args, kwargs
        hidden_states = args[0]
        kwargs = dict(kwargs)
        kwargs["attention_mask"] = _align_attn_mask(
            kwargs.get("attention_mask"), hidden_states, unmask_fn=unmask_fn
        )
        position_ids = kwargs.get("position_ids")
        if (
            position_ids is not None
            and torch.is_tensor(position_ids)
            and position_ids.device != hidden_states.device
        ):
            kwargs["position_ids"] = _to_device_via_host(position_ids, hidden_states.device)
        cache_position = kwargs.get("cache_position")
        if (
            cache_position is not None
            and torch.is_tensor(cache_position)
            and cache_position.device != hidden_states.device
        ):
            kwargs["cache_position"] = _to_device_via_host(cache_position, hidden_states.device)
        if "position_embeddings" in kwargs:
            kwargs["position_embeddings"] = _align_position_embeddings(
                kwargs["position_embeddings"], hidden_states
            )
        return args, kwargs

    for layer in layers:
        layer.register_forward_pre_hook(_layer_pre_hook, with_kwargs=True)

    model._rhoedit_attn_mask_patched = True
    print("[RhoEdit] Patched Qwen attention masks for device_map")


def _causal_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Token-level CE on the model's actual logit width.

    transformers 4.46 ``ForCausalLMLoss`` does ``view(-1, self.vocab_size)``
    inside the forward. With a 3-GPU ``device_map`` that path also runs CE on
    the pre-gather ``lm_head`` tensor and dies in CUDA ``nll_loss``. Using the
    returned logits (and ``logits.size(-1)``) is the same causal NLL.
    """
    if logits.dim() != 3:
        raise RuntimeError(f"causal NLL expected [B, T, V] logits, got {tuple(logits.shape)}")
    if labels.dtype != torch.long:
        raise RuntimeError(
            f"causal NLL expected Long labels, got {labels.dtype} shape={tuple(labels.shape)}"
        )
    shift_logits = logits[..., :-1, :].contiguous()
    # Snapshot on CPU first. A non-contiguous GPU view copied onto the logits
    # device can pick up the 4D causal-mask buffer (finfo.min / 0-1 bytes).
    shift_labels = labels[..., 1:].detach().contiguous().to(dtype=torch.long).cpu()
    shift_labels = shift_labels.to(device=shift_logits.device)
    if shift_logits.shape[:2] != shift_labels.shape[:2]:
        raise RuntimeError(
            f"causal NLL shape mismatch logits={tuple(shift_logits.shape)} "
            f"labels={tuple(shift_labels.shape)}"
        )
    n_classes = shift_logits.size(-1)
    ignored = shift_labels == -100
    if bool(ignored.all()):
        return shift_logits.sum() * 0.0
    bad = (~ignored) & ((shift_labels < 0) | (shift_labels >= n_classes))
    if bool(bad.any()):
        sample = shift_labels[bad][:16].detach().cpu().tolist()
        raise RuntimeError(
            f"causal NLL labels outside [0, {n_classes}): {sample} "
            f"logits={tuple(shift_logits.shape)} device={shift_logits.device}"
        )
    n_nan = int(torch.isnan(shift_logits).sum()) if shift_logits.is_floating_point() else 0
    if n_nan:
        raise RuntimeError(
            f"causal NLL got {n_nan} NaN logits in {tuple(shift_logits.shape)}; "
            "the forward produced NaNs (on multi-GPU device_map check the "
            "'[RhoEdit] GPU P2P copy check' line: broken P2P copies corrupt "
            "tensors that cross shard boundaries)"
        )
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, n_classes),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _sft_labels(encodings, tok: AutoTokenizer, txt: List[str], max_length: int) -> torch.Tensor:
    """Prompt/pad-masked labels on CPU so device_map forwards cannot clobber them."""
    labels = encodings["input_ids"].detach().to(dtype=torch.long).cpu().contiguous().clone()
    attn = encodings.get("attention_mask")
    if attn is not None:
        labels[attn.detach().to(dtype=torch.long).cpu() == 0] = -100
    else:
        pad_id = tok.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
    for i, prompt in enumerate(txt):
        prompt_len = len(
            tok(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
        )
        labels[i, :prompt_len] = -100
    return labels


def _layer_names(hparams) -> List[str]:
    return [hparams.rewrite_module_tmp.format(layer) for layer in hparams.layers]


def _cache_dtype_name(hparams) -> str:
    return getattr(hparams, "mom2_n_dtype", getattr(hparams, "mom2_dtype", "float32"))


def _cache_sample_size(hparams) -> int:
    return int(getattr(hparams, "mom2_n_sample", getattr(hparams, "mom2_n_samples", 10000)))


def _build_cov_cache_from_hparams(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams,
    force_recompute: bool = False,
) -> Dict[str, Dict]:
    layer_names = _layer_names(hparams)
    dtype_name = _cache_dtype_name(hparams)
    sample_size = _cache_sample_size(hparams)
    if sample_size <= 0:
        print(
            "[RhoEdit] n=0: skip capability K-FAC; unconstrained Adam on edit weights.",
            flush=True,
        )
        return {}

    raw_task_dataset = getattr(hparams, "task_mom2_dataset", None)
    task_mom2_dataset = normalize_kfac_dataset(raw_task_dataset)
    task_sample_size= getattr(hparams, "task_mom2_n_samples", None)
    if raw_task_dataset and task_mom2_dataset != raw_task_dataset:
        print(f"[RhoEdit] K-FAC task dataset {raw_task_dataset} -> {task_mom2_dataset}")
    print("[RhoEdit] Computing/loading base KFAC stats.")
    stats_dict = layer_stats_kfac_one_pass(
        model=model,
        tokenizer=tok,
        layer_names=layer_names,
        stats_dir=STATS_DIR,
        ds_name=hparams.mom2_dataset,
        to_collect=["mom2"],
        sample_size=sample_size if not force_recompute else sample_size,
        precision=dtype_name,
        force_recompute=force_recompute
    )

    task_stats_dict=None
    if task_mom2_dataset is not None:
        print("[RhoEdit] Computing/loading task KFAC stats.")
        task_stats_dict = layer_stats_kfac_one_pass(
        model=model,
        tokenizer=tok,
        layer_names=layer_names,
        stats_dir=STATS_DIR,
        ds_name=task_mom2_dataset,
        to_collect=["mom2"],
        sample_size=task_sample_size if not force_recompute else task_sample_size,
        precision=dtype_name,
        force_recompute=force_recompute
    )

    layer_to_cov_cache = {}
    for layer_name, (A, B, n) in stats_dict.items():
        cov_cache = {
            "A": A.to("cpu", dtype=torch.float32),
            "B": B.to("cpu", dtype=torch.float32),
            "num_samples": n,
        }
        if task_stats_dict is not None and layer_name in task_stats_dict:
            task_A, task_B, task_n = task_stats_dict[layer_name]
            cov_cache.update(
                {
                    "task_A": task_A.to("cpu", dtype=torch.float32),
                    "task_B": task_B.to("cpu", dtype=torch.float32),
                    "task_num_samples": task_n,
                }
            )
        layer_to_cov_cache[layer_name] = cov_cache
    return layer_to_cov_cache


def _to_cpu_float32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to("cpu", dtype=torch.float32).contiguous()


def calculate_projection_cache_with_kfac(A, B):
    return {
        "A": _to_cpu_float32(A),
        "B": _to_cpu_float32(B),
    }


def get_weights(
    model: AutoModelForCausalLM,
    hparams: AdamHyperParams,
    to_cpu: bool = False,
) -> Dict[str, torch.Tensor]:
    return {
        n: (p.detach().cpu().clone() if to_cpu else p)
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n and "bias" not in n
    }


def calculate_cov_cache_with_old_data(model, tok, hparams, force_recompute=False) -> Dict[str, Dict]:
    return _build_cov_cache_from_hparams(model, tok, hparams, force_recompute)


def calculate_cov_cache_with_request(
    txt,
    tgt,
    model,
    tok,
    hparams,
    add_pretrain_data: Optional[bool] = None,
    sample_size: Optional[int] = None,
):
    if add_pretrain_data is None:
        add_pretrain_data = getattr(hparams, "edit_cache_style", "new") == "mix"
    if sample_size is None:
        sample_size = getattr(hparams, "edit_n_samples", 10)
    cov_stats_dict = layer_stats_kfac_with_txt_tgt(
        model,
        tok,
        layer_names=_layer_names(hparams),
        txt=txt,
        tgt=tgt,
        precision=hparams.mom2_dtype,
        sample_size=sample_size,
        to_collect=["mom2"],
        add_pretrain_data=add_pretrain_data,
        pretrain_sample_size=hparams.mom2_n_samples,
    )

    layer_to_cov_cache = {}
    for layer_name in _layer_names(hparams):
        A, B, num_samples = cov_stats_dict.pop(layer_name)
        layer_to_cov_cache[layer_name] = {
            "A": A.to("cpu", dtype=torch.float32),
            "B": B.to("cpu", dtype=torch.float32),
            "num_samples": num_samples,
        }
        del A, B
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return layer_to_cov_cache


def cache_weights_to_cpu(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not isinstance(weights, dict):
        raise ValueError("Input must be a dict of tensors.")
    return {name: param.detach().cpu().clone() for name, param in weights.items()}


def is_weights_changed(current_weights, cached_weights, threshold: float) -> bool:
    for name, param in current_weights.items():
        cached_param = cached_weights[name]
        denom = torch.norm(cached_param).clamp(min=1e-8)
        change = torch.norm(param.detach().cpu() - cached_param) / denom
        if change > threshold:
            print(f"Weight {name} changed by {change:.4f}, exceeding threshold {threshold}.")
            return True
    return False


def recalculate_cov_cache_if_weights_changed(
    model,
    tok,
    hparams,
    current_weights_cpu,
    layer_to_cov_cache,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict], bool]:
    if (
        not getattr(hparams, "recalculate_cache", False)
        or current_weights_cpu is None
    ):
        return current_weights_cpu, layer_to_cov_cache, False

    weights = get_weights(model, hparams)
    threshold = getattr(hparams, "recalculate_weight_threshold", 0.01)
    if not is_weights_changed(weights, current_weights_cpu, threshold):
        return current_weights_cpu, layer_to_cov_cache, False

    del layer_to_cov_cache, weights
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    layer_to_cov_cache = calculate_cov_cache_with_old_data(
        model, tok, hparams, force_recompute=True
    )
    weights = get_weights(model, hparams)
    current_weights_cpu = cache_weights_to_cpu(weights)
    return current_weights_cpu, layer_to_cov_cache, True


def calculate_old_loss(model, tok, hparams):
    if getattr(hparams, "disable_old_loss_check", False):
        return {}
    with torch.no_grad():
        old_task_loss = calculate_cache_loss(
            model,
            tok,
            hparams.mom2_dataset,
            sample_size=100,
        )
    return {"General capabilities of the model Loss": old_task_loss}


def calculate_old_edit_loss(txt_chunks, tgt_chunks, model, tok):
    if len(txt_chunks) == 0:
        return {}

    mets = {}
    with torch.no_grad():
        for i, (txt, tgt) in enumerate(zip(txt_chunks, tgt_chunks)):
            request_loss = calculate_request_loss(model, tok, txt, tgt, sample_size=10)
            mets.update({f"OLD_EDIT_LOSS/Old Edit Loss Chunk {i}": request_loss})
    avg_loss = sum(mets.values()) / len(mets)
    mets.update({"Task 2 Loss": avg_loss})
    return mets


def _valid_cov_caches(layer_to_cov_caches: Optional[List[Dict[str, Dict]]]) -> List[Dict[str, Dict]]:
    if not layer_to_cov_caches:
        return []
    return [cache for cache in layer_to_cov_caches if cache]


def build_optimizer_with_cov_caches(
    model,
    hparams,
    layer_to_cov_caches: List[Dict[str, Dict]],
    opt=None,
):
    valid_caches = _valid_cov_caches(layer_to_cov_caches)
    if valid_caches:
        combined_layer_to_cov_cache = combine_layer_to_cov_caches(valid_caches)
        weight_to_projection_cache = calculate_projection_caches_from_cov_caches(
            model,
            hparams,
            combined_layer_to_cov_cache,
        )
    else:
        weight_to_projection_cache = {}

    if opt is not None:
        opt.reset_cache(weight_to_projection_cache)
        return opt

    weights = get_weights(model, hparams)
    opt = ProjectedAdam(
        [v for _, v in weights.items()],
        projection_cache_map=weight_to_projection_cache,
        soft_lambda=getattr(hparams, "soft_lambda", 1.0),
        factor_damping=getattr(hparams, "newton_damping", 1e-5),
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    opt.precompute_bases()
    return opt


def combine_layer_to_cov_caches(
    layer_to_cov_caches: List[Dict[str, Dict]],
) -> Dict[str, Dict]:
    layer_to_cov_caches = _valid_cov_caches(layer_to_cov_caches)
    if len(layer_to_cov_caches) == 0:
        return {}
    if len(layer_to_cov_caches) == 1:
        return layer_to_cov_caches[0]

    combined_layer_to_cov_caches = {}
    for layer_name in layer_to_cov_caches[0].keys():
        A_list = [layer_to_cov[layer_name]["A"] for layer_to_cov in layer_to_cov_caches]
        B_list = [layer_to_cov[layer_name]["B"] for layer_to_cov in layer_to_cov_caches]
        num_samples_list = [
            max(int(layer_to_cov[layer_name].get("num_samples", 0)), 1)
            for layer_to_cov in layer_to_cov_caches
        ]
        total_samples = sum(num_samples_list)

        combined_A = sum(
            A * num_sample for A, num_sample in zip(A_list, num_samples_list)
        ) / total_samples
        combined_B = sum(
            B * num_sample for B, num_sample in zip(B_list, num_samples_list)
        ) / total_samples

        combined_layer_to_cov_caches[layer_name] = {
            "A": combined_A,
            "B": combined_B,
            "num_samples": total_samples,
        }
        task_caches = [
            layer_to_cov[layer_name]
            for layer_to_cov in layer_to_cov_caches
            if "task_A" in layer_to_cov[layer_name] and "task_B" in layer_to_cov[layer_name]
        ]
        if task_caches:
            task_A_list = [cache["task_A"] for cache in task_caches]
            task_B_list = [cache["task_B"] for cache in task_caches]
            task_num_samples_list = [
                max(int(cache.get("task_num_samples", cache.get("num_samples", 0))), 1)
                for cache in task_caches
            ]
            task_total_samples = sum(task_num_samples_list)
            combined_layer_to_cov_caches[layer_name].update(
                {
                    "task_A": sum(
                        A * num_sample
                        for A, num_sample in zip(task_A_list, task_num_samples_list)
                    ) / task_total_samples,
                    "task_B": sum(
                        B * num_sample
                        for B, num_sample in zip(task_B_list, task_num_samples_list)
                    ) / task_total_samples,
                    "task_num_samples": task_total_samples,
                }
            )
    print(f"Combined samples {num_samples_list}")
    return combined_layer_to_cov_caches


def attach_task_factors(
    cap_caches: Optional[Dict[str, Dict]],
    source_caches: Optional[Dict[str, Dict]],
) -> Optional[Dict[str, Dict]]:
    """Copy edit/task K-FAC factors onto a cap cache without blending A/B."""
    if not cap_caches or not source_caches:
        return cap_caches
    for layer_name, cap_cache in cap_caches.items():
        source = source_caches.get(layer_name)
        if source is None or "task_A" not in source or "task_B" not in source:
            continue
        cap_cache["task_A"] = source["task_A"]
        cap_cache["task_B"] = source["task_B"]
        if "task_num_samples" in source:
            cap_cache["task_num_samples"] = source["task_num_samples"]
    return cap_caches


def estimate_chunk_kfac(txt, tgt, model, tok, hparams) -> Dict[str, Dict]:
    """Estimate K-FAC on the current edit chunk only. No wiki mix, no subsample."""
    return calculate_cov_cache_with_request(
        txt,
        tgt,
        model,
        tok,
        hparams,
        add_pretrain_data=False,
        sample_size=len(txt),
    )


def _with_sample_count(caches: Dict[str, Dict], n: int) -> Dict[str, Dict]:
    """Copy A/B and set the Algorithm-1 mix weight (M or T), dropping task factors."""
    n = max(int(n), 1)
    return {
        layer_name: {
            "A": cache["A"],
            "B": cache["B"],
            "num_samples": n,
        }
        for layer_name, cache in caches.items()
    }


def pair_retention_with_edit(
    retention: Dict[str, Dict],
    edit_cache: Dict[str, Dict],
) -> Dict[str, Dict]:
    """Pair retention-side (A, B) with current-chunk edit-side (task_A, task_B)."""
    paired = {}
    for layer_name, ret in retention.items():
        edit = edit_cache[layer_name]
        paired[layer_name] = {
            "A": ret["A"],
            "B": ret["B"],
            "num_samples": max(int(ret.get("num_samples", 0)), 1),
            "task_A": edit["A"],
            "task_B": edit["B"],
            "task_num_samples": max(int(edit.get("num_samples", 0)), 1),
        }
    return paired


def update_retention_eq13(
    retention: Dict[str, Dict],
    new_chunk_cache: Dict[str, Dict],
    t_k: int,
) -> Dict[str, Dict]:
    """Algorithm 1 sample-weighted mix of post-update edit factors into retention.

    M_k = M_{k-1} + T_k
    A_bar_k = (M_{k-1} A_bar_{k-1} + T_k A_k) / M_k
    where T_k = |D_edit^{(k)}| and (A_k, B_k) are estimated on theta_k.
    """
    incoming = _with_sample_count(new_chunk_cache, t_k)
    return combine_layer_to_cov_caches([retention, incoming])


def _retention_sample_count(retention: Dict[str, Dict]) -> int:
    if not retention:
        return 0
    return max(int(next(iter(retention.values())).get("num_samples", 0)), 0)


def _find_weight_for_layer(weights: Dict[str, torch.Tensor], layer_name: str):
    if layer_name in weights:
        return weights[layer_name]

    clean = layer_name[:-len(".weight")] if layer_name.endswith(".weight") else layer_name
    for weight_name, weight in weights.items():
        if layer_name in weight_name or clean in weight_name:
            return weight
    raise KeyError(f"Could not find trainable weight for layer {layer_name}")


def calculate_projection_caches_from_cov_caches(
    model,
    hparams,
    layer_to_cov_caches,
):
    weight_to_projection_cache = {}
    weights = get_weights(model, hparams)

    for layer_name, cov_cache in layer_to_cov_caches.items():
        A = cov_cache["A"].to(dtype=torch.float32)
        B = cov_cache["B"].to(dtype=torch.float32)

        if not _is_llama_or_phi(hparams.model_name):
            A, B = B, A

        projection_cache = calculate_projection_cache_with_kfac(
            A, B
        )
        projection_cache.update(
            {
                "cap_A": projection_cache["A"],
                "cap_B": projection_cache["B"],
            }
        )

        if "task_A" not in cov_cache or "task_B" not in cov_cache:
            raise ValueError(
                f"RhoEdit projection for {layer_name} is missing task_A/task_B. "
                "Pass the original capability cache, or call attach_task_factors, "
                "so edit curvature is preserved."
            )
        task_A = cov_cache["task_A"].to(dtype=torch.float32)
        task_B = cov_cache["task_B"].to(dtype=torch.float32)
        task_num_samples = cov_cache.get("task_num_samples")

        if not _is_llama_or_phi(hparams.model_name):
            task_A, task_B = task_B, task_A

        task_projection_cache = calculate_projection_cache_with_kfac(task_A, task_B)
        projection_cache.update(
            {
                "edit_A": task_projection_cache["A"],
                "edit_B": task_projection_cache["B"],
                "task_num_samples": task_num_samples,
            }
        )
        projection_cache["layer_name"] = layer_name
        weight_to_projection_cache[_find_weight_for_layer(weights, layer_name)] = projection_cache

        del A, B, task_A, task_B
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return weight_to_projection_cache


def update_model_and_tokenizer_with_appropriate_padding_token(model, tokenizer, hparams):
    if "Qwen" in hparams.model_name:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def setup_requests_for_safeedit(requests: List[Dict]) -> List[Dict]:
    """Adapt SafeEdit records to the prompt/target format used by RhoEdit."""
    if not requests:
        return []
    if "target_new" in requests[0]:
        return requests

    return [
        {
            "prompt": request["question"],
            "target_new": request["target_unsafe"],
        }
        for request in requests
    ]

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def _peak_memory_gb() -> float:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return 0.0
    return max(
        torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
    ) / 1e9


def _write_table23_train_stats(hparams, opt, train_t0: float) -> None:
    energy_fn = getattr(opt, "mean_retained_edit_gradient_energy", None)
    stats = {
        "n": int(getattr(hparams, "mom2_n_samples", 0) or 0),
        "lambda": float(getattr(hparams, "soft_lambda", 0.0)),
        "lr": float(getattr(hparams, "lr", 0.0)),
        "newton_damping": float(getattr(hparams, "newton_damping", 0.0)),
        "time_h": (time.time() - train_t0) / 3600.0,
        "memory_gb": _peak_memory_gb(),
        "retained_edit_gradient_energy": (
            float(energy_fn()) if callable(energy_fn) else 1.0
        ),
        "run_name": getattr(hparams, "run_name", None),
    }
    print(f"[RhoEdit] TABLE23_TRAIN {json.dumps(stats)}", flush=True)
    ExperimentTracker.log(
        {
            "table23/time_h": stats["time_h"],
            "table23/memory_gb": stats["memory_gb"],
            "table23/retained_edit_gradient_energy": stats[
                "retained_edit_gradient_energy"
            ],
        }
    )
    run_name = stats["run_name"]
    if not run_name:
        return
    out_dir = os.path.join("logs", str(run_name))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "table23_train.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
        handle.write("\n")


def execute_sft_adam(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AdamHyperParams,
    **kwargs: Any,
) -> AutoModelForCausalLM:
    print("[execute_sft_adam]Enter the function")
    train_t0 = time.time()
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(idx)
    patch_qwen_device_map_attention(model)
    # Keep Dropout/SDPA on the inference kernels. train() plus a Long tensor
    # on the lm_head GPU makes 3-GPU SDPA return all-NaN logits.
    model.eval()
    # Reusing DynamicCache across batches with different padded lengths NaNs
    # from the first layer on the edit GPU. Disable cache for the whole SFT.
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
    device = _embed_device(model)
    if tok.padding_side != "right":
        tok.padding_side = "right"
    
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"] and request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]
    
    layer_to_cov_cache_old = calculate_cov_cache_with_old_data(
        model, tok, hparams, force_recompute=False
    )
    
    opt = build_optimizer_with_cov_caches(model, hparams, [layer_to_cov_cache_old])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    weights = get_weights(model, hparams)
    current_weights_cpu = cache_weights_to_cpu(weights)
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    old_loss = calculate_old_loss(model, tok, hparams)
    ExperimentTracker.log(old_loss)
    loss_meter = AverageMeter()
    pbar = trange(hparams.num_steps)
    print("[execute_sft_adam]Start training\n")
    for it in pbar:
        loss_meter.reset()

        random.shuffle(requests)
        texts = [r["prompt"] for r in requests]
        targets = [r["target_new"] for r in requests]

        # split into batches
        for txt, tgt in zip(
            chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            inputs_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
            encodings = tok(inputs_targets, return_tensors="pt", padding="max_length", truncation=True, max_length=hparams.max_length).to(device)
            labels = _sft_labels(encodings, tok, txt, hparams.max_length)
            opt.zero_grad(set_to_none=True)
            outputs = model(**encodings, use_cache=False)
            loss = _causal_nll(outputs.logits, labels)
                
            loss_meter.update(loss.item(), n=labels.size(0))
            if loss.item() >= 1e-2:
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                opt.step()
                current_weights_cpu, layer_to_cov_cache_old, should_recalculate = recalculate_cov_cache_if_weights_changed(
                    model,
                    tok,
                    hparams,
                    current_weights_cpu,
                    layer_to_cov_cache_old,
                )
                if should_recalculate:
                    opt = build_optimizer_with_cov_caches(model, hparams, [layer_to_cov_cache_old], opt=opt)
        metrics =calculate_old_loss(model, tok, hparams)
        metrics.update({"FT Loss": loss_meter.avg})
        ExperimentTracker.log(metrics)
        
        pbar.write(f"FT Loss: {loss_meter.avg:.4f}")
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

        if loss_meter.avg < 1e-2:
            break

    _write_table23_train_stats(hparams, opt, train_t0)
    return model

def execute_sft_adam_sequential(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AdamHyperParams,
    tracker=None,
    **kwargs: Any,
) -> AutoModelForCausalLM:
    """Sequential RhoEdit (Algorithm 1 when ``edit_cache_style=sequential``).

    ``tracker`` (SequentialEditTracker) is optional and only adds per-round
    evaluation/plotting; the edit path is unchanged.
    """
    patch_qwen_device_map_attention(model)
    model.eval()
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
    device = _embed_device(model)
    cache_style = getattr(hparams, "edit_cache_style", "sequential")
    
    if tok.padding_side != "right":
        tok.padding_side = "right"
    
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"] and request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]
    max_samples = getattr(hparams, "max_samples", None)
    data_seed = int(getattr(hparams, "data_seed", 69))
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("hparams.max_samples must be > 0")
        rng = random.Random(data_seed)
        rng.shuffle(requests)
        requests = requests[:max_samples]
        print(
            f"[RhoEdit] first-chunk screen: data_seed={data_seed} "
            f"n={len(requests)} num_edits={hparams.num_edits}"
        )
    else:
        random.shuffle(requests)
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]
    rephrases = [r.get("rephrase_prompt") for r in requests]
    txt_chunks, tgt_chunks, reph_chunks = [], [], []
    num_rounds = -(-len(texts) // hparams.num_edits)


    layer_to_cov_cache_old = calculate_cov_cache_with_old_data(
        model, tok, hparams, force_recompute=False
    )
    # Algorithm 1: (A_bar_0, B_bar_0, M_0) <- (A_cap, B_cap, N), N = |D_cap|.
    retention = _with_sample_count(
        layer_to_cov_cache_old, _cache_sample_size(hparams)
    )

    # mix/disable still train the first chunk with the startup wiki+task pair.
    # sequential waits until the current-chunk edit factors are estimated.
    if cache_style == "sequential":
        print(
            "[RhoEdit] edit_cache_style=sequential uses Algorithm 1 "
            f"(post-update mix, M_0={_retention_sample_count(retention)})."
        )
        opt = None
    else:
        opt = build_optimizer_with_cov_caches(model, hparams, [layer_to_cov_cache_old])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    weights = get_weights(model, hparams)
    current_weights_cpu = cache_weights_to_cpu(weights)
        
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    old_loss = calculate_old_loss(model, tok, hparams)
    ExperimentTracker.log(old_loss)
    if tracker is not None:
        tracker.set_baseline(old_loss.get(CAP_LOSS_KEY), num_rounds)
    
    layer_to_cov_cache_data = None
    loss_meter = AverageMeter()

    # split into batches
    for round_idx, (txt_edit, tgt_edit, reph_edit) in enumerate(zip(
        chunks(texts, hparams.num_edits),
        chunks(targets, hparams.num_edits),
        chunks(rephrases, hparams.num_edits),
    )):
        if tracker is not None:
            tracker.begin_round(round_idx)
        if cache_style == "sequential":
            print(
                f"[RhoEdit] Estimate edit-side K-FAC on {len(txt_edit)} "
                "current-chunk samples at theta_{k-1}."
            )
            edit_cache = estimate_chunk_kfac(txt_edit, tgt_edit, model, tok, hparams)
            paired = pair_retention_with_edit(retention, edit_cache)
            opt = build_optimizer_with_cov_caches(model, hparams, [paired], opt=opt)

        pbar = trange(hparams.num_steps)
        for it in pbar:
            loss_meter.reset()
            for txt, tgt in zip(
                chunks(txt_edit, hparams.batch_size), chunks(tgt_edit, hparams.batch_size)
            ):
                inputs_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
                encodings = tok(
                    inputs_targets,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=hparams.max_length,
                ).to(device)
                labels = _sft_labels(encodings, tok, txt, hparams.max_length)
                opt.zero_grad()
                outputs = model(**encodings, use_cache=False)
                loss = _causal_nll(outputs.logits, labels)

                if loss.item() >= 1e-2:
                    loss.backward()
                    opt.step()
                    # Algorithm 1 does not recompute wiki/task caches mid-round.
                    if cache_style == "sequential":
                        should_recalculate = False
                    else:
                        current_weights_cpu, layer_to_cov_cache_old, should_recalculate = recalculate_cov_cache_if_weights_changed(
                            model,
                            tok,
                            hparams,
                            current_weights_cpu,
                            layer_to_cov_cache_old,
                        )
                    if should_recalculate:                            
                        if hparams.edit_n_samples > 0 and len(txt_chunks) > 0:
                            old_txt_list = [item for sublist in txt_chunks for item in sublist]
                            old_tgt_list = [item for sublist in tgt_chunks for item in sublist]

                            layer_to_cov_cache_data = calculate_cov_cache_with_request(
                                old_txt_list,
                                old_tgt_list,
                                model,
                                tok,
                                hparams,
                            )
                            opt = build_optimizer_with_cov_caches(model, hparams, [layer_to_cov_cache_old, layer_to_cov_cache_data], opt=opt)
                        else:
                            opt = build_optimizer_with_cov_caches(model, hparams, [layer_to_cov_cache_old] if layer_to_cov_cache_data is None else [layer_to_cov_cache_old, layer_to_cov_cache_data], opt=opt)
                loss_meter.update(loss.item(), n=labels.size(0))
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})
            if loss_meter.avg < 1e-2:
                break
        print(f"Loss after editing number of samples {len(txt_edit)}: {loss_meter.avg}")
        
        txt_chunks.append(txt_edit)
        tgt_chunks.append(tgt_edit)
        reph_chunks.append(reph_edit)
        # Editor state reported by the tracker (K-FAC factors kept across rounds).
        state_ref = layer_to_cov_cache_old
        
        if cache_style == "sequential":
            t_k = len(txt_edit)
            print(
                f"[RhoEdit] Re-estimate edit K-FAC on {t_k} current-chunk "
                f"samples at theta_k, then mix into retention (T_k={t_k})."
            )
            post_cache = estimate_chunk_kfac(txt_edit, tgt_edit, model, tok, hparams)
            retention = update_retention_eq13(retention, post_cache, t_k)
            print(f"[RhoEdit] Retention M_k={_retention_sample_count(retention)}.")
            state_ref = retention

        elif cache_style == "mix":
            old_txt_list = [item for sublist in txt_chunks for item in sublist]
            old_tgt_list = [item for sublist in tgt_chunks for item in sublist]

            layer_to_cov_cache_data_pretrain_mix = attach_task_factors(
                calculate_cov_cache_with_request(
                    old_txt_list,
                    old_tgt_list,
                    model,
                    tok,
                    hparams,
                ),
                layer_to_cov_cache_old,
            )

            opt = build_optimizer_with_cov_caches(
                model, hparams, [layer_to_cov_cache_data_pretrain_mix], opt=opt
            )
            state_ref = layer_to_cov_cache_data_pretrain_mix
        elif cache_style == "disable":
            print("[RhoEdit] edit_cache_style=disable; projection cache not updated.")

        metrics = calculate_old_loss(model, tok, hparams)
        old_edit_loss = calculate_old_edit_loss(txt_chunks, tgt_chunks, model, tok)
        metrics.update(old_edit_loss)
        metrics.update({"FT Loss": loss_meter.avg})
        ExperimentTracker.log(metrics)

        if tracker is not None:
            tracker.end_round(
                round_idx,
                model,
                tok,
                txt_chunks,
                tgt_chunks,
                reph_chunks,
                state_bytes=cov_cache_nbytes(state_ref),
                cap_loss=metrics.get(CAP_LOSS_KEY),
                ft_loss=loss_meter.avg,
            )

    if tracker is not None:
        tracker.finalize()

    return model
