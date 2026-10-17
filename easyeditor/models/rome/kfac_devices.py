"""Place independent per-layer K-FAC statistics on visible CUDA devices."""

import torch


def validate_kfac_stats_devices(value):
    value = str(value).strip()
    if value in ("auto_two", "auto_three", "model"):
        return value
    parts = value.split(",")
    if len(parts) < 2 or any(not part.strip().isdigit() for part in parts):
        raise ValueError("kfac_stats_devices must be auto_two, auto_three, model, or at least two CUDA indices (e.g. 0,1,2)")
    ids = [int(part) for part in parts]
    if len(set(ids)) != len(ids):
        raise ValueError("kfac_stats_devices requires distinct CUDA indices")
    return ",".join(map(str, ids))


def kfac_layer_devices(layer_names, model_device, selection="model"):
    """Split ordered layers into contiguous, nearly equal groups across GPUs.

    Indices refer to the process's CUDA_VISIBLE_DEVICES namespace. Selection
    changes statistics placement only, not model placement or sample ordering.
    """
    selection = validate_kfac_stats_devices(selection)
    model_device = torch.device(model_device)
    automatic = selection in ("auto_two", "auto_three")
    if selection == "model" or (automatic and model_device.type == "cpu"):
        devices = [model_device]
    else:
        count = torch.cuda.device_count()
        requested = (3 if selection == "auto_three" else 2) if automatic else len(selection.split(','))
        if count < requested:
            raise ValueError(f"K-FAC selection {selection} requires at least {requested} visible GPUs; "
                             "use kfac_stats_devices=model for one GPU")
        if automatic:
            free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(count)]
            ids = [i for _, i in sorted(free, key=lambda pair: (-pair[0], pair[1]))[:requested]]
        else:
            ids = [int(i) for i in selection.split(",")]
            if any(i >= count for i in ids):
                raise ValueError(f"K-FAC device index outside {count} visible GPUs: {selection}")
        devices = [torch.device("cuda", i) for i in ids]
    size, remainder = divmod(len(layer_names), len(devices))
    mapping = {}
    start = 0
    for i, device in enumerate(devices):
        end = start + size + (i < remainder)
        names = layer_names[start:end]
        mapping.update((name, device) for name in names)
        start = end
        print(f"[K-FAC statistics] {device}: {len(names)} layers: {', '.join(names)}")
    return mapping


@torch.no_grad()
def accumulate_kfac_factors(factors, inputs, output_grads, valid_mask):
    """Mask on source devices, then move only valid rows to the factor device."""
    device, dtype = factors["A"].device, factors["A"].dtype
    inputs = inputs[:, :-1, :]
    output_grads = output_grads[:, :-1, :]
    input_flat = inputs[valid_mask.to(inputs.device)].to(device=device, dtype=dtype)
    grad_flat = output_grads[valid_mask.to(output_grads.device)].to(device=device, dtype=dtype)
    factors["A"].addmm_(input_flat.T, input_flat)
    factors["B"].addmm_(grad_flat.T, grad_flat)
