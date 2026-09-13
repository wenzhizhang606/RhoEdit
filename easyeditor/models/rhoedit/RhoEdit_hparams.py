from dataclasses import dataclass, fields
from typing import List

import yaml

from ...util.hparams import HyperParams


@dataclass
class AdamHyperParams(HyperParams):
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: bool
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str
    device: int
    alg_name: str
    model_name: str
    objective_optimization: str
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    task_mom2_dataset: str
    task_mom2_n_samples: int
    task_dtype: str
    newton_damping: float
    soft_lambda: float
    batch_size: int = 32
    max_length: int = 40
    model_parallel: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        if not config or config.get("alg_name") != "RHOEDIT":
            raise ValueError(
                f"AdamHyperParams can not load from {hparams_name_or_path}, "
                f"alg_name is {None if not config else config.get('alg_name')}"
            )
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in config.items() if key in allowed})
