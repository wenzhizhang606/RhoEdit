import os
import re
from typing import Any, Dict, Optional

from dotenv import load_dotenv, find_dotenv

try:
    import wandb
except ImportError:
    wandb = None

try:
    import swanlab
except ImportError:
    swanlab = None


def _sanitize_metric_key(key: Any) -> str:
    text = str(key).strip()
    text = text.replace(" ", "_")
    text = re.sub(r"[^0-9A-Za-z_./-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "metric"


def _to_scalar(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item") and getattr(value, "ndim", 0) == 0:
        try:
            value = value.item()
        except (ValueError, RuntimeError, TypeError):
            return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return float(value) if isinstance(value, float) else int(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sanitize_metrics(metrics: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dicts and coerce values to SwanLab/wandb scalars."""
    if not isinstance(metrics, dict):
        scalar = _to_scalar(metrics)
        if scalar is None:
            return {}
        return {_sanitize_metric_key(prefix or "metric"): scalar}

    out = {}
    for raw_key, raw_value in metrics.items():
        key = _sanitize_metric_key(raw_key)
        if prefix:
            key = f"{prefix}/{key}" if key else prefix
        if isinstance(raw_value, dict):
            out.update(sanitize_metrics(raw_value, prefix=key))
            continue
        scalar = _to_scalar(raw_value)
        if scalar is None:
            continue
        out[key] = scalar
    return out


class ExperimentTracker:
    """单例实验追踪器，通过类名直接调用

    """
    _instance = None

    # ── 类级状态 ──
    _mode: bool = True
    _use_wandb: bool = False
    _use_swanlab: bool = False
    _project: str = ""
    _name: str = ""

    @classmethod
    def init(cls,
             project: str,
             name: str = None,
             config: dict = None,
             tracker_type: str = None,
             mode: bool = True):
        """初始化单例追踪器。

        :param project: 项目名称
        :param name: 实验/Run名称
        :param config: 超参数配置字典
        :param tracker_type: "wandb", "swanlab", 或 "none"
        :param mode: 是否启用追踪
        """
        load_dotenv(find_dotenv())

        cls._mode = mode
        if tracker_type is None:
            cls._use_wandb = False
            cls._use_swanlab = False
        else:
            ttype = tracker_type.lower()
            cls._use_wandb = (ttype == "wandb")
            cls._use_swanlab = (ttype == "swanlab")

        cls._project = project
        cls._name = name or ""
        cls._config = config or {}

        if cls._mode:
            cls._setup_backend()
            cls._launch_run()

        cls._instance = cls.__new__(cls)

    @classmethod
    def _setup_backend(cls):
        """登录对应的后端服务。"""
        if cls._use_wandb:
            if wandb is None:
                raise ImportError("未安装 wandb 库!")
            key = os.getenv("WANDB_API_KEY")
            if not key:
                raise ValueError(".env 中未找到 WANDB_API_KEY")
            wandb.login(key=key)

        elif cls._use_swanlab:
            if swanlab is None:
                raise ImportError("未安装 swanlab 库!")
            key = os.getenv("SWANLAB_API_KEY")
            if not key:
                raise ValueError(".env 中未找到 SWANLAB_API_KEY")
            swanlab.login(api_key=key)

    @classmethod
    def _launch_run(cls):
        """启动对应的实验 run。"""
        if cls._use_wandb:
            wandb.init(project=cls._project, name=cls._name, config=cls._config)
            print(f"Wandb initialized -> Project: {cls._project}, Run: {cls._name}")
        elif cls._use_swanlab:
            swanlab.init(project=cls._project, experiment_name=cls._name, config=cls._config)
            print(f"SwanLab initialized -> Project: {cls._project}, Experiment: {cls._name}")

    @classmethod
    def log(cls, metrics: dict, step: int = None):
        """记录指标。"""
        if not cls._mode:
            return
        payload = sanitize_metrics(metrics)
        if not payload:
            return
        if cls._use_wandb:
            wandb.log(payload, step=step)
        elif cls._use_swanlab:
            swanlab.log(payload, step=step)

    @classmethod
    def finish(cls):
        """结束当前 run。"""
        if not cls._mode:
            return
        if cls._use_wandb:
            wandb.finish()
        elif cls._use_swanlab:
            if hasattr(swanlab, 'finish'):
                swanlab.finish()

