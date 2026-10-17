"""Small, deterministic helpers for sequential edit history replay."""

import math
import random
from typing import List, Sequence, Tuple


def validate_history_replay(ratio, max_samples, weight):
    ratio = float(ratio)
    max_samples = int(max_samples)
    weight = float(weight)
    if not math.isfinite(ratio) or ratio < 0:
        raise ValueError("history_replay_ratio must be finite and nonnegative")
    if max_samples < 0:
        raise ValueError("history_replay_max must be nonnegative")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("history_replay_weight must be finite and nonnegative")
    return ratio, max_samples, weight


def history_replay_count(current_count: int, ratio: float, max_samples: int) -> int:
    """Return replay examples per optimizer batch.

    ``max_samples=0`` means no explicit cap.  A positive ratio always gives
    at least one replay example once the history buffer is non-empty.
    """
    if current_count <= 0 or ratio <= 0:
        return 0
    count = max(1, int(round(current_count * ratio)))
    return min(count, max_samples) if max_samples > 0 else count


def sample_history_pairs(
    history: Sequence[Tuple[str, str]], count: int, rng=None
) -> List[Tuple[str, str]]:
    """Sample previous prompt/target pairs, with replacement when necessary."""
    if count <= 0 or not history:
        return []
    rng = random if rng is None else rng
    if len(history) >= count:
        return list(rng.sample(list(history), count))
    return [rng.choice(history) for _ in range(count)]
