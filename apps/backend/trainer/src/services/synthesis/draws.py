import numpy as np


def draw_count(weights: dict[int, float], rng: np.random.Generator) -> int:
    """Draw a count with the given weight for each count."""
    counts = list(weights)
    p = np.array(list(weights.values()))
    return int(rng.choice(counts, p=p / p.sum()))


def draw_log_uniform(
    value_range: tuple[float, float], rng: np.random.Generator
) -> float:
    """Draw a value from the range, uniform on a log scale."""
    low, high = np.log(value_range)
    return float(np.exp(rng.uniform(low, high)))
