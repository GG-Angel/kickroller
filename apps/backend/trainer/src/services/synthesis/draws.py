import numpy as np


def draw_count(weights: dict[int, float], rng: np.random.Generator) -> int:
    """Draw a count with the given weight for each count."""
    counts = list(weights)
    p = np.array(list(weights.values()))
    return int(rng.choice(counts, p=p / p.sum()))
