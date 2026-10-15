from math import ceil

import librosa
import numpy as np

from src.core.config import CONFIG, LayerConfig, SynthConfig
from src.models.audio import LabeledDrop
from src.services.synthesis.bank import SampleBank
from src.services.synthesis.effects import Filter, Sidechain
from src.services.synthesis.grid import Grid
from src.services.synthesis.mix import Mix
from src.services.synthesis.patterns.kick import draw_kick_track
from src.services.synthesis.patterns.loop import draw_loop_track
from src.services.synthesis.patterns.one_shot import draw_one_shot_track
from src.services.synthesis.track import Track


def draw_grid(config: SynthConfig, rng: np.random.Generator) -> Grid:
    bpm = int(rng.choice(config.bpms))
    bar_seconds = 60 * Grid.beats_per_bar / bpm
    # One extra bar leaves room for a random crop start.
    bars = ceil(config.drop_seconds / bar_seconds) + 1
    return Grid(
        bars=bars,
        bpm=bpm,
        steps_per_beat=int(rng.choice(config.steps)),
    )


def _draw_track_count(layer: LayerConfig, rng: np.random.Generator) -> int:
    counts = list(layer.track_count_weights)
    weights = np.array(list(layer.track_count_weights.values()))
    return int(rng.choice(counts, p=weights / weights.sum()))


def _add_layer_effects(
    track: Track,
    layer: LayerConfig,
    kick_starts: np.ndarray,
    rng: np.random.Generator,
) -> None:
    if layer.high_pass_hz is not None:
        track.add_effects(Filter(cutoff=layer.high_pass_hz, btype="highpass"))
    if layer.sidechain_ratio_range is not None:
        ratio = rng.uniform(*layer.sidechain_ratio_range)
        track.add_effects(Sidechain(kick_starts, ratio=ratio))


def generate_drop(
    bank: SampleBank,
    rng: np.random.Generator,
    config: SynthConfig = CONFIG.synth,
) -> LabeledDrop:
    """Generate a random drop and the onset of each kick in it."""
    grid = draw_grid(config, rng)
    mix = Mix()

    # 1. Generate the kick track
    kick = draw_kick_track(grid, bank, rng)
    mix.add_tracks(kick.track)

    # 2. Generate the tracks of each layer
    kick_starts = grid.to_samples(kick.beats)
    for layer in config.layers:
        for _ in range(_draw_track_count(layer, rng)):
            track = (
                draw_loop_track(grid, bank, layer, kick.level, rng)
                if layer.is_loop
                else draw_one_shot_track(grid, bank, layer, kick.level, rng)
            )
            _add_layer_effects(track, layer, kick_starts, rng)
            mix.add_tracks(track)

    onsets = librosa.samples_to_time(kick_starts, sr=grid.sr)
    return LabeledDrop(signal=mix.mix(), onsets=onsets)
