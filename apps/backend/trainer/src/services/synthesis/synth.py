from math import ceil

import librosa
import numpy as np

from src.core.config import CONFIG, LayerConfig, SynthConfig
from src.models.audio import LabeledDrop
from src.services.synthesis.bank import SampleBank
from src.services.synthesis.crop import crop_drop, draw_crop
from src.services.synthesis.draws import draw_count, draw_log_uniform
from src.services.synthesis.effects import Filter, Sidechain
from src.services.synthesis.grid import Grid
from src.services.synthesis.master import draw_master_effects
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


def _add_layer_effects(
    track: Track,
    layer: LayerConfig,
    sidechain: Sidechain,
    rng: np.random.Generator,
) -> None:
    if layer.high_pass_hz_range is not None:
        cutoff = draw_log_uniform(layer.high_pass_hz_range, rng)
        track.add_effects(Filter(cutoff=cutoff, btype="highpass"))
    if layer.is_sidechained:
        track.add_effects(sidechain)


def generate_drop(
    bank: SampleBank,
    rng: np.random.Generator,
    config: SynthConfig = CONFIG.synth,
) -> LabeledDrop:
    """Generate a random drop and the onset of each kick in it.

    The drop peaks at ``config.peak_db`` before the master effects.
    """
    grid = draw_grid(config, rng)
    mix = Mix(sr=grid.sr)

    # 1. Generate the kick track
    kick = draw_kick_track(
        grid,
        bank,
        config.kick_gaps,
        config.kick_rolls,
        config.kick_fx,
        config.kick_off_beat_range,
        rng,
        config.kick_tail_note_range,
        config.kick_tail_skip_seconds_range,
    )
    mix.add_tracks(kick.track)

    # 2. Generate the tracks of each layer
    kick_starts = grid.to_samples(kick.beats)
    is_pumping = rng.random() < config.sidechain_probability
    sidechain = Sidechain(
        kick_starts,
        ratio=rng.uniform(*config.sidechain_ratio_range) if is_pumping else 0.0,
        duration=rng.uniform(*config.sidechain_seconds_range),
    )
    for layer in config.layers:
        for _ in range(draw_count(layer.track_count_weights, rng)):
            track = (
                draw_loop_track(grid, bank, layer, kick.level, rng)
                if layer.is_loop
                else draw_one_shot_track(grid, bank, layer, kick.level, rng)
            )
            _add_layer_effects(track, layer, sidechain, rng)
            mix.add_tracks(track)

    # 3. Normalize and augment the mix
    mix.add_effects(*draw_master_effects(config.master, config.peak_db, rng))

    onsets = librosa.samples_to_time(kick_starts, sr=grid.sr)
    return LabeledDrop(signal=mix.mix(), onsets=onsets)


def generate_cropped_drop(
    bank: SampleBank,
    rng: np.random.Generator,
    config: SynthConfig = CONFIG.synth,
) -> LabeledDrop:
    """Generate a drop and crop a random window of ``drop_seconds`` from it."""
    drop = generate_drop(bank, rng, config)
    duration = len(drop.signal) / CONFIG.sr
    return crop_drop(drop, draw_crop(rng, duration, config.drop_seconds))
