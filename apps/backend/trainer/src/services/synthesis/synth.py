from dataclasses import dataclass, field
from math import ceil
from typing import Protocol

import librosa
import numpy as np

from src.core.config import CONFIG, SynthConfig
from src.models.audio import LabeledDrop
from src.services.synthesis.bank import Sample, SampleBank, SampleKind
from src.services.synthesis.effects import Filter, Sidechain, trim
from src.services.synthesis.mix import Mix
from src.services.synthesis.track import Grid, Track


class PatternGenerator(Protocol):
    def generate(self) -> np.ndarray: ...


@dataclass
class KickPatternGenerator(PatternGenerator):
    """Randomly place kicks on a grid, favoring steps on coarser subdivisions.

    Each step in a beat fires with probability ``probability * weight ** depth``,
    where ``depth`` is how many subdivision levels the step sits below the beat
    (e.g. ``steps_per_beat=4`` gives ``p, p·w², p·w, p·w²``).
    """

    grid: Grid
    """Grid the pattern is generated for."""
    probability: float = 0.98
    """Kick probability on the beat itself, between 0 and 1."""
    weight: float = 0.12
    """Multiplier applied per subdivision level, between 0 and 1."""
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    """Random number generator used to draw the pattern."""

    def generate(self) -> np.ndarray:
        """Return the beat positions of the kicks drawn across the whole grid.

        Raises:
            ValueError: If ``probability`` or ``weight`` is outside [0, 1], or
                ``steps_per_beat`` is not positive.
        """
        if not 0 <= self.probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")
        if not 0 <= self.weight <= 1:
            raise ValueError("Weight must be between 0 and 1.")

        beat_weights = self.probability * self.weight ** self._calculate_depths(
            self.grid.steps_per_beat
        )
        weights = np.tile(beat_weights, self.grid.beats)
        steps = (
            np.arange(self.grid.beats * self.grid.steps_per_beat)
            / self.grid.steps_per_beat
        )
        return steps[self.rng.random(len(steps)) < weights]

    @staticmethod
    def _calculate_depths(steps_per_beat: int) -> np.ndarray:
        """Return the subdivision depth of each step within a beat.

        The beat is split by its largest prime factors first and by smaller ones
        at finer levels, e.g. ``4 -> [0, 2, 1, 2]`` and ``6 -> [0, 2, 1, 2, 1, 2]``.
        """
        if steps_per_beat == 1:
            return np.zeros(1, dtype=int)

        factor = next(
            (
                f
                for f in range(2, int(steps_per_beat**0.5) + 1)
                if steps_per_beat % f == 0
            ),
            steps_per_beat,
        )
        coarse = KickPatternGenerator._calculate_depths(
            steps_per_beat // factor
        )
        depths = np.full(steps_per_beat, coarse.max() + 1)
        depths[::factor] = coarse
        return depths


@dataclass
class LoopPatternGenerator(PatternGenerator):
    grid: Grid
    sample: Sample
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def generate(self) -> np.ndarray:
        if not self.sample.is_loop:
            raise ValueError(f"Sample '{self.sample.name}' is not a loop.")
        loop_beats = self._calculate_loop_bars() * self.grid.beats_per_bar
        return np.arange(0, self.grid.beats, loop_beats, dtype=float)

    def _calculate_loop_bars(self) -> int:
        length = len(self.sample.signal)
        if length < self.grid.bar:
            raise ValueError(
                f"Loop '{self.sample.name}' must be at least one bar long."
            )
        bars = 1
        while length >= 2 * bars * self.grid.bar:
            bars *= 2
        return bars


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


def generate_drop(
    bank: SampleBank,
    rng: np.random.Generator,
    config: SynthConfig = CONFIG.synth,
) -> LabeledDrop:
    """Generate a random drop and the onset of each kick in it."""
    grid = draw_grid(config, rng)
    mix = Mix()

    # 1. Generate kick track
    kick = trim(bank.draw(SampleKind.KICK, rng).signal)
    kick_pattern = KickPatternGenerator(grid, rng=rng).generate()
    kick_track = Track(grid)
    for beat in kick_pattern:
        kick_track.insert(kick, beat, blend=False)
    mix.add_tracks(kick_track)

    # 2. Generate 0-2 melody tracks
    # for _ in range(2):
    #     melody_track = Track(grid)
    #     melody = bank.draw(SampleKind.MELODY, rng, loops_only=True).stretch(
    #         melody_track
    #     )
    #     for beat in LoopPatternGenerator(grid, melody, rng=rng).generate():
    #         melody_track.insert(melody.signal, beat, blend=False)
    #     melody_track.add_effects(
    #         Filter(cutoff=200.0, btype="highpass"),
    #         Sidechain(grid.to_samples(kick_pattern), ratio=0.4),
    #     )
    #     mix.add_tracks(melody_track)

    # 3. Generate drum track
    drum_track = Track(grid)
    drum = bank.draw(SampleKind.DRUM, rng, loops_only=True).stretch(drum_track)
    for beat in LoopPatternGenerator(grid, drum, rng=rng).generate():
        drum_track.insert(drum.signal, beat, blend=False)
    drum_track.add_effects(
        Filter(cutoff=200.0, btype="highpass"),
    )
    mix.add_tracks(drum_track)

    # 4. Generate vocal track
    vocal_track = Track(grid)
    vocal = bank.draw(SampleKind.VOCAL, rng, loops_only=True).stretch(
        vocal_track
    )
    for beat in LoopPatternGenerator(grid, vocal, rng=rng).generate():
        vocal_track.insert(vocal.signal, beat, blend=False)
    vocal_track.add_effects(
        Filter(cutoff=200.0, btype="highpass"),
        Sidechain(grid.to_samples(kick_pattern), ratio=0.25),
    )
    mix.add_tracks(vocal_track)

    onsets = librosa.samples_to_time(grid.to_samples(kick_pattern), sr=grid.sr)
    return LabeledDrop(signal=mix.mix(), onsets=onsets)
