from math import floor

import numpy as np

from src.models.audio import Signal
from src.services.synthesis.effects import cut
from src.services.synthesis.grid import Grid


class Track:
    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self.samples = floor(grid.bars * grid.bar)
        self.signal = np.zeros(self.samples, dtype=np.float32)

    def insert(
        self,
        signal: Signal,
        beat_position: float,
        blend: bool = True,
    ) -> None:
        if beat_position < 0:
            raise ValueError("Beat cannot be negative.")
        start = self.grid.to_sample(beat_position)
        if start >= self.samples:
            return
        cut_signal = cut(signal, max(0, self.samples - start))
        end = start + len(cut_signal)
        if blend:
            self.signal[start:end] += cut_signal
        else:
            self.signal[start:end] = cut_signal

    def sidechain(
        self,
        triggers: np.ndarray,
        ratio: float = 0.5,
        duration: float = 0.1,
    ) -> None:
        """Apply sidechain compression to this track based on trigger samples."""
        if not np.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError("Sidechain ratio must be between 0 and 1.")
        if not np.isfinite(duration) or duration < 0:
            raise ValueError(
                "Sidechain duration must be a finite non-negative value."
            )

        if ratio == 0 or duration == 0:
            return  # only sidechain needed if both ratio and duration are non-zero

        duration_samples = max(1, round(duration * self.grid.sr))
        fade_samples = min(
            max(1, round(0.005 * self.grid.sr)),
            duration_samples // 2,
        )
        envelope = np.zeros(self.samples, dtype=np.float32)
        ramp = 0.5 - 0.5 * np.cos(
            np.pi * np.linspace(0.0, 1.0, fade_samples + 1, dtype=np.float32)
        )
        for trigger in triggers:
            start = int(trigger)
            if start < 0:
                raise ValueError(
                    "Sidechain triggers must be non-negative sample indices."
                )
            end = min(start + duration_samples, self.samples)
            length = end - start
            if length <= 0:
                continue
            event_envelope = np.ones(length, dtype=np.float32)
            if fade_samples:
                fade = min(fade_samples, length // 2)
                if fade:
                    event_envelope[:fade] = ramp[:fade]
                    event_envelope[-fade:] = ramp[:fade][::-1]
            envelope[start:end] = np.maximum(
                envelope[start:end], event_envelope
            )
        self.signal *= 1.0 - ratio * envelope


class Mix:
    def __init__(self) -> None:
        self.tracks: list[Track] = []

    def add(self, *track: Track) -> None:
        self.tracks.extend(track)

    def mix(self) -> Signal:
        if not self.tracks:
            return np.zeros(0, dtype=np.float32)
        max_samples = max(track.samples for track in self.tracks)
        mixed_signal = np.zeros(max_samples, dtype=np.float32)
        for track in self.tracks:
            mixed_signal[: track.samples] += track.signal
        return mixed_signal
