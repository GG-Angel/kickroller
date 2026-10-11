import numpy as np

from src.models.audio import Signal
from src.services.synthesis.track import Track


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
