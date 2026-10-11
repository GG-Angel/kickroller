import numpy as np

from src.models.audio import Signal
from src.services.synthesis.track import Track


class Mix:
    def __init__(self) -> None:
        self.tracks: list[Track] = []

    def add_tracks(self, *track: Track) -> None:
        self.tracks.extend(track)

    def mix(self) -> Signal:
        if not self.tracks:
            return np.zeros(0, dtype=np.float32)

        signals = [track.process() for track in self.tracks]
        max_samples = max(len(signal) for signal in signals)
        mixed_signal = np.zeros(max_samples, dtype=np.float32)
        for signal in signals:
            mixed_signal[: len(signal)] += signal

        return mixed_signal
