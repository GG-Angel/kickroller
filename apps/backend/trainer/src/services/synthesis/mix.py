import librosa
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
        size = max(len(signal) for signal in signals)
        return np.sum(
            [librosa.util.fix_length(signal, size=size) for signal in signals],
            axis=0,
            dtype=np.float32,
        )
