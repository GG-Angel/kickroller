import librosa
import numpy as np

from src.core.config import CONFIG
from src.models.audio import Signal
from src.services.synthesis.effects import Effect
from src.services.synthesis.track import Track


class Mix:
    def __init__(self, sr: int = CONFIG.sr) -> None:
        self.sr = sr
        self.tracks: list[Track] = []
        self.effects: list[Effect] = []
        """Master effects, applied in order to the sum of the tracks."""

    def add_tracks(self, *track: Track) -> None:
        self.tracks.extend(track)

    def add_effects(self, *effects: Effect) -> None:
        self.effects.extend(effects)

    def mix(self) -> Signal:
        if not self.tracks:
            return np.zeros(0, dtype=np.float32)

        signals = [track.process() for track in self.tracks]
        size = max(len(signal) for signal in signals)
        mixed = np.sum(
            [librosa.util.fix_length(signal, size=size) for signal in signals],
            axis=0,
            dtype=np.float32,
        )
        for effect in self.effects:
            mixed = effect.process(mixed, self.sr)
        return mixed
