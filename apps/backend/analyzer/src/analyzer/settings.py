"""All settings that can change, with their defaults.

ANALYZER_* environment variables, or a .env file in the working folder, change
them. Use "__" between nested names, and JSON for lists and tables:

    ANALYZER_DETECTOR__MIN_CONFIDENCE=0.4
    ANALYZER_TRAINING__STEPS=20000
    ANALYZER_SYNTH__KICKS__ROLL_CHANCE=0.2
    ANALYZER_SYNTH__TEMPOS='[[0.6, 160, 160], [0.4, 150, 170]]'

A name that ends in `_chance` is a probability (0-1). A pair such as `level_db`
is a range: each drop draws a value between the two. A table such as
`roll_beats` gives the probability of each value. Times are in seconds and
levels in dB unless the name says otherwise. CLI options change a setting for
one run.
"""

from typing import Annotated, Literal, NamedTuple

from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from analyzer.audio.io import SAMPLE_RATE

Chance = Annotated[float, Field(ge=0.0, le=1.0)]
Seconds = Annotated[float, Field(gt=0.0)]
Count = Annotated[int, Field(ge=1)]


def check_range(value: tuple[float, float]) -> tuple[float, float]:
    if value[0] > value[1]:
        raise ValueError(f"the first value of a range must be the lower one: {value}")
    return value


def check_distribution(value: dict[int, float]) -> dict[int, float]:
    if min(value.values(), default=-1.0) < 0.0 or abs(sum(value.values()) - 1.0) > 1e-6:
        raise ValueError(
            f"the probabilities must be 0 or more and add up to 1: {value}"
        )
    return value


Range = Annotated[tuple[float, float], AfterValidator(check_range)]
Distribution = Annotated[dict[int, float], AfterValidator(check_distribution)]


class Section(BaseModel):
    """A group of settings. An unknown name is an error, to catch typing errors."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DetectorSettings(Section):
    """Kicks: the peaks of the kick probability. Beats: beat_this, made regular."""

    peak_window: Seconds = 0.02  # a kick is the maximum within +/- this time
    # Remove kicks closer than this to a more probable kick, in beats (1/10 of a
    # beat is 80% of a 1/32 note), but never more than `min_distance` seconds.
    min_distance_beats: float = Field(0.1, gt=0.0)
    min_distance: Seconds = 0.04
    min_confidence: Chance = 0.5
    chunk_seconds: Seconds = 60.0  # the model reads long tracks in chunks
    beat_checkpoint: str = "final0"  # the beat_this model
    # The beat grid tempo is moved by octaves into this range.
    min_bpm: float = Field(150.0, gt=0.0)
    max_bpm: float = Field(170.0, gt=0.0)
    min_beat_interval: float = Field(0.75, gt=0.0)  # in beat periods; closer is extra
    beat_window: Seconds = 0.04  # a beat moves to a confident kick within +/- this
    anchor_confidence: Chance = 0.5  # the confidence of a kick that a beat moves to


class ModelSettings(Section):
    """The kick model input and layers. Each trained model keeps its own copy."""

    hop: Count = 441  # samples from one frame to the next (10 ms)
    windows: tuple[Count, ...] = (1024, 2048, 4096)  # STFT sizes: 23, 46 and 93 ms
    bands: Count = 80  # mel bands
    fmin: float = Field(27.5, gt=0.0)  # in Hz
    fmax: float = Field(16000.0, gt=0.0)  # in Hz
    log_compression: float = Field(1000.0, gt=0.0)  # log(1 + this * magnitude)
    # Channels of the 3x3 convolution layers; each halves the bands.
    frontend_channels: tuple[Count, ...] = (16, 32, 32)
    channels: Count = 64  # of the dilated layers
    dilations: tuple[Count, ...] = (1, 2, 4, 8, 16, 8)
    dropout: Chance = 0.1

    @property
    def fps(self) -> float:
        return SAMPLE_RATE / self.hop

    @property
    def context_frames(self) -> int:
        """Frames of context on each side of a frame.

        Each 3x3 convolution adds 1 frame and each dilated layer its dilation.
        """
        return len(self.frontend_channels) + sum(self.dilations)


class TrainingSettings(Section):
    steps: Count = 15000
    batch_size: Count = 16  # drops per step
    workers: int = Field(10, ge=0)  # processes that make drops
    device: Literal["mps", "cuda", "cpu"] = "mps"
    seed: int = 0
    learning_rate: float = Field(2e-3, gt=0.0)  # the peak of the one-cycle schedule
    weight_decay: float = Field(1e-4, ge=0.0)
    neighbor_target: Chance = 0.5  # the target of the frames next to an onset frame
    validate_every: Count = 500  # steps
    log_every: Count = 100  # steps
    validation_drops: Count = 200
    stats_drops: Count = 64  # drops that set the input standardization
    tolerance: Seconds = 0.02  # a found onset matches a true onset within this
    thresholds: tuple[Chance, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)  # validation
    prefetch: Count = 4  # batches that each worker makes in advance


class TempoBand(NamedTuple):
    chance: float
    low_bpm: int
    high_bpm: int


def check_tempos(value: tuple[TempoBand, ...]) -> tuple[TempoBand, ...]:
    check_distribution(dict(enumerate(band.chance for band in value)))
    for band in value:
        check_range((band.low_bpm, band.high_bpm))
    return value


class KickSettings(Section):
    kickless_drop_chance: Chance = 0.05
    kickless_bar_chance: Chance = 0.05
    off_beat_rate: Range = (0.0, 0.3)  # the chance of a 1/8 off-beat kick per beat
    missing_rate: Range = (0.0, 0.15)  # the chance of no kick on a beat
    syncopated_chance: Chance = 0.03  # a kick on one of `syncopated_ticks`
    syncopated_ticks: tuple[int, ...] = (3, 4, 8, 9)  # 1/16 and triplet positions
    roll_chance: Chance = 0.15  # per bar
    phrase_end_roll_chance: Chance = 0.4  # in the last bar of 4
    roll_beats: Distribution = {1: 0.5, 2: 0.25, 4: 0.25}  # at the end of the bar
    # Ticks (1/12 beat) between roll kicks: 1/16 notes, triplets and 1/8 notes.
    roll_steps: Distribution = {3: 0.45, 4: 0.3, 6: 0.25}
    late_roll_chance: Chance = 0.2  # the roll starts one step after the beat
    key_change_bars: Count = 2
    key_change_chance: Chance = 0.5  # every `key_change_bars`
    level_db: Range = (-1.0, 0.0)
    roll_level_db: Range = (-6.0, 0.0)
    cut_fade: Seconds = 0.003  # fade-out when the next kick cuts the tail


class LoopSettings(Section):
    layers: Distribution = {0: 0.1, 1: 0.35, 2: 0.35, 3: 0.2}
    off_bar_chance: Chance = 0.2  # the loop starts at any time, not on a bar
    high_pass_chance: Chance = 0.85  # the kick owns the sub band in a drop
    high_pass_hz: Range = (100.0, 250.0)
    level_db: Range = (-18.0, 3.0)  # relative to the kick, when high-passed
    full_band_level_db: Range = (-18.0, -6.0)  # when not high-passed
    duck_chance: Chance = 0.85  # sidechain ducking at each kick
    duck_depth: Range = (0.4, 1.0)  # 1 is silence at the kick
    duck_release: Range = (0.05, 0.25)  # time constant of the recovery


class ClapSettings(Section):
    chance: Chance = 0.6  # a drop has claps on beats 2 and 4
    beat_chance: Chance = 0.9  # each of these beats has a clap
    level_db: Range = (-12.0, -2.0)  # relative to the kick


class HitSettings(Section):
    mean_count: float = Field(8.0, ge=0.0)  # per drop (Poisson)
    off_grid_chance: Chance = 0.2  # moved off the 1/16 grid
    off_grid_shift: Range = (-0.05, 0.05)
    level_db: Range = (-18.0, -3.0)  # relative to the kick


class ImpactSettings(Section):
    chance: Chance = 0.3
    bars: tuple[int, ...] = (0, 4)  # an impact starts on bar 1 or bar 5
    level_db: Range = (-12.0, 0.0)  # relative to the kick


class MasterSettings(Section):
    eq_tilt_chance: Chance = 0.5
    eq_corner_hz: Range = (300.0, 3000.0)  # the low and high bands meet here
    eq_gain_db: Range = (-4.0, 4.0)  # of each band
    drive_db: Range = (0.0, 12.0)  # into the clipper, above the target loudness
    clipper_chance: Chance = 0.7  # tanh soft clipper
    limiter_chance: Chance = 0.6
    limiter_window: Seconds = 0.005
    limiter_ceiling: float = Field(0.95, gt=0.0, le=1.0)  # peak level
    output_gain_db: Range = (-6.0, 6.0)  # after loudness normalization


class SynthSettings(Section):
    """The synthetic training drops."""

    # The drop tempo: a whole BPM from a band, drawn with the band's chance.
    tempos: Annotated[tuple[TempoBand, ...], AfterValidator(check_tempos)] = (
        TempoBand(0.5, 160, 160),
        TempoBand(0.25, 150, 159),
        TempoBand(0.15, 161, 170),
        TempoBand(0.1, 171, 200),
    )
    drop_seconds: Seconds = 12.0  # 8 bars at 160 BPM (7.5 to 10 bars at 150-200 BPM)
    held_out_fraction: Chance = 0.1  # of the kick designs, only for validation
    kicks: KickSettings = KickSettings()
    loops: LoopSettings = LoopSettings()
    claps: ClapSettings = ClapSettings()
    hits: HitSettings = HitSettings()
    impacts: ImpactSettings = ImpactSettings()
    master: MasterSettings = MasterSettings()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ANALYZER_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    detector: DetectorSettings = DetectorSettings()
    model: ModelSettings = ModelSettings()
    training: TrainingSettings = TrainingSettings()
    synth: SynthSettings = SynthSettings()
