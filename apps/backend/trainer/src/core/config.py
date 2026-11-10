from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

from src.models.audio import SampleKind


class Placement(BaseModel):
    """Where a one-shot can play."""

    period: float | None = None
    """Beats between positions. None uses one grid step."""
    offset: float = 0.0
    """Beat of the first position."""
    probability_range: tuple[float, float] = (1.0, 1.0)
    """Chance range for each position."""


class LayerConfig(BaseModel):
    """Settings for one sample layer."""

    kind: SampleKind
    is_loop: bool
    track_count_weights: dict[int, float]
    """Weight for each track count."""
    level_db_range: tuple[float, float]
    """Level below the kick's loudest-window RMS, in dB."""
    high_pass_hz_range: tuple[float, float] | None = (200.0, 200.0)
    """Cutoff range for each track. None disables the filter."""
    is_sidechained: bool = False
    """Whether the layer ducks at each kick."""
    placements: list[Placement] = []
    """Positions for one-shot tracks."""

    @model_validator(mode="after")
    def _check_placements(self) -> "LayerConfig":
        if not self.is_loop and not self.placements:
            raise ValueError(
                f"The {self.kind} one-shot layer has no placements."
            )
        return self


BAR = Placement(period=4, probability_range=(0.2, 0.5))
BACKBEAT = Placement(period=2, offset=1, probability_range=(0.6, 1.0))
BEAT = Placement(period=1, probability_range=(0.2, 0.5))
OFFBEAT = Placement(period=1, offset=0.5, probability_range=(0.3, 0.8))
STEP = Placement(probability_range=(0.02, 0.1))

DEFAULT_LAYERS = [
    LayerConfig(
        kind=SampleKind.MELODY,
        is_loop=True,
        track_count_weights={0: 0.2, 1: 0.8},
        level_db_range=(-8, 0),
        high_pass_hz_range=(60.0, 200.0),
        is_sidechained=True,
    ),
    LayerConfig(
        kind=SampleKind.MELODY,
        is_loop=False,
        track_count_weights={0: 0.85, 1: 0.15},
        level_db_range=(-10, -2),
        high_pass_hz_range=(60.0, 200.0),
        is_sidechained=True,
        placements=[OFFBEAT, STEP],
    ),
    LayerConfig(
        kind=SampleKind.VOCAL,
        is_loop=True,
        track_count_weights={0: 0.85, 1: 0.15},
        level_db_range=(-12, -4),
        is_sidechained=True,
    ),
    LayerConfig(
        kind=SampleKind.VOCAL,
        is_loop=False,
        track_count_weights={0: 0.85, 1: 0.15},
        level_db_range=(-12, -4),
        is_sidechained=True,
        placements=[BAR, BEAT],
    ),
    LayerConfig(
        kind=SampleKind.DRUM,
        is_loop=True,
        track_count_weights={0: 0.5, 1: 0.5},
        level_db_range=(-16, -8),
        is_sidechained=True,
    ),
    LayerConfig(
        kind=SampleKind.DRUM,
        is_loop=False,
        track_count_weights={0: 0.4, 1: 0.5, 2: 0.1},
        level_db_range=(-12, -3),
        placements=[BACKBEAT, BEAT, OFFBEAT, STEP],
    ),
    LayerConfig(
        kind=SampleKind.FX,
        is_loop=False,
        track_count_weights={0: 0.8, 1: 0.2},
        level_db_range=(-18, -8),
        is_sidechained=True,
        placements=[BAR, STEP],
    ),
]


class SectionConfig(BaseModel):
    """Settings for non-overlapping sections of whole beats."""

    section_count_weights: dict[int, float]
    """Weight for each section count."""
    section_beats: list[int] = [1, 2, 4]
    """Section lengths in beats."""


class KickFxConfig(SectionConfig):
    """Sections that use a resonant high-pass filter."""

    section_count_weights: dict[int, float] = {
        0: 0.15,
        1: 0.25,
        2: 0.25,
        3: 0.2,
        4: 0.15,
    }
    high_pass_hz_range: tuple[float, float] = (300.0, 1250.0)
    """Cutoff range, drawn on a log scale."""
    resonance_q_range: tuple[float, float] = (1.5, 10.0)
    """Q range. 0.7 gives no resonance. Drawn on a log scale."""
    filter_orders: list[int] = [2, 4, 6]
    """Filter orders. Each adds 6 dB/octave."""


class KickRollConfig(SectionConfig):
    """Sections where kicks play on every step."""

    section_count_weights: dict[int, float] = {0: 0.5, 1: 0.35, 2: 0.15}
    section_beats: list[int] = [1, 2]
    steps_per_beat: list[int] = [2, 3, 4]
    """Steps per beat: 2 gives 1/8, 3 gives triplets, and 4 gives 1/16 notes."""
    level_db_range: tuple[float, float] = (-3.0, 0.0)
    """Roll level relative to other kicks, in dB."""


class MasterConfig(BaseModel):
    """Random master effects: tilt, clip, limit, then MP3. Gain is always last."""

    tilt_probability: float = Field(default=0.5, ge=0, le=1)
    tilt_db_range: tuple[float, float] = (-6.0, 6.0)
    """Highs level relative to lows, in dB."""
    tilt_pivot_hz: float = 1000.0

    clip_probability: float = Field(default=0.5, ge=0, le=1)
    clip_drive_db_range: tuple[float, float] = (0.0, 12.0)
    """Clipper drive range, in dB."""

    limit_probability: float = Field(default=0.5, ge=0, le=1)
    limit_drive_db_range: tuple[float, float] = (0.0, 9.0)
    """Limiter drive range, in dB."""
    limit_ceiling_db: float = -0.3

    mp3_probability: float = Field(default=0.4, ge=0, le=1)
    mp3_compression_range: tuple[float, float] = (0.0, 0.75)
    """MP3 compression range: 0 is about 320 kbps; 0.75 is about 96 kbps."""

    gain_db_range: tuple[float, float] = (-6.0, 6.0)
    """Output gain range, in dB."""


class SynthConfig(BaseModel):
    layers: list[LayerConfig] = DEFAULT_LAYERS

    kick_fx: KickFxConfig = KickFxConfig()
    kick_gaps: SectionConfig = SectionConfig(
        section_count_weights={0: 0.4, 1: 0.35, 2: 0.25}
    )
    """Where kicks stop while other layers continue."""
    kick_rolls: KickRollConfig = KickRollConfig()

    kick_off_beat_range: tuple[float, float] = (0.0, 0.3)
    """Extra 1/8 off-beat chance per beat, drawn once per drop."""
    sidechain_probability: float = Field(default=0.75, ge=0, le=1)
    """Chance that layers duck at each kick. Do not rely on ducking to find kicks."""
    sidechain_ratio_range: tuple[float, float] = (0.3, 0.95)
    """Duck depth range at each kick."""
    sidechain_seconds_range: tuple[float, float] = (0.06, 0.25)
    """Duck length range after each kick, including attack and release."""

    steps: list[int] = [3, 4, 6, 8]
    """Allowed step values for the synthesizer."""
    bpms: list[int] = [140, 150, 155, 160, 165, 170, 180]
    """Allowed BPM values for the synthesizer."""

    drop_seconds: float = 12.0
    """Drop length in seconds."""
    peak_db: float = -1.0
    """Peak before master effects, in dBFS."""

    master: MasterConfig | None = MasterConfig()
    """Master effects for augmentation. None disables them."""

    held_out_kick_fraction: float = Field(default=0.2, ge=0, le=1)
    """Fraction of kicks used only for validation."""


class FeatureConfig(BaseModel):
    """Log spectrogram input for the model."""

    hop: int = 441
    """Samples between frames."""
    windows: tuple[int, ...] = (1024, 2048, 4096)
    """FFT window lengths in samples. Each is one model input channel."""
    fmin: float = 30.0
    """Center frequency of the lowest band, in Hz."""
    fmax: float = 16000.0
    """Highest band center, in Hz."""
    bands_per_octave: int = 12
    """Log-spaced bands per octave."""
    neighbor_target: float = 0.5
    """Target value for frames next to an onset."""


class TrainConfig(BaseModel):
    """Training and validation settings."""

    steps: int = 3000
    """Optimizer steps per run."""
    batch_size: int = 16
    """Examples per batch."""
    learning_rate: float = 1e-3
    """AdamW learning rate."""
    positive_weight: float = 5.0
    """Positive target weight. Kicks are rare frames."""
    validation_drops: int = 64
    """Drops with held-out kicks in validation."""
    validation_seed: int = 0
    """Seed for the fixed validation set."""
    stats_drops: int = 32
    """Training drops used for standardization."""
    validate_every: int = 250
    """Steps between validations. Save a checkpoint when F-measure improves."""
    log_every: int = 50
    """Steps between training-loss log lines."""
    workers: int = 4
    """DataLoader processes. 0 uses the main process."""
    thresholds: list[float] = [round(0.05 * i, 2) for i in range(2, 19)]
    """Peak thresholds to test. Save the best one."""
    tolerance: float = 0.02
    """Maximum onset error counted as a hit, in seconds."""
    min_onset_distance: float = 0.04
    """Minimum time between detected onsets, in seconds."""


class Config(BaseSettings):
    sr: int = 44100

    synth: SynthConfig = SynthConfig()
    features: FeatureConfig = FeatureConfig()
    train: TrainConfig = TrainConfig()


CONFIG = Config()
