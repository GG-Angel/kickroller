from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

from src.models.audio import SampleKind


class Placement(BaseModel):
    """Where a one-shot can play: every ``period`` beats from ``offset``."""

    period: float | None = None
    """Beats between the allowed positions, or None for one grid step."""
    offset: float = 0.0
    """Beat of the first allowed position."""
    probability_range: tuple[float, float] = (1.0, 1.0)
    """Range of the chance that each allowed position plays."""


class LayerConfig(BaseModel):
    """The tracks of one sample kind, either loops or one-shots."""

    kind: SampleKind
    is_loop: bool
    track_count_weights: dict[int, float]
    """Weight of each number of tracks in a drop."""
    level_db_range: tuple[float, float]
    """Range of the level relative to the kick, in dB of ``loudest_rms``."""
    high_pass_hz: float | None = 200.0
    """High-pass cutoff, or None for no high-pass."""
    is_sidechained: bool = False
    """Whether the tracks duck at each kick, with the sidechain of the drop."""
    placements: list[Placement] = []
    """Placements of a one-shot track, of which each track draws one."""

    @model_validator(mode="after")
    def _check_placements(self) -> "LayerConfig":
        if not self.is_loop and not self.placements:
            raise ValueError(
                f"The {self.kind} one-shot layer has no placements."
            )
        return self


PHRASE = Placement(period=16, probability_range=(0.5, 1.0))
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
        is_sidechained=True,
    ),
    LayerConfig(
        kind=SampleKind.MELODY,
        is_loop=False,
        track_count_weights={0: 0.85, 1: 0.15},
        level_db_range=(-10, -2),
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
        kind=SampleKind.IMPACT,
        is_loop=False,
        track_count_weights={0: 0.7, 1: 0.3},
        level_db_range=(-6, 0),
        high_pass_hz=None,
        placements=[PHRASE],
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
    """Sections of whole beats, which start on any beat and do not overlap."""

    section_count_weights: dict[int, float]
    """Weight of each number of sections in a drop."""
    section_beats: list[int] = [1, 2, 4]
    """Lengths of a section in beats."""


class KickFxConfig(SectionConfig):
    """Sections of the kick track that go through a resonant high-pass."""

    section_count_weights: dict[int, float] = {
        0: 0.15,
        1: 0.25,
        2: 0.25,
        3: 0.2,
        4: 0.15,
    }
    high_pass_hz_range: tuple[float, float] = (300.0, 1250.0)
    """Range of the high-pass cutoff, drawn log-uniform."""
    resonance_q_range: tuple[float, float] = (1.5, 10.0)
    """Range of the filter Q (0.7 is no resonance), drawn log-uniform."""
    filter_orders: list[int] = [2, 4, 6]
    """Filter orders to draw from: each order adds 6 dB/oct of slope."""


class MasterConfig(BaseModel):
    """Random master effects on each drop, which make the drops more varied.

    Each effect is used with its probability, in this order: tilt, clip,
    limit, MP3. The gain is always used, as the last effect.
    """

    tilt_probability: float = Field(default=0.5, ge=0, le=1)
    tilt_db_range: tuple[float, float] = (-6.0, 6.0)
    """Range of the gain of the highs relative to the lows."""
    tilt_pivot_hz: float = 1000.0

    clip_probability: float = Field(default=0.5, ge=0, le=1)
    clip_drive_db_range: tuple[float, float] = (0.0, 12.0)
    """Range of the gain into the soft clipper."""

    limit_probability: float = Field(default=0.5, ge=0, le=1)
    limit_drive_db_range: tuple[float, float] = (0.0, 9.0)
    """Range of the gain into the limiter."""
    limit_ceiling_db: float = -0.3

    mp3_probability: float = Field(default=0.4, ge=0, le=1)
    mp3_compression_range: tuple[float, float] = (0.0, 0.75)
    """Range of the MP3 compression level: 0 is about 320 kbps, 0.75 is about
    96 kbps."""

    gain_db_range: tuple[float, float] = (-6.0, 6.0)
    """Range of the output gain."""


class SynthConfig(BaseModel):
    layers: list[LayerConfig] = DEFAULT_LAYERS
    kick_fx: KickFxConfig = KickFxConfig()
    kick_gaps: SectionConfig = SectionConfig(
        section_count_weights={0: 0.4, 1: 0.35, 2: 0.25}
    )
    """Sections where the kick stops and the other layers continue."""
    sidechain_ratio_range: tuple[float, float] = (0.7, 0.95)
    """Range of the duck depth at each kick."""
    sidechain_seconds_range: tuple[float, float] = (0.06, 0.25)
    """Range of the duck length after each kick."""

    steps: list[int] = [3, 4, 6, 8]
    bpms: list[int] = [140, 150, 155, 160, 165, 170, 180]

    drop_seconds: float = 12.0

    peak_db: float = -1.0
    """Peak of each drop before the master effects, in dBFS."""

    master: MasterConfig | None = MasterConfig()
    """Master effects for augmentation, or None for no augmentation."""

    held_out_kick_fraction: float = Field(default=0.1, ge=0, le=1)
    """Part of the kicks that only validation drops use."""


class Config(BaseSettings):
    sr: int = 44100

    synth: SynthConfig = SynthConfig()


CONFIG = Config()
