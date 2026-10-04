from pydantic import BaseModel
from pydantic_settings import BaseSettings


class SynthSettings(BaseModel):
    drop_seconds: float = 12.0
    beats_per_bar: int = 4
    sixteenths_per_beat: int = 4
    phrase_bars: int = 4
    peak_db: float = -1.0

    bpm_weights: tuple[tuple[int, float], ...] = (
        (140, 0.02),
        (150, 0.20),
        (155, 0.15),
        (160, 0.40),
        (165, 0.10),
        (170, 0.05),
        (180, 0.04),
        (200, 0.04),
    )

    missing_rate: tuple[float, float] = (0.0, 0.15)
    kickless_beat_rate: tuple[float, float] = (0.05, 0.12)
    triplet_chance: float = 0.5
    kickroll_chance: tuple[float, float] = (0.10, 0.30)
    kickroll_resolution_weights: tuple[float, float, float] = (4.0, 2.0, 1.0)
    kick_level_db: tuple[float, float] = (-1.0, 0.0)
    roll_level_db: tuple[float, float] = (-6.0, 0.0)
    cut_fade_seconds: float = 0.003

    max_loops: int = 3
    loop_level_db: tuple[float, float] = (-18.0, 3.0)
    high_pass_hz: tuple[float, float] = (100.0, 250.0)
    high_pass_order: int = 2
    duck_depth: tuple[float, float] = (0.4, 1.0)
    duck_release_seconds: tuple[float, float] = (0.05, 0.25)
    clap_chance: float = 0.6
    clap_level_db: tuple[float, float] = (-12.0, -2.0)
    mean_hits: float = 8.0
    hit_level_db: tuple[float, float] = (-18.0, -3.0)
    impact_chance: float = 0.3
    impact_level_db: tuple[float, float] = (-12.0, 0.0)

    trim_top_db: float = 40.0
    trim_frame_length: int = 64
    trim_hop_length: int = 16
    attack_seconds: float = 0.1


class Settings(BaseSettings):
    debug: bool = False

    sample_rate: int = 44100
    synth: SynthSettings = SynthSettings()


SETTINGS = Settings()
