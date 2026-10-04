from pydantic import BaseModel
from pydantic_settings import BaseSettings


class SynthSettings(BaseModel):
    drop_seconds: float = 12.0
    beats_per_bar: int = 4
    sixteenths_per_beat: int = 4
    phrase_bars: int = 4
    peak_db: float = -1.0

    target_bpm: int = 160
    target_bpm_chance: float = 0.5
    bpm_range: tuple[int, int] = (150, 170)

    kickless_bar_chance: float = 0.05
    missing_rate: tuple[float, float] = (0.0, 0.15)
    off_beat_rate: tuple[float, float] = (0.0, 0.3)
    syncopated_chance: float = 0.03
    syncopated_positions: tuple[float, ...] = (1 / 4, 1 / 3, 2 / 3, 3 / 4)
    roll_chance: float = 0.2
    roll_beats: tuple[int, ...] = (1, 2, 4)
    roll_steps: tuple[float, ...] = (1 / 4, 1 / 3, 1 / 2)
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
