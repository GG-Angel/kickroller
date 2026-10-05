from pydantic import BaseModel
from pydantic_settings import BaseSettings


class SynthSettings(BaseModel):
    drop_duration_seconds: float = 12.0
    beats_per_bar: int = 4
    sixteenth_notes_per_beat: int = 4
    bars_per_phrase: int = 4
    drop_peak_level_db: float = -1.0

    bpm_probability_weights: tuple[tuple[int, float], ...] = (
        (150, 0.1),
        (155, 0.15),
        (160, 0.50),
        (165, 0.20),
        (180, 0.05),
    )

    kickless_drop_probability_range: tuple[float, float] = (0.0, 0.15)
    kickless_beat_probability_range: tuple[float, float] = (0.05, 0.15)
    triplet_grid_probability: float = 0.3
    per_beat_kickroll_probability_range: tuple[float, float] = (0.2, 0.4)
    kickroll_hits_per_beat_weights: tuple[float, float, float] = (1.0, 4.0, 2.0)
    regular_kick_level_db_range: tuple[float, float] = (-1.0, 0.0)
    kickroll_level_db_range: tuple[float, float] = (-6.0, 0.0)
    kick_tail_fade_seconds: float = 0.003

    max_simultaneous_loops: int = 3
    loop_level_db_range: tuple[float, float] = (-18.0, 3.0)
    loop_high_pass_cutoff_hz_range: tuple[float, float] = (100.0, 300.0)
    loop_high_pass_filter_order: int = 2
    loop_duck_depth_range: tuple[float, float] = (0.4, 1.0)
    loop_duck_release_seconds_range: tuple[float, float] = (0.05, 0.25)

    clap_layer_probability: float = 0.6
    clap_level_db_range: tuple[float, float] = (-12.0, -2.0)

    mean_random_hits_per_drop: float = 8.0
    random_hit_level_db_range: tuple[float, float] = (-18.0, -3.0)

    impact_layer_probability: float = 0.25
    impact_level_db_range: tuple[float, float] = (-12.0, 0.0)

    one_shot_silence_trim_threshold_db: float = 40.0
    one_shot_trim_frame_samples: int = 64
    one_shot_trim_hop_samples: int = 16
    one_shot_attack_window_seconds: float = 0.1

    label_click_frequency_hz: float = 3000.0  # above the kick body, easy to hear


class Settings(BaseSettings):
    debug: bool = False

    sample_rate: int = 44100
    synth: SynthSettings = SynthSettings()


SETTINGS = Settings()
