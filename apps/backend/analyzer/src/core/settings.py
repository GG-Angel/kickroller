from pydantic import BaseModel
from pydantic_settings import BaseSettings


class SynthSettings(BaseModel):
    drop_duration_seconds: float = 12.0
    beats_per_bar: int = 4
    sixteenth_notes_per_beat: int = 4
    bars_per_phrase: int = 4
    drop_peak_level_db: float = -1.0
    level_window_seconds: float = 0.1  # a sample level is the RMS of its loudest window

    # bpm_probability_weights: tuple[tuple[int, float], ...] = (
    #     (150, 0.1),
    #     (155, 0.15),
    #     (160, 0.50),
    #     (165, 0.20),
    #     (180, 0.05),
    # )

    bpm_probability_weights: tuple[tuple[int, float], ...] = ((165, 1.00),)

    kickless_drop_probability: float = 0.075
    kickless_bar_probability: float = 0.05
    kickless_beat_probability_range: tuple[float, float] = (0.05, 0.1)
    off_beat_kick_probability_range: tuple[float, float] = (0.0, 0.3)  # 1/8 off-beat
    syncopated_kick_probability: float = 0.03  # per beat with no off-beat kick
    syncopated_kick_offsets_beats: tuple[float, ...] = (0.25, 1 / 3, 2 / 3, 0.75)
    kickroll_probability_per_bar: float = 0.15
    phrase_end_kickroll_probability: float = 0.4  # in the last bar of a phrase
    kickroll_length_beats_weights: tuple[tuple[int, float], ...] = (
        (1, 0.5),
        (2, 0.25),
        (4, 0.25),
    )  # a kickroll fills the last beats of its bar
    kickroll_kicks_per_beat_weights: tuple[tuple[int, float], ...] = (
        (4, 0.45),  # 1/16 notes
        (3, 0.30),  # 1/8 triplets
        (2, 0.25),  # 1/8 notes
    )
    late_kickroll_probability: float = 0.2  # the first kick is one step after the beat
    regular_kick_level_db_range: tuple[float, float] = (-1.0, 0.0)
    kickroll_level_db_range: tuple[float, float] = (-6.0, 0.0)
    kick_tail_fade_seconds: float = 0.003
    held_out_kick_fraction: float = 0.1  # only for validation drops

    loop_count_weights: tuple[tuple[int, float], ...] = (
        (0, 0.10),
        (1, 0.75),
        (2, 0.15),
    )  # the number of loops played at the same time
    loop_level_db_range: tuple[float, float] = (-12.0, 0.0)
    loop_high_pass_cutoff_hz_range: tuple[float, float] = (100.0, 300.0)
    loop_high_pass_filter_order: int = 2
    loop_duck_depth_range: tuple[float, float] = (0.4, 1.0)
    loop_duck_release_seconds_range: tuple[float, float] = (0.05, 0.25)

    clap_layer_probability: float = 0.6
    clap_level_db_range: tuple[float, float] = (-12.0, -2.0)

    mean_random_hits_per_drop: float = 8.0
    random_hit_level_db_range: tuple[float, float] = (-15.0, -6.0)

    impact_layer_probability: float = 0.25
    impact_level_db_range: tuple[float, float] = (-12.0, 0.0)

    one_shot_silence_trim_threshold_db: float = 40.0
    one_shot_trim_frame_samples: int = 64
    one_shot_trim_hop_samples: int = 16

    master_eq_tilt_probability: float = 0.5
    master_eq_tilt_corner_hz_range: tuple[float, float] = (300.0, 3000.0)
    master_eq_tilt_gain_db_range: tuple[float, float] = (-4.0, 4.0)  # of each band
    master_eq_tilt_filter_order: int = 1
    master_reference_loudness_lufs: float = -14.0  # the loudness before the drive
    master_drive_db_range: tuple[float, float] = (0.0, 12.0)
    master_soft_clipper_probability: float = 0.7
    master_limiter_probability: float = 0.6
    master_limiter_window_seconds: float = 0.005
    master_limiter_ceiling: float = 0.95  # peak level

    label_click_frequency_hz: float = 3000.0  # above the kick body, easy to hear


class Settings(BaseSettings):
    debug: bool = False

    sample_rate: int = 44100
    synth: SynthSettings = SynthSettings()


SETTINGS = Settings()
