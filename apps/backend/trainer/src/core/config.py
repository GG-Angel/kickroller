from pydantic import BaseModel
from pydantic_settings import BaseSettings


class SynthConfig(BaseModel):
    bpms: list[int] = [140, 150, 155, 160, 165, 170, 180]
    bpm: int = 160

    steps: list[int] = [3, 4, 6, 8]

    kick_db_range: tuple[int, int] = (-1, 0)

    drop_seconds: float = 12.0


class Config(BaseSettings):
    sr: int = 44100

    synth: SynthConfig = SynthConfig()


CONFIG = Config()
