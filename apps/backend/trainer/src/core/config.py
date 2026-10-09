from pydantic_settings import BaseSettings


class Config(BaseSettings):
    sr: int = 44100
    bpm: int = 160


CONFIG = Config()
