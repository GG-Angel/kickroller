from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    debug: bool = False

    sample_rate: int = 44100


SETTINGS = Settings()
