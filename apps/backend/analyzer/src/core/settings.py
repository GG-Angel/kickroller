from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    debug: bool = False


settings = Settings()
