from pydantic_settings import BaseSettings
from pydantic import BaseModel


class BankConfig(BaseModel):
    pass


class Config(BaseSettings):
    sr: int = 44100


CFG = Config()
