from pydantic import BaseSettings
from torch import cuda
from typing import Optional


class Settings(BaseSettings):
    AUTH_TOKEN: Optional[str] = None
    # The compute device available.
    DEVICE: str = "cuda" if cuda.is_available() else "cpu"
    USE_GPU: bool = cuda.is_available()
    MODEL: str = "EleutherAI/gpt-neo-2.7B"
    MODEL_TYPE: str = "text-generation"
    MODEL_CACHE_DIR: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        use_enum_values = True


settings = Settings()
