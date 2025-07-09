# coding: utf-8
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class BasicConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='bts_server_')

class ServerConfig(BasicConfig):
    host: str = '0.0.0.0'
    port: int = 8502

class ModelConfig(BasicConfig):
    cfg_path: str = 'baselines/ChronosBolt/config/chronos_base.py'
    ckpt_path: str = 'utsf_ckpt/ChronosBolt-base-BLAST.pt'
    device_type: str = 'gpu'
    gpus: Optional[str] = '0'
    context_length: int = 72
    prediction_length: int = 24
    