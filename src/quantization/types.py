from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class TileConfig:
    tile_m: int = 64
    tile_n: int = 64


@dataclass(frozen=True)
class ApplyConfig:
    model_type: str
    dtype_out: str = "fp16"
    tile: TileConfig = field(default_factory=TileConfig)
    module_filters: List[str] = field(default_factory=list)
    quant_style: str = "gi"
    iaa_channel_stats_path: Optional[str] = None
