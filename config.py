from dataclasses import dataclass

@dataclass
class FoundationConfig:
    MIXED_PRECISION : bool = True

@dataclass
class FloodConfig:
    MIXED_PRECISION : bool = False
