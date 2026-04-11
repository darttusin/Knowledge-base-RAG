from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    id: int
    text: str
    source: str
    score: float
