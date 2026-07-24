from dataclasses import dataclass

@dataclass
class TrainingParams:
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 1