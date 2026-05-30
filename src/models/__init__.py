"""Model interfaces and placeholders."""

from src.models.interfaces import Model, ModelTrainer, ValidationScheme
from src.models.placeholders import NoOpModel, NoOpModelTrainer

__all__ = ["Model", "ModelTrainer", "NoOpModel", "NoOpModelTrainer", "ValidationScheme"]
