from abc import ABC, abstractmethod
from typing import Any


class Serializable(ABC):
    """Abstract base class for objects that can be serialized to/from JSON dictionaries."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize the object to a dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "Serializable":
        """Reconstruct the object from a dictionary."""
        pass
