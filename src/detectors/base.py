"""Common interface for all flicker detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """A detector that evaluates one shared, precomputed feature object."""

    @abstractmethod
    def detect(self, features: object) -> object:
        """Return a detector-specific result."""
        raise NotImplementedError
