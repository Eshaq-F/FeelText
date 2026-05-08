from abc import ABC, abstractmethod


class BaseAnalyzer(ABC):
    """
    Contract every analyzer implementation must satisfy.
    The web layer depends only on this interface, never on concrete types.
    """

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @property
    @abstractmethod
    def supported_languages(self) -> list[str]: ...

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def analyze(self, text: str) -> dict: ...

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        return [self.analyze(t) for t in texts]
