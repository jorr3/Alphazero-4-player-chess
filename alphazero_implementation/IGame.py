from abc import ABC, abstractmethod
import torch


class IGame(ABC):
    """
    Interface for game implementations.
    Defines the essential methods that any game should implement.
    """
    @abstractmethod
    def __repr__(self):
        pass

    @classmethod
    def take_action(cls, state, action, player):
        """Takes action and returns the game state post-action."""
        pass

    @classmethod
    def get_legal_actions_mask(cls, state: list, device: str) -> torch.Tensor:
        """Returns a binary vector indicating valid moves for the current state."""
        pass

    @classmethod
    def get_terminated(cls, state) -> tuple[bool, float]:
        """Returns the value of the current state and a flag indicating if the game is over."""
        pass

    @classmethod
    def get_opponent(cls, player: int) -> int:
        """Returns the opponent of the given player."""
        pass

    @staticmethod
    def get_opponent_value(value: float) -> float:
        """Returns the value from the perspective of the opponent."""
        pass

    @classmethod
    def get_encoded_states(cls, states: list, device: str) -> torch.Tensor:
        """Returns an encoded representation of the game state for neural network inputs."""
        pass
