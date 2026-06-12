from abc import ABC, abstractmethod


class BaseEnv(ABC):
    """
    Abstract base for all simulation environments.
    Every concrete task implements these three methods.
    """

    @abstractmethod
    def reset(self) -> dict:
        """
        Reset the environment to a fresh episode.
        Returns the initial observation dict.
        """
        ...

    @abstractmethod
    def step(self, action: dict) -> tuple[dict, float, bool]:
        """
        Apply one action.
        Returns (observation, reward, done).
        """
        ...

    @abstractmethod
    def get_obs(self) -> dict:
        """
        Return the current observation without stepping.
        Used for snapshot polling by the renderer.
        """
        ...

    @abstractmethod
    def close(self):
        """Disconnect and clean up the physics client."""
        ...
