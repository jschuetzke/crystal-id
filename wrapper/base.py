from abc import ABC, abstractmethod


class BaseSimulator(ABC):
    @abstractmethod
    def __init__(self, x_min, x_max):
        """
        Abstract method that should be implemented by each specific simulator.
        """
        pass

    @abstractmethod
    def get_simulation(self):
        """
        Abstract method that should be implemented by each specific simulator.
        """
        # should return positions, intensities (scaled to max 1)
        pass
        
