import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from .base import BaseSimulator


class Simulator(BaseSimulator):
    def __init__(self, x_min, x_max, radiation):
        self.x_min = x_min
        self.x_max = x_max
        self._get_calculator(radiation)
        return

    def _get_calculator(self, radiation):
        if type(radiation) is float:
            self.calculator = XRDCalculator(radiation)
        elif radiation == "CuKa_double":
            self.calculator = [XRDCalculator("CuKa1"), XRDCalculator("CuKa2")]
        else:  # assume radiation spec in pymatgen dict
            self.calculator = XRDCalculator(radiation)
        return

    def get_simulation(self, structure):
        if type(self.calculator) is list:
            pat1 = self.calculator[0].get_pattern(
                structure, two_theta_range=(self.x_min, self.x_max)
            )
            pat2 = self.calculator[1].get_pattern(
                structure, two_theta_range=(self.x_min, self.x_max)
            )
            positions = np.concatenate((pat1.x, pat2.x), axis=0)
            intensities1 = pat1.y
            intensities2 = pat2.y
            intensities1 = np.array([
                f / max(intensities1) for f in intensities1
            ])
            intensities2 = np.array([
                f / max(intensities2) * 0.5 for f in intensities2
            ])
            intensities = np.concatenate((intensities1, intensities2), axis=0)
            return positions, intensities
        # else
        pat = self.calculator.get_pattern(
            structure, two_theta_range=(self.x_min, self.x_max)
        )
        positions = pat.x
        intensities = pat.y
        intensities = [
            f / max(intensities) for f in intensities
        ]  # correct intensities (set max to 1)
        return positions, intensities
