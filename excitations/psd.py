"""
A library containing diverse power spectral density (PSD) functions
"""

import numpy as np
import abc


class PSD(abc.ABC):

    @abc.abstractmethod
    def __call__(self, f):
        """
        Return the power spectral density at frequency f
        """
        pass


class RollOffPSD(PSD):

    def __init__(self, a_v=4.032e-7, omega_r=0.0206, omega_c=0.8246):
        self.a_v = a_v
        self.omega_r = omega_r
        self.omega_c = omega_c

    def __call__(self, f):
        omega = 2 * np.pi * f
        return (
            self.a_v
            * self.omega_c**2
            / ((omega**2 + self.omega_r**2) * (omega**2 + self.omega_c**2))
        )


class FlatNoisePSD(PSD):

    def __init__(self, a_v=1.0):
        self.a_v = a_v

    def __call__(self, f):
        return self.a_v * np.ones_like(f)


class BandPassPSD(PSD):

    def __init__(self, a_v=1.0, f_1=0.1, f_2=1.0):
        self.a_v = a_v
        self.f_1 = f_1
        self.f_2 = f_2

    def __call__(self, f):
        return self.a_v * np.where(
            (f >= self.f_1) & (f <= self.f_2),
            np.ones_like(f),
            np.zeros_like(f),
        )
