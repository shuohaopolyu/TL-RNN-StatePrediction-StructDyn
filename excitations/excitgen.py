'''
Random excitations for the dynamical system
'''
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import abc

class ExcitationGenerator(abc.ABC):

    @abc.abstractmethod
    def __call__(self):
        """
        Return the callable random excitation function
        """
        pass

    @abc.abstractmethod
    def generate(self):
        """
        Generate the time and random excitation signal in numpy arraies of dimension (n_t, )
        """
        pass

class PSDExcitationGenerator:
    """
    Generate a random excitation signal based on the given power spectral density
    """
    def __init__(self, psd, tmax, fmax):
        """
        :param psd: power spectral density object with the following methods:
            - psd(f): returns the power spectral density at frequency f
            Note: the frequency f is in Hz, and the psd function is one-sided (unilateral),
            i.e. psd is defined for f >= 0, and 
            is two times the psd of the positive frequencies for discrete Fourier transform (DFT) results.

        :param tmax: maximum time for the excitation signal
        :param fmax: truncation frequency  
        """
        self.psd = psd
        self.tmax = tmax
        self.fmax = fmax
        self.fs = 2*self.fmax
        self.fmin = 1/self.tmax
        self.num_samples = int(self.fs*self.tmax)
        self.t = np.linspace(0, self.tmax, self.num_samples)
        self.f = np.fft.rfftfreq(self.num_samples, 1/self.fs)
        

    def __call__(self):
        """
        Return the callable random excitation function with linear interpolation
        """
        return interp1d(self.t, self.generate()[1], kind='linear', bounds_error=True)
        

    def generate(self):
        """
        Generate the time and random excitation signal in numpy arraies of dimension (n_t, )
        """
        return self.t, np.fft.irfft(self._calculate_fourier_components(), n=self.num_samples, axis=0, norm='forward')

    def _calculate_fourier_components(self):
        """
        Calculate the Fourier components of the excitation signal
        """
        phase = np.random.uniform(0, 2*np.pi, self.num_samples//2+1)
        amplitude = np.sqrt(self.psd(self.f)*self.fmin/2)
        amplitude[0] = 0
        return amplitude*np.exp(1j*phase)
    
    def plot_psd(self, format='normal'):
        """
        Plot the power spectral density
        :param format: 'normal' or 'logx' or 'logy' or 'logxlogy'
        """
        plt.figure(figsize=(4, 4))
        plt.plot(self.f, self.psd(self.f), color='b', marker='o', markerfacecolor='r')
        if 'logx' in format:
            plt.xscale('log')
        if 'logy' in format:
            plt.yscale('log')
        plt.xlabel('frequency (Hz)')
        plt.ylabel('PSD')
        plt.show()

    def plot(self):
        """
        Plot the excitation signal
        """
        plt.figure(figsize=(12, 4))
        plt.plot(self.t, self.generate()[1], color='b')
        plt.xlabel('time (s)')
        plt.ylabel('excitation signal')
        plt.show()

class PeriodicExcitationGenerator:
    """
    Generate a periodic excitation signal with a given period
    """
    def __init__(self, period, amplitude=1.0, phase=0.0, tmax=100.0, num_samples=10000):
        """
        :param period: period of the excitation signal
        :param amplitude: amplitude of the excitation signal
        :param phase: phase of the excitation signal
        """
        self.period = period
        self.amplitude = amplitude
        self.phase = phase
        self.t = np.linspace(0, tmax, num_samples)

    def __call__(self):
        """
        Return the callable random excitation function with linear interpolation
        """
        return interp1d(self.t, self.generate()[1], kind='linear', bounds_error=True)

    def generate(self):
        """
        Generate the time and random excitation signal in numpy arraies of dimension (n_t, )
        """
        return self.t, self.amplitude*np.sin(2*np.pi*self.t/self.period + self.phase)

    def plot(self):
        """
        Plot the excitation signal
        """
        plt.plot(self.t, self.generate(), color='b')
        plt.xlabel('time (s)')
        plt.ylabel('excitation signal')
        plt.show()