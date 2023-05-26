# Created by Diogo André @FCT-NOVA, Jun 2023

import pdb

import os
import traceback
from warnings import warn
from dataclasses import dataclass

from numpy.fft import fft, fftfreq, fftshift
from numpy import ndarray, nan # core types
from numpy import array, arange, abs, log10,  sqrt, diff # core functions
from numpy import sum as npsum
from numpy import min as npmin
from numpy import max as npmax
from numpy import argmax, argmin, argwhere # core array searching utilities

from scipy.signal import hann, hamming, blackmanharris, blackman, gaussian, kaiser, cosine, parzen


windows = {
    "rectangular": lambda x, np: x,
    "hanning": lambda x, np: x*hann(np),
    "hamming": lambda x, np: x*hamming(np),
    "blackmanharris": lambda x, np: x*blackmanharris(np),
    "blackman": lambda x, np: x*blackman(np),
    "gaussian": lambda x, np: x*gaussian(np),
    "kaiser": lambda x, np: x*kaiser(np),
    "cosine": lambda x, np: x*cosine(np),
    "parzen": lambda x, np: x*parzen(np),
}

@dataclass
class FFTEngine:
    """ Fast Fourier Transform (FFT) computation engine and spectral analyser.
    """
    def __init__(
        self,
        signal,
        time: ndarray = None,
        unit="V"
    ) -> None:
        # signal
        self.signal:ndarray = signal
        #time axis
        self.time:ndarray = time

        # power spectrum
        self.ps_f:ndarray = None
        # frequency axis
        self.ff:ndarray = None
        # bins and power levels of main harmonic components
        self.harmonic_bin_idxs:ndarray = None
        self.harmonic_bins:ndarray = None
        self.harmonic_powers:ndarray = None
        
        
        # spectral characteristics
        self.snr:float = None # signal-to-noise ratio
        self.sfdr:float = None # spurious-free dynamic range
        self.sndr:float = None # signal-to-noise-distortion ratio
        self.thd:float = None # total harmonic distortion
        self.enob:float = None # effective number of bits
        self.h2:float = None # second order fractional harmonic distortion
        self.h3:float = None # third order fractional harmonic distortion
        self.dc:float = None # dc component level

    def set_time_axis(self, time:ndarray):
        self.time = time

    def fft(
        self,
        n_points: int,
        fs: float = None,
        window: str = "rectangular",
    ) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        """_summary_
        Compute the FFT of a signal.
        Args:
            signal: signal to be analysed
            fs: sampling frequency of the signal
            n_points: number of points to be used in the FFT
            window: window function to be applied to the signal
        Returns:
            tt: time axis of the signal
            signal: signal
            ff: frequency axis of the signal's power spectrum
            power_f: power spectrum of the signal
        Examples:
        """
        signal = self.signal
        assert len(signal) > 0, "signal must have at least one point."
        assert n_points > 0, "n_points must be an integer greater and 0 - and preferably a power of 2 for fft computation speed up."
        assert n_points <= len(signal), "n_points must be less than or equal to the signal length."
        downsampling = 1
        
        # sample the signal if necessary
        downsampling = len(signal) // n_points
        downsampling = 1 if downsampling == 0 else downsampling
        # downsample the signals to the sampling frequency effectively parsed as input
        signal = self.signal[::downsampling]
        np_fft = len(signal)
        # apply the window function if requested
        if window not in list(windows.keys()):
            warn(f"Window function {window} not implemented. Setting to default rectangular window.")
            print("Available windows: ", '\n'.join(list(windows.keys())))
            window = "rectangular"
        signal = windows[window](signal, np_fft)
        if self.time is not None:
            if fs is not None:
                warn("fs - sampling frequency will be ignored since a time axis is provided.")
            tt = self.time[::downsampling]
            ts = npmin(diff(tt))
            ff = fftshift(fftfreq(np_fft, ts))  # [Hz]
        else:
            assert fs is not None, "fs - sampling frequency must be provided if no time axis is provided."
            assert fs > 0, "fs - sampling frequency must be positive."
            # generate time axis
            ts = 1 / fs # sampling period
            tt = ts * arange(np_fft)
            # compute the frequency axis
            ff = fftshift(fftfreq(np_fft, ts))  # [Hz]
        # compute the FFT
        signal_f = abs(fftshift(fft(signal) / np_fft) ) # [V]
        
        power_f = signal_f*signal_f # [V^2] - square the voltage spectrum to obtain the power spectrum
        self.ps_f = power_f
        self.ff = ff
        self.time = tt if self.time is not None else self.time
        return tt, signal, ff, power_f
    
    def compute_harmonics(
        self,
        fs: float,
        span: int = 1,
        harmonics: int = 7,
    ) -> tuple[ndarray, ndarray]:
        """Compute the main harmonic components of a signal,
        returning their respective bins and power levels.
        Args:
            fs (float): sampling frequency of the signal
            span (int, optional): number of bins the signal bin occupies. Defaults to 1.
                - rectangular = 1
                - hann, hamming, blackman = 3
                - blackmanharris, gaussian, kaiser, cosine, parzen = 5
            harmonics (int, optional): Number of harmonics to be computed. Defaults to 7.
        """
        assert span > 0, "span must be an integer greater than 0."
        assert harmonics > 1, "harmonics must be an integer greater than 1."
        assert fs > 0, "fs - sampling frequency must be positive."
        # get the positive frequency spectrum
        pps = self.ps_f[self.ff >= 0]
        pff = self.ff[self.ff >= 0]
        # obtain the signal frequency bin
        # while discarding the DC component
        signal_bin = pff[argwhere(pps == npmax(pps[span:]))[0]]
        # obtain the harmonics of the signal from the signal bin
        harmonic_bins = [
            pff[ argmin( abs(pff - (mult * signal_bin)) ) ]
            for mult in range(1, harmonics + 1)
            if mult * signal_bin <= npmax(pff)
        ]
        # tones that surpass Fs are aliased back to [0, Fs/2] spectrum
        harmonic_bins = array([
            pff[ argmin( abs(pff - (fs - bin)) ) ]
            if bin > fs / 2
            else bin
            for bin in harmonic_bins
        ])
        # indexes of the harmonic bins
        harmonic_bins_idxs = array([argwhere(pff == bin)[0] for bin in harmonic_bins]).reshape(-1)
        harmonics_power = array([
            npsum(pps[harmonic_bin_idx - span : harmonic_bin_idx + span])
            for harmonic_bin_idx in harmonic_bins_idxs
        ])
        self.harmonic_bin_idxs = harmonic_bins_idxs
        self.harmonic_bins = harmonic_bins
        self.harmonic_powers = harmonics_power
        return harmonic_bins_idxs, harmonic_bins, harmonics_power
        
    def get_dc(
        self,
        span: int = 1
    ):
        """ Compute the DC component of a signal.
        Args:
            span (int, optional): number of bins the signal bin occupies. Defaults to 1.
                - rectangular = 1
                - hann, hamming, blackman = 3
                - blackmanharris, gaussian, kaiser, cosine, parzen = 5
        """
        pps = self.ps_f[self.ff >= 0]
        signal_dc_power = npsum(pps[0 : span])
        dc_level = sqrt(signal_dc_power)
        self.dc = dc_level
        return dc_level
        
    def get_snr(
        self,
        span: int = 1
    ):
        """ Compute the signal-to-noise ratio of a signal.
        Args:
            span (int, optional): number of bins the signal bin occupies. Defaults to 1.
                - rectangular = 1
                - hann, hamming, blackman = 3
                - blackmanharris, gaussian, kaiser, cosine, parzen = 5
        """
        assert self.harmonic_bins is not None, "harmonic_bins must be computed before computing the SNR."
        assert self.harmonic_powers is not None, "harmonic_powers must be computed before computing the SNR."
        assert len(self.harmonic_bins) > 1, "harmonic_bins must have at least two elements."
        assert span > 0, "span must be an integer greater than 0."
        signal_power = self.harmonic_powers[0]
        pps = self.ps_f[self.ff >= 0]
        total_distortion_power = npsum(self.harmonic_powers[1:])
        dc_power = self.get_dc(span)**2
        noise_power = (
            npsum(pps)
            - dc_power
            - signal_power
            - total_distortion_power
        )
        self.snr = 10 * log10(signal_power / noise_power)
        return self.snr
    
    def get_thd(
        self
    ):
        """ Compute the total harmonic distortion of a signal.
        """
        assert self.harmonic_bins is not None, "harmonic_bins must be computed before computing the SFDR."
        assert self.harmonic_powers is not None, "harmonic_powers must be computed before computing the SFDR."
        assert len(self.harmonic_bins) > 1, "harmonic_bins must have at least two elements."
        signal_power = self.harmonic_powers[0]
        total_distortion_power = npsum(self.harmonic_powers[1:])
        self.thd = 10 * log10(signal_power / total_distortion_power)
        return self.thd
    
    def get_sfdr(
        self,
        span: int = 1
    ):
        """Compute the spurious-free dynamic range of a signal.
        Args:
            span (int, optional): number of bins the signal bin occupies. Defaults to 1.
                - rectangular = 1
                - hann, hamming, blackman = 3
                - blackmanharris, gaussian, kaiser, cosine, parzen = 5
        """
        assert self.harmonic_bins is not None, "harmonic_bins must be computed before computing the SFDR."
        assert self.harmonic_powers is not None, "harmonic_powers must be computed before computing the SFDR."
        assert len(self.harmonic_bins) > 1, "harmonic_bins must have at least two elements."
        assert span > 0, "span must be an integer greater than 0."
        pps = self.ps_f[self.ff >= 0]
        signal_bin_idx = self.harmonic_bin_idxs[0]  # get the index of the signal bin
        spurious_spectrum = pps.copy()
        # erase the signal bin from the spurious spectrum
        spurious_spectrum[signal_bin_idx - span : signal_bin_idx + span] = npmin(pps)
        # erase the signal's DC component from the spurious spectrum
        spurious_spectrum[0 : span] = npmin(pps)
        # find the strongest spurious component
        spur_bin_idx = argmax(spurious_spectrum)
        # measure the power of the strongest spurious component
        spur_power = npsum(
            spurious_spectrum[spur_bin_idx - span : spur_bin_idx + span]
        )
        signal_power = self.harmonic_powers[0]
        self.sfdr = 10 * log10(signal_power / spur_power)
        return self.sfdr
    
    def get_sndr(
        self,
        span: int = 1,
    ):
        """Compute the signal-to-noise-distortion ratio of a signal.
        Args:
            span (int, optional): number of bins the signal bin occupies. Defaults to 1.
                - rectangular = 1
                - hann, hamming, blackman = 3
                - blackmanharris, gaussian, kaiser, cosine, parzen = 5
        """
        assert self.harmonic_bins is not None, "harmonic_bins must be computed before computing the SFDR."
        assert self.harmonic_powers is not None, "harmonic_powers must be computed before computing the SFDR."
        assert len(self.harmonic_bins) > 1, "harmonic_bins must have at least two elements."
        assert span > 0, "span must be an integer greater than 0."
        pps = self.ps_f[self.ff >= 0]
        dc_power = self.get_dc(span)**2
        signal_power = self.harmonic_powers[0]
        total_distortion_power = npsum(self.harmonic_powers[1:])
        noise_power = (
            npsum(pps)
            - dc_power
            - signal_power
            - total_distortion_power
        )
        self.sndr = 10 * log10(signal_power / (noise_power + total_distortion_power))
        return self.sndr

    def get_h2(
        self
    ):
        """Compute the second harmonic of a signal.
        Args:
            span (int, optional): number of bins the signal bin occupies. Defaults to 1.
                - rectangular = 1
                - hann, hamming, blackman = 3
                - blackmanharris, gaussian, kaiser, cosine, parzen = 5
        """
        assert self.harmonic_bins is not None, "harmonic_bins must be computed before computing the SFDR."
        assert self.harmonic_powers is not None, "harmonic_powers must be computed before computing the SFDR."
        assert len(self.harmonic_bins) > 1, "harmonic_bins must have at least two elements."
        self.h2 = 10 * log10(self.harmonic_powers[1] / self.harmonic_powers[0])
        return self.h2
    
    def get_h3(
        self
    ):
        """Compute the second harmonic of a signal.
        Args:
            span (int, optional): _description_. Defaults to 1.
        """
        assert self.harmonic_bins is not None, "harmonic_bins must be computed before computing the SFDR."
        assert self.harmonic_powers is not None, "harmonic_powers must be computed before computing the SFDR."
        assert len(self.harmonic_bins) > 2, "harmonic_bins must have at least three elements."
        self.h3 = 10 * log10(self.harmonic_powers[2] / self.harmonic_powers[0])
        return self.h3
    
    def get_enob(
        self,
        span: int = 1,
    ):
        """Compute the effective number of bits of an equivalent ADC with 
        the same linearity as the system having as output the signal.
        Args:
            Args:
            span (int, optional): number of bins the signal bin occupies. Defaults to 1.
                - rectangular = 1
                - hann, hamming, blackman = 3
                - blackmanharris, gaussian, kaiser, cosine, parzen = 5
        """
        assert self.harmonic_bins is not None, "harmonic_bins must be computed before computing the SFDR."
        assert self.harmonic_powers is not None, "harmonic_powers must be computed before computing the SFDR."
        assert span > 0, "span must be an integer greater than 0."
        sndr = self.get_sndr(span)
        self.enob = (sndr - 1.76) / 6.02
        return self.enob
    
    def spectral_analysis(
        self,
        fs: float = None,
        span: int = 1,
        harmonics: int = 7,
    ):
        """ Compute the spectral markers of a signal, such 
        as the harmonic components, total harmonic distortion (thd),
        spurious free dynamic range (sfdr), signal-to-noise ratio (snr),
        signal-to-noise-distortion ratio (sndr), and equivalent resolution 
        of an adc with the spectral characteristics of the analysed signal.
        Args:
            fs: sampling frequency of the signal
            span: number of bins the signal bin occupies
                - rectangular = 1
                - hann, hamming, blackman = 3
                - blackmanharris, gaussian, kaiser, cosine, parzen = 5
            harmonics: number of harmonics to be analysed
        Returns:
            dc: signal DC level
            sfdr: spurious free dynamic range
            thd: total harmonic distortion
            snr: signal-to-noise ratio
            sndr: signal-to-noise-distortion ratio
            enob: equivalent number of bits
            hd2: second order fractional harmonic distortion 
            hd3: third order fractional harmonic distortion
        Examples:
        """
        assert span > 0, "span must be an integer greater than 0."
        assert harmonics > 1, "harmonics must be an integer greater than 1."
        if fs is None:
            fs = 1/(npmin(self.time[1:] - self.time[:-1]))
        self.compute_harmonics(fs, span, harmonics)
        sa = {
            "dc": self.get_dc(span),
            "snr": self.get_snr(span),
            "sndr": self.get_sndr(span),
            "sfdr": self.get_sfdr(span),
            "thd": self.get_thd(),
            "enob": self.get_enob(span),
            "h2": self.get_h2(),
            "h3": self.get_h3(),     
        }
        return sa

# TODO: Add specifications testing for pass/fail analysis of the spetral markers of the signal
class SpecificationTester:
    pass