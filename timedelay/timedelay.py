# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import LombScargle
from tqdm import tqdm
import seaborn as sns
from lightkurve import search_lightcurvefile

from .estimator import estimate_frequencies

class TimeDelay():

    uHz_conv = 1e-6 * 24 * 60 * 60  # Factor to convert between day^-1 and uHz

    def __init__(self, times, mags, freqs=None, fmin=None, 
                fmax=None, **kwargs):

        self.times = times
        self.mags = mags

        if fmin is None:
            fmin = 1e-3
        if fmax is None:
            fmax = 0.5 / np.median(np.diff(self.times))        
        self.fmin = fmin
        self.fmax = fmax

        if freqs is None:
            freqs = estimate_frequencies(self.times, self.mags, fmin=self.fmin,
                                        fmax=self.fmax,**kwargs)
            
        self.freqs = freqs

    @staticmethod
    def from_archive(target, **kwargs):
        lc_collection = search_lightcurvefile(target).download_all()
        lc = lc_collection[0].PDCSAP_FLUX.normalize()
        for l in lc_collection[1:]:
            lc = lc.append(l.PDCSAP_FLUX.normalize())

        lc = lc.remove_nans()
        magnitude = -2.5 * np.log10(lc.flux)
        magnitude = magnitude - np.average(magnitude)
        return TimeDelay(lc.time, magnitude, **kwargs)

    def time_delay(self, segment_size=10):
        """ Calculates the time delay signal, splitting the lightcurve into 
        chunks of width segment_size """
        
        times, mags = self.times, self.mags
        time_0 = times[0]
        time_slice, mag_slice, phase = [], [], []
        self.time_delays, self.time_midpoints = [], []

        # Loop over lightcurve
        for t, y  in tqdm(zip(times, mags), total=len(times)):
            time_slice.append(t)
            mag_slice.append(y)
            
            # In each segment
            if t - time_0 > segment_size:
                # Append the time midpoint
                self.time_midpoints.append(np.mean(time_slice))
                
                # And the phases for each frequency
                phase.append(self.dft_phase(time_slice, mag_slice, self.freqs))
                time_0 = t
                time_slice, mag_slice = [], []
                
        phase = np.array(phase).T
        # Phase wrapping patch
        for ph, f in zip(phase, self.freqs):
            mean_phase = np.mean(ph)
            ph[np.where(ph - mean_phase > np.pi/2)] -= np.pi
            ph[np.where(ph - mean_phase < -np.pi/2)] += np.pi
            ph -= np.mean(ph)

            td = ph / (2*np.pi*(f / self.uHz_conv * 1e-6))
            self.time_delays.append(td)
        return self.time_midpoints, self.time_delays

    def dft_phase(self, x, y, freqs):
        """ Discrete fourier transform to calculate the ASTC phase
        given x, y, and an array of frequencies"""

        freqs = np.asarray(freqs)
        
        x = np.array(x)
        y = np.array(y)
        phase = []
        for f in freqs:
            expo = 2.0 * np.pi * f * x
            ft_real = np.sum(y * np.cos(expo))
            ft_imag = np.sum(y * np.sin(expo))
            # Why is this broken?
            #phase.append(np.arctan2(ft_imag,ft_real))
            phase.append(np.arctan(ft_imag/ft_real))
        return phase

    def periodogram(self, N=100000):
        """ Calculates the periodogram of the lightcurve """

        fmin, fmax = self.fmin, self.fmax
        freq = np.linspace(fmin, fmax, N)
        model = LombScargle(self.times, self.mags)
        power = model.power(freq, method="fast", normalization="psd")

        # Convert to amplitude
        fct = np.sqrt(4./len(self.times))
        amp = np.sqrt(np.abs(power)) * fct

        return freq, amp

    def plot_periodogram(self, ax=None, **kwargs):
        """ Plots the periodogram of the lightcurve """
        per_freq, per_amp = self.periodogram(**kwargs)

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(per_freq, per_amp, "k", linewidth=0.5)
        ax.set_xlabel("Frequency [cpd]")
        ax.set_ylabel("Amplitude [mag]")

        ax.set_xlim(per_freq[0], per_freq[-1])
        ax.set_xlabel(r"frequency $[d^{-1}]$")

        nyquist = 0.5 / np.median(np.diff(self.times))
        ax.axvline(nyquist, c='r')
        ax.set_ylabel("Amplitude")
        ax.set_ylim([0,None])
        return ax

    def unique_colors(self, n, cmap="hls"):
        colors = np.array(sns.color_palette(cmap, 
                        n_colors=n))
        return colors

    def wavelet(self, fmin, fmax, windows=500, nfreq=1000, gwidth=1, ax=None, cmap='jet'):
        """ Calculates a simple Gaussian based wavelet to 
        check the stability of the frequency over the LC"""

        t, y = self.times, self.mags

        def gaussian(x, mu, sig):
            return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)
    
        num_days = t[-1] - t[0]
        ds = num_days/windows

        freq = np.linspace(fmin, fmax, nfreq)
        dynam, btime = [], []
        
        # Loop over windows
        for i, n in enumerate(tqdm(range(windows))):
            window = gaussian(t, t[0]+i*ds, gwidth)
            btime.append(t[0]+i*ds)
            
            # determine FT power
            model = LombScargle(t, window*y)
            power = model.power(freq, method="fast", normalization="psd")

            for j in range(len(power)):
                dynam.append(power[j])
        if ax is None:
            fig, ax = plt.subplots()
            
        # define shape of results array
        dynam = np.array(dynam, dtype='float64')
        dynam.shape = len(btime), len(power)
        
        ax.imshow(dynam.T, origin='lower', aspect='auto', cmap=cmap,
                extent=[btime[0], btime[-1], fmin, fmax],
                interpolation='bicubic')        
        return ax

    def first_look(self, segment_size=10, **kwargs):
        fig, axes = plt.subplots(3,1,figsize=[12,12])
        t, y = self.times, self.mags

        # Lightcurve
        ax = axes[0]
        ax.plot(t, y, "k", linewidth=0.5)
        ax.set_xlabel('Time [BJD]')
        ax.set_ylabel('Magnitude [mag]')
        ax.set_xlim([t.min(), t.max()])
        ax.invert_yaxis()

        # Time delays
        ax = axes[2]
        time_midpoints, time_delays = self.time_delay(segment_size, **kwargs)
        colors = self.unique_colors(len(time_delays))
        for delay, color in zip(time_delays, colors):
            ax.scatter(time_midpoints,delay, alpha=1, s=8,color=color)
            ax.set_xlabel('Time [BJD]')
            ax.set_ylabel(r'$\tau [s]$')
        ax.set_xlim([t.min(), t.max()])

        # Periodogram
        ax = axes[1]
        periodogram_freq, periodogram_amp = self.periodogram()
        ax.plot(periodogram_freq, periodogram_amp, "k", linewidth=0.5)
        ax.set_xlabel("Frequency [$d^{-1}$]")
        ax.set_ylabel("Amplitude [mag]")
        for freq, color in zip(self.freqs, colors):
                ax.scatter(freq, np.max(periodogram_amp), color=color)
        ax.set_xlim([periodogram_freq[0], periodogram_freq[-1]])
        ax.set_ylim([0,None])

        return

    def to_model(self):
        from .tdmodel import TDModel
        return TDModel(self.time_midpoints, self.time_delays, self.freqs)

    def wavelet_td(self, windows=500, gwidth=1, ax=None, cmap='jet'):
        
        t,y = self.times, self.mags
        freq = self.freqs
        def gaussian(x, mu, sig):
            return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)
        
        num_days = t[-1] - t[0]
        ds = num_days/windows
        dynam, self.btime = [], []
        
        # Loop over windows
        for i, n in enumerate(tqdm(range(windows))):
            window = gaussian(t, t[0]+i*ds, gwidth)
            self.btime.append(t[0]+i*ds)
            
            # determine TD
            ph = self.dft_phase(t, window*y, freq)
            dynam.append(ph)
        
        dynam = np.array(dynam).T

        self.sliding_time_delays = []
        for ph, f in zip(dynam, freq):
            mean_phase = np.mean(ph)
            ph[np.where(ph - mean_phase > np.pi/2)] -= np.pi
            ph[np.where(ph - mean_phase < -np.pi/2)] += np.pi
            ph -= np.mean(ph)
            td = ph / (2*np.pi*(f / self.uHz_conv * 1e-6))
            self.sliding_time_delays.append(td)
        return self.btime, self.sliding_time_delays

    def plot_wavelet_td(self, ax=None, **kwargs):
        btime, time_delays = self.wavelet_td(**kwargs)
        if ax is None:
            fig, ax = plt.subplots()
        for td in time_delays:
            ax.plot(btime,td)

        ax.set_xlabel('Time [BJD]')
        ax.set_ylabel('Time delay (s)')
        return ax