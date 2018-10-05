import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.stats import LombScargle

class TimeDelay():

    def __init__(self, times, mags, freqs=None, min_freq=5, max_freq=49,**kwargs):

        self.times = times
        self.mags = mags

        if freqs is None:
            freqs = self.estimate_frequencies(**kwargs)
            freqs = freqs[freqs<max_freq]
            freqs = freqs[freqs>min_freq]
        self.freqs = freqs
        self.nu = self.freqs

    @staticmethod
    def from_archive(target, **kwargs):
        """Instantiates a TimeDelay object from target KIC ID by downloading
        photometry from MAST. 
        Args:
            target: (string) target ID (i.e. 'KIC9651065')
            **kwargs: Optional args to pass to TimeDelay
        """
        try:
            from lightkurve import KeplerLightCurveFile
        except ImportError:
            raise ImportError('LightKurve package is required for MAST.')

        lcs = KeplerLightCurveFile.from_archive(target, quarter='all', 
                                                cadence='long')
        lc = lcs[0].PDCSAP_FLUX.remove_nans()
        lc.flux = -2.5 * np.log10(lc.flux)
        lc.flux = lc.flux - np.average(lc.flux)

        for i in lcs[1:]:
            i = i.PDCSAP_FLUX.remove_nans()
            i.flux = -2.5 * np.log10(i.flux)
            i.flux = i.flux - np.average(i.flux)
            lc = lc.append(i)
            
        return TimeDelay(lc.time, lc.flux, **kwargs)

    def time_delay(self, segment_size=10):
        """ Calculates the time delay signal, splitting the lightcurve into 
        chunks of width segment_size """
        uHz_conv = 1e-6 * 24 * 60 * 60  # Factor to convert between day^-1 and uHz
        times, mags = self.times, self.mags

        self.time_delays, self.time_midpoints = [], []

        for freq in self.nu:
            times_0 = times[0]
            phase, time_mid = [], []

            mod, time_mod = [], []
            for i, j in zip(times, mags):
                mod.append(j)
                time_mod.append(i)

                if i-times_0 > segment_size:
                    phase.append(self.ft_single(time_mod, mod, freq))
                    time_mid.append(np.mean(time_mod))
                    times_0 = i
                    mod, time_mod = [], []

            phase -= np.mean(phase)
            tau = phase / (2*np.pi*(freq / uHz_conv * 1e-6))
            self.time_delays.append(tau)
            self.time_midpoints.append(time_mid)
        return self.time_midpoints, self.time_delays

    def plot_td(self, periodogram=True, **kwargs):

        time_midpoints, time_delays = self.time_delay(**kwargs)
        colors = ['red','darkorange','gold','seagreen','dodgerblue','darkorchid','mediumvioletred']

        if periodogram:
            fig, ax = plt.subplots(2,1,figsize=[10,10])
            periodogram_freq, periodogram_amp = self.periodogram()
            ax[1].plot(periodogram_freq, periodogram_amp, "k", linewidth=0.5)
            ax[1].set_xlabel("frequency [cpd]")
            ax[1].set_ylabel("Amplitude [mag]")
            for freq, color in zip(self.nu, colors):
                ax[1].scatter(freq, np.max(periodogram_amp), c=color)
        else:
            fig, ax = plt.subplots(figsize=[10,5])
            ax = [ax]

        for midpoint, delay, color in zip(time_midpoints, time_delays, colors):
            ax[0].scatter(midpoint,delay, alpha=1, s=8,c=color)
            ax[0].set_xlabel('Time [BJD]')
            ax[0].set_ylabel(r'Time delay $\tau$ [s]')
        return ax

    def estimate_frequencies(self, max_peaks=7, oversample=4.0, tflow=True):

        """ This function does some fancy peak fitting to estimate the main
        frequencies of the lightcurve. """
        x = self.times
        y = self.mags

        tmax = x.max()
        tmin = x.min()
        dt = np.median(np.diff(x))
        df = 1.0 / (tmax - tmin)
        ny = 0.5 / dt

        freq = np.arange(df, 2 * ny, df / oversample)
        power = LombScargle(x, y).power(freq)

        # Find peaks
        peak_inds = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
        peak_inds = np.arange(1, len(power)-1)[peak_inds]
        peak_inds = peak_inds[np.argsort(power[peak_inds])][::-1]
        peaks = []
        for j in range(max_peaks):
            i = peak_inds[0]
            freq0 = freq[i]
            alias = 2.0*ny - freq0

            m = np.abs(freq[peak_inds] - alias) > 25*df
            m &= np.abs(freq[peak_inds] - freq0) > 25*df

            peak_inds = peak_inds[m]
            peaks.append(freq0)
        
        if tflow:
            try:
                import tensorflow as tf
            except ImportError:
                raise ImportError('tensorflow package is required for this')
            # Optimize the model
            T = tf.float64
            t = tf.constant(x, dtype=T)
            f = tf.constant(y, dtype=T)
            nu = tf.Variable(peaks, dtype=T)
            arg = 2*np.pi*nu[None, :]*t[:, None]
            D = tf.concat([tf.cos(arg), tf.sin(arg),
                        tf.ones((len(x), 1), dtype=T)],
                        axis=1)

            # Solve for the amplitudes and phases of the oscillations
            DTD = tf.matmul(D, D, transpose_a=True)
            DTy = tf.matmul(D, f[:, None], transpose_a=True)
            w = tf.linalg.solve(DTD, DTy)
            model = tf.squeeze(tf.matmul(D, w))
            chi2 = tf.reduce_sum(tf.square(f - model))

            opt = tf.contrib.opt.ScipyOptimizerInterface(chi2, [nu],
                                                        method="L-BFGS-B")
            with tf.Session() as sess:
                sess.run(nu.initializer)
                opt.minimize(sess)
                return sess.run(nu)
        else:
            return np.array(peaks)

    def periodogram(self, oversample=2., samples=100000):
        """ Calculates the periodogram of the lightcurve """
        t = self.times
        y = self.mags

        uHz_conv = 1e-6 * 24 * 60 * 60  # Factor to convert between day^-1 and uHz

        nyquist = 0.5 / np.median(np.diff(t))
        nyquist = nyquist / uHz_conv
        freq_uHz = np.linspace(1e-2, nyquist * oversample, samples)
        freq = freq_uHz * uHz_conv

        model = LombScargle(t, y)
        power = model.power(freq, method="fast", normalization="psd")

        # Convert to amplitude
        fct = np.sqrt(4./len(t))
        amp = np.sqrt(np.abs(power)) * fct
        return freq_uHz * uHz_conv, amp

    def plot_periodogram(self, ax=None):
        """ Plots the periodogram of the lightcurve """
        per_freq, per_amp = self.periodogram()
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(per_freq, per_amp, "k", linewidth=0.5)
        ax.set_xlabel("frequency [cpd]")
        ax.set_ylabel("Amplitude [mag]")
        return ax

    def ft_single(self, x, y, freq):
        """ This calculates the discrete fourier transform of the signal 
        at a specified signal, returning the phase. Should really be vectorized
        """
        x = np.asarray(x)
        y = np.asarray(y)
        notnans = (~np.isnan(x)) & (~np.isnan(y))
        x = x[notnans]
        y = y[notnans]

        ft_real, ft_imag, power, fr = [], [], [], []
        len_x = len(x)
        ft_real.append(0.0)
        ft_imag.append(0.0)
        omega = 2.0 * np.pi * freq
        for i in range(len_x):
            expo = omega * x[i]
            c = np.cos(expo)
            s = np.sin(expo)
            ft_real[-1] += y[i] * c
            ft_imag[-1] += y[i] * s
        phase = np.arctan(ft_imag[0]/ft_real[0])
        return phase

