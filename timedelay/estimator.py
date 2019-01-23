# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from astropy.stats import LombScargle
from scipy import optimize
    
def estimate_frequencies(x, y, fmin=None, fmax=None, 
                            max_peaks=9, oversample=4.0,
                            optimize_f=True):
    tmax = x.max()
    tmin = x.min()
    dt = np.median(np.diff(x))
    df = 1.0 / (tmax - tmin)
    ny = 0.5 / dt

    if fmin is None:
        fmin = df
    if fmax is None:
        fmax = ny

    freq = np.arange(fmin, fmax, df / oversample)
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
    peaks = np.array(peaks)

    if optimize_f:
        def chi2(nu):
            arg = 2*np.pi*nu[None, :]*x[:, None]
            D = np.concatenate([np.cos(arg), np.sin(arg),
                        np.ones((len(x), 1))],
                        axis=1)

            # Solve for the amplitudes and phases of the oscillations
            DTD = np.matmul(D.T, D)
            DTy = np.matmul(D.T, y[:, None])
            w = np.linalg.solve(DTD, DTy)
            model = np.squeeze(np.matmul(D, w))

            chi2_val = np.sum(np.square(y - model))
            return chi2_val

        res = optimize.minimize(chi2, [peaks], method="L-BFGS-B")
        return res.x
    else:
        return peaks