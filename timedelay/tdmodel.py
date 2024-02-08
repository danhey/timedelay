# # -*- coding: utf-8 -*-

# from __future__ import division, print_function

# import numpy as np
# import matplotlib.pyplot as plt
# from astropy.stats import LombScargle

# import theano.tensor as tt
# import pymc3 as pm
# from exoplanet.orbits import get_true_anomaly

# class TDModel(object):
#     def __init__(self, times, tds, freqs, **kwargs):
#         self.times = times
#         self.tds = tds
#         self.freqs = freqs

#         # Define pm model
#         self.model = pm.Model()
#         self.setup_orbit_model(**kwargs)

#     def setup_orbit_model(self, period=None):

#         # Estimate initial period from TD's

#         # Get period estimate
#         ls_model = LombScargle(self.times,self.tds[0])
#         f = np.linspace(1e-3,0.5/np.median(np.diff(self.times)),10000)
#         power = ls_model.power(f, method="fast", normalization="psd")
#         period_t = 1/f[np.argmax(power)]

#         with self.model as model:

#             # Parameters
#             self.period = pm.Normal("period", mu=period_t, sd=100)
#             self.tref = pm.Uniform("tref", lower=-5000, upper=5000)
#             self.varpi = pm.Uniform("varpi", lower=0, upper=50)
#             self.eccen = pm.Uniform("eccen", lower=1e-3, upper=0.999)

#             self.lighttime = pm.Uniform('lighttime', lower=-2000,
#                                 upper=2000, shape=(len(self.freqs)))

#             # Deterministic transformations
#             # Mean anom
#             M = 2.0 * np.pi * (self.times - self.tref) / self.period
#             # True anom
#             f = get_true_anomaly(M, self.eccen + tt.zeros_like(M))

#             factor = 1.0 - tt.square(self.eccen)
#             factor /= 1.0 + self.eccen * tt.cos(f)
#             psi = -factor * tt.sin(f + self.varpi)

#             tau = self.lighttime[:,None] * psi[None,:]
#             taumodel = pm.Deterministic('taumodel', tau - tt.mean(tau))

#             # Condition on the observations
#             pm.Normal("obs", mu=taumodel, sd=None, observed=self.tds)

#     def sample(self, draws=1000, tune=1000, chains=4, **kwargs):
#         with self.model as model:
#             map_params = pm.find_MAP()
#             self.trace = pm.sample(draws=draws, tune=tune,
#                             chains=chains, start=map_params,**kwargs)

#         return pm.summary(self.trace, varnames=["period", "lighttime",
#                                     "tref", "varpi", "eccen"])

#     def trace_plot(self):
#         pm.traceplot(self.trace, varnames=["period", "lighttime",
#                                     "tref", "varpi", "eccen"])

#     def corner_plot(self):
#         pass

#     def plot_model(self, ax=None):
#         pass
