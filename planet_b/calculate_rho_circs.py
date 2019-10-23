from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import photoeccentric as pe

rho_meas, e_rho_meas = 0.68, 0.064    # measured stellar density
nsamp, npts = 1e5, 220  # for sampling, keep as is
rhostar_circs = np.load('rho_circ_chain.npy')
ecc, om, gs, rho = pe.photoeccentric_maxprob(rhostar_circs, None, rho_meas, \
    	e_rho_meas, nsamp=nsamp,npts=npts, plotfig=True, retvals=True)


#pe.photoeccentric_maxprob(rhostar_circs, None, rho_meas, e_rho_meas,nsamp=-1,npts=240, plotfig=True)
