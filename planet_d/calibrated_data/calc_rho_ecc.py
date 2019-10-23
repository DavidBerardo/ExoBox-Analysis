import photoeccentric as pe
import numpy as np
import matplotlib.pyplot as plt 

def calc(num):
	print(num)
	chain = np.load('sampler_chain_' + str(num) + '.npy')
	chain = chain[:,20000:,:]
	chain = np.asarray([i for i in chain if all(i[:,-1] < -3.6)])
	f = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))

	#do everything in cgs
	a = f[:,2]
	p = f[:,1] * 24 * 3600
	G = 6.674e-8

	rho_circ = 3 * np.pi * a**3 / p**2 / G

	rho_meas, e_rho_meas = 0.68, 0.064    # measured stellar density
	nsamp, npts = 1e5, 220  # for sampling, keep as is
	ecc, om, gs, rho = pe.photoeccentric_maxprob(rho_circ, None, rho_meas, \
    	e_rho_meas, nsamp=nsamp,npts=npts, plotfig=False, retvals=True,verbose=True)

	np.save('eccentricity_posterior_' + str(num),ecc)
	np.save('omega_posterior_' + str(num),om)
	#plt.savefig('eccentricity_plot_' + str(num) + '.pdf')

periods = range(1,24)

for p in periods:
	calc(p)
