import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt as mf
from scipy import interpolate
import batman
import emcee
import pandas as pd
import pickle
import time
import sys

#from best fit
t0 = 2457163.16066
t0_err = 0.5 * (0.0033470103 + 0.003469955641)
per = 31.707048
a = 21.747136
inc = 87.44889
rp = 0.023
q1 = 0.49275
q2 = 0.311020
s1 = 5.227458738918901e-05
s2 = 0.0003659040249140631
s3 = 0.00025238680750403326

####LOAD DATA
#C5
data = pd.read_csv('C5_planet_c_only.csv')
t1 = np.asarray(data['BJD'])
f1 = np.asarray(data['flux'])


#C18
#data = pd.read_csv('C18_planet_c_only.csv')
data = pd.read_csv('../C18_detrended.csv')
#t3 = np.asarray(data['BJD'])
#f3 = np.asarray(data['flux'])

#C18 short cadence
#data = pd.read_csv('C18_short_cadence_planet_c_only.csv')

data = pd.read_csv('../C18_short_cadence.csv')
t3 = np.asarray(data['BJD'])
f3 = np.asarray(data['flux'])

#data = pd.read_csv('C18_ian.csv')
#t3 = np.asarray(data['bjd']+2454833)
#f3 = np.asarray(data['phot'])

def generate_transit(t0,per,a,inc,rp,u,t):
	params = batman.TransitParams()
	params.per = per
	params.rp = rp
	params.a = a
	params.inc = inc
	params.ecc = 0
	params.w = 0
	params.limb_dark = 'quadratic'
	params.u = u
	params.t0 = t0
	
	m = batman.TransitModel(params,t)
	l = m.light_curve(params)

	return l


def lnbounds(p):
	return 0

def lnprob(theta):
	lp = lnbounds(theta)
	if not np.isfinite(lp):
		return -np.inf
	L = lp + lnlike(theta)
	if np.isnan(L): 
		return -np.inf
	return lp + lnlike(theta)

#######C5###############
print(t1[0])
print(t1[-1])

n1 = int(np.ceil((t1[0] - t0)/per))
n2 = int(np.ceil((t1[-1] - t0)/per))

for i in range(n1,n2):
	p0 = [t1[0] + (t0 - t1[0]) + i * per]
	err = [t0_err]
	print(p0[0])
	f = np.asarray([f1[j] for j in range(len(f1)) if abs(t1[j] - p0[0])  < 0.5])
	t = np.asarray([t1[j] for j in range(len(t1)) if abs(t1[j] - p0[0])  < 0.5])
	print(len(t))
	

	#theta = t0, per, a, inc, rp, [u1,u2]
	def lnlike(theta):

		light_c5 = generate_transit(theta[0],per,a,inc,rp,[q1,q2],t)

		like = -0.5*(np.sum((f-light_c5)**2/s1**2))

		return like

	nwalkers = 20
	ndim = 1
	nsteps = 1000

	pos = emcee.utils.sample_ball(p0,std = err,size = nwalkers)
	sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,threads = 20)
	sampler.run_mcmc(pos,nsteps)

	chain = sampler.chain
	chain = chain[:,500:,:]
	np.save('single_events/chain_transit_' + str(i),chain)
	flat = np.reshape(chain,(20*500))
	t0_median = np.median(flat)

	fit = generate_transit(t0_median,per,a,inc,rp,[q1,q2],t)
	plt.scatter(t,f,s = 2)
	plt.plot(t,fit,c = 'C1',label = 'Single Event Fit')
	original = generate_transit(t0,per,a,inc,rp,[q1,q2],t)
	plt.plot(t,original,'k--',label = 'Combined Model Fit')
	plt.legend(loc = 0)
	plt.savefig('single_events/transit_' + str(i) + '.pdf')
	plt.clf()

#######C18###############
n1 = int(np.ceil((t3[0] - t0)/per))
n2 = int(np.ceil((t3[-1] - t0)/per))

for i in range(n1,n2):
	print(i)
	p0 = [t3[0] + (t0 - t3[0]) + i * per]
	err = [t0_err]
	print(p0[0])
	f = np.asarray([f3[j] for j in range(len(f3)) if abs(t3[j] - p0[0])  < 0.5])
	t = np.asarray([t3[j] for j in range(len(t3)) if abs(t3[j] - p0[0])  < 0.5])
	print(len(t))
	

	#theta = t0, per, a, inc, rp, [u1,u2]
	def lnlike(theta):

		light_c18_short_cad = generate_transit(theta[0],per,a,inc,rp,[q1,q2],t)

		like = -0.5*(np.sum((f-light_c18_short_cad)**2/s1**2))

		return like

	nwalkers = 20
	ndim = 1
	nsteps = 1000

	pos = emcee.utils.sample_ball(p0,std = err,size = nwalkers)
	sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,threads = 20)
	sampler.run_mcmc(pos,nsteps)

	chain = sampler.chain
	chain = chain[:,500:,:]
	np.save('single_events/chain_transit_' + str(i),chain)
	flat = np.reshape(chain,(20*500))
	t0_median = np.median(flat)

	fit = generate_transit(t0_median,per,a,inc,rp,[q1,q2],t)
	plt.scatter(t,f,s = 2)
	plt.plot(t,fit,c = 'C1',label = 'Single Event Fit')
	original = generate_transit(t0,per,a,inc,rp,[q1,q2],t)
	plt.plot(t,original,'k--',label = 'Combined Model Fit')
	plt.legend(loc = 0)
	plt.savefig('single_events/transit_' + str(i) + '.pdf')
	plt.clf()






