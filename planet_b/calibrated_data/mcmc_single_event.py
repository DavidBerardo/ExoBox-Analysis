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
t0 = 2457152.2816418963
t0_err = 0.5 * (0.0012066122144460678 + 0.0012264265678822994)
per = 15.572102958164814
a = 19.49803428542964
inc = 88.267222986
rp = 0.018860574757576
q1 = 0.464720246
q2 = 0.060670518316
s1 = 10**(-4.399579024)
s2 = 10**(-4.2985239)
s3 = 10**(-3.83404356)

'''
#from best fit
t0 = 2457152.281839
t0_err = 0.5 * (0.0012066122144460678 + 0.0012264265678822994)
per = 15.5720984
a = 21.696601
inc = 88.857050
rp = 0.018431858
q1 = 0.46253948
q2 =  0.0637072
s1 = 4.055559270368129e-05
s2 =  5.0566548370700936e-05
s3 = 0.00014628202638586536
'''
'''
#from best fit
t0 = 2457152.2812403617
t0_err = 0.5 * (0.0018356344662606716 + 0.0019747051410377026)
per = 15.572154364423948
a = 9.54059176
inc = 84.48802
rp = 0.02290
q1 = 0.36095
q2 = 0.412241
s1 = 5.279491196164024e-05
s2 = 0.0003588
s3 = 0.0002186
'''
####LOAD DATA
#C5
data = pd.read_csv('C5_planet_b_only.csv')
t1 = np.asarray(data['BJD'])
f1 = np.asarray(data['flux'])


#C18
data = pd.read_csv('C18_calibrated_planet_b_only.csv')
t2 = np.asarray(data['BJD'])
f2 = np.asarray(data['flux'])


#C18 short cadence
data = pd.read_csv('C18_short_cadence_calibrated_planet_b_only.csv')
t3 = np.asarray(data['BJD'])
f3 = np.asarray(data['flux'])

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
	nsteps = 5000

	pos = emcee.utils.sample_ball(p0,std = err,size = nwalkers)
	sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,threads = 20)
	sampler.run_mcmc(pos,nsteps)

	chain = sampler.chain
	chain = chain[:,1000:,:]
	np.save('single_events/chain_transit_' + str(i),chain)
	flat = np.reshape(chain,(20*4000))
	t0_median = np.median(flat)

	fit = generate_transit(t0_median,per,a,inc,rp,[q1,q2],t)
	plt.scatter(t,f,s = 2)
	plt.plot(t,fit,c = 'C1',label = 'Single Event Fit')
	original = generate_transit(t0,per,a,inc,rp,[q1,q2],t)
	plt.plot(t,original,'k--',label = 'Combined Model Fit')
	plt.legend(loc = 0)
	plt.savefig('single_events/transit_' + str(i) + '.pdf')
	plt.clf()

#######C5###############
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
	nsteps = 5000

	pos = emcee.utils.sample_ball(p0,std = err,size = nwalkers)
	sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,threads = 20)
	sampler.run_mcmc(pos,nsteps)

	chain = sampler.chain
	chain = chain[:,1000:,:]
	np.save('single_events/chain_transit_' + str(i),chain)
	flat = np.reshape(chain,(20*4000))
	t0_median = np.median(flat)

	fit = generate_transit(t0_median,per,a,inc,rp,[q1,q2],t)
	plt.scatter(t,f,s = 2)
	plt.plot(t,fit,c = 'C1',label = 'Single Event Fit')
	original = generate_transit(t0,per,a,inc,rp,[q1,q2],t)
	plt.plot(t,original,'k--',label = 'Combined Model Fit')
	plt.legend(loc = 0)
	plt.savefig('single_events/transit_' + str(i) + '.pdf')
	plt.clf()







