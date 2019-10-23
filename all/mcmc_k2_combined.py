from __future__ import division
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


####LOAD DATA
#C5
data = pd.read_csv('../C5_detrended.csv')
t1 = np.asarray(data['BJD'])
f1 = np.asarray(data['flux'])
sigma2c5 = (1.1428985942e-05)**2

#C18
data = pd.read_csv('../C18_detrended.csv')
t2 = np.asarray(data['BJD'])
f2 = np.asarray(data['flux'])
sigma2c18 = (3.8629440215e-05)**2



def generate_transit(t0,per,a,inc,rp,u):
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
	params.fp = 1
	
	m1 = batman.TransitModel(params,t1)
	m2 = batman.TransitModel(params,t2)

	l1 = m1.light_curve(params)
	l2 = m2.light_curve(params)
	return l1, l2

#theta = [offset, ci's, D, f, g]
def lnlike(theta):

	curves_c5 = []
	curves_c18 = []

	for i in range(5):
		l1, l2 = generate_transit(theta[5*i],theta[5*i + 1],theta[5*i + 2],theta[5*i + 3],theta[5*i + 4],theta[-2:])
		curves_c5.append(l1)
		curves_c18.append(l2)

	light_c5 = np.asarray([min([i[j] for i in curves_c5]) for j in range(len(t1))])
	light_c18 = np.asarray([min([i[j] for i in curves_c18]) for j in range(len(t2))])

	return -0.5*(np.sum((f1-light_c5)**2/sigma2c5)) -0.5*(np.sum((f2-light_c18)**2/sigma2c18))

def lnbounds(p):
	if not(all(i > 0 for i in p)):
		return -np.inf

	elif p[-1]**2 + p[-2]**2 > 1:
		return -np.inf
	else:
		return 0

def lnprob(theta):
	lp = lnbounds(theta)
	if not np.isfinite(lp):
		return -np.inf
	L = lp + lnlike(theta)
	if np.isnan(L): 
		return -np.inf
	return lp + lnlike(theta)

ndim = 27
nwalkers = 150
p0 = [2457152.2844,15.5712,19.5,88.4,0.0188] + \
	 [2457163.1659,31.6978,73,89.58,0.0166] + \
	 [2457166.2620,156,140,90,0.0259] + \
	 [2457142.01656,131,140,90,0.03613] + \
	 [2457186.91451,271,112,90,0.0672] + \
	 [0.3,0.3]

err = [0.002,0.002,5,0.5,0.001] + \
	  [0.002,0.002,20,0.5,0.001] + \
	  [0.002,100,25,0.2,0.001] + \
	  [0.002,100,25,0.2,0.001] + \
	  [0.002,0.5,5,0.2,0.001] + \
	  [0.1,0.1]

p = emcee.utils.sample_ball(p0,std = err, size = nwalkers)
sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,threads = 20)

#sampler shape is (nwalkers,steps,dims)
def plot_chains(sampler,fname,steps):
	fig = plt.figure(figsize = (10,10))
	chain = sampler.chain
	nwalkers = np.shape(sampler.chain)[0]
	dim = np.shape(sampler.chain)[-1]
	w = int(round(np.sqrt(dim)))
	h = int(np.ceil(dim/float(w)))

	for i in range(dim):
		ax = fig.add_subplot(w,h,i+1)
		for j in range(nwalkers):
			plt.plot(range(steps),chain[j,:steps,i],c = 'C1',alpha = 0.5)
	plt.savefig(fname + '.png')
	plt.close()
	return

def run_sampler(sampler,p0):
	Nburn = 10000
	Nsteps = 40000
	#burn in phase
	elapsed_time = 0 #used to keep track of time between print outs, estimate remaining time
	t0 = time.time()
	print('Initiating burn in: ')
	for i, (pos, lnp, state) in enumerate(sampler.sample(p0, iterations=Nburn)):
		if (i+1) % (Nburn/100) == 0:
			
			#get an estimate of how much time is left
			elapsed_time += time.time() - t0 #add the total time elapsed since previous 1% of runs finished
			remaining_estimate = int(elapsed_time/((i+1)/(Nburn/100))*(1 - float(i)/ Nburn)*100)
			remaining_estimate = str(int(remaining_estimate/60)) + ':' + str(remaining_estimate%60).zfill(2)			
			#remaining_estimate = str(int(remaining_estimate/60)) + ':' + '0'*(2 - len(str(remaining_estimate%60))) + str(remaining_estimate%60)

			sys.stdout.write('\r' + ' '*10 + "burn in: " +  "{0:.1f}%".format(100 * float(i + 1) / Nburn) + ', Approx time left: ' + remaining_estimate + ' '*10)
			sys.stdout.flush()

			t0 = time.time()

	#plot burn in chains


	print("Mean acceptance fraction during burn in: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))

	#plot_chains(sampler,'burn_in_chains',Nburn)
	#clear out the current chain, after burn in
	sampler.reset()

	print #skip to a new line

	
	#Actual mcmc run
	elapsed_time = 0 #used to keep track of time between print outs, estimate remaining time
	t0 = time.time()
	print('Starting actual run: ')
	for i, (pos, lnp, state) in enumerate(sampler.sample(pos, iterations=Nsteps)):
		if (i+1) % (Nsteps/100) == 0:
			#plot_chains(sampler,'mcmc_output_orbit_params/post_burn_chains',i)
			#get an estimate of how much time is left
			elapsed_time += time.time() - t0 #add the total time elapsed since previous 1% of runs finished
			remaining_estimate = int(elapsed_time/((i+1)/(Nsteps/100))*(1 - float(i)/ Nsteps)*100)
			remaining_estimate = str(int(remaining_estimate/60)) + ':' + str(remaining_estimate%60).zfill(2)			
			#remaining_estimate = str(int(remaining_estimate/60)) + ':' + '0'*(2 - len(str(remaining_estimate%60))) + str(remaining_estimate%60)

			sys.stdout.write('\r' + ' '*10 + "{0:.1f}%".format(100 * float(i + 1) / Nsteps) + ', Approx time left: ' + remaining_estimate + ' '*10)
			sys.stdout.flush()

			t0 = time.time()
	print #skip to a new line
	#plot_chains(sampler,'post_burn_chains',Nsteps)
	print("Mean acceptance fraction: {0:.3f}"
               .format(np.mean(sampler.acceptance_fraction)))

	return sampler

sampler = run_sampler(sampler,p)
c = sampler.chain
np.save('sampler_chain',c)
f = sampler.flatchain
np.save('sampler_flatchain',f)
p_max = f[np.argmax(sampler.flatlnprobability)]
np.save('max_likelihood_point',p_max)


#sampler = pickle.load(open('mcmc_output_combined/mcmc_sampler_result.p','rb'))
#analyze_mcmc_output(sampler,'mcmc_output.data',27,100,1000)
#plot_model(sampler)


