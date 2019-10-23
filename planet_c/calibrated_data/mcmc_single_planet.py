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
data = pd.read_csv('C5_planet_c_only.csv')
t1 = np.asarray(data['BJD'])
f1 = np.asarray(data['flux'])

#C18
data = pd.read_csv('C18_calibrated_planet_c_only.csv')
t2 = np.asarray(data['BJD'])
f2 = np.asarray(data['flux'])

#C18 short cadence
data = pd.read_csv('C18_short_cadence_calibrated_planet_c_only.csv')
t3 = np.asarray(data['BJD'])
f3 = np.asarray(data['flux'])

t_combined = np.asarray(list(t1) + list(t2) + list(t3))
len_1 = len(t1)
len_2 = len(t2)
len_3 = len(t3)

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
	
	#m = batman.TransitModel(params,t_combined,supersample_factor = 10, exp_time = 30.0 / 60.0 / 24.0)
	m1 = batman.TransitModel(params,t1,supersample_factor = 10, exp_time = 30.0 / 60.0 / 24.0)
	#m1 = batman.TransitModel(params,t1)
	m2 = batman.TransitModel(params,t2,supersample_factor = 10, exp_time = 30.0 / 60.0 / 24.0)
	#m2 = batman.TransitModel(params,t2)
	m3 = batman.TransitModel(params,t3)
	
	l1 = m1.light_curve(params)
	l2 = m2.light_curve(params)
	l3 = m3.light_curve(params)
	
	return l1, l2, l3

#theta = t0, per, a, inc, rp, [u1,u2]
def lnlike(theta):

	light_c5, light_c18, light_c18_short_cad = generate_transit(theta[0],theta[1],theta[2],theta[3],theta[4],theta[-5:-3])
	#light_c5 = generate_transit(theta[0],theta[1],theta[2],theta[3],theta[4],theta[-3:-1])

	err1 = (10**theta[-3])**2
	err2 = (10**theta[-2])**2
	err3 = (10**theta[-1])**2

	like1 = -0.5*(np.sum((f1-light_c5)**2/err1 + np.log(err1)))
	like2 = -0.5*(np.sum((f2-light_c18)**2/err2 + np.log(err2)))
	like3 = -0.5*(np.sum((f3-light_c18_short_cad)**2/err3 + np.log(err3)))

	return like1 + like2 + like3
	#return like1

def lnprior(p):
	if abs(p[0] - 2457163.1659) > 1 or not(31.704 <p[1] < 31.708) or p[2] > 200 or p[3] > 180 or p[4] > 1:
		return -np.inf
	
	if all(i > 0 for i in p[:-3]) and (p[-5] + p[-4] < 1) and all(-6 < j < -2 for j in p[-3:]):
		prior_like = 0
		
		#gaussian prior for q1 from f fit
		mu_q1 = 0.4551
		sigma_q1 = 0.016
		q1 = p[5]
		prior_like += np.log(1.0/(np.sqrt(2*np.pi)*sigma_q1))-0.5*(q1-mu_q1)**2/sigma_q1**2

		#gaussian prior for q2 from f fit
		mu_q2 = 0.0441
		sigma_q2 = 0.03
		q2 = p[6]
		prior_like += np.log(1.0/(np.sqrt(2*np.pi)*sigma_q2))-0.5*(q2-mu_q2)**2/sigma_q2**2

		return prior_like
	else:
		return -np.inf

def lnprob(theta):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	L = lp + lnlike(theta)
	if np.isnan(L): 
		return -np.inf
	return lp + lnlike(theta)



p0 = [2457163.1659,31.7065,70,89,0.0166] + \
	 [0.3]*2 + [-4]*3

err = [0.004,0.005,10,0.5,0.005] + \
	  [0.1]*2 + [0.5]*3

nwalkers = 150
ndim = len(p0)

p = emcee.utils.sample_ball(p0,std = err, size = nwalkers)
sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,threads = 20)

def run_sampler(sampler,p0):
	Nburn = 0
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
	for i, (pos, lnp, state) in enumerate(sampler.sample(p0, iterations=Nsteps)):
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
p_max = f[np.argmax(sampler.flatlnprobability)]
np.save('max_likelihood_point',p_max)

#analyze_mcmc_output(sampler,'mcmc_output.data',50,10,10)
#plot_model(sampler)
samples = sampler.chain[:, :, :].reshape((-1, ndim))
samples[:, -1] = 10**(samples[:, -1])
samples[:, -2] = 10**(samples[:, -2])
samples[:, -3] = 10**(samples[:, -3])
results = map(lambda v: (v[1], v[1]-v[0], v[2]-v[1]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
for i in results:
	print(i)
