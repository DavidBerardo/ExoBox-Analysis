import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt as mf
import pickle
import pandas as pd
import batman
####LOAD DATA
#C5
data = pd.read_csv('../C5_detrended.csv')
t1 = np.asarray(data['BJD'])
f1 = np.asarray(data['flux'])
sigma2c5 = np.std(f1)**2

#C18
data = pd.read_csv('../C18_detrended.csv')
t2 = np.asarray(data['BJD'])
f2 = np.asarray(data['flux'])
sigma2c18 = np.std(f2)**2


def lnprob():
	return

#calculate gelman_rubin statistic
def gelman_rubin(chain,M,ndim):
	R = []
	N = len(chain[0]) #number of steps
	for i in range(ndim):
		tm = 1.0/M*np.sum([np.mean(chain[j,:,i]) for j in range(M)])
		B = N/(M-1.0)*np.sum([(np.mean(chain[j,:,i])-tm)**2 for j in range(M)])
		W = (1.0/M)*np.sum([np.sum([1.0/(N-1.0)*(np.mean(chain[j,:,i]) - chain[j,k,i])**2 for k in range(N)])for j in range(M)])
		V = ((N-1.0)/N)*W + 1.0/(N)*B
		Rcur = round((V/W)**0.5,3)
		R.append(Rcur)
	return R

def analyze_mcmc_output(sampler,output_name,Nwalkers,Nburn,Nsteps):
	output = open(output_name,'w')

	output.write('########################\n')
	output.write('###### INPUT INFO ######\n')
	output.write('########################\n')
	output.write('\n')
	output.write('Nwalkers = ' + str(Nwalkers) + '\n')
	output.write('Nburn = ' + str(Nburn) + '\n')
	output.write('Nsteps = ' + str(Nsteps) + '\n')
	output.write('\n')

	f = sampler.flatchain
	p_max = f[np.argmax(sampler.flatlnprobability)]
	output.write(','.join([str(i) for i in p_max]) + '\n')
	output.write('\n')

	Ndim = len(f[0])

	#get median of each value
	mcmc_median_results = np.median(f, axis=0)

	#get errors for each value
	samples = sampler.chain[:,:,:].reshape((-1, Ndim))
	errors = [np.percentile(np.array(samples[:, i]), [(100 - 68.3) / 2, 50 + 68.3 / 2]) for i in range(Ndim)]

	#calculate gelman_rubin statistic
	Rs = gelman_rubin(sampler.chain[:, :, :],Nwalkers,Ndim)

	output.write('----Final Results----- \n')
	output.write('\n')
	#print out median value with plus/minus errors
	output.write('Parameter Name, Median Value, Err-, Err+, GR statistic\n')
	param_names = ['t0','per','a','inc','rp'] + ['q1','q2']
	for i in range(Ndim):
		output.write(param_names[i] + ': ' + str(mcmc_median_results[i]) + 
				', ' + str(mcmc_median_results[i] - errors[i][0]) + 
				', ' + str(-mcmc_median_results[i] + errors[i][1]) + 
				', ' + str(Rs[i]) + '\n')

def generate_transit(t0,per,a,inc,rp,u,fold = False):
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
	
	if fold:
		t = np.asarray(list(t1) + list(t2))%per
		m = batman.TransitModel(params,t)
		l = m.light_curve(params)
		return l
	else:
		m1 = batman.TransitModel(params,t1)
		m2 = batman.TransitModel(params,t2)

		l1 = m1.light_curve(params)
		l2 = m2.light_curve(params)
		return l1, l2

def plot_model(sampler):
	#mcmc median results
	f = sampler.flatchain

	theta = p_max = f[np.argmax(sampler.flatlnprobability)]

	light_c5_max, light_c18_max = generate_transit(theta[5],theta[1],theta[2],theta[3],theta[4],theta[-2:])

	theta = np.median(f, axis=0)

	light_c5_med, light_c18_med = generate_transit(theta[5],theta[1],theta[2],theta[3],theta[4],theta[-2:])

	fig = plt.figure()
	plt.scatter(t1,f1,s = 2)
	plt.plot(t1,light_c5_max,c = 'C1')
	plt.plot(t1,light_c5_med,c = 'C2')
	plt.savefig('C5_lightcurve_data.pdf')

	fig = plt.figure()
	plt.scatter(t2,f2,s = 2)
	plt.plot(t2,light_c18_max,c = 'C1')
	plt.plot(t2,light_c18_med,c = 'C2')
	plt.savefig('C18_lightcurve_data.pdf')

	fig = plt.figure()
	light_fold = generate_transit(theta[0],theta[1],theta[2],theta[3],theta[4],theta[-2:],fold = True)
	plt.scatter(t2,f2,s = 2)
	plt.plot(np.asarray(list(t1) + list(t2))%theta[0],light_fold,c = 'C1')
	plt.savefig('folded_light_curve_with_fit')

if __name__ == "__main__":
	sampler = pickle.load(open('mcmc_sampler_result.p','rb'))
	analyze_mcmc_output(sampler,100,100,1000)
	plot_model(sampler)