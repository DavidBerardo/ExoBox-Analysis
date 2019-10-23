from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt as mf
import pickle
import pandas as pd
import batman
import corner

def gelman_rubin(chain):
    ssq = np.var(chain, axis=1, ddof=1)
    W = np.mean(ssq, axis=0)
    tb = np.mean(chain, axis=1)
    tbb = np.mean(tb, axis=0)
    m = chain.shape[0]
    n = chain.shape[1]
    B = n / (m - 1) * np.sum((tbb - tb)**2, axis=0)
    var_t = (n - 1) / n * W + 1 / n * B
    R = np.sqrt(var_t / W)
    return R

def analyze_mcmc_output(output_name,Nwalkers,Nburn,Nsteps,chain,f):
	output = open(output_name,'w')

	output.write('########################\n')
	output.write('###### INPUT INFO ######\n')
	output.write('########################\n')
	output.write('\n')
	output.write('Nwalkers = ' + str(Nwalkers) + '\n')
	output.write('Nburn = ' + str(Nburn) + '\n')
	output.write('Nsteps = ' + str(Nsteps) + '\n')
	output.write('\n')

	p_max = np.load('max_likelihood_point_1.npy')
	output.write(','.join([str(i) for i in p_max]) + '\n')
	output.write('\n')

	Ndim = len(f[0])

	#get median of each value
	mcmc_median_results = np.median(f, axis=0)

	#get errors for each value
	samples = chain[:,:,:].reshape((-1, Ndim))
	errors = [np.percentile(np.array(samples[:, i]), [(100 - 68.3) / 2, 50 + 68.3 / 2]) for i in range(Ndim)]

	#calculate gelman_rubin statistic
	Rs = gelman_rubin(chain)

	output.write('----Final Results----- \n')
	output.write('\n')
	#print out median value with plus/minus errors
	output.write('Parameter Name, Median Value, Err-, Err+, GR statistic\n')
	param_names = ['t0','per','a','inc','rp'] + ['q1','q2'] + ['f1','f2','f3']

	for i in range(Ndim):
		output.write('	' + param_names[i] + ': ' + str(mcmc_median_results[i]) + 
				', ' + str(mcmc_median_results[i] - errors[i][0]) + 
				', ' + str(-mcmc_median_results[i] + errors[i][1]) +
				', ' + str(Rs[i]) + '\n')


def triangle_plots(f,append=''):
	param_names = ['t0','per','a','inc','rp','q1','q2','f1','f2','f3']
	#param_names = ['t0','per','a','inc','rp','q1','q2','f1']
	fig = corner.corner(f,labels = param_names,range = [0.98]*len(f[0]))
	plt.savefig('planet_f_posterior' + append +'.png')
	plt.clf()

	return


if __name__ == "__main__":

	for i in range(1,24):
		print(i)
		chain = np.load('sampler_chain_'+str(i) + '.npy')
		chain = chain[:,20000:,:]
		chain = np.asarray([i for i in chain if all(i[:,-1] < -3.7)])
		f = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))
		f[:,3] = f[:,3] - 2 * (f[:,3] // 90 * (f[:,3]%90))
		f[:,-1] = 10**f[:,-1]
		f[:,-2] = 10**f[:,-2]
		f[:,-3] = 10**f[:,-3]

		Ndim = len(f[0])
                
		analyze_mcmc_output('mcmc_output_'+str(i)+'.data',150,20000,20000,chain,f)
		continue
		#get median of each value
		mcmc_median_results = np.median(f, axis=0)

		#get errors for each value
		samples = chain[:,:,:].reshape((-1, Ndim))
		errors = [np.percentile(np.array(samples[:, i]), [(100 - 68.3) / 2, 50 + 68.3 / 2]) for i in range(Ndim)]
		
		print(mcmc_median_results[2]**3 / mcmc_median_results[1]**2)

		for j in range(10):
			plt.figure(j)
			plt.errorbar(i,mcmc_median_results[j],[[mcmc_median_results[j] - errors[j][0]],[-mcmc_median_results[j] + errors[j][1]]],fmt = 'x')

	
	#plt.show()
