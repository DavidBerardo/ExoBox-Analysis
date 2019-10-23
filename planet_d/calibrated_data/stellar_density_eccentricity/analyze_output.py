from __future__ import division
import matplotlib.pyplot as plt
import numpy as np


def analyze_mcmc_output(sampler,output_name,max_like,Nwalkers,Nburn,Nsteps):
	output = open(output_name,'w')

	output.write('########################\n')
	output.write('###### INPUT INFO ######\n')
	output.write('########################\n')
	output.write('\n')
	output.write('Nwalkers = ' + str(Nwalkers) + '\n')
	output.write('Nburn = ' + str(Nburn) + '\n')
	output.write('Nsteps = ' + str(Nsteps) + '\n')
	output.write('\n')

	s = np.shape(chain)
	f = np.reshape(chain,(s[0] * s[1],s[2]))
	#point of highest likelihood
	p_max = max_like
	output.write(','.join([str(i) for i in p_max]) + '\n')
	output.write('\n')

	#Calculate BIC
	#max_like = lnprob(p_max)
	#BIC = np.log(len(times)) * len(p_max) - 2 * np.log(max_like)
	#output.write('BIC: ' + str(BIC) + '\n')
	#output.write('\n')

	Ndim = len(f[0])

	#get median of each value
	mcmc_median_results = np.median(f, axis=0)

	#get errors for each value
	#samples = sampler.chain[:,:,:].reshape((-1, Ndim))

	errors = [np.percentile(np.array(f[:, i]), [(100 - 68.3) / 2, 50 + 68.3 / 2]) for i in range(Ndim)]

 	#calculate gelman_rubin statistic
	#Rs = gelman_rubin(sampler.chain[:, :, :],Nwalkers,Ndim)

	output.write('----Final Results----- \n')
	output.write('\n')
	#print out median value with plus/minus errors
	output.write('Parameter Name, Median Value, Err-, Err+, GR statistic\n')

	param_names = ['t0','per','a/rs','inc','rp/rs','ecosw','esinw','σ1','σ2','σ3']

	for i in range(Ndim):
		output.write(param_names[i] + ': ' + str(mcmc_median_results[i]) + 
				', ' + str(mcmc_median_results[i] - errors[i][0]) + 
				', ' + str(-mcmc_median_results[i] + errors[i][1]) +'\n')

for h in range(1,20):
	chain = np.load('sampler_chain_' + str(h) + '.npy')
	max_like = np.load('max_likelihood_point_' + str(h) + '.npy')

	analyze_mcmc_output(chain,'mcmc_output_' + str(h) + '.data',max_like,Nwalkers = 150,Nburn = 10000,Nsteps = 10000)