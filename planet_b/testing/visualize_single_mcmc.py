from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt as mf
import pickle
import pandas as pd
import batman
import corner

####LOAD DATA
#C5
data = pd.read_csv('../../C5_detrended.csv')
t1 = np.asarray(data['BJD'])
f1 = np.asarray(data['flux'])


#C18
data = pd.read_csv('../../C18_detrended.csv')
t2 = np.asarray(data['BJD'])
f2 = np.asarray(data['flux'])

#C18 high cad
data = pd.read_csv('../../C18_short_cadence.csv')
t3 = np.asarray(data['BJD'])
f3 = np.asarray(data['flux'])


'''
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
'''

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

#sampler shape is (nwalkers,steps,dims)
def plot_chains(chain,fname):
	fig = plt.figure(figsize = (10,10))
	nwalkers = np.shape(chain)[0]
	dim = np.shape(chain)[-1]
	steps = np.shape(chain)[1]
	w = int(round(np.sqrt(dim)))
	h = int(np.ceil(dim/float(w)))

	for i in range(dim):
		ax = fig.add_subplot(w,h,i+1)
		for j in range(nwalkers):
			plt.plot(range(steps),chain[j,:steps,i],c = 'C1',alpha = 0.1)
	plt.savefig(fname + '.png')
	plt.close()
	return

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

	p_max = np.load('max_likelihood_point.npy')
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
		t = np.linspace(-per/2.0,per/2.0,10000)
		m = batman.TransitModel(params,t)
		l = m.light_curve(params)
		return l

	m1 = batman.TransitModel(params,t1)
	m2 = batman.TransitModel(params,t2)
	m3 = batman.TransitModel(params,t3)

	l1 = m1.light_curve(params)
	l2 = m2.light_curve(params)
	l3 = m3.light_curve(params)
	return l1, l2, l3

def triangle_plots(f,append=''):
	param_names = ['t0','per','a','inc','rp','q1','q2','f1','f2','f3']
	fig = corner.corner(f,labels = param_names,range = [0.95]*len(f[0]))
	plt.savefig('planet_b_posterior' + append +'.png')
	plt.clf()

	return

def plot_model(f,append = ''):

	theta = np.load('max_likelihood_point.npy')

	light_c5, light_c18, light_c18_high_cad = generate_transit(theta[0],theta[1],theta[2],theta[3],theta[4],theta[-2:])

	fig = plt.figure()
	plt.scatter(t1,f1,s = 2)
	plt.plot(t1,light_c5,c = 'C1')
	plt.savefig('C5_lightcurve_data'+append+'.pdf')
	'''
	fig = plt.figure()
	plt.scatter(t2,f2,s = 2)
	plt.plot(t2,light_c18,c = 'C1')
	plt.savefig('C18_lightcurve_data'+append+'.pdf')

	fig = plt.figure()
	plt.scatter(t3,f3,s = 2)
	plt.plot(t3,light_c18_high_cad,c = 'C1')
	plt.savefig('C18_high_cad_lightcurve_data'+append+'.pdf')
	'''
	#plt.show()


def plot_individual_transits(f,append=''):
	theta = np.load('max_likelihood_point.npy')

	y_up = [1.0002]
	y_down = [0.9995]

	lightcurve = generate_transit(0,theta[1],theta[2],theta[3],theta[4],theta[-2:],fold = True)
	p = theta[1]
	t0 = theta[0]
	
	plt.scatter(24*((t1 - t0)%p - float(p) * ((t1 - t0)%p // (p/2.0))),f1,s=4,c='C0')
	#plt.scatter(24*((t2 - t0)%p - float(p) * ((t2 - t0)%p // (p/2.0))),f2,s=4,c='C0')
	#plt.scatter(24*((t3 - t0)%p - float(p) * ((t3 - t0)%p // (p/2.0))),f3,s=4,c='C0')
	

	t = np.linspace(-p/2.0 * 24,p/2.0 * 24,10000)
	plt.plot(t,lightcurve,c='C1')
	
	plt.xlim([-20,20])
	plt.ylim([y_down[0],y_up[0]])

	plt.title('HIP 41378 b')
	plt.xlabel('Hours from midtransit')
	plt.ylabel('Relative Brightness')


	plt.savefig('folded_planet_b'+append+'.pdf')	
	return

if __name__ == "__main__":

	chain = np.load('sampler_chain.npy')
	#flatchain
	f = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))
	f[:,3] = f[:,3] - 2 * (f[:,3] // 90 * (f[:,3]%90))
	f[:,-1] = 10**f[:,-1]
	theta = np.load('max_likelihood_point.npy')

	plot_chains(chain,'post_burn_in_chains_only_c5')
	plot_model(f,'_only_c5')
	triangle_plots(f,'_only_c5')
	plot_individual_transits(f,'_only_c5')
	analyze_mcmc_output('mcmc_output_only_c5.data',100,20000,20000,chain,f)