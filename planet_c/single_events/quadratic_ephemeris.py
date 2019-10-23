from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import os
import emcee

chains = []
for i in os.listdir('.'):
	if '.npy' in i:
		chains.append(np.load(i))

N = len(chains)

nwalk = np.shape(chains[0])[0]
nstep = np.shape(chains[0])[1]
size = nwalk * nstep
flats = []
for i in range(N):
	flats.append(np.reshape(chains[i],(size)))

w = int(round(np.sqrt(N)))
h = int(np.ceil(N/float(w)))
fig = plt.figure(figsize = (8,8*w/h))
#for i in range(N):
#	ax = fig.add_subplot(h,w,i+1)
#	plt.hist(flats[i],bins = 30)
#plt.savefig('posteriors.pdf')
#plt.clf()

t0 = 2457163.16066
t0_err =0.5 * (0.0033470103 + 0.003469955641)
per = 31.707048
per_err = 0.5 * (0.0002143703810 + 0.00023410304)

epochs = []
t0s = []
errs = []


fig = plt.figure()
for i in range(N):
	t0_cur = np.median(flats[i])
	t0s.append(t0_cur)
	cur_epoch = np.round((t0_cur - t0) / per)
	epochs.append(cur_epoch)
	err_cur = np.percentile(flats[i], [(100 - 68.3) / 2, 50 + 68.3 / 2])
	#print(t0_cur,t0_cur-err_cur[0],err_cur[1]-t0_cur)
	#t0s.append((t0 + cur_epoch * per - t0_cur) * 24 * 60)
	errs.append([(t0_cur-err_cur[0]),(err_cur[1]-t0_cur)])

###add in spitzer point                                                         
t0_cur = 57606.486 + 2400000.5
#t0_cur = 2457606.9237
t0s.append(t0_cur)
cur_epoch = np.round((t0_cur - t0) / per)
epochs.append(cur_epoch)
errs.append([(0.004),(0.004)])
print(epochs)
avg_errs = [np.mean(i) for i in errs]
N += 1
	
avg_errs = [np.mean(i) for i in errs]

def lnlike(theta, x, y, yerr):
    m1,m2, b = theta
    #print(m,x,b)
    model = [m2 * i**2 + m1 * i + b for i in x]
    like = -0.5*np.sum([(y[i]-model[i])**2 / yerr[i]**2 for i in range(len(x))])
    return like

def lnprob(theta, x, y, yerr):
    return lnlike(theta, x, y, yerr)

ndim, nwalkers = 3, 100
pos = [[per,0,t0s[0]] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(epochs, t0s, avg_errs))

sampler.run_mcmc(pos, 1000)
samples = sampler.chain[:, 200:, :].reshape((-1, ndim))
m1_mcmc,m2_mcmc, b_mcmc   = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))

print(m1_mcmc,m2_mcmc,b_mcmc)
#plt.scatter(epochs,t0s)
model = [m2_mcmc[0] * e**2 + m1_mcmc[0] * e + b_mcmc[0] for e in epochs]

residuals = [-(t0s[i] - model[i]) * 24 * 60 for i in range(len(t0s))]
print(residuals)

#m,b = np.polyfit(epochs, t0s, 1)
#print(m,b)
#model = [m * e + b for e in epochs]
#residuals = [(t0s[i] - model[i]) * 24 * 60 for i in range(len(t0s))]
#print(residuals)
#plt.errorbar(epochs,residuals)
print(errs)
errs = np.reshape(errs,(N,2))
errs = errs * 24 * 60
plt.errorbar(epochs,residuals,yerr=errs.T,c ='C1',linestyle='None',fmt = 'x')
print(epochs,residuals,errs)
plt.plot(epochs,[0] * len(epochs),'k-')
plt.show()


