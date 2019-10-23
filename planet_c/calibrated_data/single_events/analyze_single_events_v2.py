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
for i in range(N):
	ax = fig.add_subplot(h,w,i+1)
	plt.hist(flats[i],bins = 30)
plt.savefig('posteriors.png')
plt.clf()

t0 = 2457163.160883
t0_err =0.5 * (0.002684689126908779 + 0.00225356547)
per = 31.7064804
per_err = 0.5 * (0.00018840797354968686 +  0.0002426719)

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

#t0s[-1] = 2458272.75
#errs[-1] = [(0.005),(0.005)]
#print(t0s)
#print(errs)
#sys.exit()

###################################
###add in spitzer points                                                         
t0_cur = 57606.486 + 2400000.5 #davids point

#t0_cur = 2773.98582609 + 2454833 #johns point
#t0_cur = 2457606.9237 #kevins point
t0s.append(t0_cur)

cur_epoch = np.round((t0_cur - t0) / per)
epochs.append(cur_epoch)

errs.append([(0.0036),(0.0036)]) #davids errors
N += 1

###################################
#new DDT point
#t0_cur = 58557.6570 + 2400000.5 #davids point
#t0_cur = 58557.8498 + 2400000.5
t0_cur = 58557.8427 + 2400000.5
#t0_cur = 58557.468479 + 2400000.5

#t0s.append(t0_cur)

cur_epoch = np.round((t0_cur - t0) / per)
#epochs.append(cur_epoch)

#errs.append([(0.00555),(0.0024350192)])
#N += 1

##################
#new DDT point #2
#t0_cur = 58716.3056197 + 2400000.5
t0_cur = 58716.369278839 + 2400000.5

t0s.append(t0_cur)

cur_epoch = np.round((t0_cur - t0) / per)
epochs.append(cur_epoch)

#errs.append([(0.01),(0.01)])
errs.append([(0.01),(0.01)])
N += 1

#########

###################################
#maybe tess point
#t0_cur = 1494.93 + 2457000 #davids point

#t0s.append(t0_cur)

#cur_epoch = np.round((t0_cur - t0) / per)
#epochs.append(cur_epoch)

#errs.append([(0.04),(0.04)]) #davids errors
#N += 1
###################################

print(epochs)
avg_errs = [np.mean(i) for i in errs]

	
avg_errs = [np.mean(i) for i in errs]

def lnlike(theta, x, y, yerr):
    m, b = theta
    #print(m,x,b)
    model = [m * i + b for i in x]
    like = -0.5*np.sum([(y[i]-model[i])**2 / yerr[i]**2 for i in range(len(x))])
    return like

def lnprob(theta, x, y, yerr):
    return lnlike(theta, x, y, yerr)

ndim, nwalkers = 2, 100
pos = [[per,t0s[0]] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(epochs, t0s, avg_errs))

sampler.run_mcmc(pos, 500)
samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
m_mcmc, b_mcmc   = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))

print(m_mcmc,b_mcmc)
#plt.scatter(epochs,t0s)
model = [m_mcmc[0] * e + b_mcmc[0] for e in epochs]

residuals = [-(t0s[i] - model[i]) * 24 * 60 for i in range(len(t0s))]
print(residuals)

residuals = [(t0s[i] - model[i]) * 24 * 60 for i in range(len(t0s))]

outline1 = 'Observed'
outline2 = 'Calculated'
sigs = [5] * N
for i in range(len(t0s)):
	line = '$' + str(round(t0s[i] - 2454833,sigs[i])) + '^{+'+str(round(errs[i][1],sigs[i]))+ '}_{-'+str(round(errs[i][0],sigs[i])) + '}$'
	line += ' & ' +  str(round(model[i] - 2454833,sigs[i]))
	print (errs[i][1])
	line += ' & $' + str(round(residuals[i],1)) +'^{+'+str(round(errs[i][1]*24*60,1))+'}_{'+str(round(errs[i][0]*24*60,1))+'}$'
	line += ' \\\\'
	print(line)

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
plt.savefig('O-C.png')
#plt.show()


