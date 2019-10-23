from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import os

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
plt.savefig('posteriors.pdf')
plt.clf()

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
	cur_epoch = np.round((t0_cur - t0) / per)
	epochs.append(cur_epoch)
	err_cur = np.percentile(flats[i], [(100 - 68.3) / 2, 50 + 68.3 / 2])
	print(t0_cur,t0_cur-err_cur[0],err_cur[1]-t0_cur)
	t0s.append((t0 + cur_epoch * per - t0_cur) * 24 * 60)
	errs.append([(t0_cur-err_cur[0]) * 24 * 60,(err_cur[1]-t0_cur) * 24 * 60])
	
print(epochs)
errs = np.reshape(errs,(N,2))
x = np.linspace(-1,epochs[-1],1000)
y_err_plus = np.sqrt(t0_err**2 + (x*per_err)**2) * 24 * 60
y_err_minus = - np.sqrt(t0_err**2 + (x*per_err)**2) * 24 * 60
plt.plot(x,[0]*len(x),'C0')
plt.plot(x,y_err_plus,'k--')
plt.plot(x,y_err_minus,'k--')
plt.plot(x,2*y_err_plus,'r--')
plt.plot(x,2*y_err_minus,'r--')
plt.errorbar(epochs,t0s,yerr=errs.T,c ='C1',linestyle='None',fmt = 'x')
plt.xlabel('Epoch')
plt.ylabel('O - C (minutes)')
plt.title('HIP 41378 c TTV plot')
plt.savefig('O-C.pdf')

