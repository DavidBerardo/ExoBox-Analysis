from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

#load sampler chains
chain = np.load('../planet_b/calibrated_data/sampler_chain.npy')
chain = chain = chain[:,30000:,:]

rs = np.random.normal(1.343,0.032,len(chain) * len(chain[1]))
flat = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))
#print(np.shape(flat))


per = flat[:,1]
a = flat[:,2]
inc = flat[:,3] * np.pi / 180.0
rp = flat[:,4]

#plt.hist(((1 + rp)**2 - a**2 * np.cos(inc)**2),alpha = 0.5)
#plt.hist(((1 - rp)**2 - a**2 * np.cos(inc)**2),alpha = 0.5)
#plt.show()

#t14 = (per / np.pi) * np.arcsin(((1 + rp)**2 - a**2 * np.cos(inc)**2)**0.5 / (a * np.sin(inc)))
#t23 = (per / np.pi) * np.arcsin(((1 - rp)**2 - a**2 * np.cos(inc)**2)**0.5 / (a * np.sin(inc)))
#good = [i for i in range(len(t23)) if ~np.isnan(t23[i])]
#t14 = np.asarray([t14[i] for i in good])
#t23 = np.asarray([t23[i] for i in good])
#per = np.asarray([per[i] for i in good])
#rp = np.asarray([rp[i] for i in good])
#a = np.asarray([a[i] for i in good])
print('hi')
t14 = np.asarray([(per[i] / np.pi) * np.arcsin((max((1 + rp[i])**2 - a[i]**2 * np.cos(inc[i])**2,0))**0.5 / (a[i] * np.sin(inc[i]))) for i in range(len(per))])
print('hi 2')
t23 = np.asarray([(per[i] / np.pi) * np.arcsin((max((1 - rp[i])**2 - a[i]**2 * np.cos(inc[i])**2,0))**0.5 / (a[i] * np.sin(inc[i]))) for i in range(len(per))])
print('hi 3')

G = 2942 # in units of solar radius^3 / solar mass / day^2
rho_circ = (2 * rp**0.5 / (t14**2 - t23**2)**0.5)**3 * (3 * per / G / np.pi**2)
print(rho_circ)
print(min(rho_circ),max(rho_circ))
#plt.hist(rho_circ)
plt.scatter(a,rho_circ,s=1)
plt.plot([0,100],[0.134,0.134],'k-')
plt.plot([0,100],[0.134 - 0.006,0.134 - 0.006],'k--')
plt.plot([0,100],[0.134 + 0.006,0.134 + 0.006],'k--')
plt.show()