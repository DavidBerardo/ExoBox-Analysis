from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import batman
from random import randint
from collections import OrderedDict

times = np.linspace(3724.5,3725.7,1000)

def generate_transit(data,index1, index2,offset = 0):
	params = batman.TransitParams()

	params.per = data[index1][index2][1]
	params.rp = data[index1][index2][4]
	params.a = data[index1][index2][2]
	params.inc = data[index1][index2][3]
	params.ecc = 0
	params.w = 0
	params.limb_dark = 'quadratic'
	params.u = [data[index1][index2][5],data[index1][index2][6]]
	params.t0 = data[index1][index2][0] - 2454833 + offset
	
	m = batman.TransitModel(params,times)
	l = m.light_curve(params)

	return l

def generate_transit_from_spitzer(data,index1, index2,offset = 0):
	params = batman.TransitParams()

	params.per = 31.6978
	params.rp = np.sqrt(data[index1][index2][-5])
	params.a = data[index1][index2][-2]
	params.inc = data[index1][index2][-1]
	params.ecc = 0
	params.w = 0
	params.limb_dark = 'linear'
	params.u = [0.14]
	params.t0 = data[index1][index2][0] + 2400000.5 - 2454833 + offset

	print(params.per)
	print(params.rp)
	print(params.inc)
	print(params.t0)
	sys.exit()
	
	m = batman.TransitModel(params,times)
	l = m.light_curve(params)

	return l


#import posterior chain for c
c_data = np.load('sampler_chain_c_spitzer.npy')

#import posterio chain for d
d_data = np.load('../planet_d/sampler_chain_4.npy')

n1 = 150 - 1
n2 = 20000 - 1
count = 0
avg = np.asarray([0] * 1000)
for i in range(200):
	#compute transit signal of c 
	lightcurve_c = generate_transit_from_spitzer(c_data,randint(0,100-1),randint(0,5000-1),offset = -0/24.0) - 1
	if min(lightcurve_c) == 0 or min(lightcurve_c[500:]) == 0 or np.isnan(lightcurve_c).any():
		continue
	#compute transit signal of d
	lightcurve_d = generate_transit(d_data,randint(0,n1),20000 + randint(0,n2)) - 1
	if min(lightcurve_d) == 0:
		continue
	#compute total lightcurve
	
	lightcurve = 1 + lightcurve_c + lightcurve_d
	avg = [avg[zz] + lightcurve[zz] for zz in range(1000)]
	count += 1
	#just c
	plt.plot(times,1+0.0003 + lightcurve_c,alpha = 0.05,color = 'C1',zorder=0)
	#plot lightcurve on top of previous
	plt.plot(times,lightcurve,alpha = 0.05,color = 'C0',zorder=0)

plt.plot([0,1],[1,1],color='C1',label = 'Just planet c')
plt.plot([0,1],[1,1],color='C0',label = 'Planet c & d')

#plt.scatter(times,[zz / count for zz in avg], s = 3,color = 'C3',zorder=1)
#second c
plt.plot([3725.2755 - 5/24]*2,[0.9990,1.0001],'k--')
plt.plot([3725.2755 + 5/24]*2,[0.9990,1.0001],'k--')

#plt.plot([t0 - 1.5/24,t0 + 1.5/24],[1.0001]*2,'k')

plt.plot([3725.15]*2,[1,1+0.5 * (95 * 1e-6) / (444*1e-6)**0.], color = 'r')
plt.xlim([3724.6,3725.65])

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(),loc=0)

plt.ylabel('Relative Flux')
plt.xlabel('BJD - 2454833')

#plt.savefig('spitzer_transit_window_minus_offset.pdf')
plt.show()

