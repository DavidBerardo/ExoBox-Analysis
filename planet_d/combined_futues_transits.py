from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time


t_cur_bjd = 2458401.22590
def calc_future(harmonic,nplus = False,other = False,t0 = -1,p = -1):

	#print '##########################'
	#print 'Period: ' + path + ' days'
	#print ' '
	#if other:
	#	chain = np.load('../planet_f/sampler_chain_' + str(harmonic) + '.npy')
	#else:
	#	chain = np.load('sampler_chain_' + str(harmonic) + '.npy')
	chain = np.load('sampler_chain_' + str(harmonic) + '.npy')
	chain = chain[:,20000:,:]
	chain = np.asarray([i for i in chain if all(i[:,9] < -3.6)])

	f = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))
	#f[:,3] = f[:,3] - 2 * (f[:,3] // 90 * (f[:,3]%90))

	t0 = f[:,0] 
	#a = f[:,2]
	p = f[:,1]

	#i = f[:,3]
	#rp = f[:,4]
	#b = a * np.cos(i * np.pi / 180.0)

	n = np.ceil((t_cur_bjd - np.median(t0)) / np.median(p))

	t = t0 + n * p
	per = np.median(p)
	#plt.plot([0,],[per]*2,'k-')
	once = True #so that long periods get at least one line
	while not(np.median(t) > 2458730.50000 and once):
		#plt.scatter([np.median(t- 2454833)],[1 + 0.05 * harmonic],c = 'C' + str((harmonic- 1)%10))
		plt.scatter([np.median(t-2454833)],per,color='C0')

		#if other:
		#	plt.plot([np.median(t-2454833)]*2,[0.99,1.01],c = 'C0',alpha = 0.3)
		#else:
		#	plt.plot([np.median(t-2454833)]*2,[0.99,1.01],c = 'C1',alpha = 0.3)
		#once = True

		t += p

#planet d
plot_color = 0
for i in range(1,20):
	calc_future(i)
	#print('\\hline')

#planet f
#for i in range(1,20):
#	calc_future(i,other = True)

t0_c = 2330.1613
per_c = 31.70714

n1 = int(np.ceil((3570 - t0_c) / per_c))
n2 = int(np.ceil((3900 - t0_c) / per_c))

for i in range(n1,n2):
	plt.plot([t0_c + per_c * i]*2, [0,380],color = 'r',alpha = 0.5)

sp_1_start = 2458520.66667 
sp_1_end = 2458570.22917

plt.fill_between([sp_1_start - 2454833,sp_1_end-2454833],0,380,alpha = 0.3,color = 'k',label = 'Spitzer')

sp_2_start = 2458691.66667
sp_2_end = 2458730.47917

plt.fill_between([sp_2_start - 2454833,sp_2_end-2454833],0,380,alpha = 0.3,color = 'k')

tess_start = 2458490.50000
tess_end = 2458516.50000
plt.fill_between([tess_start - 2454833,tess_end-2454833],0,380,alpha = 0.3,color = 'r',label = 'TESS')


plt.xlabel('BJD - 2454833')
plt.ylabel('HIP41378 D period (days)')

plt.text(3570,1.015,'Start = 10-09-2018')
#plt.text(3725,1.015,'End = 06-01-2019')

#plt.xlim([3725,3900])
#plt.plot([0],[1],c = 'C1',label = 'Planet d')
#plt.plot([0],[1],c = 'C1',label = 'Planet d')


#plt.plot([3660,3660 + 16.0 / 24],[1.0125]*2,'k--')
plt.savefig('future_transits_cdf.pdf')
plt.show()
