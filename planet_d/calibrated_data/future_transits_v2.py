from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time


t_cur_bjd = 2458668.15101
def calc_future(harmonic,nplus = False,other = False):

	#print '##########################'
	#print 'Period: ' + path + ' days'
	#print ' '
	if other:
		chain = np.load('../planet_f/sampler_chain_' + str(harmonic) + '.npy')
	else:
		chain = np.load('sampler_chain_' + str(harmonic) + '.npy')

	chain = chain[:,20000:,:]
	chain = np.asarray([i for i in chain if all(i[:,9] < -3.6)])

	f = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))
	f[:,3] = f[:,3] - 2 * (f[:,3] // 90 * (f[:,3]%90))

	t0 = f[:,0] 
	a = f[:,2]
	p = f[:,1]

	i = f[:,3]
	rp = f[:,4]
	b = a * np.cos(i * np.pi / 180.0)

	n = np.ceil((t_cur_bjd - np.median(t0)) / np.median(p))

	t = t0 + n * p
	once = False #so that long periods get at least one line
	count = 0
	while not(np.median(t) > 2458635.98412 and once) and (count < 4):
		#plt.scatter([np.median(t- 2454833)],[1 + 0.05 * harmonic],c = 'C' + str((harmonic- 1)%10))
		#if other:
		#	plt.plot([np.median(t-2454833)]*2,[0.99,1.01],c = 'C0',alpha = 0.3)
		#else:
		#	plt.plot([np.median(t-2454833)]*2,[0.99,1.01],c = 'C1',alpha = 0.3)

		once = True
		outline =  str(round(np.median(p),2)) + ' '

		dur = p / np.pi * np.arcsin(a**(-1) * np.sqrt((1 + rp)**2 - (b)**2))
		start = t - dur / 2.0
		end = t + dur / 2.0


		errs = np.percentile(np.array(t), [(100 - 68.3) / 2, 50 + 68.3 / 2])
		errs = [np.median(t) - errs[0],errs[1] - np.median(t)]
		sigs = min(str(errs[0]).split('.')[1].count('0'),str(errs[1]).split('.')[1].count('0'))
		sigs = 3

		outline += '& ' + str(round(np.median(t- 2454833),sigs+1)) + ' $^{+' + \
			str(round(errs[1],sigs+1)) + '}_{-' + str(round(errs[0],sigs+1)) + '}$ '



		#print 'Start of transit: '
		#print ' '
		errs = np.percentile(np.array(start), [(100 - 68.3) / 2, 50 + 68.3 / 2])
		#print np.median(start), np.median(start) - errs[0], errs[1] - np.median(start)
		med = np.median(start)
		#print Time(min(start),format = 'jd').iso
		outline += '& ' + str(Time(med,format='jd').iso).split('.')[0] + ' '
		#print Time(max(start),format = 'jd').iso

		#print '\n'

		#print 'Center of transit: '
		#print ' '

		errs = np.percentile(np.array(t), [(100 - 68.3) / 2, 50 + 68.3 / 2])
		#print np.median(t), np.median(t) - errs[0], errs[1] - np.median(t)
		med = np.median(t)
		#print Time(min(t),format = 'jd').iso
		outline += '& ' + str(Time(med,format='jd').iso).split('.')[0] + ' '
		#print Time(max(t),format = 'jd').iso

		#print '\n'

		#print 'End of transit: '
		#print ' '

		errs = np.percentile(np.array(end), [(100 - 68.3) / 2, 50 + 68.3 / 2])
		#print np.median(end), np.median(end) - errs[0], errs[1] - np.median(end)
		med = np.median(end)
		#print Time(min(end),format = 'jd').iso
		outline += '& ' + str(Time(med,format='jd').iso).split('.')[0] + ' '
		#print Time(max(end),format = 'jd').iso

		outline += '\\\\'
		#print ' '
		#print '##########################'
		#fig = plt.figure()
		#ax = fig.add_subplot()
		print(outline)
		t += p
		count += 1

for i in range(1,24):
	calc_future(i)
	print('\\hline')
plt.show()
