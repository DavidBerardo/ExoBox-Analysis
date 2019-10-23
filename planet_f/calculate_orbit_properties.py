from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

Rs = 1.343 * 9.731


for i in range(5):
	chain = np.load('sampler_chain_' + str(i+1) + '.npy')
	chain = chain[:,20000:,:]
	chain = np.asarray([i for i in chain if all(i[:,9] < -3.6)])


	#flatchain
	f = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))
	f[:,3] = f[:,3] - 2 * (f[:,3] // 90 * (f[:,3]%90))
	f[:,-1] = 10**f[:,-1]
	f[:,-2] = 10**f[:,-2]
	f[:,-3] = 10**f[:,-3]

	per = f[:,1]
	a = f[:,2]
	inc = f[:,3]
	rp = f[:,4]

	b = a * np.cos(inc*np.pi / 180.0)
	dur = per / np.pi * np.arcsin((1.0 / a) * np.sqrt((1 + rp)**2 - b**2))



	outline = 'HIP 41378f.' + str(i+1) + ' '*7
	outline += '& ' + str(int(round(np.median(per))))
	outline += '& ' + str(round(np.median(rp) * Rs,2))
	outline += '& ' + str(round(np.median(dur),2))
	print(outline)