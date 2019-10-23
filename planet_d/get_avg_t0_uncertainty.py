from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import math

harmonics = [4,8,12,16,20]
harmonics = [1]

t0_uncertainty = 0
for h in harmonics:
	chain = np.load('sampler_chain_' + str(h) + '.npy')
	chain = chain[:,35000:,:]
	print(np.mean(chain[:,:,0]))
	print(np.median(chain[:,:,0]))
	chain = np.asarray([i for i in chain if all(i[:,9] < -3.6)])
	plt.hist(chain[:,:,0])
	plt.show()
	print(np.std(chain[:,:,0]) * 24 * 60)
	t0_uncertainty += 0.2 * np.std(chain[:,:,0]) * 24 * 60

print(t0_uncertainty)