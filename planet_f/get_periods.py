from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
output = open('periods_t0s_planet_f.txt','w+')
for i in range(1,23):
	chain = np.load('sampler_chain_' + str(i) + '.npy')
	chain = chain[:,20000:,:]
	chain = np.asarray([i for i in chain if all(i[:,9] < -3.6)])

	f = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))

	t0 = np.median(f[:,0])
	p = np.median(f[:,1])

	output.write(str(t0) + ',' + str(p) + '\n')

