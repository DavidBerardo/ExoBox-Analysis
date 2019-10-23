from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

#load sampler chains
chain = np.load('calibrated_data/sampler_chain.npy')
chain = chain = chain[:,20000:,:]

rs = np.random.normal(1.343,0.032,len(chain) * len(chain[1]))
flat = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))
#print(np.shape(flat))


per = flat[:,1]
a = flat[:,2]
inc = flat[:,3]

a = a * rs 

M = 4 * np.pi**2 * a**3 / per**2 / 2940
#M = [i for i in M if i < 10]

plt.hist(M,bins = 50,alpha = 0.5)


plt.show()