from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

for i in range(19):
	print(i)
	#load sampler chains
	chain = np.load('calibrated_data/sampler_chain_' + str(i+1) + '.npy')
	chain = chain = chain[:,20000:,:]

	rs = np.random.normal(1.343,0.032,len(chain) * len(chain[1]))
	flat = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))
	#print(np.shape(flat))


	per = flat[:,1]
	a = flat[:,2]

	a = a * rs 

	M = 4 * np.pi**2 * a**3 / per**2 / 2940
	M = [i for i in M if i < 10]
	#plt.scatter(np.median(per),np.median(M),c='C1')
	plt.errorbar(np.median(per),np.median(M),[[np.std(M)],[np.std(M)]],c='C1')
	#plt.scatter(np.log(np.std(per)),np.log(np.std(M)),c='C1')
	#plt.hist(M,bins = 50,alpha = 0.5)

plt.plot([0,1100],[1.168]*2,'k-')
plt.plot([0,1100],[1.168 - 0.072]*2,'r-')
plt.plot([0,1100],[1.168 + 0.072]*2,'r-')
plt.show()