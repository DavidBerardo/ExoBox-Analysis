import numpy as np
import matplotlib.pyplot as plt

def plot_period(n):
	chain = np.load('sampler_chain_' + str(n) + '.npy')

	chain = chain[:,20000:,:]
	chain = np.asarray([i for i in chain if all(i[:,9] < -3.6)])

	f = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))
	f[:,3] = f[:,3] - 2 * (f[:,3] // 90 * (f[:,3]%90))

	p = f[:,1]

	m = np.median(p)
	errs = np.percentile(np.array(m), [(100 - 68.3) / 2, 50 + 68.3 / 2])
	errs = [np.median(m) - errs[0],errs[1] - np.median(m)]
	
	plt.scatter(1084/n,m)
	#plt.errorbar(1084 / n,m)

for i in range(1,20):
	print(i)
	plot_period(i)
plt.show()