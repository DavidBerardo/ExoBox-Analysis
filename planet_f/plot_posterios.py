import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize = (10,8))

posteriors = [np.load('sampler_chain_' + str(i) + '.npy') for i in range(1,15)]

longest_per = 1084
ax = fig.add_subplot(1,2,1)

for i in range(len(posteriors)):
	chain = posteriors[i]
	f = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))
	a = f[:,2]

	plt.hist(a,histtype = 'step',bins = 50, normed = True,label = str(int(longest_per / (i + 1))))

plt.xlabel('a / r_s')
plt.title('Semi major axis posteriors')
plt.legend(loc = 0,title = 'Period')
ax = fig.add_subplot(1,2,2)

for i in range(len(posteriors)):
	chain = posteriors[i]
	f = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))
	f[:,3] = f[:,3] - 2 * (f[:,3] // 90 * (f[:,3]%90))
	inc = f[:,3]
	inc = [zz for zz in inc if zz > 88]
	plt.hist(inc,histtype = 'step',normed = True)

plt.xlabel('inclination')
plt.title('Inclination posteriors')
plt.tight_layout()
plt.savefig('hip41378f_period_posterios')
plt.show()
