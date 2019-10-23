from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import math

chain = np.load('sampler_chain.npy')
durations = []
print(chain[0][0])
for i in range(150):
	for j in range(20000):
		w = chain[i][j]
		p = w[1]
		a = w[2]
		inc = w[3]
		b = a * np.cos(inc)
		r = w[4]

		dur = p / np.pi  * np.arcsin(a**(-1) * np.sqrt((1 + r)**2 - b**2)) * 24
		if math.isnan(dur) or dur > 10:
			continue
		durations.append(dur)
	print(i,np.median(durations))


print(np.median(durations))
plt.hist(durations,bins = 30)
plt.show()
