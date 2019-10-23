import numpy as np 
from random import randint
import matplotlib.pyplot as plt 

weights = [0.9769388245173086,1.0,0.9605620293479114,0.8527748404018931,0.698491894619396,0.6149058106125308,0.5900523691995945,0.5286079005180453,0.5234072714187523,0.5804138323111319,0.522033263386457,0.5154815359941507,0.4659030097122696,0.4772863154792229,0.3927166827031249,0.48078944844023697,0.4976910152660104,0.4992337881584008,0.3725458742085914]


weight_sum = np.sum(weights[:19])
new_posterior = []

num_points = len(np.load('eccentricity_posterior_1.npy'))
for i in range(19):
	chain = np.load('eccentricity_posterior_' + str(i + 1) + '.npy')
	pick = int(weights[i] * num_points / weight_sum)
	print(pick)
	for j in range(pick):
		new_posterior.append(chain[randint(0,len(chain) - 1)])

print(len(new_posterior))
m = np.median(new_posterior)
errs = np.percentile(new_posterior, [(100 - 68.3) / 2, 50 + 68.3 / 2])
print(m,m-errs[0],errs[1]-m)
plt.hist(new_posterior)
plt.show()