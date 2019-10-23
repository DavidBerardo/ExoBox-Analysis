import numpy as np 
from random import randint
import matplotlib.pyplot as plt 

weights = [1.0,0.9238920323802421,0.7709170639313215,0.6165628153026017,0.541813414261163,0.5574319829461212,0.5888748387485149,0.3776725237066467,0.507346024608193,0.35917444757041767,0.47753368006762154,0.40166311646754976,0.38470677849569446,0.3959217966426934,0.42827684389327564,0.42070845916730226,0.42943090514606846,0.4058498641659757,0.44508547165500256]
weight_sum = np.sum(weights)
new_posterior = []

num_points = len(np.load('eccentricity_posterior_1.npy'))
for i in range(19):
	chain = np.load('eccentricity_posterior_' + str(i + 1) + '.npy')
	pick = int(weights[i] * num_points / weight_sum)
	for j in range(pick):
		new_posterior.append(chain[randint(0,len(chain) - 1)])

print(len(new_posterior))
m = np.median(new_posterior)
errs = np.percentile(new_posterior, [(100 - 68.3) / 2, 50 + 68.3 / 2])

print(m,m-errs[0],errs[1]-m)
plt.hist(new_posterior)
plt.show()