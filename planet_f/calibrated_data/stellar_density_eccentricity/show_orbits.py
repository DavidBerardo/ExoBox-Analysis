from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

def plot_orbit(a,e,w,style = 'single'):
	t = np.linspace(0,2*np.pi,1000)
	r = a * (1.0-e**2) / (1.0 + e * np.cos(t - w))

	if style == 'median':
		plt.plot(r * np.cos(t),r * np.sin(t),alpha = 1,c = 'C1')
	if style == 'mean':
		plt.plot(r * np.cos(t),r * np.sin(t),alpha = 1,c = 'C2')
	else:
		plt.plot(r * np.cos(t),r * np.sin(t),alpha = 0.01,c = 'C0')
	return

chain = np.load('sampler_chain.npy')

shape = np.shape(chain)
flatchain = np.reshape(chain,(shape[0] * shape[1],shape[2]))

flatchain = flatchain[::500]
for p in flatchain:
	a = p[2]
	ecosw = p[5]
	esinw = p[6]
	e = np.sqrt(ecosw**2 + esinw**2)
	w = np.arctan(esinw/ecosw)

	plot_orbit(a,e,w)


a = np.median(flatchain[:,2])
ecosw = np.median(flatchain[:,5])
esinw = np.median(flatchain[:,6])
e = np.sqrt(ecosw**2 + esinw**2)
w = np.arctan(esinw/ecosw)

plot_orbit(a,e,w,style = 'median')

a = np.mean(flatchain[:,2])
ecosw = np.mean(flatchain[:,5])
esinw = np.mean(flatchain[:,6])
e = np.sqrt(ecosw**2 + esinw**2)
w = np.arctan(esinw/ecosw)

plot_orbit(a,e,w,style = 'mean')



plt.show()