from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import corner

def make_plots(chain):
	#make a corner plot
	f = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))
 
	param_names = ['t0','per','a','inc','rp','ecosw','esinw','σ1','σ2','σ3']
	fig = corner.corner(f,labels = param_names,range = [0.98]*len(f[0]))
	#plt.show()
	plt.savefig('triangle.png')
	#plt.clf()
	return 
	
def plot_chains(chain):
	fig = plt.figure(figsize = (10,10))
	nwalkers = np.shape(chain)[0]
	dim = np.shape(chain)[-1]
	steps = np.shape(chain)[1]
	w = int(round(np.sqrt(dim)))
	h = int(np.ceil(dim/float(w)))

	for i in range(dim):
		ax = fig.add_subplot(w,h,i+1)
		for j in range(nwalkers):
			plt.plot(range(steps),chain[j,:steps,i],c = 'C1',alpha = 0.5)
	plt.savefig('post_burn_in_chains.png')
	plt.close()
	return


chain = np.load('sampler_chain.npy')
print(np.shape(chain))
make_plots(chain)
#plot_chains(chain,)
print(np.shape(chain))