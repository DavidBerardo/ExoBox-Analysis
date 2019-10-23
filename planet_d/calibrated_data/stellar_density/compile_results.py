from __future__ import division
import matplotlib.pyplot as plt
import numpy as np



def make_plot(ind_1,ind_2):
	for h in range(1,24):
		data = open('mcmc_output_' + str(h) + '.data','r').readlines()
		perline = data[ind_1]
		perline = perline.replace(',','')
		perline = [float(i) for i in perline.split()[1:]]
		
		aline = data[ind_2]
		aline = aline.replace(',','')
		aline = [float(i) for i in aline.split()[1:]]

		plt.scatter(perline[0],aline[0],c='k',s=1)
		plt.plot([perline[0] - perline[1],perline[0] + perline[2]],[aline[0]]*2,c='r')
		plt.plot([perline[0]]*2,[aline[0] - aline[1],aline[0] + aline[2]],c='r')

make_plot(14,15)
plt.show()