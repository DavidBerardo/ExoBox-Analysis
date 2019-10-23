from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

data = open('results.txt','r').readlines()

lines = [i.strip().split() for i in data[1:]]

pers = [float(i[0]) for i in lines]
hits = [float(i[1]) for i in lines]

plt.plot(pers,hits,label = 'without planet e')


data = open('results_with_e.txt','r').readlines()

lines = [i.strip().split() for i in data[1:]]

pers = [float(i[0]) for i in lines]
hits = [float(i[1]) for i in lines]

plt.plot(pers,hits,label = 'with planet e')


#plot periods
longest = 1113.45
for i in range(1,20):
	plt.plot([longest/i]*2,[0,100],'k--',alpha = 0.2)

plt.xlabel('Period of HIP41378 d (days)')
plt.ylabel('"Collision Metric" (% posterior overlap)')
plt.xscale('log')
plt.legend(loc = 0)
plt.savefig('planet_d_period_stability.png')
plt.show()