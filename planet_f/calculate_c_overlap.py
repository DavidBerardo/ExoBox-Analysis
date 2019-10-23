import matplotlib.pyplot as plt
import numpy as np
import time
from random import randint

#check if two orbital paths will overlap
def check_overlap(a1,w1,e1,a2,w2,e2):

	#calculate the difference between the two orbits
	x = np.linspace(0,2*np.pi,100)
	y = a1 * (1.0-e1**2) / (1.0 + e1 * np.cos(x - w1)) - a2 * (1.0-e2**2) / (1.0 + e2 * np.cos(x - w2))

	#if sign doesnt change, no overlap
	if max(y)*min(y) > 0:
		return False

	#if sign changes, orbits overlap
	return True

#solar mass in MJ
M = 1223.6

periods_f = range(1,20)
periods_d = range(1,20)
d_longest = 1113.45
f_longest = 1084.16
trials = 10000
#output = open('planets_f_d_overlap_check.txt','w+')
output = open('planets_f_overlap_with_c.txt','w+')
output.write('Period f, overlap percentage\n')

#load semi major axis chain for c
chain = np.load('../planet_c/sampler_chain.npy')
chain = chain[:,20000:,:]
flat = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))
semi_major_c = flat[:,2]
def check_two_planets():
	for pf in periods_f:
		print(f_longest / pf)
		#load semi major axis mcmc chain for f
		chain = np.load('sampler_chain_' + str(pf) + '.npy')
		chain = chain[:,20000:,]
		flat = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))
		semi_major_f = flat[:,2]

		h_chain = []
		#load ecc, omega mcmc chain for f
		ecc_chain_f = np.load('eccentricity_posterior_'+str(pf)+'.npy')
		w_chain_f = np.load('omega_posterior_'+str(pf)+'.npy')
		counter = 0 
		for i in range(trials):
			#draw from mcmc distributions
			r = randint(0,len(semi_major_f)-1)
			a = semi_major_f[r]
			r = randint(0,len(w_chain_f)-1)
			w = w_chain_f[r]
			e = ecc_chain_f[r]

			m = 0.02 #in jupiter masses
			h = a * (1 - e) * (m / 3/ M)**(1/3)
			h_chain.append(h / a)

			r = randint(0,len(semi_major_c)-1)
			ac = semi_major_c[r]
			ans = check_overlap(a - h,w,e,ac,0,0)

			if ans:
				counter += 1

		print('	' + str(round(float(counter) / trials  * 100,2)))
		#print('planet d hill radius: ' + str(np.median(h_chain_d) / np.median(semi_major_d)))
		output.write(str(f_longest / pf) + ' ' + str(round(float(counter) / trials  * 100,2)) + '\n')
		#print('planet d hill radius: ' +  str(np.median(h_chain_f) / np.median(semi_major_f)))
		#output.write(' \n')

check_two_planets()
