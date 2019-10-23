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
output = open('planets_f_d_overlap_check_no_c_overlap.txt','w+')
output.write('Period f, Period d, overlap percentage\n')

#load semi major axis chain for c
chain = np.load('planet_c/sampler_chain.npy')
chain = chain[:,20000:,:]
flat = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))
semi_major_c = flat[:,2]
def check_two_planets():
	for pf in periods_f:
		print(f_longest / pf)
		#load semi major axis mcmc chain for f
		chain = np.load('planet_f/sampler_chain_' + str(pf) + '.npy')
		chain = chain[:,20000:,]
		flat = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))
		semi_major_f = flat[:,2]

		h_chain_f = []
		#load ecc, omega mcmc chain for f
		ecc_chain_f = np.load('planet_f/eccentricity_posterior_'+str(pf)+'.npy')
		w_chain_f = np.load('planet_f/omega_posterior_'+str(pf)+'.npy')
		for pd in periods_d:
			h_chain_d = []
			print(d_longest/pd)
			#load semi major axis mcmc chain for d
			chain = np.load('planet_d/sampler_chain_'+ str(pd) + '.npy')
			chain = chain[:,20000:,]
			flat = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))
			semi_major_d = flat[:,2]

			#load ecc, omega mcmc chain for d
			ecc_chain_d = np.load('planet_d/eccentricity_posterior_' + str(pd) + '.npy')
			w_chain_d = np.load('planet_d/omega_posterior_' + str(pd) + '.npy')
			counter = 0
			for i in range(trials):
				#draw from mcmc distributions for d
				r1 = randint(0,len(semi_major_d)-1)
				a1 = semi_major_d[r1]
				r2 = randint(0,len(w_chain_d)-1)
				w1 = w_chain_d[r2]
				e1 = ecc_chain_d[r2]

				m1 = 0.02 #in jupiter masses
				h1 = a1 * (1 - e1) * (m1 / 3/ M)**(1.0/3)
				h_chain_d.append(h1 / a1)

				#draw from mcmc distributions for f
				r1 = randint(0,len(semi_major_f)-1)
				a2 = semi_major_f[r1]
				r2 = randint(0,len(w_chain_f)-1)
				w2 = w_chain_f[r2]
				e2 = ecc_chain_f[r2]

				m2 = 0.02
				h2 = a2 * (1 - e2) * (m2 / 3.0/ M)**(1.0/3)
				h_chain_f.append(h2 / a2)
				#check if they overlap
				ans1 = check_overlap(a1 + h1,w1,e1,a2 - h2,w2,e2)
				#if ans1:
				#	counter += 1
				#	continue

				ans2 = check_overlap(a1 - h1,w1,e1,a2 + h2,w2,e2)
				#if ans2:
				#	counter +=1
				#	continue
				#also check overlap with c, draw from dist
				
				r3 = randint(0,len(semi_major_c)-1)
				ac = semi_major_c[r3]
				ans3 = check_overlap(a1 - h1,w1,e1,ac,0,0)
				#if ans3:
				#	counter +=1
				#	continue

				ans4 = check_overlap(a2 - h2,w2,e2,ac,0,0)
				#if ans4:
				#	counter +=1
				#	continue
				#check if it overlapped with d, but NOT with c
				if (ans1 or ans2) and not(ans3 or ans4):
					counter += 1

			print('	' + str(round(float(counter) / trials  * 100,2)))
			print('planet d hill radius: ' + str(np.median(h_chain_d)))
			output.write(str(f_longest / pf) + ', ' + str(d_longest / pd) + ' ' + str(round(float(counter) / trials  * 100,2)) + '\n')
		print('planet f hill radius: ' +  str(np.median(h_chain_f)))
		#output.write(' \n')

check_two_planets()
