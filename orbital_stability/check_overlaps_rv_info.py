import matplotlib.pyplot as plt
import numpy as np
import time
from random import randint

#check if two orbital paths will overlap
def check_orbits(a1,w1,e1,a2,w2,e2):

	#calculate the difference between the two orbits
	x = np.linspace(0,2*np.pi,100)
	y = a1 * (1.0-e1**2) / (1.0 + e1 * np.cos(x - w1)) - a2 * (1.0-e2**2) / (1.0 + e2 * np.cos(x - w2))

	#if sign doesnt change, no overlap
	if max(y)*min(y) > 0:
		return False

	#if sign changes, orbits overlap
	return True


def check_overlap(a1,e1,w1,m1,a2,e2,w2,m2):
	#mutual hill radius of planets
	rh = 0.5 * (a1 + a2) * ((m1 + m2) / M)**(1.0/3)
	delta = 3.5 * rh

	#check for overlap of one planet +/- delta
	ans1 = check_orbits(a1 + delta,w1,e1,a2,w2,e2)
	if ans1:
		return True
	ans2 = check_orbits(a1 - delta,w1,e1,a2,w2,e2)
	if ans2:
		return True
	
	else:
		return False

#solar mass in MJ
M = 1223.6

periods_d = range(1,20)
d_longest = 1113.45
trials = 10000

#output = open('planets_f_d_overlap_check.txt','w+')
output = open('results.txt','w+')
output.write('Period d, overlap percentage\n')

#load posterior data for b
chain = np.load('../planet_b/calibrated_data/stellar_density_eccentricity/sampler_chain.npy')
flat_b = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))

#load posterior data for c
chain = np.load('../planet_c/calibrated_data/stellar_density_eccentricity/sampler_chain.npy')
flat_c = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))

#load semi major axis chain for f, fixed at period max / 2
chain = np.load('../planet_f/calibrated_data/stellar_density_eccentricity/sampler_chain.npy')
flat_f = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))


def check_planet_d():
	semi_major_f = flat_f[:,2]

	for pd in periods_d:
		print(d_longest/pd)

		#load semi major axis mcmc chain for d
		chain = np.load('../planet_d/calibrated_data/stellar_density_eccentricity/sampler_chain_'+ str(pd) + '.npy')
		flat_d = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))

		counter = 0
		killers = [0] * 5
		for i in range(trials):


			#draw from mcmc distributions for d
			rd = randint(0,len(flat_d)-1)
			ad = flat_d[rd][2]
			ecosw = flat_d[rd][5]
			esinw = flat_d[rd][6]
			ed = np.sqrt(ecosw**2 + esinw**2)
			wd = np.arctan(esinw/ecosw)

			#hill radius of d
			md = 0.025 #in jupiter masses
			hd = ad * (1 - ed) * (md / 3/ M)**(1.0/3) 

			###############################################################
			#check overlap with planet b
			rb = randint(0,len(flat_b)-1)
			ab = flat_b[rb][2]
			ecosw = flat_b[rb][5]
			esinw = flat_b[rb][6]
			eb = np.sqrt(ecosw**2 + esinw**2)
			wb = np.arctan(esinw/ecosw)

			mb = 0.021 #mass in Mj from Santerne

			overlap = check_overlap(ad,ed,wd,md,ab,eb,wb,mb)
			if overlap:
				counter += 1 
				killers[0] += 1
				continue

			###############################################################
			#check overlap with planet c
			rc = randint(0,len(flat_c)-1)
			ac = flat_c[rc][2]
			ecosw = flat_c[rc][5]
			esinw = flat_c[rc][6]
			ec = np.sqrt(ecosw**2 + esinw**2)
			wc = np.arctan(esinw/ecosw)

			mc = 0.014 #mass in Mj from Santerne

			overlap = check_overlap(ad,ed,wd,md,ac,ec,wc,mc)
			if overlap:
				counter += 1 
				killers[1] += 1
				continue

			###############################################################
			#check overlap with planet g
			pg = np.random.normal(62.06,0.32)
			rs = np.random.normal(1.273,0.015)
			ms = np.random.normal(1.16,0.04)
			#calculate using kep 3rd
			ag = 4.208 * pg**(2.0/3.0) * ms**(1/3.0) *rs**(-1.0)
			eg = max(np.random.normal(0.06,0.04),0)
			wg = 0
			mg = 0

			overlap = check_overlap(ad,ed,wd,md,ag,eg,wg,mg)
			if overlap:
				counter += 1 
				killers[2] += 1
				continue

			###############################################################
			#check overlap with planet e
			pe = np.random.normal(369,9.5)
			rs = np.random.normal(1.273,0.015)
			ms = np.random.normal(1.16,0.04)
			#calculate using kep 3rd
			ae = 4.208 * pe**(2.0/3.0) * ms**(1/3.0) *rs**(-1.0)
			ee = max(np.random.normal(0.14,0.09),0)
			we = 0
			me = 0.03

			#overlap = check_overlap(ad,ed,wd,md,ae,ee,we,me)
			#if overlap:
			#	counter += 1 
			#	killers[3] += 1
			#	continue


			###############################################################
			#check overlap with planet f
			rf = randint(0,len(flat_f)-1)
			af = flat_f[rf][2]
			ecosw = flat_f[rf][5]
			esinw = flat_f[rf][6]
			ef = np.sqrt(ecosw**2 + esinw**2)
			wf = np.arctan(esinw/ecosw)

			mf = 0.04 #in jupiter masses from santerne

			overlap = check_overlap(ad,ed,wd,md,af,ef,wf,mf)
			if overlap:
				counter += 1 
				killers[4] += 1
				continue
		
		print('	' + str(round(float(counter) / trials  * 100,2)))
		#print('planet d hill radius: ' + str(np.median(h_chain_d)))
		output.write(str(d_longest / pd) + ' ' + str(round(float(counter) / trials  * 100,2)) + '\n')


check_planet_d()
