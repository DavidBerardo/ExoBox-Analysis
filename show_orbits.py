import matplotlib.pyplot as plt 
import numpy as np 
from random import randint

#check if two orbital paths will overlap                                                    
def check_overlap(a1,w1,e1,a2,w2,e2):
        #calculate the difference between the two orbits                                    
        x = np.linspace(0,2*np.pi,1000)
        y = a1 * (1.0-e1**2) / (1.0 + e1 * np.cos(x - w1)) - a2 * (1.0-e2**2) / (1.0 + e2 *np.cos(x - w2))
        
        #if sign doesnt change, no overlap                                                 
        if max(y)*min(y) > 0:
                return False

        #if sign changes, orbits overlap                                                   
        return True




nf = 3
chain = np.load('planet_f/sampler_chain_' + str(nf) + '.npy')
chain = chain[:,20000:,]
flat = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))
semi_major_f = flat[:,2]

nd = 5
chain = np.load('planet_d/sampler_chain_' + str(nd) + '.npy')
chain = chain[:,20000:,]
flat = np.reshape(chain,(len(chain) * len(chain[0]),len(chain[0][0])))
semi_major_d = flat[:,2]

ecc_chain_f = np.load('planet_f/eccentricity_posterior_' + str(nf) + '.npy')
w_chain_f = np.load('planet_f/omega_posterior_' + str(nf) + '.npy')

ecc_chain_d = np.load('planet_d/eccentricity_posterior_' + str(nd) + '.npy')
w_chain_d = np.load('planet_d/omega_posterior_' + str(nd) + '.npy')
mins = []
counter = 0 
for i in range(1000):
	r1 = randint(0,len(semi_major_f)-1)
	a1 = semi_major_f[r1]
	r2 = randint(0,len(ecc_chain_f)-1)
	e1 = ecc_chain_f[r2]
	w1 = w_chain_f[r2]

	r1 = randint(0,len(semi_major_d)-1)
	a2 = semi_major_d[r1]
	r2 = randint(0,len(ecc_chain_d)-1)
	e2 = ecc_chain_d[r2]
	w2 = w_chain_d[r2]
        
        
	t = np.linspace(0,2 * np.pi-0.01,100)
	if not(check_overlap(a1,w1,e1,34,0,0)):
               orb1 = 0.0062456 * a1 * (1-e1**2) / (1 + e1 * np.cos(t - w1))
               plt.plot(orb1*np.cos(t),orb1 * np.sin(t),c = 'C0',alpha = 0.1,zorder=0)
	if not(check_overlap(a2,w2,e2,34,0,0)):
               orb2 = 0.0062456 * a2 * (1-e2**2) / (1 + e2 * np.cos(t - w2))
               plt.plot(orb2*np.cos(t),orb2 * np.sin(t),c = 'C1',alpha = 0.1, zorder=0)
	#plt.plot(t,orb1-orb2,alpha = 0.2)
	#small = min(orb1 - orb2)
	#big = max(orb1 - orb2)
	#if small * big < 0:
	#	counter += 1

x = np.linspace(0,2*np.pi,1000)
plt.plot(0.2056*np.cos(x),0.2056*np.sin(x),c='k',label = 'HIP31378 c')
plt.scatter(0,0,marker='*',color='k',s = 30,zorder=1)

plt.plot(0,0,color = 'C1',label = "HIP41378 d")
plt.plot(0,0,color = 'C0', label = "HIP41378 f")

plt.arrow(1.6,0,0.2,0,color = 'k',head_width = 0.1)

plt.text(1.5,-0.2,'To Earth')
plt.xlim([-2,2])
plt.ylim([-2,2])
print(float(counter) / 1000 * 100)
plt.xlabel('x position (AU)')
plt.ylabel('y position (AU)')
plt.legend(loc = 0)
#plt.hist(mins)
plt.title('Period d = ' + str(int(round(1114 / nd))) + ' days, Period f = ' +str(int(round(1084 / nf))) + ' days')
plt.savefig('Orbital_view_d_5_f_3_no_c_cross.pdf')
plt.show()
