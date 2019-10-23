import numpy as np 
output = open('photoeccentric_results.txt','w+')
p_longest = 1114
def analyze(n):
	ecc_chain = np.load('eccentricity_posterior_' + str(n) + '.npy')
	omega_chain = np.load('omega_posterior_' + str(n) + '.npy')
	g = (1 + ecc_chain * np.sin(omega_chain)) / (1 - ecc_chain**2)**(0.5)
	esinw = ecc_chain * np.sin(omega_chain)

	output.write('Period = ' + str(round(p_longest / n,2)) + '\n')

	printline = str(int(p_longest / n))
	
	error = np.percentile(np.array(g), [(100 - 68.3) / 2, 50 + 68.3 / 2])
	med = round(np.median(g),3)
	err_plus, err_minus = round(error[1] - med,3), round(med - error[0],3)
	output.write('	g: ' + str(med) + ' -' + str(err_minus) + ' +' + str(err_plus) + '\n')

	printline += '  & $' + str(med) + '_{-' + str(err_minus) + '}^{+' + str(err_plus) + '}$'


	error = np.percentile(np.array(ecc_chain), [(100 - 68.3) / 2, 50 + 68.3 / 2])
	med = round(np.median(ecc_chain),3)
	err_plus, err_minus = round(error[1] - med,3), round(med - error[0],3)
	output.write('	eccentricity: ' + str(med) + ' -' + str(err_minus) + ' +' + str(err_plus) + '\n')

	printline += ' & $' + str(med) + '_{-' + str(err_minus) + '}^{+' + str(err_plus) + '}$'

	error = np.percentile(np.array(omega_chain), [(100 - 68.3) / 2, 50 + 68.3 / 2])
	med = round(np.median(omega_chain),3)
	err_plus, err_minus = round(error[1] - med,3), round(med - error[0],3)
	output.write('	omega: ' + str(med) + ' -' + str(err_minus) + ' +' + str(err_plus) + '\n')

	error = np.percentile(np.array(esinw), [(100 - 68.3) / 2, 50 + 68.3 / 2])
	med = round(np.median(esinw),3)
	err_plus, err_minus = round(error[1] - med,3), round(med - error[0],3)
	output.write('	esinw: ' + str(med) + ' -' + str(err_minus) + ' +' + str(err_plus) + '\n')
	output.write('\n')

	printline += ' & $' + str(med) + '_{-' + str(err_minus) + '}^{+' + str(err_plus) + '}$'

	printline += ' \\\\'
	print(printline)



for i in range(1,20):
	analyze(i)