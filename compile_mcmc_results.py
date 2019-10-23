from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import re

results = open('planet_d_mcmc_results.txt','w+')

for i in range(1,24):
	file = open('planet_d/calibrated_data/mcmc_output_' + str(i) + '.data','r').readlines()

	per = re.sub(':','',file[14])
	per = re.sub(',','',per)
	print(per.split()[1:-1])
	a = re.sub(':','',file[15])
	a = re.sub(',','',a)
	print(a.split()[1:-1])

