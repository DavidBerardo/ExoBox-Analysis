from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
####LOAD DATA
#C5
data = pd.read_csv('C5_planet_d_only.csv')
t1 = np.asarray(data['BJD'])
f1 = np.asarray(data['flux'])

#C18
data = pd.read_csv('C18_calibrated_planet_d_only.csv')
t2 = np.asarray(data['BJD'])
f2 = np.asarray(data['flux'])

#C18 short cadence
data = pd.read_csv('C18_short_cadence_calibrated_planet_d_only.csv')
t3 = np.asarray(data['BJD'])
f3 = np.asarray(data['flux'])

plt.scatter(t1,f1,s = 2)
plt.scatter(t2,f2,s = 2)
plt.scatter(t3,f3,s = 2)

plt.show()