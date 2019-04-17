import matplotlib.pyplot as plt
import matplotlib
import csv
import pandas as pd
import numpy as np


matplotlib.style.use('ggplot')

bayes_opt_data_power = []
bayes_opt_data_velocity = []


for i in range(0, 12):
	df = pd.read_csv('./bayes_opt_data/bayes_opt_data_'+str(i)+'.csv', header=None, skiprows=1)
	a = df.values
	a = np.amax(a, axis=0)
	bayes_opt_data_power.append(a[0])
	bayes_opt_data_velocity.append(a[1])


# print type(bayes_opt_data_power)
plt.plot(bayes_opt_data_velocity, bayes_opt_data_power, 'ro')
plt.show()
