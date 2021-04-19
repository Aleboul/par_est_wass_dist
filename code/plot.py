# cd C:/Users/gboul/Documents/ensae_3a/transport/projet
# python plot.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from matplotlib import cm
import seaborn as sns

df_mewe = pd.read_csv('df_cloud.csv')
true_theta = [1,1]
df_mewe['mu_scaled'] = (df_mewe.mu - true_theta[0]) * np.sqrt(df_mewe.n)
df_mewe['sigma_scaled'] = (df_mewe.sigma - true_theta[1]) * np.sqrt(df_mewe.n)
print(df_mewe)

M = 100
N = 20
m = 30
n = [5,10,15,20,25,30]

def scatter_plot(df, xlabel = r'$\mu$', ylabel = r'$\sigma$'):
	"""
		plot the scatter plot of the two estimators
		
		Inputs
		------
		df : pd.DataFrame, must have two estimators in there
		xlabel : label of x-axis
		ylabel : label of y-axis
		
		Outputs
		-------
		fig, ax
	"""
	cmap = cm.get_cmap('OrRd')
	
	# extract all colors form the map
	
	cmaplist = [cmap(i) for i in range(cmap.N)]
	
	# create the new map
	
	cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)
	bounds = np.linspace(n[0],n[-1],len(n))
	norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
	
	# plot the values of the estimators and the true value
	
	fig, ax = plt.subplots()
	ax.scatter(df.mu, df.sigma, c = df.n, cmap = cmap, norm = norm, s=10, alpha = 0.5)
	ax.axvline(x = true_theta[0], color = 'grey', lw = 0.5, linestyle = '-') 
	ax.axhline(y = true_theta[1], color = 'grey', lw = 0.5, linestyle = '-')
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	return fig, ax
	
fig, ax = scatter_plot(df_mewe)
plt.show()
	
# plot the distribution of the estimators
	
fig, ax = plt.subplots()
cmap = cm.get_cmap('OrRd')
sns.histplot(data = df_mewe, x = "mu", hue = "n", palette= cmap)
plt.show()
sns.histplot(data = df_mewe, x = "sigma", hue = "n", palette= cmap)
plt.show()