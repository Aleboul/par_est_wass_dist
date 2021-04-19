# cd C:/Users/gboul/Documents/ensae_3a/transport/projet
# python mom.py

import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from matplotlib import cm
import seaborn as sns

def MoM(obs, K):
	"""
		Calcul la médiane des moyenne formée de K blocs
	"""
	
	sample_ = np.random.choice(obs, len(obs))
	n = len(sample_)
	if n % K != 0 :
		N = ( n // K ) * K
		sample_ = sample_[0:N]
	sample_ = np.split(sample_, K)
	means_ = np.mean(sample_, axis = 1)
	median = np.median(means_)
	return median

def preprocessing(obs, K):
	"""
		Génère un échantillon formé de K blocs de médiane de moyenne
	"""
	
	sample = np.random.choice(obs, len(obs), replace = False) # on réechantillon l'array
	n = len(sample)
	if n % K != 0 :
		N = ( n // K ) * K
		sample = sample[0:N]
	sample = np.split(sample, K) # on partitionne l'array en K blocs disjoint
	MoMs_ = []
	for x in sample :
		MoMs_.append(MoM(x,10))
	MoMs = np.array(MoMs_)
	return MoMs

def generate_randomness(nobservations):
	return (np.random.normal(0,  1, nobservations))

def robservation(theta, randomness):
	normals_ = theta[0] + theta[1] * randomness
	return normals_

def contamination_huber(theta_1, theta_2, epsilon, randomness, indices = False):
	"""
		Simule une contamination à la Huber lorsque les deux lois
		sont toutes deux gaussiennes mais avec des paramètres différents
		
		Inputs
		------
		N : Nombre d'échantillon que l'on souhaite tirer
		mu_1 : espérance des inliers
		sigma_1 : matrice de covariance de inliers
		mu_2 : espérance des outliers
		sigma_2 : matrice de covariance des outliers
		epsilon : probabilité d'appartenir à l'échantillon des outliers
		
		Outputs
		-------
		X : np.array de dimension d x N
		index : indice observations contaminés
	"""
	N = len(randomness)
	binom = np.random.binomial(1, epsilon, N) # on effectue N tirs d'une loi de bernouilli
	# avec probabilité epsilon
	X = np.zeros(N) # initialisation
	index = np.where(binom == 0) 
	mask = np.ones(N, dtype = bool)
	mask[index] = False # indice des observations non contaminée
	N_ = N - np.sum(mask) # nombre d'observation non contaminée
	X[~mask] = theta_1[0] + theta_1[1] * randomness[~mask] # tirage 
	N_ = np.sum(mask) # nombre d'observation contaminée
	X[mask] = theta_2[0] + theta_1[1] * randomness[mask] # tirage
	if indices == False:
		return X
	else :
		return X, mask
    
def mewe(M,N,K,m,n,target):
	"""
		Compute the mewe estimator between the empirical measure and the a priori
		
		Inputs
		------
		M      : Number of estimators
		N      : Number of simulated sample from the a priori
		m      : Length of the sample from the a priori
		n      : Length of the sample observed
		target : List of parameters to detail for the estimation
				 - generate_randomness(nobservations)
					function that generate basic random sample that would be modified by continuous transformation
				 - observation(theta, randomness) :
					function that are really observed. In a well specified model, the function observation and robservation are identical
				 - robservation(theta, randomness)
					function that generate the data observed from the a priori measure using a parameter and the randomness from generate_randomness
				 - dist
					a function that specify how the distance is computed between the observed sample and the simulated one
				 - true_theta 
					the parameter used for generate observed data
				 - theta_dim 
					length of the parameter
		
		Outputs
		-------
		pd.DataFrame
	"""
	
	output = []
	for k in range(0,M):
		print(k)
		# Allocate space for output
		mewe_store = np.zeros((len(n),target['thetadim']))
		mewe_runtimes = np.zeros(len(n))
		mewe_evals = np.zeros(len(n))
		
		# generate all observations and sets of randomness to be used
		
		obs_rand = target['generate_randomness'](np.max(n))
		obs_all = target['observation'](target['true_theta'], false_theta, epsilon,obs_rand)
		
		# generate the synthetic randomness, sort.
		
		randomness = [target['generate_randomness'](m) for i in range(N)]
		
		for i in range(0,len(n)):
			# subset observations and sort
			obs_ = obs_all[:n[i]]
			obs = preprocessing(obs_,K) 
			sort_obs = np.sort(obs)
			sort_obs_mult = np.repeat(sort_obs, m / K, axis = 0)
			
			# Define the objective to be minimized to find the MEWE
			
			def obj1(theta):
				if(theta[1] < 0 ):
					out = 10e6
				else :
					wass_dists = [target['dist'](sort_obs_mult, np.sort(target['robservation'](theta, x))) for x in randomness]
					out = np.mean(wass_dists)
				
				return out
				
			# Optimization
			
			t_mewe = time.process_time()
			mewe = minimize(fun = obj1, x0 = true_theta)
			t_mewe = time.process_time() - t_mewe
			
			# Save the results
			mewe_store[i] = mewe.x
			mewe_runtimes[i] = t_mewe
			mewe_evals[i] = mewe.nit
		
		output_cbind = np.c_[mewe_store, mewe_runtimes, mewe_evals, n, np.arange(len(n))]
		output.append(output_cbind)
		
	return output


def mewe_K(M,N,K,m,n,target):
	"""
		Compute the mewe estimator between the empirical measure and the a priori
		
		Inputs
		------
		M      : Number of estimators
		N      : Number of simulated sample from the a priori
		m      : Length of the sample from the a priori
		n      : Length of the sample observed
		target : List of parameters to detail for the estimation
				 - generate_randomness(nobservations)
					function that generate basic random sample that would be modified by continuous transformation
				 - observation(theta, randomness) :
					function that are really observed. In a well specified model, the function observation and robservation are identical
				 - robservation(theta, randomness)
					function that generate the data observed from the a priori measure using a parameter and the randomness from generate_randomness
				 - dist
					a function that specify how the distance is computed between the observed sample and the simulated one
				 - true_theta 
					the parameter used for generate observed data
				 - theta_dim 
					length of the parameter
		
		Outputs
		-------
		pd.DataFrame
	"""
	
	output = []
	for k in range(0,M):
		print(k)
		# Allocate space for output
		mewe_store = np.zeros((len(K),target['thetadim']))
		mewe_runtimes = np.zeros(len(K))
		mewe_evals = np.zeros(len(K))
		
		# generate all observations and sets of randomness to be used
		
		obs_rand = target['generate_randomness'](np.max(n))
		obs_all = target['observation'](target['true_theta'], false_theta, epsilon,obs_rand)
		
		# generate the synthetic randomness, sort.
		
		randomness = [target['generate_randomness'](m) for i in range(N)]
		
		for i in range(0,len(K)):
			# subset observations and sort
			obs_ = obs_all[:n]
			obs = preprocessing(obs_,K[i]) 
			sort_obs = np.sort(obs)
			sort_obs_mult = np.repeat(sort_obs, m / K[i], axis = 0)
			
			# Define the objective to be minimized to find the MEWE
			
			def obj1(theta):
				if(theta[1] < 0 ):
					out = 10e6
				else :
					wass_dists = [target['dist'](sort_obs_mult, np.sort(target['robservation'](theta, x))) for x in randomness]
					out = np.mean(wass_dists)
				
				return out
				
			# Optimization
			
			t_mewe = time.process_time()
			mewe = minimize(fun = obj1, x0 = true_theta)
			t_mewe = time.process_time() - t_mewe
			
			# Save the results
			mewe_store[i] = mewe.x
			mewe_runtimes[i] = t_mewe
			mewe_evals[i] = mewe.nit
		
		output_cbind = np.c_[mewe_store, mewe_runtimes, mewe_evals, K, np.arange(len(K))]
		output.append(output_cbind)
		
	return output	
	
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

def scatter_plot_K(df, xlabel = r'$\mu$', ylabel = r'$\sigma$'):
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
	cmap = cm.get_cmap('Blues')
	
	# extract all colors form the map
	
	cmaplist = [cmap(i) for i in range(cmap.N)]
	
	# create the new map
	
	cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)
	bounds = np.linspace(K[0],K[-1],len(K))
	norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
	
	# plot the values of the estimators and the true value
	
	fig, ax = plt.subplots()
	ax.scatter(df.mu, df.sigma, c = df.K, cmap = cmap, norm = norm, s=10, alpha = 0.5)
	ax.axvline(x = true_theta[0], color = 'grey', lw = 0.5, linestyle = '-') 
	ax.axhline(y = true_theta[1], color = 'grey', lw = 0.5, linestyle = '-')
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	return fig, ax
	
def metricL1(xvec, yvec):
	return (np.mean(np.absolute(xvec - yvec)))
	
target = {}

target = {'generate_randomness' : generate_randomness, 'observation' : contamination_huber, 'robservation' : robservation}

true_theta = [0,1]
false_theta = [10,9]
target['dist'] = metricL1
target['true_theta'] = true_theta
target['thetadim'] = 2
epsilon = 0.2

M = 1000
N = 20
m = 500
n = [5000,6000,7000,8000,9000]
# n = 10000
K = 500 # prendre une valeur inférieure à 500 sinon NA car il ne peut avoir 50 blocs composé d'au moins 10 données
t = time.process_time()
dataset = mewe(M,N,K,m,n, target)
t = time.process_time() - t
print(t)
df_mewe = pd.DataFrame(np.concatenate(dataset))
df_mewe.columns = ["mu","sigma","runtime","fn.evals","n","gp"]
print(df_mewe)

fig, ax = scatter_plot(df_mewe)
plt.show()
# cmap = cm.get_cmap('OrRd')
# sns.histplot(data = df_mewe, x = "mu", hue = "n", palette = cmap)
# plt.show()
# sns.histplot(data = df_mewe, x = "sigma", hue = "n", palette = cmap)
# plt.show()

df_mewe.to_csv('df_mome.csv', index=False)