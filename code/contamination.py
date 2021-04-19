# cd C:/Users/gboul/Documents/ensae_3a/transport/projet
# python contamination.py

import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

###############################
## Contamination à la Huber  ##
###############################

np.random.seed(42)

target = {}

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
	X[mask] = np.sign(np.random.uniform(-1,1, N_))*(theta_2[0] + theta_1[1] * randomness[mask]) # tirage
	if indices == False:
		return X
	else :
		return X, mask

def adversarial_contamination(theta_1, theta_2, epsilon, randomness):
	"""
		Simule une contamination adverse lorsque les deux lois
		sont toutes deux gaussiennes mais avec des paramètres différents
		
		Inputs
		------
		N : Nombre d'échantillon que l'on souhaite tirer
		mu_1 : espérance des inliers
		sigma_1 : matrice de covariance de inliers
		mu_2 : espérance des outliers
		sigma_2 : matrice de covariance des outliers
		epsilon : fraction de l'échantillon à corrompre
		
		Outputs
		-------
		X : np.array de dimension d x N
		index : indice observations contaminés
	"""
	N = len(randomness)
	nb_corromp = np.int(N * epsilon)
	X = np.zeros(N)
	X = theta_1[0] + theta_1[1] * randomness
	# L'adversaire rentre dans le jeu de données et va corrompre une fraction epsilon
	index = np.argsort(np.abs(X - theta_1[0]))
	X = X[index] # on trie en fonction de la proximité à mu_1
	# on va retirer les plus petite valeurs et en ajouter des nouvelles selon une nouvelle loi
	X[0:nb_corromp] = np.sign(np.random.uniform(-1,1, nb_corromp))*(theta_2[0] + theta_1[1] * randomness[0:nb_corromp])
	X = np.random.choice(X,len(X), replace = False)
	
	return X		

def mewe(M,N,m,n,target):
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
			obs = obs_all[:n[i]]
			sort_obs = np.sort(obs)
			sort_obs_mult = np.repeat(sort_obs, m / n[i], axis = 0)
			
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
	
target = {'generate_randomness' : generate_randomness, 'observation' : contamination_huber, 'robservation' : robservation}

def metricL1(xvec, yvec):
	return (np.mean(np.absolute(xvec - yvec)))
	
true_theta = [0,1]
false_theta = [10,9]
target['dist'] = metricL1
target['true_theta'] = true_theta
target['thetadim'] = 2
epsilon = 0.2

M = 20
N = 20
m = 500
n = [50,100,250,500]

obs_rand = target['generate_randomness'](np.max(n))
obs_all = target['observation'](true_theta, false_theta, 0.2,obs_rand)
sns.histplot(obs_all)
plt.show()

# t = time.process_time()
# dataset = mewe(M,N,m,n, target)
# t = time.process_time() - t
# print(t)
# df_mewe = pd.DataFrame(np.concatenate(dataset))
# df_mewe.columns = ["mu","sigma","runtime","fn.evals","n","gp"]
# print(df_mewe)

target['contamination'] = adversarial_contamination
obs_all = target['contamination'](true_theta, false_theta, 0.2,obs_rand)
print(obs_all)
sns.histplot(obs_all)
plt.show()