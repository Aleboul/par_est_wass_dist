# cd C:/Users/gboul/Documents/ensae_3a/transport/projet
# python nwell_specified.py

import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize
from tqdm import tqdm
def mewe_misspecified(M,N,m,n,target):
	"""
	Compute the mewe estimator for misspecified models
	Inputs
	------
	M      : Number of estimators
	N      : Number of simulated sample from prior
	m      : Length of the sample from the prior
	n      : Length of the observed sample
	target : List of parameters to detail for the estimation
	- generate_randomness(nobservations)
	function that generates basic random sample that would be modified by continuous transformation
	- simulation(theta, randomness)
	function that generates data from the misspecified model, with parameters theta and randomness provided by generate_randomness
	- dist
	a function that specify how the distance is computed between the observed sample and the simulated one
	- true_theta 
	the parameter used to generate observed data
	- observed_law: distribution of observed data (Cauchy or Gamma)
	- theta_dim 
	length of the parameter
	Outputs
	-------
	pd.DataFrame
	"""
	output = []
	for k in tqdm(range(0,M)):
		# Allocate space for output
		mewe_store = np.zeros((len(n),target['thetadim']))
		mewe_runtimes = np.zeros(len(n))
		mewe_evals = np.zeros(len(n))
		
		# generate all observations and sets of randomness to be used
		
		if target["observed_law"] == "Gamma":
			obs_all = np.random.gamma(true_theta[0], true_theta[1],np.max(n))
		elif target["observed_law"] == "Cauchy":
			obs_all = np.random.standard_cauchy(np.max(n))
		else : 
			return("Not implemented law")
			break
		# la ligne du dessus est modifiée pour générer un échantillon contaminé
		
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
					wass_dists = [target['dist'](sort_obs_mult, np.sort(target['simulation'](theta, x))) for x in randomness]
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
	
np.random.seed(11)

target = {}

def generate_randomness(nobservations):
	return (np.random.normal(0,  1, nobservations))
	
def gaussian_simulation(theta, x):
	return(theta[0] + theta[1] * x)

	
target = {'generate_randomness' : generate_randomness, 'simulation' : gaussian_simulation}

def metricL1(xvec, yvec):
	return (np.mean(np.absolute(xvec - yvec)))
	
target['dist'] = metricL1

target['thetadim'] = 2
target['observed_law'] = "Gamma"
true_theta = [10,1/5]
M = 1000
N = 20
m = 10**4
n = [50,100,250,500,1000,5000,10000]
t = time.process_time()
dataset = mewe_misspecified(M,N,m,n,target)
t = time.process_time() - t
print(t)
df_mewe = pd.DataFrame(np.concatenate(dataset))
df_mewe.columns = ["mu","sigma","runtime","fn.evals","n","gp"]
print(df_mewe)

df_mewe.to_csv('df_gamma.csv', index=False)