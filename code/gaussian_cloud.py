# cd C:/Users/gboul/Documents/ensae_3a/transport/projet
# python gaussian_cloud.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize

np.random.seed(11)


target = {}

def generate_randomness(nobservations):
	return (np.random.normal(0,  1, nobservations))
 
def robservation(theta, randomness):
	nobservations = np.int(len(randomness))
	dt = 1 / nobservations
	t = np.linspace(0,1, nobservations)
	W_ = np.sqrt(t) * randomness
	W_ = np.cumsum(W_)*np.sqrt(dt)
	X_ = (theta[0] - 0.5*theta[1]**2)*t + theta[1]*W_
	S_ = np.exp(X_)
	return S_
	
target = {'generate_randomness' : generate_randomness, 'robservation' : robservation}

# Commentaire 03/04/2021 : รงa marche bien.

def metricL1(xvec, yvec):
	return (np.mean(np.absolute(xvec - yvec)))
	
true_theta = [0,1]
target['thetadim'] = 2

#Pick m to be larger than or equal to max(n) and a multiple of each entry in n

# M = 1000
N = 20
m = 10**4
n = [50,100,250,500,1000,5000,10000]
M = 100
# n = [50,100,250]

import time

t = time.process_time()
def mewe_brownian(M,N,m,n):
	
	output = []
	for k in range(0,M):
		print(k)
		# Allocate space for output
		mewe_store = np.zeros((len(n),target['thetadim']))
		mewe_runtimes = np.zeros(len(n))
		mewe_evals = np.zeros(len(n))
		
		# generate all observations and sets of randomness to be used
		
		obs_rand = target['generate_randomness'](np.max(n))
		obs_all = target['robservation'](true_theta, obs_rand)
		
		# generate the synthetic randomness, sort.
		
		randomness = [target['generate_randomness'](m) for i in range(N)]
		
		for i in range(0,len(n)):
			# subset observations and sort
			obs = obs_all[:n[i]]
			sort_obs = np.sort(obs)
			sort_obs_mult = np.repeat(sort_obs, m / n[i], axis = 0)
			
			# Define the objective to be minimzed to find the MEWE
			
			def obj1(theta):
				if(theta[1] < 0 ):
					out = 10e6
				else :
					wass_dists = [metricL1(sort_obs_mult, np.sort(target['robservation'](theta, x))) for x in randomness]
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
	
dataset = mewe_brownian(M,N,m,n)
print(dataset)

t = time.process_time() - t
df_mewe = pd.DataFrame(np.concatenate(dataset))
df_mewe.columns = ["mu","sigma","runtime","fn.evals","n","gp"]
print(df_mewe)

df_mewe.to_csv('df_brownian.csv', index=False)
