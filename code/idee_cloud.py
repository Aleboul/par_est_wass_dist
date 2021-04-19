# cd C:/Users/gboul/Documents/ensae_3a/transport/projet
# python idee_cloud.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize



def generate_randomness(nobservations):
	return (np.random.multivariate_normal([0,0],  [[1,0],[0,1]], nobservations))
def robservation(theta, randomness):
	normals_ = theta + randomness
	return normals_

target = {'generate_randomness' : generate_randomness, 'robservation' : robservation}

def cost_matrix_Lp(x, y, p = 2):
	"""
		Takes two marices of size N1 times d and N2 times d, return an N1 times N2 cost matrix
		
		Inputs
		------
		x : matrix of size N1 times d
		y : matrix of size N2 times d
		p : order of norm
		
		Outputs
		-------
		np.array of size N1 times N2
	"""
	C = np.zeros((x.shape[0], y.shape[0]))
	for i in range(x.shape[0]):
		for j in range(y.shape[0]):
			C[i,j] = np.linalg.norm(x[i] - y[j], ord = p)
	
	return C

def wasserstein(p_, q_, C_, epsilon, niterations):
	"""
		Compute distance between p and q
		p corresponds to the weighs of a N-sample
		each q corresponds to the weights of a M-sample
		Thus cost_matrix must be a N x M cost matrix
		epsilon is a regularization parameter.
		
		Inputs
		------
		p_ 			: weight of a N-sample
		q_ 			: weight of a M-sample
		C_ 			: cost matrix
		epsilon 	: regularization parameter
		niterations : number of iterations
		
		Outputs
		-------
		return a distance and optimal plan
	"""
	N = len(p_)
	M = len(q_)
	
	K = np.exp(-C_ / epsilon)
	
	v = np.ones(M)
	
	for _ in range(1, niterations):
		u = p_ / K.dot(v)
		v = q_ / K.T.dot(u)
	
	transport_matrix = np.diag(u).dot(K).dot(np.diag(v))
	d = np.sum(transport_matrix * C_, axis = 0)
	output = {'distance' : d, 'transport_matrix' : transport_matrix}
	return output


def sinkhorn_distance(x1, x2, p = 1, ground_p = 2, eps = 0.05, niterations = 100):
	"""
		Estimate Wasserstein distance between two clouds
		
		Inputs
		------
		x1 : first cloud
		x2 : second cloud
		p  : order in power
		ground_p : order of cost_matrix
		eps : regularization parameter
		niterations : number of iterations in sinkhorn
		
		Outputs
		-------
		sinkhorn distance
	"""
	w1 = np.repeat(1/x1.shape[0], x1.shape[0])
	w2 = np.repeat(1/x2.shape[0], x2.shape[0])
	C = np.power(cost_matrix_Lp(x1,x2, ground_p),p)
	epsilon = eps * np.median(C)
	wass = wasserstein(w1, w2, C, epsilon, niterations)
	Phat = wass['transport_matrix']
	return np.power((np.sum(Phat * C)),1/p)
  
np.random.seed(11)

target = {}
	
target = {'generate_randomness' : generate_randomness, 'robservation' : robservation}

true_theta = [1,1]
target['dist'] = sinkhorn_distance
target['true_theta'] = true_theta
target['thetadim'] = 2


#Pick m to be larger than or equal to max(n) and a multiple of each entry in n

M = 100
N = 20
m = 30
n = [5,10,15,20,25,30]

import time

t = time.process_time()
def mewe_cloud(M,N,m,n, target):
	"""
		Compute the mewe estimator
		
		Inputs
		------
		M      : Number of estimators
		N      : Number of simulated sample from the a priori
		m      : Length of the sample from the a priori
		n      : Length of the sample observed
		target : List of parameters to detail for the estimation
				 - generate_randomness(nobservations)
					function thet generate basic random sample that would be modified by continuous transformation
				 - robservation(theta, randomness)
					function that generate the data observed using a parameter and the randomness from generate_randomness
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
		obs_all = target['robservation'](true_theta, obs_rand)
		
		# generate the synthetic randomness, sort.
		
		randomness = [target['generate_randomness'](m) for i in range(N)]
		
		for i in range(0,len(n)):
			# subset observations and sort
			obs = obs_all[:n[i]]
			# Define the objective to be minimzed to find the MEWE
			
			def obj1(theta):
				wass_dists = [target['dist'](obs, target['robservation'](theta, x)) for x in randomness]
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
	
dataset = mewe_cloud(M,N,m,n)
print(dataset)

t = time.process_time() - t
df_mewe = pd.DataFrame(np.concatenate(dataset))
df_mewe.columns = ["mu","sigma","runtime","fn.evals","n","gp"]
print(df_mewe)

df_mewe.to_csv('df_cloud.csv', index=False)