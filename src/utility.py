from scipy.stats import norm
import random
import numpy as np



       

"""
Given the current block index, return the corresponding position in SAVE method
For example, 
for current_block_idx = (1,2), which means the first block of samples in second time block.
it will return the start position of first block of samples and second time block as well as the remanining time length
This function is usful in updating the inference block

"""

def get_idx_pos(current_block_idx, n, T, n_min, T_min):
    K_n = n//n_min
    K_T = T//T_min
    k_n, k_T = current_block_idx
    if k_n < K_n:
        if k_T < K_T:
            return (k_n - 1)* n_min, (k_n) * n_min, T_min
        else:
            return (k_n - 1)* n_min, (k_n) * n_min, T_min + T - K_T * T_min 
    else:
        if k_T < K_T:
            return (k_n - 1)* n_min, n , T_min
        else:
            return (k_n - 1)* n_min, n , T_min + T - K_T * T_min 
   
"""
Given the current block index, return the next block index
"""
def get_next_block_idx(current_block_idx, K_n, K_T):
    n, t = current_block_idx
    if n < K_n: return [n + 1, t]
    else:
        if t < K_T:
            return [1, t + 1]
        else: return None

"""
Utility functions
"""        
class normcdf():
	def transform(self, S):
		return norm.cdf(S)

class iden():
	def transform(self, S):
		return S
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

"""
construct a gym-like class in simulation study Eq(5.21), denoted as setting class
"""

class action_space(): 
	def __init__(self, dim_action = 2):
		self.n = dim_action
	def sample(self):
		return random.randint(0, self.n - 1)

class observation_space():
	def __init__(self, dim):
		self.shape = [dim]

class setting():
	def __init__(self ,T = 48, dim = 2):
		self.action_space = action_space(dim_action = 2)
		self.observation_space = observation_space(dim)
		self.last_ob = None
		self.T = T ## length of trajectories
		self.count = 0
	def reset(self, S_init = None):
		if S_init is None:
			self.last_ob = np.random.normal(0, 1 , self.observation_space.shape[0])
		else:
			self.last_ob = S_init
		return self.last_ob 
	def seed(self, seed):
		np.random.seed(seed) 
	def step(self, A):
		if self.observation_space.shape[0] == 2:
			self.count += 1
			s1,s2 = self.last_ob
			s1_next =  3/4 * (2 * A - 1) * s1  + np.random.normal(0, 0.5, 1)
			s2_next =  3/4 * (1 - 2 * A) * s2  + np.random.normal(0, 0.5, 1)
			S_next = np.array((s1_next,s2_next)).reshape(-1)
			self.last_ob = [s1_next[0], s2_next[0]] ## update last ob
			U =  np.array(2 * s1_next  + s2_next - 1/4 * (2 * A - 1)) 
			if self.count >= self.T:
				done = True
				self.count = 0
			else:
				done = False
		else:
			self.count += 1
			if self.count >= self.T:
				done = True
				self.count = 0
			else:
				done = False
			S_next = np.random.normal(0, 1, self.observation_space.shape[0])
			U = [np.random.normal(0,1)]
		return S_next, U[0], done, None







