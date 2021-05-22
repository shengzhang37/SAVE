from scipy.stats import norm
import random
import numpy as np


def get_next_block_idx(current_block_idx, K_n, K_T):
    n, t = current_block_idx
    if n < K_n: return [n + 1, t]
    else:
        if t < K_T:
            return [1, t + 1]
        else: return None
        
## given the current block index return the corresponding position

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
        
        
class normcdf():
    def transform(self, S):
        return norm.cdf(S)


class iden():
    def transform(self, S):
        return S

class cliffwalk():
    def transform(self, S):
        try:
            #return [(s/47) + np.random.normal(0, 0.001) for s in S]
            return [(s/47) for s in S]
        except:
            return S/47

class cliffwalk_noise():
    def transform(self, S):
        try:
            return [(s/47) + np.random.normal(0, 0.001) for s in S]
            #return [(s/47) for s in S]
        except:
            return S/47

    
class maxmin():
    def __init__(self):
        self.min = float("inf")
        self.max = -float("inf")
    def transform(self, S): 
        ## S is list
        self.min = min(min(S), self.min)
        self.max = max(max(S), self.max)

        #print("after transform",S, self.min, self.max, [(s - self.min)/ (self.max - self.min) for s in S])
        return [(s - self.min)/ (self.max - self.min) for s in S]

