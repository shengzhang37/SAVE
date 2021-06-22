
import operator
from itertools import product
from itertools import accumulate
import numpy as np
import random
import pickle
from scipy.interpolate import BSpline
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from numpy.linalg import inv
from functools import reduce
from scipy.stats import norm
from scipy import integrate
from .utility import *
from .AGENT import *
from tqdm import tqdm

## construct target policy
def target_policy(S):
    if S[0] > 0 and S[1] > 0:
        return 1
    else: return 0

## get the true value estimation by MC repetition

def main_get_value(T_List = [30,50,70], rep = 1000000):
    value_store = {}
    for T in T_List:
        value_store[T] = []
        n = 100
        env = setting(T = T)
        a = simulation(env)
        a.gen_buffer(total_N = 64000, S_init = None, policy = a.obs_policy )
        a.B_spline(L = 7, d = 3)
        output, A_percent, _ = a.evaluate_policy(policy = target_policy, seed = None, S_init = None, n = rep)
        est_mean = np.mean(output)
        value_store[T].append(est_mean)
        
    filename = 'value_int_store' 
    outfile = open(filename,'wb')
    pickle.dump(value_store, outfile)

    
## main function to obtain fixed initial inference

def main(seed = 1, T = 30, n = 25, N = 50, beta = 3/7, U_int_store = None):
    """
    input: 
        seed: random seed
        T : trajectory length
        n: sample size
        N: repetitions
        beta: it is used to calculate the size of bspline basis
        U_int_store = None : we use MC to get numerical integration for U 
    output:
        store inference result in filename_CI
    """
    ### CI store
    filename_CI = 'CI_store_T_%d_n_%d_S_init_int_simulation_1_2' %(T, n)
    outfile_CI = open(filename_CI, 'ab')
    #####
    total_N = T * n
    L = int(np.sqrt((n*T)**beta))
    env = setting(T = T)
    a = simulation(env)
    try:
        filename = 'value_int_store'
        outfile = open(filename,'rb')
        est_mean = pickle.load(outfile)[T][0]
        outfile.close()
    except:
        est_mean = 0.288
    count = 0
    for i in range(N):
        np.random.seed(((1 + seed) * N + (i + 1)) * 1234567)
        a.buffer = {} ## when using gen_buffer, we should empty the buffer first!!
        a.gen_buffer(total_N = total_N, S_init = None, policy = a.obs_policy ) ## generate buffer
        a.B_spline(L = max(7,(L + 3)), d = 3) ## constrcut B spline
        L = max(7,(L + 3)) - 3 ## L is the size of basis
        lower_bound, upper_bound = a.inference_int(policy = target_policy, U_int_store = U_int_store)
        pickle.dump([lower_bound, upper_bound], outfile_CI)
        if lower_bound < est_mean and est_mean < upper_bound:
            count += 1
    print(count / N)
    outfile_CI.close()
    f = open("result_T_%d_n_%d_S_integration_L_%d.txt" %(T,n, L ), "a+")
    f.write("Count %d in %d \r\n" % (count, N))
    f.close()

