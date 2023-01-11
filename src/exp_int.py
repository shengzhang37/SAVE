
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

def main_get_value(T_List = [30], rep = 1000000, output_path = "./output"):
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
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    filename = 'value_int_store' 
    outfile = open(os.path.join(output_path, filename),'wb')
    pickle.dump(value_store, outfile)

    
## main function

def main(seed = 1, T = 30, n = 25, N = 50, beta = 3/7, U_int_store = None, output_path = "./output",):
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
    outfile_CI = open(os.path.join(output_path, filename_CI), 'ab')
    #####
    total_N = T * n
    L = int(np.sqrt((n*T)**beta))
    env = setting(T = T)
    a = simulation(env)
    try:
        filename = 'value_int_store'
        outfile = open(os.path.join(output_path, filename),'rb')
        est_mean = pickle.load(outfile)[T][0]
        outfile.close()
    except:
        est_mean = 0.288
    count = 0
    for i in range(N):
        np.random.seed(((1 + seed) * N + (i + 1)) * 1234567)
        a.buffer = {} ## when using gen_buffer, we should empty the buffer first!!
        a.gen_buffer(total_N = total_N, S_init = None, policy = a.obs_policy )
        a.B_spline(L = max(7,(L + 3)), d = 3)
        L = max(7,(L + 3)) - 3 ## L can not be 3 .. should be at least 4
        lower_bound, upper_bound = a.inference_int(policy = target_policy, U_int_store = U_int_store)
        pickle.dump([lower_bound, upper_bound], outfile_CI)
        if lower_bound < est_mean and est_mean < upper_bound:
            count += 1
    print("Count of covered CI over all repetition : ", count / N)
    outfile_CI.close()
    f = open(os.path.join(output_path, "result_T_%d_n_%d_S_integration_L_%d.txt" %(T,n, L )), "a+")
    f.write("Count %d in %d \r\n" % (count, N))
    f.close()

##############################################################################################################
############################################################################################################## 
 
"""
Below are implement about DR Method (Doubly robust off-policy value evaluation for reinforcement learning)
"""



#V^H = Vhat + rho(r + gamma V^{H-1} - Q)
#V^H - 1 = Vhat + rho(r + gamma V^{H-2} - Q)
#V^H - 2 = Vhat + rho(r + gamma V^{H-3} - Q)
#.
#.
#.
#V^1 = Vhat + rho(r_H + gamma V^{0} - Q_H)

def train_target_policy(T = 30, n = 25, policy = target_policy):

    total_N = T * n
    beta = 3/7
    L = int(np.sqrt((n*T)**beta))
    env = setting(T = T)
    a = simulation(env)
    a.gen_buffer(total_N = total_N, S_init = None, policy = a.obs_policy )
    a.B_spline(L = max(7,(L + 3)), d = 3)
    ### estimate beta
    a._beta_hat(policy)
    a._store_para(a.est_beta)
    ### pickle the trained model
    filename_train = 'train_target_policy_with_T_%d_n_%d' %(T, n)
    outfile_train = open(filename_train, 'wb')
    pickle.dump({'scaler' : a.scaler, 'bspline' : a.bspline, 'para' : a.para}, outfile_train) ## get the model which can obtain Q function for target policy
    outfile_train.close()

#train_target_policy(T = 140, n = 1000)




   
def main_DR(seed = 1, T = 30, n = 25,  alpha = 0.05, policy = target_policy, filename_train = 'train_target_policy_with_T_140_n_1000'):
    ### CI store
    filename_CI = 'CI_store_T_%d_n_%d_S_init_int_DR' % (T, n)
    outfile_CI = open(filename_CI, 'ab')
    
    ## get true value
    filename = 'value_int_store'
    outfile = open(filename,'rb')
    est_mean = pickle.load(outfile)[T][0]
    outfile.close()
    
    count = 0
    N = 50
    for i in range(N):
        np.random.seed(((1 + seed) * N + (i + 1)) * 1234567) 
        V_output = V_DR(T = T, n = n, policy = target_policy, filename_train = filename_train ) ## obtain value estimation
        lower_bound = np.mean(V_output) - norm.ppf(1 - alpha/2) * np.std(V_output)/(n**0.5)
        upper_bound = np.mean(V_output) + norm.ppf(1 - alpha/2) * np.std(V_output)/(n**0.5)
        CI_length = 2 * norm.ppf(1 - alpha/2) * np.std(V_output)/(n**0.5)
        
        pickle.dump([lower_bound,upper_bound], outfile_CI)
        if lower_bound < est_mean and est_mean < upper_bound:
            count += 1
            
        print("Lower bound %.3f and upper bound %.3f for true mean %.3f" %(lower_bound,upper_bound, est_mean))
    outfile_CI.close()
    print(count / N)
    f = open("result_T_%d_n_%d_S_integration_DR.txt" %(T,n), "a+")
    f.write("Count %d in %d \r\n" % (count, N))
    f.close()
    
def V_DR(T, n, policy = target_policy, filename_train = 'train_target_policy_with_T_140_n_1000'):
    ### filename_train 就是只用当前的数据（就是切一半）来算Q 和 V 值
    if filename_train is None:
        ## obtain trained model
        ### use half data to train the model and get Q function store in objective b
        total_N = T * (n // 2)
        beta = 3/7
        L = int(np.sqrt((n*T)**beta))

        ## generate data and basis spline
        env = setting(T = T)
        b = simulation(env)

        b.gen_buffer(total_N = total_N, S_init = None, policy = b.obs_policy )
        b.B_spline(L = max(7,(L + 3)), d = 3)
        ### estimate beta
        b._beta_hat(policy)
        b._store_para(b.est_beta)
        
        ## use the rest data to get V output 
        
        total_N = T * n - total_N
        beta = 3/7
        L = int(np.sqrt((n*T)**beta))
        env = setting(T = T)
        a = simulation(env)
        a.gen_buffer(total_N = total_N, S_init = None, policy = a.obs_policy )
        ## unpack the trained model
        a.scaler = b.scaler
        a.bspline = b.bspline
        a.para = b.para

        ## 
        V_output = []
        for traj in a.buffer.values():
            S, A, U, T = traj
            dp = [0] * (T + 1)
            for i in reversed(range(T)):
                Q_hat = a.Q(S[i], A[i]) ## 用trained model 来算Q和V
                r = U[i]
                V_hat = a.V(S[i], policy)
                if policy(S[i]) == A[i]:
                    rho = 0.5
                else:
                    rho = 0
                dp[i] = V_hat + rho * (r + a.gamma * dp[i + 1] - Q_hat)
            V_output.append(dp[0])
        return V_output
    
    ## filename_train 有的话 就是可以用（其他数据）train 好的模型来得到Q，然后来算V
    else:
        outfile_train = open(filename_train, 'rb')
        output  =  pickle.load(outfile_train)
        outfile_train.close()

        total_N = T * n
        beta = 3/7
        L = int(np.sqrt((n*T)**beta))
        env = setting(T = T)
        a = simulation(env)
        a.gen_buffer(total_N = total_N, S_init = None, policy = a.obs_policy )
        ## unpack the trained model
        a.scaler = output['scaler']
        a.bspline = output['bspline']
        a.para = output['para']

        ## 
        V_output = []
        for traj in a.buffer.values():
            S, A, U, T = traj
            dp = [0] * (T + 1)
            for i in reversed(range(T)):
                Q_hat = a.Q(S[i], A[i]) ## 用trained model 来算Q和V
                r = U[i]
                V_hat = a.V(S[i], policy)
                if policy(S[i]) == A[i]:
                    rho = 0.5
                else:
                    rho = 0
                dp[i] = V_hat + rho * (r + a.gamma * dp[i + 1] - Q_hat)
            V_output.append(dp[0])
        return V_output



        