import operator
from itertools import product
from itertools import accumulate
import numpy as np
import random

import pickle
import os.path


from scipy.interpolate import BSpline

from sklearn import linear_model
from sklearn.linear_model import LinearRegression

from numpy.linalg import inv
from functools import reduce

from scipy.stats import norm
from scipy import integrate

import numpy as np
import operator
import pandas as pd
from sklearn import preprocessing

import operator

from .utility import *
from .AGENT import *

#######################################################################################
#######################################################################################
## 

def get_sign(s):
    if s > 0:
        return "+"
    else: return "-"

#### obtain true value for estimated optimal policy by MC repetitions

    
    
def off_policy_value(T = 120, n = 1000, n_train = 100, beta = 3/7, S_init = (0.5, 0.5), error_bound = 0.005, terminate_bound = 50, rep = 100, output_path = "./output"):
    total_N = T * n
    L = int(np.sqrt((n_train *T)**beta)) # note the number of basis should match the number of n in training period
    
    env = setting(T = T)
    a = simulation(env, n = n) # need specify n
    a.gen_buffer(S_init = None, policy = a.obs_policy )
    a.B_spline(L = max(7,(L + 3)), d = 3)
    
    error = 1
    terminate_loop = 0 
    ## if error < error_bound, it converges
    print("start updating.......")
    while error > error_bound and terminate_loop < terminate_bound:
        a._stretch_para()
        tmp = a.all_para
        a.update_op()
        a._stretch_para()
        error = np.sqrt(np.mean((tmp - a.all_para)**2))
        terminate_loop += 1
        print("current error is %.4f, error bound is %.4f" %( error, error_bound))
    print("end updating....")
    ################################
    ################################
    # need  re-specify n ###########
    ################################
    ################################
    a.n = n_train 
    ## evaluate the trained policy
    output, A_percent, _  = a.evaluate_policy(policy = a.opt_policy, seed = None, S_init = S_init, n = rep) 
    print("output: %.3f(%.2f)" %(np.mean(output), np.std(output)/(rep **0.5)))
    
    filename = 'opt_value_store_T_%d_n_%d_S_init_%s%s' %(T, n_train, get_sign(S_init[0]), get_sign(S_init[1]))
    outfile = open(os.path.join(output_path, filename),'wb')
    pickle.dump(output, outfile)
    outfile.close()


### Construct CI for True value ###

def main(seed = 1, T = 120, n = 100, T_min = 30, n_min = 25, beta = 3/7, S_init = (0.5, 0.5), error_bound = 0.01, N = 10, alpha = 0.05, terminate_bound = 15, output_path = "./output"):
    """
    input: 
        seed: random seed
        T : trajectory length
        n: sample size
        T_min: trajectory length in each block
        n_min: sample size in each block
        beta: it is used to calculate the size of bspline basis
        S_init : make inference of specifc state initial 
        error_bound: stop error bound for double fitted q learning
        terminate_bound: iteration bound for double fitted q learning
        alpha: significance level
        N: repetitions
    output:
        store inference result in filename_CI
    """
    ## obtain estimated mean
    try:
        filename = 'opt_value_store_T_%d_n_%d_S_init_%s%s' %(120, 100, get_sign(S_init[0]), get_sign(S_init[1])) ## make decision
        outfile = open(os.path.join(output_path, filename),'rb')
        output = pickle.load(outfile)
        outfile.close()
        est_mean = np.mean(output)
        mc_error = np.std(output)/np.sqrt(len(output))
    except:
        est_mean = 0
        mc_error = 0
    ## Store the CI!!
    filename_CI = 'CI_store_T_%d_n_%d_S_init_%s%s' %(T, n, get_sign(S_init[0]), get_sign(S_init[1]))
    outfile_CI = open(os.path.join(output_path, filename_CI), 'ab')
    ## use our method to get CI for the est_mean
    count = 0
    V_tilde_list = [] # store the V_tilde in N repetition
    CI_length_list = []
    ## repeat it N times:
    for i in range(N):
        np.random.seed(((1 + seed) * N + (i + 1)) * 123456)
        result_V, result_sigma = [], [] # store V and sigma in each block in one repetition
        env = setting(T = T)
        a = simulation(env, n = n)
        L = int(np.sqrt((n * T) ** beta))
        K_n = n // n_min
        K_T = T // T_min
        a.buffer_next_block(n_min, T_min, T, n = None )
        for rep in range(K_n * K_T - 1):
            a.append_next_block_to_buffer()
            a.B_spline(L = max(7,(L + 3)), d = 3)
            error = 1
            terminate_loop = 0
            while error > error_bound and terminate_loop < terminate_bound: 
                a._stretch_para()
                tmp = a.all_para
                a.update_op()
                a._stretch_para()
                error = np.sqrt(np.mean((tmp - a.all_para)**2))
                terminate_loop += 1
                print("loop %d, in k = %d ,error is %.3f" %(terminate_loop, rep, error))
            a.buffer_next_block(n_min, T_min, T, n = None )
            ## calculate sigma and V
            a._sigma(a.opt_policy, S_init, block = True) ## estimate the beta
            V = a.V(S_init, a.opt_policy)
            ## store sigma and V
            print("current index is (%d, %d), length of current buffer %d , length of first one %d, value is %.2f, sigma2 is %.2f "%(a.current_block_idx[0], a.current_block_idx[1], len(a.buffer), a.buffer[0][3], V, a.sigma2))
            result_V.append(V)
            result_sigma.append(np.sqrt(a.sigma2))
        K = len(result_sigma) + 1
        V_tilde = np.sum([result_V[i] / result_sigma[i] for i in range(K - 1)]) /  np.sum([1/ result_sigma[i] for i in range(K - 1)])
        sigma_tilde = (K - 1) / np.sum([1/ result_sigma[i] for i in range(K - 1)])
        lower_bound = V_tilde - norm.ppf(1 - alpha/2) * sigma_tilde / (n * T * (K - 1) /(K))**0.5
        upper_bound = V_tilde + norm.ppf(1 - alpha/2) * sigma_tilde / (n * T * (K - 1) /(K))**0.5
        print("sigma_tilde", sigma_tilde, "V_tilde", V_tilde, "n", n, "T", T, lower_bound, upper_bound)
        ####################################################
        pickle.dump([lower_bound,upper_bound], outfile_CI) ## store CI
        ####################################################
        V_tilde_list.append(V_tilde)
        CI_length_list.append(upper_bound - lower_bound)
        if est_mean > lower_bound and est_mean < upper_bound:
            count += 1
    outfile_CI.close()
    print("Count of covered CI over all repetition : ", count / N)
    f = open(os.path.join(output_path, "result_opt_pol_T_%d_n_%d_S_init_%s%s_(K_n_%d_K_T_%d).txt" %(T,n, get_sign(S_init[0]), get_sign(S_init[1]), K_n, K_T)), "a+")
    f.write("Count %d in %d, estimated mean: %f(MC error: %f), V_tilde_mean: %f (CI length : %f) \r\n" % (count, N, est_mean, mc_error, np.mean(V_tilde_list), np.mean(CI_length_list) ))
    f.close()
    
    

##############################################################################################################
############################################################################################################## 

"""
Below are application of OhioT1DM datasets
"""

## The source data is from  http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html
## the data is preprocessed by R code Ohio_data/data_clean.R

## impute Nan with colmean

def imputeNaN(a): 
    if len(a.shape) == 1:
        l = a.copy().reshape(-1,1)
    else:
        l = a.copy()
    col_mean = np.nanmean(l, axis=0)
    inds = np.where(np.isnan(l))
    l[inds] = np.take(col_mean, inds[1])
    return l
    
## Extract MDP components including State (S), Action (A), Reward (Y) for Ohio Data

def extract_mdp(csv_name = 'person1-5min.csv', time_interval = 36, gamma_carb = 0.9, cutoff = 0.1): # time_interval = 36 ## 12 means 1 hour, 36 means 3 hours 
    ### read data
    data = pd.read_csv(csv_name)
    ### get carb_discount
    carb = []
    discount = []
    count = 0
    last_carb = 0
    for i in range(len(data)):
        lastmeal_carb = data.iloc[i]["lastmeal_carb"]
        carb.append(lastmeal_carb)
        if not np.isnan(lastmeal_carb):
            if last_carb != lastmeal_carb:
                count = 0
                discount.append(gamma_carb ** count)
                last_carb = lastmeal_carb
            else:
                count += 1
                discount.append(gamma_carb ** count)
        else:
            discount.append(lastmeal_carb)
    carb_discount = list(map(operator.mul, carb, discount))
    ## get S,A,Y 
    rate = [] 
    glucose = []
    dose = []
    carb = []
    stability = []
    for i in range(len(data)//time_interval):
        start = i * time_interval
        end = i * time_interval + time_interval
        rate.append(np.mean(data.iloc[start : end]["rate"]))
        glucose.append(np.nanmean(data.iloc[start : end]["glucose"]))
        ## use discount-carb to get carb by discount way
        carb.append(np.nanmean(carb_discount[start : end]))
        dose.append(np.sum(data.iloc[start : end]["dose"]))
        ## get stability
        stability.append(np.nanmean(data.iloc[start : end]["stability"]))
    #### construct S,A,Y
    S = np.array((rate, carb, glucose)).T ## rate carb glucose
    #S = np.array((rate, carb)).T
    A = 1 * (np.array(dose) > cutoff) ## cutoff for effective action
    Y = np.array(stability)
    S, A, Y = imputeNaN(S), imputeNaN(A), imputeNaN(Y)
    ## scale the state! because in normcdf, un-scale will push to 1
    S = preprocessing.scale(S)
    return S, A.reshape(-1), Y.reshape(-1)


## find valid initial time point where second dimension of state (carb) is greater than 0.1

def find_valid_init(S):
    for i in range(len(S)):
        if np.abs(S[i][1]) > 0.1:
            break
    return i


## apply SAVE method to Ohio Data

def main_realdata(patient = 0, error_bound = 0.01, terminate_bound = 15, alpha = 0.05, time_interval = 36, gamma_carb = 0.9, cutoff = 0.1, product_tensor = False, Lasso = False, reward_dicount = 0.5, S_init_time = 396): 
    """
    Input:
        patient = 0 ~ 5 represent different patient
        error_bound: stop error bound for double fitted q learning
        terminate_bound: iteration bound for double fitted q learning
        alpha: significance level
        time_interval = 36 means 36 * 5 = 180 mins as one datapoint
        gamma_carb : decay rate for carb
        cutoff : > cutoff means valid action
        product_tensor: if True we use product bspline, otherwise, we use additive bslpline
        Lasso: if True, use Lasso loss in  double fitted q learning update
        reward_dicount: reward discount decay rate
        S_init_time: it means the initial time: 396 = (9 + 24) * 60 / 5 corresponds to day 1 9:00 am
    """

    S = {}
    A = {}
    Y = {}
    S[0], A[0], Y[0] = extract_mdp(csv_name = 'Ohio_data/person1-5min.csv', time_interval = time_interval, gamma_carb = gamma_carb, cutoff = cutoff)
    S[1], A[1], Y[1] = extract_mdp(csv_name = 'Ohio_data/person2-5min.csv', time_interval = time_interval, gamma_carb = gamma_carb, cutoff = cutoff)
    S[2], A[2], Y[2] = extract_mdp(csv_name = 'Ohio_data/person3-5min.csv', time_interval = time_interval, gamma_carb = gamma_carb, cutoff = cutoff)
    S[3], A[3], Y[3] = extract_mdp(csv_name = 'Ohio_data/person4-5min.csv', time_interval = time_interval, gamma_carb = gamma_carb, cutoff = cutoff)
    S[4], A[4], Y[4] = extract_mdp(csv_name = 'Ohio_data/person5-5min.csv', time_interval = time_interval, gamma_carb = gamma_carb, cutoff = cutoff)
    S[5], A[5], Y[5] = extract_mdp(csv_name = 'Ohio_data/person6-5min.csv', time_interval = time_interval, gamma_carb = gamma_carb, cutoff = cutoff)
    
    #### choose the initial point and get initial state (S_init) and observed value (true_value)
    init_time_point = int(S_init_time/time_interval) ## init_time_point means the S_init_time's corresponding 
    S_init = S[patient][init_time_point]
    discount = [ reward_dicount **(i) for i in range(len(Y[patient][init_time_point:]))]
    true_value = np.sum(list(map(operator.mul, Y[patient][init_time_point:], discount)))
    result_V, result_sigma = [], []
    ### remove the unused data for S, A, Y
    for i in range(6):
        cut_point = find_valid_init(S[i])
        S[i] = S[i][cut_point : ]
        Y[i] = Y[i][cut_point : ]
        A[i] = A[i][cut_point : ]
    T = min(S[i].shape[0] for i in range(6))
    n = 6 ## 6 patients
    beta = 3/7 ## tuning parameter for number of basis
    n_min = 6 ## ## 6 patient per block
    T_min = int(100  * 36 / time_interval) 
    env = setting(T = T, dim = 3)
    #### we can manipulate reward discount too
    a = simulation(env, n = n, product_tensor = product_tensor, reward_dicount = reward_dicount) ## control the product tensor
    #a.gamma = 0.9 ## choose longer tail?
    L = int(np.sqrt((n * T_min) ** beta))
    print("number of basis: ", L)
    K_n = n // n_min
    K_T = T // T_min
    a.buffer_next_block(n_min, T_min, T, n = None )
    ## replace the next block (simulated data) by the real data
    next_block = {}
    for i in range(6):
        next_block[i] = [ list(S[i][0:(T_min + 1)]), list(A[i][0:T_min]), list(Y[i][0:T_min]), len(list(A[i][0:T_min]))]
    a.next_block = next_block
    for rep in range(K_n * K_T - 1):
        a.append_next_block_to_buffer()
        if product_tensor:
            a.B_spline(L = 5, d = 2)
        else:
            a.B_spline(L = max(7,(L + 3)), d = 3)
        error = 1
        terminate_loop = 0
        while error > error_bound and terminate_loop < terminate_bound:
            a._stretch_para()
            tmp = a.all_para
            a.update_op(Lasso = Lasso)
            a._stretch_para()
            error = np.sqrt(np.mean((tmp - a.all_para)**2))
            terminate_loop += 1
            print("in k = %d, terminate_loop %d, error is %.3f" %(rep, terminate_loop, error))

        a.buffer_next_block(n_min, T_min, T, n = None )
        next_block = {}
        for i in range(6):
            next_block[i] = [S[i][(rep + 1) * T_min : ((rep + 1)* T_min + T_min + 1)], list(A[i][(rep + 1) * T_min : ((rep + 1)* T_min + T_min  )]), list(Y[i][(rep + 1) * T_min : ((rep + 1)* T_min + T_min )]), T_min]
        a.next_block = next_block
        a._sigma(a.opt_policy, S_init, block = True)
        V = a.V(S_init, a.opt_policy)
        ## store sigma and V
        print("current index is (%d, %d), length of current buffer %d , length of first one %d, value is %.2f, sigma2 is %.2f "%(a.current_block_idx[0], a.current_block_idx[1], len(a.buffer), a.buffer[0][3], V, a.sigma2))
        result_V.append(V)
        result_sigma.append(np.sqrt(a.sigma2))
    print("dimension of basis spline", a.para_dim)
    K = len(result_sigma) + 1
    V_tilde = np.sum([result_V[i] / result_sigma[i] for i in range(K - 1)]) /  np.sum([1/ result_sigma[i] for i in range(K - 1)])
    sigma_tilde = (K - 1) / np.sum([1/ result_sigma[i] for i in range(K - 1)])
    lower_bound = V_tilde - norm.ppf(1 - alpha/2) * sigma_tilde / (n * T * (K -1) /(K))**0.5
    upper_bound = V_tilde + norm.ppf(1 - alpha/2) * sigma_tilde / (n * T * (K -1) /(K))**0.5
    if upper_bound > true_value:
        useful = 1
    else:
        useful = 0
    f = open("Real_data_gamma_carb_%.2f_cutoff_%.2f_time_inteval_%d_Lasso_%d_product_tensor_%d_reward_dicount_%.2f_S_init_time_%d.txt" % (gamma_carb, cutoff, time_interval, int(Lasso), int(product_tensor), reward_dicount, S_init_time ), "a+")
    f.write("For patient %d, lower_bound is %.3f, upper bound is %.3f, true_value is %.3f \r\n useful : %d \r\n" % (patient, lower_bound, upper_bound, true_value, useful))
    f.close()





