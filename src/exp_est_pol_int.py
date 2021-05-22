
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
    
#######################################################################################
#######################################################################################

def train_opt_policy(T = 30, n = 25, n_train = 25, error_bound = 0.005, terminate_bound = 50):

    total_N = T * n
    beta = 3/7
    L = int(np.sqrt((n_train * T)**beta))

    ## generate data and basis spline
    env = setting(T = T)
    
    a = simulation(env)

    a.gen_buffer(total_N = total_N, S_init = None, policy = a.obs_policy )
    a.B_spline(L = max(7,(L + 3)), d = 3)
    ### estimate beta
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
    
    ### pickle the trained model
    filename_train = 'train_opt_policy_with_T_%d_n_%d' %(T, n)
    outfile_train = open(filename_train, 'wb')
    pickle.dump({'scaler' : a.scaler, 'bspline' : a.bspline, 'para' : a.para}, outfile_train)
    outfile_train.close()

    
def store_data_set(L_list = [4,5,6,7], total_N = 12800):
    output = {}
    
    for L in L_list : 
        env = setting(T = 32)
        a = simulation(env)
        a.gen_buffer(total_N = total_N, S_init = None, policy = a.obs_policy )
        a.B_spline(L = (L + 3), d = 3)
        output[str(L)] = {'buffer' : a.buffer, 'bspline' : a.bspline, 'para' : a.para, 'para_dim' : a.para_dim}
    ### pickle the trained model
    filename_data_set = 'store_data_set_total_N_%d' %(total_N)
    outfile_data_set = open(filename_data_set, 'wb')
    pickle.dump(output, outfile_data_set)
    outfile_data_set.close()
    
## get integrated U which is corresponding to the basis in (L_list)**2
def main_get_U(L = 4, filename_train = 'train_opt_policy_with_T_100_n_50', total_N = 64000):
    
    ## obtain trained model
    outfile_train = open(filename_train, 'rb')
    output  =  pickle.load(outfile_train)
    outfile_train.close()
    ### obtain the pretrained optimal policy
    T = 30
    n = 50
    total_N = T * n
    beta = 3/7
    L = int(np.sqrt((n*T)**beta))
    env = setting(T = T)
    ## store in b
    b = simulation(env)
    b.gen_buffer(total_N = total_N, S_init = None, policy = b.obs_policy )
    ## unpack the trained model
    b.scaler = output['scaler']
    b.bspline = output['bspline']
    b.para = output['para']
    ########################################
    
    env = setting(T = 32)
    a = simulation(env)
    a.gen_buffer(total_N = total_N, S_init = None, policy = a.obs_policy )
    store_U = {}
    
    store_U[L] = []
    a.B_spline(L = (L + 3), d = 3)
    for i in range(a.para_dim*2):
        print("integrating para %d"% i)
        f = lambda y,x : a._U(policy = b.opt_policy, S = (x,y))[i] * norm.pdf(y) * norm.pdf(x) 
        #store_U[L].append(integrate.dblquad(f, np.NINF, np.Inf, lambda x: np.NINF, lambda x: np.Inf)[0])
        store_U[L].append(integrate.dblquad(f, -5, 5, lambda x: np.NINF, lambda x: np.Inf)[0])
        
    print(store_U[L])
    filename = 'U_int_store_num_basis_%d_total_N_%d' % (L, total_N) ## 这个是对optimal policy来算的U int
    outfile = open(filename, 'wb')
    pickle.dump(store_U, outfile)
    outfile.close()
    
    
def main_get_U_distributed(L = 4, para_dim = 0, filename_train = 'train_opt_policy_with_T_60_n_500', total_N = 12800):
    if filename_train is None:
        if L == 4: 
            filename_train = 'train_opt_policy_with_T_30_n_500'
        elif L == 5:
            filename_train = 'train_opt_policy_with_T_60_n_500'
        elif L == 6:
            filename_train = 'train_opt_policy_with_T_100_n_500'
    ## obtain trained model
    outfile_train = open(filename_train, 'rb')
    output_train  =  pickle.load(outfile_train)
    outfile_train.close()
    ### obtain the pretrained optimal policy
    T = 30
    n = 50
    #beta = 3/7
    #L = int(np.sqrt((n*T)**beta))
    env = setting(T = T)
    ## store in b
    b = simulation(env)
    b.gen_buffer(total_N = T * n, S_init = None, policy = b.obs_policy )
    ## unpack the trained model
    b.scaler = output_train['scaler']
    b.bspline = output_train['bspline']
    b.para = output_train['para']
    #####################################################
    #### obtain stored data set ###################
    #####################################################
    
    filename_data_set = 'store_data_set_total_N_%d' % total_N
    ## obtain dataset
    outfile_data_set = open(filename_data_set, 'rb')
    output_data_set  =  pickle.load(outfile_data_set)
    outfile_data_set.close()
    ######
    env = setting(T = 32)
    a = simulation(env)

    a.buffer = output_data_set[str(L)]['buffer']
    a.bspline = output_data_set[str(L)]['bspline']
    a.para = output_data_set[str(L)]['para']
    a.para_dim = output_data_set[str(L)]['para_dim']


    a.B_spline(L = (L + 3), d = 3)
    ### para_dim = 0,1,2..., L^2 - 1
    i = para_dim
    print("integrating para %d"% i)
    f = lambda y,x : a._U(policy = b.opt_policy, S = (x,y))[i] * norm.pdf(y) * norm.pdf(x) 
    #print(b.para)
    result = integrate.dblquad(f, np.NINF, np.Inf, lambda x: np.NINF, lambda x: np.Inf)[0]
    #result = integrate.dblquad(f, -5, 5, lambda x: -5, lambda x: 5)[0]
    filename = 'U_int_store_num_basis_%d_para_dim_%d_total_N_%d' % (L, para_dim, total_N) ## 这个是对optimal policy来算的U int
    outfile = open(filename, 'wb')
    pickle.dump(result, outfile)
    outfile.close()
    
## get integrated value which is corresponding to the time points in (T_list)
    
def main_get_value(T_List = [30,50,70], rep = 100000, filename_train = 'train_opt_policy_with_T_60_n_500'):
    ## obtain trained model
    outfile_train = open(filename_train, 'rb')
    output  =  pickle.load(outfile_train)
    outfile_train.close()
    ### obtain the pretrained optimal policy
    T = 30
    n = 50
    total_N = T * n
    beta = 3/7
    L = int(np.sqrt((n*T)**beta))
    env = setting(T = T)
    ## store in b
    b = simulation(env)
    b.gen_buffer(total_N = total_N, S_init = None, policy = b.obs_policy )
    ## unpack the trained model
    b.scaler = output['scaler']
    b.bspline = output['bspline']
    b.para = output['para']
    ################################
    
    value_store = {}
    
    for T in T_List:
        value_store[T] = []
        n = 100
        env = setting(T = T)
        a = simulation(env)
        a.gen_buffer(total_N = 64000, S_init = None, policy = a.obs_policy )
        a.B_spline(L = 7, d = 3)
        output, A_percent, _ = a.evaluate_policy(policy = b.opt_policy, seed = None, S_init = None, n = rep)
        est_mean = np.mean(output)
        value_store[T].append(est_mean)
        
    filename = 'value_int_store_opt' 
    outfile = open(filename,'wb')
    pickle.dump(value_store, outfile)
    outfile.close()

    
##################################################################
##############   main function  ##################################
##################################################################

def main(seed = 1, T = 60, n = 25, T_min = 30, n_min = 25, beta = 3/7, error_bound = 0.01, N = 10, alpha = 0.05, terminate_bound = 30, U_int_store = None, MC_N = 10000, fitted_Q = False):
    
    ### CI store
    if MC_N is None:
        exp_MC_N = 0
    else:
        exp_MC_N = MC_N
    filename_CI = 'CI_store_T_%d_n_%d_S_init_int_simulation_2_2_%d' %(T, n, exp_MC_N)
    #filename_CI = 'CI_store_T_%d_n_%d_S_init_int_simulation_2_2' %(T, n)
    outfile_CI = open(filename_CI, 'ab')
    
    ## determine the true value
    try:
        filename = 'value_int_store_opt'
        outfile = open(filename,'rb')
        est_mean = pickle.load(outfile)[50][0] ## 在 tocluster里面我只跑了T = 50的情况
        outfile.close()
    except:
        est_mean = 0


    ## use our method to get CI for the est_mean
    count = 0
    V_tilde_list = [] # store the V_tilde in N repetition
    CI_length_list = []

    for i in range(N):
        np.random.seed(((1 + seed) * N + (i + 1)) * 1234567)
        
        result_V, result_sigma = [], [] # store V and sigma in each block in one repetition
        env = setting(T = T)
        a = simulation(env, n = n)

        L = int(np.sqrt((n_min * T_min) ** beta))
        ################
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
                a.update_op() ## 这个update是对整个current buffer来算的
                a._stretch_para()
                error = np.sqrt(np.mean((tmp - a.all_para)**2))
                terminate_loop += 1
                print("loop %d, in k = %d ,error is %.3f" %(terminate_loop, rep, error))
            
            a.buffer_next_block(n_min, T_min, T, n = None )
            L = max(7,(L + 3)) - 3 ## L can not be 3 .. should be at least 4
            
            print("start making inference on current block .... ")
            lower_bound, upper_bound = a.inference_int(policy = a.opt_policy, U_int_store = U_int_store, block = True, MC_N = MC_N) ## use inference_int to get V and sigma， 是用next buffer的数据集来算的! 
            print("end making inference on current block .... ")
            V = (lower_bound + upper_bound)/2
            print("current index is (%d, %d), length of current buffer %d , length of first one %d, value is %.2f, sigma2 is %.2f "%(a.current_block_idx[0], a.current_block_idx[1], len(a.buffer), a.buffer[0][3], V, a.sigma2))
            
            result_V.append(V)
            result_sigma.append(np.sqrt(a.sigma2))
                                
        K = len(result_sigma) + 1
        V_tilde = np.sum([result_V[i] / result_sigma[i] for i in range(K - 1)]) /  np.sum([1/ result_sigma[i] for i in range(K - 1)])
        sigma_tilde = (K - 1) / np.sum([1/ result_sigma[i] for i in range(K - 1)])
        lower_bound = V_tilde - norm.ppf(1 - alpha/2) * sigma_tilde / (n * T * (K -1) /(K))**0.5
        upper_bound = V_tilde + norm.ppf(1 - alpha/2) * sigma_tilde / (n * T * (K -1) /(K))**0.5
        
        ####################################################
        print("LOWERBOUND and UPPERBOUND is ", lower_bound, upper_bound)
        pickle.dump([lower_bound,upper_bound], outfile_CI) ## 存CI！！！
        ####################################################
        
        V_tilde_list.append(V_tilde)
        CI_length_list.append(upper_bound - lower_bound)
        if est_mean > lower_bound and est_mean < upper_bound:
            count += 1
    outfile_CI.close()
                                
    print(count / N)
    f = open("RESULT_opt_pol_T_%d_n_%d_S_init_integration_(K_n_%d_K_T_%d).txt" %(T,n, K_n, K_T), "a+")
    f.write("Count %d in %d, estimated mean: %f, V_tilde_mean: %f (CI length : %f) \r\n" % (count, N, est_mean, np.mean(V_tilde_list), np.mean(CI_length_list) ))
    f.close()
    
    