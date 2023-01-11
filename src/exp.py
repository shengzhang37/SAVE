
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
def get_sign(s):
    if s > 0:
        return "+"
    else: return "-"
    
def target_policy(S):
    if S[0] > 0 and S[1] > 0:
        return 1
    else: return 0
    
    
def main_get_value(T = 30, S_init = (0.5, 0.5), rep = 100, output_path = "./output"):
    print("+" * 45 + "calculating the value" + "+" * 45)
    ### calculate the value
    env = setting(T = T)
    a = simulation(env)
    output, A_percent, _ = a.evaluate_policy(policy = target_policy, seed = None, S_init = S_init, n = rep) 
    print("Mean of value is %.4f"  %np.mean(output))
    ### store the output
    print("+" * 45 + "dumping" + "+" * 45)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    pickle.dump( output, open( output_path + "/value_1_1_S_init_%s%s" % (get_sign(S_init[0]), get_sign(S_init[1])), "wb" ) )
    print("+" * 45 + "dumped" + "+" * 45)


def main(seed = 1, T = 30, n = 25, S_init = (0.5, 0.5), N = 50, beta = 3/7, output_path = "./output", print_every = 10):
    
    filename_CI = output_path + '/CI_store_T_%d_n_%d_S_init_%s%s_sim_1_1' %(T, n, get_sign(S_init[0]), get_sign(S_init[1]))
    outfile_CI = open(filename_CI, 'ab')
    
    #### choose parameter
    
    total_N = T * n
    L = int(np.sqrt((n*T)**beta))

    ## generate data and basis spline
    
    env = setting(T = T)
    a = simulation(env)
    
    ### get value
    value_file = output_path + "/value_1_1_S_init_%s%s" % (get_sign(S_init[0]), get_sign(S_init[1]))
    assert os.path.exists(value_file), "No value file"
    output = pickle.load( open( value_file, "rb" ) )
    est_mean = np.mean(output)
    #if S_init == (0.5, 0.5):
    #    est_mean = -0.0614164995122
    #elif S_init == (-0.5, -0.5):
    #    est_mean = 0.437036040212

    count = 0
    print("S_init is ", S_init)

    for i in range(N):
        #np.random.seed(((1 + seed) * N + (i + 1)) * 123456)
        np.random.seed(((1 + seed) * N + (i + 1)) * 1234567)
        a.buffer = {} ## when using gen_buffer, we should empty the buffer first!!
        a.gen_buffer(total_N = total_N, S_init = None, policy = a.obs_policy )
        a.B_spline(L = max(7,(L + 3)), d = 3)
        lower_bound, upper_bound = a.inference(policy = target_policy, S = S_init)
        
        pickle.dump([lower_bound,upper_bound], outfile_CI) 
        
        if lower_bound < est_mean and est_mean < upper_bound:
            count += 1
        if i % print_every == 0 :
            print("iteration,", i , count, 
              "CI", lower_bound, upper_bound, "sigma_2", a.sigma2, 
              "estimated mean", np.mean([lower_bound[0], upper_bound[0]]))
    print("Count of covered CI over all repetition : ", count / N)
    outfile_CI.close()
    f = open(output_path + "/result_T_%d_n_%d_S_init_%s%s_gamma_%.2f.txt" %(T,n, get_sign(S_init[0]), get_sign(S_init[1]), a.gamma), "a+")
    f.write("Count %d in %d, estimated mean: %f(%f) \r\n" % (count, N, est_mean, np.std(output)))
    f.close()