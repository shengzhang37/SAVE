from .utility import *
import operator
from itertools import product
from itertools import accumulate
import numpy as np
import random
import os
import pickle
import os.path
import time
from scipy.interpolate import BSpline
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from numpy.linalg import inv
from functools import reduce
from scipy.stats import norm
from scipy import integrate
from scipy.stats import norm
from tqdm import tqdm


class simulation(object):
    def __init__(self, env, n = 50, scale = "NormCdf", product_tensor = True, reward_dicount = 0.5, use_confounder = False):
        #############################################################################
        #############################################################################
        ### self.env : store the dynamic environment
        ### self.n : store the number of patients(objects)
        ### self.gamma : store the discount
        ### self.buffer : store the data buffer
        ### self.obs_policy : uniformly sample
        ### self.nums_action : store the number of discrete actions that can be chosen
        ### self.dims_state : store the dimension of the state
        #############################################################################
        ### self.last_obs : store the last observation which is particularly designed for append block to make
        ###                 sure that the append block's first state can match the last state in current buffer
        ### self.current_block_idx : store the current position of the block
        #############################################################################
        ### self.scaler : store the scaler which should be applied to bound the state into [0,1]
        #############################################################################
        ### self.knot : store the quantile knots for basis spline
        ### self.para : store the the dimension of parameter built in basis spline
        #############################################################################
        
        self.env = env
        self.n = n
        self.gamma = reward_dicount
        self.buffer = {}
        if use_confounder:
            self.obs_policy = lambda S : np.random.binomial(1, min( (sigmoid(S[0]) + sigmoid(S[1])) / 2, 1 )) ## confounder
        else:
            self.obs_policy =  lambda S : self.env.action_space.sample()
        self.nums_action = self.env.action_space.n
        self.dims_state = self.env.observation_space.shape[0]
        self.last_obs = np.random.normal(0,1,self.dims_state * self.n).reshape(self.n, self.dims_state)
        self.current_block_idx = [0,1] ## [n,t]
        if scale == "NormCdf":
            self.scaler = normcdf()
        elif scale == "Identity":
            self.scaler = iden()
        self.knot = None 
        self.para_dim = None 
        self.product_tensor = product_tensor

        
    #################################
    ###### move one step forward ####
    #################################
    
    def step_env(self, A):
        S_next, U, done, _ = self.env.step(A)
        return S_next, U, done
    
    #################################
    #### generate one trajectory ####
    #################################
    
    def gen_traj(self, evaluation = False, policy = None, seed = None, S_init = None, burn_in = None):
        #############################################################################
        ######### OUTPUT: state, action, utility trajectory and T ###################
        #############################################################################
        if policy is None:
            policy = self.obs_policy
        ## initialize the state
        if seed is None and S_init is None: 
            S = self.env.reset()
        elif seed is not None:
            self.env.seed(seed) 
            S = self.env.reset()
        elif S_init is not None:
            S = self.env.reset(S_init)
        S_traj = [S]
        A_traj = []
        U_traj = []
        done = False
        while not done:
            A = policy(S)
            S_next, U, done = self.step_env(A)
            S_traj.append(S_next)
            A_traj.append(A)
            U_traj.append(U)
            S = S_next # update current S as S_next
        T = len(U_traj)
        ## output state, action, utility trajectory and T
        if burn_in is None:
            return [S_traj, A_traj, U_traj, T]
        else:
            return [S_traj[burn_in:], A_traj[burn_in:], U_traj[burn_in:], T - burn_in]
        
    ####################################
    #### Store multiple trajectories ###
    ####################################

    def gen_buffer(self, policy = None, n = None, S_init = None, burn_in = None, total_N = None): # Get observations
        if total_N is None:
            if n is None:
                n = self.n
            for i in range(n):
                #self.buffer[(i)] = None
                self.buffer[(i)] = self.gen_traj(policy = policy, burn_in = burn_in, S_init = S_init)
        else:
            count = 0
            i = 0
            while count < total_N:
                self.buffer[(i)] = self.gen_traj(policy = policy, burn_in = burn_in, S_init = S_init)
                count += self.buffer[(i)][3]
                i += 1
            self.n = i
            self.total_N = count
            
    ####################################
    #### generate next block ###########
    ####################################
    
    def buffer_next_block(self, n_min, T_min, T, n = None, policy = None):
        #### store the next block in next_block
        if n is None: 
            n = self.n
        self.K_n = n//n_min
        self.K_T = T//T_min
        
        if self.current_block_idx[0] == self.K_n and self.current_block_idx[1] == self.K_T:
            self.next_block = {}
        else:
            self.next_block_idx = get_next_block_idx(self.current_block_idx, self.K_n, self.K_T)

            self.next_block = {}
            start_i, end_i, T_block =  get_idx_pos(self.next_block_idx, n, T, n_min, T_min) ## start_i/end_i is for N axis not T axis
            self.env.T  = T_block
            for k in range(start_i, end_i):
                if policy is None:
                    self.next_block[k] = self.gen_traj(S_init = self.last_obs[k].copy()) ## the length of trajectory depend on the DONE output which is specified in env
                else:
                    self.next_block[k] = self.gen_traj(S_init = self.last_obs[k].copy(), policy = policy)
                self.last_obs[k] = self.env.last_ob
                
    ##################################################
    #### append next block to current block ##########
    ##################################################
    
    def append_next_block_to_buffer(self):
        if len(self.next_block) > 0:
            ## update current block idx
            self.current_block_idx = self.next_block_idx.copy()
            self.next_block_idx = get_next_block_idx(self.current_block_idx, self.K_n, self.K_T)
            ## append self.next_block to self.buffer:
            for key, value in self.next_block.items():
                if self.buffer.get(key) is None:
                    self.buffer[key] = value
                else:
                    S, A, U, t = value
                    self.buffer[key][0].extend(S[1:])
                    self.buffer[key][1].extend(A)
                    self.buffer[key][2].extend(U)
                    self.buffer[key][3] += t
        
    #############################
    #### evaluate given policy###
    #############################
    
    def evaluate_policy(self, policy, n = 20, seed = None, S_init = None, lower_b = None, upper_b = None): 
        output = []
        A_percent = []
        value = []
        count = 0
        for i in tqdm(range(n)): ## evaluation on n people
            S, A, U, T = self.gen_traj(policy = policy, seed = seed, S_init = S_init)
            est_Value =  sum(map(operator.mul, 
                                      [self.gamma ** j for j in range(T)], 
                                        U))
            output.append(est_Value)
            A_percent.append(np.mean(A))
            value.append(0)
            if lower_b or upper_b is not None:
                if est_Value >= lower_b and est_Value <= upper_b:
                    count += 1
        if lower_b or upper_b is not None:
            return output, A_percent, value, count / n
        else:
            return output, A_percent, value

    
    #################################
    #### Construct Basis Spline #####
    #################################
    
    def B_spline(self, L = 10, d = 3):
        data = []
        for i in range( len(self.buffer)):
            data.extend(self.buffer[i][0])
        scale_data = (self.scaler.transform(data))
        self.knot = np.quantile(scale_data, np.linspace(0,1,L + 1), axis=0)
        self.bspline = []

        self.para_dim = [1 if self.product_tensor else 0][0]
        for i in range(self.dims_state):
            tmp = []
            for j in range(L - d):
                cof = [0] * (L - d)
                cof[j] = 1
                spf = BSpline(self.knot.T[i], cof, d)
                tmp.append(spf)
            self.bspline.append(tmp)
        ############### if dimension of state is more than 2, we use additive tensor in experiments ############
            if self.product_tensor:
                self.para_dim *= len(self.bspline[i])
            else:
                self.para_dim += len(self.bspline[i])
        ######################################################################################## 
            print("Building %d-th basis spline (total %d state dimemsion) which has %d basis " %(i, self.dims_state,len(self.bspline[i]) ))
        
        self.para = {}
        for i in range(self.nums_action):
            self.para[i] = np.random.normal(0,0,self.para_dim)
        self.para_2 = self.para.copy()  ## for double
        
    ##############################
    ###### calculate Q function ##
    ##############################
    
    def Q(self, S, A, predictor = False, double = False):
        ## input state is original
        S = self.scaler.transform(S)
        ## compute Q function
        # it is used for linear regression as a predictor
        
        ############### if dimension of state is more than 2, we use additive tensor ############
        ## us np.prod to get the product tensor of result
        if self.product_tensor:
            output = list(map(np.prod,(product(*[np.array([func(s) for func in f]) for f,s in zip(self.bspline, S)],repeat=1))))
        else:
            output = list(np.concatenate([np.array([func(s) for func in f]) for f,s in zip(self.bspline, S)]))
        ########################################################################################
        if predictor: 
            return output
        
        else: 
            # Double Q iteration
            if double:
                return sum(map(operator.mul, output, self.para_2[int(A)])) ## <- apply double Q!
            else:
                return sum(map(operator.mul, output, self.para[int(A)]))
    
    def V(self, S, policy):
        ###
        ## todo: use MC to change it..
        ##
        return self.Q(S, policy(S))
    
    def V_int(self, policy, MC_N = None):
        if MC_N is None:
            f = lambda y,x : self.V(policy = policy, S = (x,y)) * norm.pdf(y) * norm.pdf(x)
            return integrate.dblquad(f, -5, 5, lambda x: -5, lambda x: 5)[0]
        else:
            f = lambda y,x : self.V(policy = policy, S = (x,y))
            x = np.random.normal(size = MC_N)
            y = np.random.normal(size = MC_N)
            return np.mean([f(y[i],x[i]) for i in range(MC_N)])
        
    
    ##################################
    ######## update the para #########
    ##################################
    
    def update_op(self, shuffle = False, batch = None, double = True, Lasso = False): ## Q^*
        ## obtain predictor and reponse
        ## target and and predictor(f) in Q learning which is used for for linear prediction
        target = {}
        f = {}
        for i in range(self.nums_action):
            target[i] = []
            f[i] = []
        ## shuffle the buffer: if true shuffle the order, otherwise apply linear regression to all directly
        if shuffle:
            1 # to add
        else:
            for k in range(len(self.buffer)):
                S = self.buffer[k][0]
                A = self.buffer[k][1]
                Y = self.buffer[k][2]
                T = self.buffer[k][3]
                for i in range(T):
                    if i < T - 1:
                        a_star = np.argmax([self.Q(S[i + 1], j, predictor = False, double = double) 
                                                     for j in range(self.nums_action)]) ## use double Q learning..
                        target[int(A[i])].append(Y[i]  + 
                                                 self.gamma * self.Q(S[i + 1], a_star, predictor = False) )

                    else: 
                        target[int(A[i])].append(Y[i])
                    f[int(A[i])].append(self.Q(S[i],A[i], predictor = True))
                    
        ## use target and f to update the parameters 
        self.para_2 = self.para.copy()
        for i in range(self.nums_action):
            if Lasso:
                reg = linear_model.Lasso(alpha=0.1, fit_intercept = False)
            else:
                reg = LinearRegression(fit_intercept  = False)
            reg.fit(np.array(f[i]), np.array(target[i]))
            self.para[i] = reg.coef_
            


    ########################################
    ######### obtain the optimal policy ####
    ########################################
    
    def opt_policy(self, S, epsilon = 0.0): 
        # output Action
        if np.random.uniform(0,1) < epsilon:
            return self.obs_policy(S)
        else:
            return np.argmax([self.Q(S,i, predictor = False ) for i in range(self.nums_action)])
        
    def _stretch_para(self):
        self.all_para = []
        for i in self.para.values():
            self.all_para.extend(i)
        self.all_para = np.array(self.all_para)

    #############################################################################################
    ########################## make inference on beta ########################################### 
    #############################################################################################
    
   
    def _Xi(self, S, A):
        S = self.scaler.transform(S)
        if A == 0:
            ############### if dimension of state is more than 2, we use additive tensor ############
            if self.product_tensor:
                return np.array(list(map(np.prod,(product(*[np.array([func(s) for func in f]) for f,s in zip(self.bspline, S)], repeat=1)))) + [0] * self.para_dim).reshape(-1,1)
            else:
                return np.array(list(np.concatenate([np.array([func(s) for func in f]) for f,s in zip(self.bspline, S)])) + [0] * self.para_dim).reshape(-1,1)
            
        else: ## only consider A = 1/0, here it means A = 1
            if self.product_tensor:
                return np.array([0] * self.para_dim + list(map(np.prod,(product(*[np.array([func(s) for func in f]) for f,s in zip(self.bspline, S)],repeat=1))))).reshape(-1,1)
            else:
                return np.array([0] * self.para_dim + list(np.concatenate([np.array([func(s) for func in f]) for f,s in zip(self.bspline, S)]))).reshape(-1,1)
            #############################################################################################
        
            
    def _U(self, S, policy):
        return self._Xi(S, policy(S))
            
    def _Sigma(self, policy, block = False):
        output = np.zeros((self.para_dim * self.nums_action, self.para_dim * self.nums_action))
        output_2 = np.zeros((self.para_dim * self.nums_action, 1))
        total_T = 0 
        if not block:
            for i in self.buffer.keys():
                T = self.buffer[i][3]
                total_T += T
                for j in range(T):
                    S = self.buffer[i][0][j]
                    S_next = self.buffer[i][0][j + 1]
                    A = self.buffer[i][1][j]
                    Y = self.buffer[i][2][j]
                    output += (np.matmul( self._Xi(S, A) , (self._Xi(S, A) - self.gamma * self._U(S_next, policy = policy)).T))
                    output_2 += Y * self._Xi(S,A)
        else:
            for i in self.next_block.keys():
                T = self.next_block[i][3]
                total_T += T
                for j in range(T):
                    S = self.next_block[i][0][j]
                    S_next = self.next_block[i][0][j + 1]
                    A = self.next_block[i][1][j]
                    Y = self.next_block[i][2][j]
                    output += (np.matmul( self._Xi(S, A) , (self._Xi(S, A) - self.gamma * self._U(S_next, policy = policy)).T))
                    output_2 += Y * self._Xi(S,A) 
        self.total_T = total_T
        """
        Add ridge can avoid overfitting!
        """
        self.Sigma_hat =  np.diag([10 ** (-9)] * output.shape[0])  +  output / total_T
        # self.Sigma_hat =  output / total_T
        #if not block: !!
        #    self.vector = output_2 / total_T
        self.vector = output_2 / total_T
        
    def _beta_hat(self, policy, block = False):
        self._Sigma(policy, block = block)
        self.inv_Sigma_hat = inv(self.Sigma_hat)
        #if not block: !!
        #    self.est_beta = np.matmul(self.inv_Sigma_hat, self.vector)
        self.est_beta = np.matmul(self.inv_Sigma_hat, self.vector)
        
    ## store the estimated beta in self.para
    def _store_para(self, est_beta):
        for i in range(self.nums_action):
            self.para[i] = self.est_beta[ i * self.para_dim : (i + 1)* self.para_dim].reshape(-1)
            
    def _Omega_hat(self, policy, block = False):
        
        ## first we use the policy to obtain the estimation of parameters;
        ## note: if policy is opt_policy, which will use the estimated parameters
        self._beta_hat(policy, block = block)
        ## Then we store the estimated paras
        self._store_para(self.est_beta)
        
        output = np.zeros((self.para_dim * self.nums_action, self.para_dim * self.nums_action))
        output_sqrt =  np.zeros((self.para_dim * self.nums_action, 1))
        if not block:
            for i in self.buffer.keys():
                T = self.buffer[i][3]
                for j in range(T - 1):
                    S = self.buffer[i][0][j]
                    S_next = self.buffer[i][0][j + 1]
                    A = self.buffer[i][1][j]
                    U = self.buffer[i][2][j]
                    Xi = self._Xi(S,A)
                    output += ((U + self.gamma * (self.V(S_next, policy)) - (self.Q(S, A)))**2) * np.matmul(Xi, Xi.T)
                    output_sqrt += (U + self.gamma * (self.V(S_next, policy)) - (self.Q(S, A))) * Xi
        else:
            for i in self.next_block.keys():
                T = self.next_block[i][3]
                for j in range(T - 1):
                    S = self.next_block[i][0][j]
                    S_next = self.next_block[i][0][j + 1]
                    A = self.next_block[i][1][j]
                    U = self.next_block[i][2][j]
                    Xi = self._Xi(S, A)
                    output += ((U + self.gamma * (self.V(S_next, policy)) - (self.Q(S, A)))**2) * np.matmul(Xi, Xi.T)
                    output_sqrt += (U + self.gamma * (self.V(S_next, policy)) - (self.Q(S, A))) * Xi
        self.Omega = output / self.total_T
        self.Omega_sqrt = output_sqrt / (self.total_T) ** 0.5
        
    
        
    #### for S_init individual
    def _sigma(self, policy, S, block = False):
        self._Omega_hat(policy, block = block)
        self.sigma = reduce(np.matmul, [self._U(S, policy).T, self.inv_Sigma_hat, self.Omega_sqrt])
        self.sigma2 = reduce(np.matmul, [self._U(S, policy).T, self.inv_Sigma_hat, self.Omega, self.inv_Sigma_hat.T, self._U(S, policy)]) 

    def inference(self, policy, S, alpha = 0.05, block = False):
        self._sigma(policy, S, block = block) ## estimate the beta
        V = self.V(S, policy)
        return V - norm.ppf(1 - alpha/2) * self.sigma2 ** 0.5 / (self.total_T ** 0.5), V + norm.ppf(1 - alpha/2) * self.sigma2 ** 0.5 / (self.total_T ** 0.5)
    
    #############################################
    ##### for S_init with integration ###########
    #############################################
    def _sigma_int(self, policy, block = False, U_int_store = "U_int_store", MC_N = None):
        print("start calculating Omega....")
        self._Omega_hat(policy, block = block)
        print("start extracting U....")
        ## get U int from pickle file!
        if U_int_store is None:
            if MC_N is None:
                raise ValueError("NEED MC_N is NOT None..")
            U_int = []
            x = np.random.normal(size = MC_N)
            y = np.random.normal(size = MC_N) 
            f = lambda y,x : self._U(policy = policy, S = (x,y))
            for ele in range(self.para_dim*2):
                print("integrating para %d, total number of parameters is 2*%d"% (ele, self.para_dim))
                U_int.append(np.mean([f(y[i],x[i])[ele] for i in range(MC_N)]))
            U_int = np.array(U_int)
            
        else:
            filename = U_int_store
            if os.path.exists(filename):
                outfile = open(filename,'rb')
                U_int = np.array(pickle.load(outfile)).reshape(-1,1)
                outfile.close()
            else:
                if MC_N is None:
                    raise ValueError("NEED MC_N is NOT None..")
                U_int = []
                x = np.random.normal(size = MC_N)
                y = np.random.normal(size = MC_N) 
                f = lambda y,x : self._U(policy = policy, S = (x,y))
                for ele in range(self.para_dim*2):
                    print("integrating para %d, total number of parameters is 2*%d"% (ele, self.para_dim))
                    U_int.append(np.mean([f(y[i],x[i])[ele] for i in range(MC_N)]))
                U_int = np.array(U_int)
                pickle.dump(U_int, open(filename, "wb"))
            
        ## get sigma2
        print("start obtaining sigma2....")
        self.sigma2 = reduce(np.matmul, [U_int.T, self.inv_Sigma_hat, self.Omega, self.inv_Sigma_hat.T, U_int])
        
        
    def inference_int(self, policy, alpha = 0.05, U_int_store = None, block = False, MC_N = 10000, fitted_Q = False):
        
        ############################################################################################################
        ##### Note 1 : MC_N = None : we use built-in function to get numerical integration for V 
        #####         MC_N = 10000 : we use MC to get numerical integration for V 
        ##### Note 2 : U_int_store = "U_int_store" : we use stored U to calculate U 
        #####          U_int_store = None : we use MC to get numerical integration for U <-- it need MC is not None
        ##### Note 3 : fitted_Q = False : we use LSE to re-calculate the self.para
        #####          fitted_Q = True : we use current stored self.para (according to the main_est*, it is fitted-Q).
        #####          <-- fitted_Q should always be False.
        ############################################################################################################
        
        self._sigma_int(policy, U_int_store = U_int_store, block = block, MC_N = MC_N) 
        print("start getting V value (slow.. need to numerical integration)....")
        start = time.time()
        V = self.V_int(policy, MC_N) 
        print("Finshed! cost %d time" % (time.time() - start))
        return V - norm.ppf(1 - alpha/2) * (self.sigma2 ** 0.5) / (self.total_T ** 0.5), V + norm.ppf(1 - alpha/2) * (self.sigma2 ** 0.5) / (self.total_T ** 0.5)
    
    
    
    
    
    
    