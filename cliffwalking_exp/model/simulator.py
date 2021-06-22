from scipy.stats import norm
import random
import numpy as np



## construct a gym-like class 

class setting():
    def __init__(self ,T = 48, dim = 2):
        self.action_space = action_space(dim_action = 2)
        self.observation_space = observation_space(dim)
        self.last_ob = None
        ## length of trajectories
        self.T = T
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

            #s1_next =  3/4 * (- 1) * s1  + np.random.normal(0, 0.5, 1)
            #s2_next =  3/4 * (1 ) * s2  + np.random.normal(0, 0.5, 1)
            s1_next =  3/4 * (2 * A - 1) * s1  + np.random.normal(0, 0.5, 1)
            s2_next =  3/4 * (1 - 2 * A) * s2  + np.random.normal(0, 0.5, 1)


            S_next = np.array((s1_next,s2_next)).reshape(-1)
            ## update last ob
            self.last_ob = [s1_next[0], s2_next[0]]

            U =  np.array(2 * s1_next  + s2_next - 1/4 * (2 * A - 1)) 
            #U =  np.array(2 * s1_next  + s2_next ) 
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


## setting in https://arxiv.org/abs/1908.08526

def sigmoid(x):
    return(1.0/(1.0+np.exp(-0.1*x)))

def behavior_policy_DR(s, beta = 0.2):
    assert len(s) == 1
    a = beta*sigmoid(s[0]) + beta*np.random.uniform(0.0,1.0)
    return(np.random.binomial(1,a,1)[0])

def target_policy_DR(s, alpha = 0.9):
    assert len(s) == 1
    a = alpha * sigmoid(s[0]) + (1 - alpha) * np.random.uniform(0.0,1.0)
    return(np.random.binomial(1,a,1)[0])


class setting_DR():
    def __init__(self ,T = 30, dim = 1):
        self.action_space = action_space(dim_action = 2)
        self.observation_space = observation_space(dim)
        self.last_ob = None
        ## length of trajectories
        self.T = T
        self.count = 0
    def reset(self, S_init = None):
        if S_init is None:
            self.last_ob = np.random.normal(0.5, 0.2, self.observation_space.shape[0])
        else:
            self.last_ob = S_init
        return self.last_ob 
    def seed(self, seed):
        np.random.seed(seed) 

    def step(self, A):
        self.count += 1
        s = self.last_ob
        s_next =  np.random.normal(0.02*(self.count %2)+ s * 1.0-0.3*(A-0.5),0.2) 
        S_next = np.array(s_next).reshape(-1)
        ## update last ob
        self.last_ob = s_next
        U = np.random.normal(0.9 * s + 0.3 * A - 0.02*(self.count%2), 0.2) 
        if self.count >= self.T:
            done = True
            self.count = 0
        else:
            done = False
        return S_next, U[0], done, None
    
    
    
class action_space():
    def __init__(self, dim_action = 2):
        self.n = dim_action
    def sample(self):
        return random.randint(0, self.n - 1)

class observation_space():
    def __init__(self, dim):
        self.shape = [dim]

        
   