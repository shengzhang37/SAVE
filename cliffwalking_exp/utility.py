from scipy.stats import norm
import random
import numpy as np



def get_sign(s):
    if s > 0:
        return "+"
    else: return "-"
    
def target_policy(S):
    assert len(S) == 2
    if S[0] > 0 and S[1] > 0:
        return 1
    else: return 0
    
    








