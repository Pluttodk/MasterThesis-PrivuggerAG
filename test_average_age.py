import attacker
import numpy as np
import opendp.smartnoise.core as sn
from sklearn.feature_selection import mutual_info_regression
from typing import List
from scipy import stats as st
from numba import njit

def wrapper(a: List[float]) -> float:
    return np.mean(a)
## opendp program

def mode(a: List[float]) -> float:
    return np.median(a)

domain = [
    {
        "name": "age",
        "lower": 10,
        "upper": 50,
        "type": "float",
    }
]

def sums(l):
    return np.sum(l)

def diff(trace):
    return sums(np.asarray(trace["Rest_age"]))

def q(trace):
    I = mutual_info_regression(trace["Alice_age"].reshape(-1,1), trace["out"], random_state=np.random.RandomState(12345))[0]
    return -I

#Maximum is = 4.605

wrap = attacker.construct_analysis(mode, 
                            domain, 
                            q,
                            cores=1)
wrap.best_dist()
print(np.min(wrap.Y, axis=1))
# for i in range(len(X)):
#     print("="*9+str(i)+"="*9)
#     print(f"Maximum y reached after 100-iterations: {Y[i][np.argmin(Y[i])]}")
#     print(f"Maximum x reached after 100-iterations: {X[i][np.argmin(Y[i])]}")
#     print("="*19)