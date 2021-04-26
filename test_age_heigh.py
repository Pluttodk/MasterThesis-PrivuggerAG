import attacker
import numpy as np
import opendp.smartnoise.core as sn
from sklearn.feature_selection import mutual_info_regression
from typing import List
from scipy import stats as st
from numba import njit

def wrapper(a: List[float], b: List[float]) -> float:
    res = np.mean(a[b > 180])
    if np.isnan(res):
        return 0
    return res



domain = [
    {
        "name": "age",
        "lower": 10,
        "upper": 80,
        "type": "float"
    },
    {
        "name": "height",
        "lower": 150,
        "upper": 200,
        "type": "float"
    }
]

def q(trace):
    I = mutual_info_regression(trace["Alice_age"].reshape(-1,1), trace["out"], random_state=np.random.RandomState(1))[0]
    return -I

# Assumption is that dist for parameter 0 has to be low, and parameter for height below 180

X, Y = attacker.construct_analysis(wrapper, 
                            domain, 
                            q,
                            random_state=1)
for i in range(len(X)):
    print("="*9+str(i)+"="*9)
    print(f"Maximum y reached after 100-iterations: {Y[i][np.argmin(Y[i])]}")
    print(f"Maximum x reached after 100-iterations: {X[i][np.argmin(Y[i])]}")
    print("="*19)