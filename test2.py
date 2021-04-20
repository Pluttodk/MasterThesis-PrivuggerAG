import attacker
import numpy as np
import opendp.smartnoise.core as sn
from sklearn.feature_selection import mutual_info_regression
from typing import List
from scipy import stats as st

def wrapper(a: List[float]) -> float:
    return sum(a)/len(a)
## opendp program

domain = [
    {
        "name": "age",
        "lower": 10,
        "upper": 100,
        "type": "float"
    }
]

def diff(trace):
    return np.sum(trace["Rest_age"])

def q(trace):
    I = mutual_info_regression(trace["Alice_age"].reshape(-1,1), trace["out"], random_state=np.random.RandomState(12345))[0]
    return -I

#Maximum is = 4.605

attacker.construct_analysis(wrapper, 
                            domain, 
                            diff,
                            random_state=1)
# print("="*20)
# print(f"Maximum y reached after 100-iterations: {Y_step[np.argmin(Y_step)]}")
# print(f"Maximum x reached after 100-iterations: {X_step[np.argmin(Y_step)]}")
# print("="*20)