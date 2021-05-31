import privuggerag as attacker
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from typing import List

def wrapper(a: float, b: List[float]) -> float:
    o = [a]
    for bi in b[1:]:
        o.append(bi)
    return np.mean(o)

domain = [
    {
        "name": "age",
        "lower": 10,
        "upper": 80,
        "type": "float"
    },
    {
        "name": "age",
        "lower": 10,
        "upper": 80,
        "type": "float"
    }
]

def q(trace):
    I = mutual_info_regression(trace["Alice_age"].reshape(-1,1), trace["out"], random_state=np.random.RandomState(1))[0]
    return -I

# Assumption is that dist for parameter 0 has to be low, and parameter for height below 180

wrap = attacker.construct_analysis(wrapper, 
                            domain, 
                            q)
print(wrap)