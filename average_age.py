import attacker
import numpy as np
import opendp.smartnoise.core as sn
from sklearn.feature_selection import mutual_info_regression
from typing import List
from scipy import stats as st
from matplotlib import pyplot as plt
import parameters

def wrapper(a: List[float]) -> float:
    return sum(a)/len(a)
## opendp program

domain = [
    {
        "name": "age",
        "lower": 10,
        "upper": 100,
        "type": "float",
        "alice": st.norm(0,10)
    }
]

def q(trace):
    I = mutual_info_regression(trace["Alice_age"].reshape(-1,1), trace["out"], random_state=np.random.RandomState(12345))[0]
    return -I

#Maximum is = 4.605

x = [4.605] * 10
fig, ax = plt.subplot((2,2))
res = attacker.construct_analysis(wrapper, 
                                  domain, 
                                  q)