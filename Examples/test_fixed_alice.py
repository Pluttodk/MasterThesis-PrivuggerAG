import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import privuggerag as attacker
from sklearn.feature_selection import mutual_info_regression
from typing import List
import scipy.stats as st

def q(a: List[float]) -> float:
    return sum(a)/len(a)

domain = [
    {
        "name": "age",
        "lower": 10,
        "upper": 50,
        "type": "float",
        "alice": st.norm(25,10)
    }
]
def lm(trace):
    I = mutual_info_regression(trace["Alice_age"].reshape(-1,1), trace["out"])[0]
    return -I
wrap = attacker.construct_analysis(q, domain, lm, cores=1)
wrap.best_dist()