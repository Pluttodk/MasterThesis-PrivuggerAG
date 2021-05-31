import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import privuggerag as attacker
from sklearn.feature_selection import mutual_info_regression
from typing import List

def q(a: float, b: float) -> float:
    return a+b

domain = [
    {
        "name": "a",
        "lower": 10,
        "upper": 50,
        "type": "float",
    },
    {
        "name": "b",
        "lower": 10,
        "upper": 50,
        "type": "float",
    }
]
def lm(trace):
    I = mutual_info_regression(trace["Alice_a"].reshape(-1,1), trace["out"])[0]
    return -I
wrap = attacker.construct_analysis(q, domain, lm, cores=1)
wrap.best_dist()