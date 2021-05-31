import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import privuggerag as attacker
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from typing import List

def wrapper(a: int) -> int:
    return a == 1024

domain = [
    {
        "name": "guess",
        "lower": 0,
        "upper": 2000,
        "type": "int"
    }
]


def q(trace):
    res = np.sum(trace["out"])
    return -res

wrapper = attacker.construct_analysis(wrapper, 
                            domain, 
                            q)
print(wrapper)
