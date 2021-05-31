import privuggerag as attacker
import numpy as np
import opendp.smartnoise.core as sn
from sklearn.feature_selection import mutual_info_regression
from typing import List
from scipy import stats as st
from numba import njit
from matplotlib import pyplot as plt

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
