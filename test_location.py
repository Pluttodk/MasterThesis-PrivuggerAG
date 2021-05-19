import attacker
import numpy as np
import opendp.smartnoise.core as sn
from sklearn.feature_selection import mutual_info_regression
from typing import List, Tuple
from scipy import stats as st

def Location(loc: List[Tuple[float, float]]) -> float:
    return 10
domain = [
    {
        "name": "loc",
        "lower": 0,
        "upper": 100,
        "type": "float",
    },    {
        "name": "loc",
        "lower": 0,
        "upper": 100,
        "type": "float",
    }
]
def mi_loc(t):
    return -mutual_info_regression(t["Alice_loc"].reshape(-1,1), 
                                   t["out"], 
                                   discrete_features=False)[0]

locWrap = attacker.construct_analysis(Location, domain, mi_loc)