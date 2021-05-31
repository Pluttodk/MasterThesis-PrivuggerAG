import privuggerag as attacker
from sklearn.feature_selection import mutual_info_regression
from typing import List

def q(a: List[float]) -> float:
    return sum(a)/len(a)

domain = [
    {
        "name": "age",
        "lower": 10,
        "upper": 50,
        "type": "float",
    }
]
def lm(trace):
    I = mutual_info_regression(trace["Alice_age"].reshape(-1,1), trace["out"])[0]
    return -I
wrap = attacker.construct_analysis(q, domain, lm, cores=1)
wrap.best_dist()