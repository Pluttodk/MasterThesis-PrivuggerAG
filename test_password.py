import attacker
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

X, Y = attacker.construct_analysis(wrapper, 
                            domain, 
                            q,
                            random_state=1)
for i in range(2):
    print("="*9+str(i)+"="*9)
    print(f"Maximum y reached after 100-iterations: {Y[i][np.argmin(Y[i])]}")
    print(f"Maximum x reached after 100-iterations: {X[i][np.argmin(Y[i])]}")
    print("="*19)
print(X,Y)

fig,ax = plt.subplots(2,2, figsize=(18,8))

for i in range(2):
    for j in range(len(X[i][0])):
        ax[i][j].scatter(X[i][:,j], -Y[i])
plt.savefig("img/performance_difference.png")
