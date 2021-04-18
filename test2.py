import attacker
import numpy as np
import pandas as pd
import opendp.smartnoise.core as sn
import pymc3 as pm
import arviz as az
import theano.tensor as tt
import theano
from sklearn.feature_selection import mutual_info_regression
from typing import List

def wrapper(a: List[float]) -> float:
    return sum(a)/len(a)
## opendp program
@theano.compile.ops.as_op(itypes=[tt.lvector],
                            otypes=[tt.dscalar])
def dp_program(age):
    print(age)
    return sum(age)/len(age)

domain = [
    {
        "name": "age",
        "lower": 10,
        "upper": 80,
        "type": "float"
    }
]

def q(trace):
    I = mutual_info_regression(trace["Alice_age"].reshape(-1,1), trace["out"])[0]
    return -I

res = attacker.construct_analysis(wrapper, domain, q)