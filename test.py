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

def wrapper(a: List[int], s: List[int], e: List[int], r: List[int], i : List[float], m: List[int]) -> float:
    return dp_program(a,s,e,r,i,m)
## opendp program
@theano.compile.ops.as_op(itypes=[tt.lvector,tt.lvector,tt.lvector,
                                    tt.lvector,tt.dvector,tt.lvector],
                            otypes=[tt.dscalar])
def dp_program(age,sex,educ,race,income,married):
    return 10
    temp_file='temp.csv'    
    var_names = ["age", "sex", "educ", "race", "income", "married"]
    data = {
        "age":     age,
        "sex":     sex,
        "educ":    educ,
        "race":    race,
        "income":  income,
        "married": married
    }
    df = pd.DataFrame(data,columns=var_names)
    df.to_csv(temp_file)
    with sn.Analysis() as analysis:
        # load data
        data = sn.Dataset(path=temp_file,column_names=var_names)

        # get mean of age
        age_mean = sn.dp_mean(data = sn.to_float(data['income']),
                                privacy_usage = {'epsilon': .1},
                                data_lower = 0., # min income
                                data_upper = 200., # max income                   
                                data_rows = 10
                                )
    analysis.release()
    return np.float64(age_mean.value)    

domain = [
    {
        "name": "age",
        "lower": 10,
        "upper": 80,
        "type": "int"
    },
    {
        "name": "sex",
        "lower": 0,
        "upper": 2,
        "type": "int"
    },
    {
        "name": "educ",
        "lower": 0,
        "upper": 10,
        "type": "int"
    },
    {
        "name": "race",
        "lower": 0,
        "upper": 50,
        "type": "int"
    },
    {
        "name": "income",
        "lower": 0,
        "upper": 200,
        "type": "float"
    },
    {
        "name": "married",
        "lower": 0,
        "upper": 1,
        "type": "int"
    },
]

def q(trace):
    I = mutual_info_regression(trace["Alice_income"][:,0].reshape(-1,1), trace["out"][:,0])[0]
    return -I

res = attacker.construct_analysis(wrapper, domain, q)