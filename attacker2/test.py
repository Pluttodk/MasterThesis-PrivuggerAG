import pymc3 as pm
import parse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GPyOpt.methods import BayesianOptimization
import optimizer as opt
from sklearn.feature_selection import mutual_info_regression
from typing import *



def fix_domain(domain):
    LEGAL_VALS = {
        "name" : lambda x: isinstance(x, str),
        "lower": lambda x: isinstance(x, int) or isinstance(x, float),
        "upper": lambda x: isinstance(x, int) or isinstance(x, float),
        "type": lambda x: x == "int" or x == "float"
    }

    for k, v in LEGAL_VALS.items():
        if not v(domain[k]):
            #Fix the input
            raise TypeError(f"Wrong variable for domain with key {k}, expected ..., recieved {domain[k]}")
    return domain

def domain_to_dist(d):
    res = [fix_domain(domain) for domain in d]
    resulting_domain = []
    for domain in d:
        # Append a map to int
        number_of_dist = opt.CONT_DIST if domain["type"] == "float" else opt.DISC_DIST
        resulting_domain.append({
            "name": domain["name"]+"_dist",
            "type": "discrete",
            "domain": tuple(range(number_of_dist))
        })

        #Append value equal to mu
        resulting_domain.append({
            "name": domain["name"]+"_mu",
            "type": "continuous" if domain["type"] == "float" else "discrete",
            "domain": (domain["lower"], domain["upper"]) if  domain["type"] == "float" else tuple(range(int(domain["lower"]), int(domain["upper"]))),
        })
        #Append value equal to sigma
        #takin in to account that standard deviation are always floats
        upper_bound_std = 1/12*(domain["upper"] - domain["lower"])**2
        resulting_domain.append({
            "name": domain["name"]+"_std",
            "type": "continuous",
            "domain": (0.001, upper_bound_std),
        })
    return resulting_domain

def construct_analysis(f, domain, q):
    method = parse.create_analytical_method(f, q, domain)
    f_new = lambda x: method(x)
    domain = domain_to_dist(domain)
    Bopt = BayesianOptimization(f=f_new, domain=domain, 
                         acquisition_type='EI',        # Expected Improvement
                         exact_feval = True)
    Bopt.run_optimization(max_iter = 50, eps=1e-8)           # True evaluations, no sample noise)
    print("="*20)
    print("Value of (x,y) that minimises the objective:"+str(Bopt.x_opt))    
    print("Minimum value of the objective: "+str(Bopt.fx_opt))     
    print("="*20)
    return Bopt

def f(x : Tuple[List[float], List[int]]) -> float:
    x = x[0]
    print(x, "Within F")
    return sum(x)/len(x)

def q_mutual_info(alice, out):
    I = mutual_info_regression(alice[:,0].reshape(-1,1), out[:,0])[0]
    return -I

domain = [
    {
        "name": "age", 
        "lower": 0, 
        "upper": 100,
        "type": "float"
    },
    {
        "name": "height", 
        "lower": 100, 
        "upper": 150,
        "type": "int"
    },    
    {
        "name": "age", 
        "lower": 0, 
        "upper": 100,
        "type": "float"
    },
]
# domain_to_dist(domain)
construct_analysis(f, domain, q_mutual_info)
# run_opt(analys, domain).plot_convergence()
# plt.show()