import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GPyOpt.methods import BayesianOptimization
from typing import *
from attacker import parameters, parse, optimizer as opt
import logging
import time


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

def progress_bar(f, max_iter, current, width=20):
    s = "#"*int((current/max_iter)*width)
    rest = "-"*int(((max_iter-current)/max_iter)*width)

    start = time.time()
    res = f()
    end = time.time()

    time_taken = end-start
    projected_finish = time_taken*(max_iter-current)
    hours, rem = divmod(projected_finish, 60*60)
    minutes, rem = divmod(rem, 60)
    seconds = round(rem)
    
    times = "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), seconds)
    print(f"\r [{s}{rest}] - {current}/{max_iter} - Time left: {times} ", end="\r")
    
    return res

def construct_analysis(f, domain, q):
    logger = logging.getLogger('pymc3')
    logger.setLevel(logging.ERROR)
    logger.propagate = False

    logger.disabled = True
    method = parse.create_analytical_method(f, q, domain)

    parameters.PROGRESS = 0
    def f_new(x):
        parameters.PROGRESS += 1
        return progress_bar(lambda:method(x), parameters.MAX_ITER+parameters.INITIAL_DESIGN_NUMDATA, parameters.PROGRESS)

    domain = domain_to_dist(domain)
    Bopt = BayesianOptimization(f=f_new, domain=domain, 
                         acquisition_type='EI',        # Expected Improvement
                         initial_design_numdata=parameters.INITIAL_DESIGN_NUMDATA,
                         exact_feval = True)
    Bopt.run_optimization(max_iter = parameters.MAX_ITER, eps=1e-8)           # True evaluations, no sample noise)
    print("="*20)
    print("Value of (x,y) that minimises the objective:"+str(Bopt.x_opt))    
    print("Minimum value of the objective: "+str(Bopt.fx_opt))     
    print("="*20)
    
    logger.disabled = False
    return wrapper(Bopt, method)


class wrapper:
    def __init__(self, Bopt, f):
        self.f = f
        self.Bopt = Bopt
    
    def plot_convergence(self):
        self.Bopt.plot_convergence()
    
    def plot_acquisition(self):
        self.Bopt.plot_acquisition()

    def maximum(self):
        return self.Bopt.x_opt
    
    def run(self, return_trace=False):
        return self.f(np.asarray([self.Bopt.x_opt]), return_trace)
