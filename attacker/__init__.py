import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import GPyOpt
from GPyOpt.methods import BayesianOptimization
from typing import *
from attacker import parameters, parse, optimizer as opt
import logging
import time
from tqdm import tqdm
from attacker.dist import *
from cProfile import Profile


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
        number_of_dist = parameters.CONT_DIST if domain["type"] == "float" else parameters.DISC_DIST
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
        upper_bound_std = (1/12)*(domain["upper"] - domain["lower"])**2
        resulting_domain.append({
            "name": domain["name"]+"_std",
            "type": "continuous",
            "domain": (0.1, upper_bound_std),
        })
    return resulting_domain


def domain_to_dist_ids(d, ids):
    res = [fix_domain(domain) for domain in d]
    resulting_domain = []
    constraints = []
    pos = 0
    for domain in d:
        if ids == 0:
            dom, cons = normal_domain(domain)
            pos += 2
        elif ids == 1:
            dom, cons = uniform_domain(domain, pos)
            pos += 2

        for di in dom:
            resulting_domain.append(di)
        for c in cons:
            constraints.append(c)
    return resulting_domain, constraints

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

def construct_analysis(f, domain, q, random_state=None):
    method = parse.create_analytical_method(f, q, domain, random_state)

    X, Y = [],[]
    for dist in tqdm(range(parameters.CONT_DIST)):
        cur_dist, constraint = domain_to_dist_ids(domain, dist)

        feasible_region = GPyOpt.Design_space(space = cur_dist, constraints = constraint) 
        initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, 10)

        f = lambda x: method(x,dist)

        #CHOOSE the objective
        objective = GPyOpt.core.task.SingleObjective(f)

        # CHOOSE the model type
        model = GPyOpt.models.GPModel(exact_feval=True,optimize_restarts=10,verbose=False)

        #CHOOSE the acquisition optimizer
        aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)

        #CHOOSE the type of acquisition
        acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)

        #CHOOSE a collection method
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator, initial_design)

        bo.run_optimization(max_iter = parameters.MAX_ITER, eps = parameters.EPS, verbosity=False) 

        X.append(bo.X)
        Y.append(bo.Y)
    return X,Y



class wrapper:
    def __init__(self, Bopt, f):
        self.f = f
        self.Bopt = Bopt
    
    def plot_convergence(self, plot=True):
        self.Bopt.plot_convergence()
        if not plot:
            import datetime
            plt.savefig("img/"+ datetime.date.today().__str__() + "-convergence-privugger.png")
    
    def plot_acquisition(self, plot=True):
        self.Bopt.plot_acquisition()
        if not plot:
            import datetime
            plt.savefig("img/"+ datetime.date.today().__str__() + "-acquisition-privugger.png")

    def maximum(self):
        return self.Bopt.fx_opt, self.Bopt.x_opt
    
    def run(self, return_trace=False):
        return self.f(np.asarray([self.Bopt.x_opt]), return_trace)
