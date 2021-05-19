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
from scipy import stats as st
from joblib import Parallel, delayed


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

def domain_to_dist_ids(d, ids):
    res = [fix_domain(domain) for domain in d]
    resulting_domain = []
    constraints = []
    pos = 0
    for i, domain in zip(ids,d):
        if i == 0:
            if domain["type"] == "float":
                dom, cons = normal_domain(domain)
            else:
                dom, cons = poisson_domain(domain, pos)
            pos += 2
        elif i == 1:
            dom, cons = uniform_domain(domain, pos)
            pos += 2
        elif i == 2:
            dom, cons = half_normal_domain(domain, pos)
            pos += 2

        for di in dom:
            resulting_domain.append(di)
        for c in cons:
            constraints.append(c)
    return resulting_domain, constraints

def construct_analysis(f, domain, q, random_state=None, cores=1):
    method = parse.create_analytical_method(f, q, domain, random_state)

    comb = np.array([np.arange(parameters.CONT_DIST) if d["type"] == "float" else np.arange(parameters.DISC_DIST) for d in domain])

    combs = np.array(np.meshgrid(*comb)).T.reshape(-1, len(comb))

    # X, Y, fs = [],[], []
    def run_analysis(dist):
        cur_dist, constraint = domain_to_dist_ids(domain, dist)
        feasible_region = GPyOpt.Design_space(space = cur_dist, constraints = constraint) 
        initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, 10)

        f = lambda x: method(x,dist)
        # fs.append(f)
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

        

        return bo.X, bo.Y, dist, bo, method
    res = Parallel(n_jobs=cores)(delayed(run_analysis)(dist) for dist in tqdm(combs))

    return wrapper(res, [d["type"] for d in domain])



class wrapper:
    def __init__(self, res, types):
        self.X = [r[0] for r in res]
        self.Y = [r[1] for r in res]
        self.dist = [r[2] for r in res]
        self.bos = [r[3] for r in res]
        self.functions = [r[4] for r in res]
        self.types = types

    def print_dist(self, val, di, t):
        print(val)
        if t == "float":
            if di == 0:
                print(f"Normal(mu={val[0]}, sigma={val[1]})")
            elif di == 1:
                print(f"Uniform(lower={val[0]}, scale={val[1]})")
            elif di == 2:
                print(f"HalfNormal(mu={val[0]}, scale={val[1]})")
        else:
            if di == 0:
                print(f"Poisson(mu={val[0]}, scale={val[1]})")
            elif di == 1:
                print(f"Discrete Uniform(lower={val[0]}, scale={val[1]})")

    def best_dist(self):
        for i in range(len(self.X)):
            best_id = np.argmin(self.Y[i])
            best_y = self.Y[i][best_id]
            best_x = self.X[i][best_id]
            dists = self.dist[i]
            for di, t in zip(dists, self.types):
                self.print_dist(best_x, di, t)

    
    def plot_best_dist(self):
        fig, ax = plt.subplots(len(self.domain),2)
        for a, d in zip(ax, domain):
            if d["type"] == "int":
                x = np.arange(d["lower"], d["upper"])
                # mu,scale = 
                # a[0].plot(poisson)
                #a[1].plot(Uniform)

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
