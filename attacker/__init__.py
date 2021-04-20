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
        print("UPPER BOUD " + str(upper_bound_std))
        resulting_domain.append({
            "name": domain["name"]+"_std",
            "type": "continuous",
            "domain": (0.1, upper_bound_std),
        })
    return resulting_domain


def normal_domain(domain):
    upper_bound_std = (1/12)*(domain["upper"] - domain["lower"])**2
    return [
        {
            "name": domain["name"]+"_mu",
            "type": "continuous" if domain["type"] == "float" else "discrete",
            "domain": (domain["lower"], domain["upper"]) if  domain["type"] == "float" else tuple(range(int(domain["lower"]), int(domain["upper"]))),
        },
        {
            "name": domain["name"]+"_std",
            "type": "continuous",
            "domain": (0.1, upper_bound_std),
        }
    ],[
    ]

def uniform_domain(domain, pos):
    """
    We add constraints such that the value will scale correctly. 
    One being the following
    1/12*(x)^2 >= 0.1
    Where X is the scale.

    so we get that scale has to be >= sqrt(6/5)
    """
    upper = domain["upper"]
    return [
        {
            "name": domain["name"]+"_mu",
            "type": "continuous" if domain["type"] == "float" else "discrete",
            "domain": (domain["lower"], domain["upper"]) if  domain["type"] == "float" else tuple(range(int(domain["lower"]), int(domain["upper"]))),
        },
        {
            "name": domain["name"]+"_std",
            "type": "continuous",
            "domain": (0, domain["upper"]),
        }
    ],[
        {
            'name': domain["name"]+"_constr2", 
            'constraint': f'-x[:,{pos+1}]+np.sqrt(6/5)' # Variance >= 0.1
        },
        {
            'name': domain["name"]+"_constr2", 
            'constraint': f'x[:,{pos+1}]+x[:,{pos}]-{upper}' # lower+scale <= domain[upper]
        },
    ]

def domain_to_dist_ids(d, ids):
    res = [fix_domain(domain) for domain in d]
    resulting_domain = []
    constraints = []
    pos = 0
    for domain in d:
        if ids == 0:
            dom, cons = normal_domain(domain)
            for di in dom:
                resulting_domain.append(di)
            for c in cons:
                constraints.append(c)
            pos += 2
        elif ids == 1:
            dom, cons = uniform_domain(domain, pos)
            for di in dom:
                resulting_domain.append(di)
            for c in cons:
                constraints.append(c)
            pos += 2
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
    logger = logging.getLogger('pymc3')
    logger.setLevel(logging.ERROR)
    logger.propagate = False

    logger.disabled = True
    method = parse.create_analytical_method(f, q, domain, random_state)

    parameters.PROGRESS = 0

    # def f_new(x):
    #     parameters.PROGRESS += 1
    #     return progress_bar(lambda:method(x), parameters.MAX_ITER+parameters.INITIAL_DESIGN_NUMDATA, parameters.PROGRESS)
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

        print("="*20)
        print("Value of (x,y) that minimises the objective:"+str(bo.x_opt))    
        print("Minimum value of the objective: "+str(bo.fx_opt))     
        print("="*20)

    # domain = domain_to_dist(domain)
    # x_init = np.asarray([[d["domain"][0] for d in domain[1:]]]) #Lower x0 guess
    # y_init = np.asarray([[method(np.append(np.asarray([[0]]), x_init).reshape((1,-1)))]])
    # X_step, Y_step = x_init, y_init

    # # For each distribution, iterate through here
    # for dist in tqdm(range(parameters.CONT_DIST)):
    #     x_init = np.asarray([[d["domain"][0] for d in domain[1:]]]) #Lower x0 guess
    #     y_init = np.asarray([[method(np.append(np.asarray([[0]]), x_init).reshape((1,-1)))]])
    #     X_step, Y_step = x_init, y_init
    #     best = 0
    #     for i in tqdm(range(100), leave=False):
    #         bo_step = BayesianOptimization(f = None, domain = domain[1:], X = X_step, Y = Y_step)
    #         x_next = bo_step.suggest_next_locations()
    #         y_next = method(np.append(np.asarray([[dist]]), x_next).reshape((1,-1)))
            
    #         if y_next < best:
    #             best = y_next
            
    #         X_step = np.vstack((X_step, x_next))
    #         Y_step = np.vstack((Y_step, y_next))
    #     print(f"best {best} for dist {dist}")
    #     print(f"Maximum x reached after 100-iterations: {X_step[np.argmin(Y_step)]}")

    

    # Bopt = BayesianOptimization(f=f_new, domain=domain, 
    #                      evaluator_type = 'local_penalization',
    #                      acquisition_type = parameters.ACQUISITION,       # LCB acquisition
    #                      acquisition_weight = 0.1,
    #                      initial_design_numdata=parameters.INITIAL_DESIGN_NUMDATA,
    #                      exact_feval = True)
    # Bopt.run_optimization(max_iter = parameters.MAX_ITER, eps=1e-8, context={"age_dist": 0})           # True evaluations, no sample noise)
    # print("="*20)
    # print("Value of (x,y) that minimises the objective:"+str(Bopt.x_opt))    
    # print("Minimum value of the objective: "+str(Bopt.fx_opt))     
    # print("="*20)
    
    # logger.disabled = False
    # return X_step, Y_step


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
