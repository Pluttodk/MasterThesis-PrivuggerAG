from typing import * 
import pymc3 as pm
import numpy as np
from attacker import parameters, optimizer as opt
from numba import njit

def convert_primitive_to_dist(p, i, is_list, rng=None):
    if p == float:
        def float_convert(x, domain):
            size = parameters.DB_SIZE if is_list else 1
            return opt.map_int_to_cont_dist(domain[i]["name"], x, domain[i], shape=size, rng=rng)
        return float_convert
    elif p == int:
        def int_convert(x, domain):
            size = parameters.DB_SIZE if is_list else 1
            return opt.map_int_to_discrete_dist(domain[i]["name"], x, domain[i], shape=size)
        return int_convert

def convert_non_primitive_to_dist(p,i, rng=None):
    inner = p.__args__
    partial = []
    for pi in inner:
        if pi == int or pi == float:
            is_list = p.__origin__ == List or p.__origin__ == list
            partial.append(convert_primitive_to_dist(pi, i, is_list, rng=rng))
            i+=1
        else:
            temp, i = convert_non_primitive_to_dist(pi, i, rng=rng)
            partial.append(temp)
    return partial, i

def parse(f,rng=None):
    """
    For each parameter - p:
        For each type of p:
            if p is continuous:
                Create lambda method for map_int_to_cont
            else:
                create lambda method for map_int_to_disc
            append to a broader perspective
    """
    methods = []
    parameters = list(f.__annotations__.values())
    i = 0
    for p in parameters[:-1]:
        if p == float or p == int:
            methods.append(convert_primitive_to_dist(p, i, False, rng=rng))
        else:
            # Under this assumption we now have a list or tuple
            temp, i = convert_non_primitive_to_dist(p,i,rng=rng)
            methods.append(temp)
    return methods

@njit
def make_output(f, dist):
    db = np.zeros(len(dist))
    for i,d in enumerate(dist):
        db[i] = f(d)
    return db

@njit
def unwrap_fast(X):
    db = np.empty((len(X[0]), len(X)))
    for i in range(len(X[0])):
        for j in range(len(X)):
            db[i,j] = X[j][i]
    return db

def create_analytical_method(f, q, domain, random_state=None):
    if isinstance(random_state, int):
        methods = parse(f,random_state)
    else:
        methods = parse(f)
    if len(list(f.__annotations__.values())) == 2:
        if isinstance(methods[0], list):

            def inner(x, i, return_trace=False):
                x = np.append(np.asarray([[i]]), x).reshape((1,-1))
                items = eval_methods(methods[0], 0, x, domain)
                dist = [di(parameters.SAMPLES) for di in items]
                dist_reshape = unwrap_fast(dist)

                out = make_output(f, dist_reshape)
                trace = {
                    f"Alice_{domain[0]['name']}": dist[0],
                    f"Rest_{domain[0]['name']}": dist[1:],
                    "out": out
                }
                return q(trace)
            return inner
        else:
            def inner(x, i, return_trace=False):
                x = np.append(np.asarray([[i]]), x).reshape((1,-1))
                np.random.seed(12345)
                with pm.Model() as model:
                    res = methods[0](x[0])
                    pm.Deterministic("out", f(res))
                    trace = pm.sample(parameters.SAMPLES, progressbar = 0, return_inferencedata=False, cores=2)
                    if return_trace:
                        return q(trace), trace
                    return q(trace)
            return inner
    else:
        # We will have to do a larger test, such that every parameter becomes a 
        def inner(x, i, return_trace=False):
            x = np.append(np.asarray([[i]]), x).reshape((1,-1))
            np.random.seed(12345)
            with pm.Model() as model:
                objects = len(np.asarray(methods).flatten())
                items = eval_methods(methods, 0, x, domain)
                out = pm.Deterministic("out", f(items[0],items[1],items[2],items[3],items[4],items[5]))
                trace = pm.sample(parameters.SAMPLES, progressbar = 0, return_inferencedata=False, cores=2)
                if return_trace:
                        return q(trace), trace
                return q(trace)
        return inner

def eval_methods(method, i, x, domain):
    res = np.empty(len(method), dtype=object)
    for j, m in enumerate(method):
        if isinstance(m, list):
            res[j] = eval_methods(m, i, x, domain)
        else:
            c = m(x[0][i:i+3], domain)
            i+=3
            if len(method) == 1:
                return c
            res[j]=c
    return res