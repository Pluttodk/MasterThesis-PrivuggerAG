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

def make_output(f, dist):
    db = np.zeros(len(dist), )
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
    if len(list(f.__annotations__.values())) == 2 and len(domain) == 1:
        if isinstance(methods[0], list):
            #Method with one List as input
            def inner(x, i, return_trace=False):
                x = np.append(np.asarray([[i]]), x).reshape((1,-1))
                dist = eval_methods(methods[0], x, domain)
                dist_reshape = unwrap_fast(dist)
                out = make_output(f, dist_reshape)
                trace = {
                    f"Alice_{domain[0]['name']}": dist[0],
                    f"Rest_{domain[0]['name']}": dist[1:],
                    "out": out
                }
                if return_trace:
                    return q(trace),trace
                return q(trace)
            return inner
        else:
            #Method with one primitive data type
            def inner(x, i, return_trace=False):
                x = np.append(np.asarray([[i]]), x).reshape((1,-1))
                items = methods[0](x[0], domain)
                items = items[0]
                dist = items(parameters.SAMPLES)
                try:
                    out = f(dist)
                except:
                    out = [f(di) for di in dist]
                trace = {
                    f"{domain[0]['name']}": dist,
                    "out": out
                }
                if return_trace:
                    return q(trace), trace
                return q(trace)
            return inner
    else:
        # We will have to do a larger test, such that every parameter becomes a 
        def inner(x, i, return_trace=False):
            x_new = np.zeros(len(x[0])+len(x[0])//2)
            for j in range(len(x_new)):
                if not j % 3:
                    x_new[i] = i
                else:
                    pos = j-((j//3)+1)
                    x_new[j] = x[0][pos]
            res = eval_methods(methods, x_new.reshape((1,-1)), domain)

            data = np.empty(tuple([len(res)] + list(res[0].shape)[::-1]))
            r = res
            if len(list(f.__annotations__.values())) == 2:
                while len(r) == 1:
                    r = r[0]
                for j in range(len(r)):
                    r[j] = r[j].flatten()
                data = np.array(list(zip(*r)))
            #print(list(zip(*r)))
            else:
                for j in range(len(r)):
                    data[j] = list(zip(*res[j]))
            out = np.zeros(parameters.SAMPLES)
            for j in range(len(out)):
                if len(list(f.__annotations__.values())) == 2:
                    out[j] = f(data[j])
                else:
                    out[j] = f(*data[:,j])
            trace = {"out": out}
            for j,d in enumerate(domain):
                #Tuple
                if len(r[j]) == parameters.SAMPLES:
                    trace["Alice_" + d["name"]] = r[j]
                else:
                    trace["Alice_" + d["name"]] = r[j][0]
                    trace["Rest_" + d["name"]] = r[j][1:]
            if return_trace:
                return q(trace), trace
            return q(trace)
        return inner

def eval_methods(method, x, domain, i=0):
    res = np.empty(len(method), dtype=object)
    for j, m in enumerate(method):
        if isinstance(m, list):
            res[j] = eval_methods(m, x, domain, i)
        else:
            c = np.asarray([di(parameters.SAMPLES) for di in m(x[0][i:i+3], domain)])
            i+=3
            if len(method) == 1:
                return c
            res[j]=c
    return res