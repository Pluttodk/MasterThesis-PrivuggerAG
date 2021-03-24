from typing import * 
import optimizer as opt
import pymc3 as pm
import numpy as np
import parameters

def convert_primitive_to_dist(p, i, is_list):
    if p == float:
        def float_convert(x, domain):
            print(i, domain)
            size = parameters.DB_SIZE if is_list else 1
            print(f"float_{i}")
            return opt.map_int_to_cont_dist(f"float_{i}", x, domain[i], shape=size)
        return float_convert
    elif p == int:
        def int_convert(x, domain):
            print("int", i)
            size = parameters.DB_SIZE if is_list else 1
            return opt.map_int_to_cont_dist(f"int_{i}", x, domain[i], shape=size)
        return int_convert

def convert_non_primitive_to_dist(p,i):
    inner = p.__args__
    partial = []
    for pi in inner:
        if pi == int or pi == float:
            is_list = p.__origin__ == List
            partial.append(convert_primitive_to_dist(pi, i, is_list))
            i+=1
        else:
            temp, i = convert_non_primitive_to_dist(pi, i)
            partial.append(temp)
    return partial, i

def parse(f):
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
            methods.append(convert_primitive_to_dist(p, i, False))
        else:
            # Under this assumption we now have a list or tuple
            temp, i = convert_non_primitive_to_dist(p,i)
            methods.append(temp)
    return methods


def create_analytical_method(f, q, domain):
    methods = parse(f)
    if len(list(f.__annotations__.values())) == 2:
        if isinstance(methods[0], list):
            def inner(x):
                np.random.seed(12345)
                with pm.Model() as model:
                    items = eval_methods(methods[0], 0, x, domain)
                    out = pm.Deterministic("out", f(items))
                    trace = pm.sample(1_000, return_inferencedata=False, cores=1)
                    return q(trace["float_1"],trace["out"])
            return inner
        else:
            def inner(x):
                np.random.seed(12345)
                with pm.Model() as model:
                    res = methods[0](x[0])
                    pm.Deterministic("out", f(res))
                    trace = pm.sample(1_000, return_inferencedata=False, cores=1)
                    return q(trace["alice"],trace["out"])
            return inner
    else:
        # We will have to do a larger test, such that every parameter becomes a 
        def inner(x):
            np.random.seed(12345)
            with pm.Model() as model:
                objects = len(np.asarray(methods).flatten())
                items = eval_methods(methods, 0, x, domain)
                out = pm.Deterministic("out", f(*items))
                trace = pm.sample(1_000, return_inferencedata=False, cores=1)
                return q(trace["float_0"],trace["out"])
        return inner

def eval_methods(method, i, x, domain):
    res = np.empty(len(method), dtype=object)
    for j, m in enumerate(method):
        if isinstance(m, list):
            res[j] = eval_methods(m, i, x, domain)
        else:
            c = m(x[0][i:i+3], domain)
            db = np.empty(parameters.DB_SIZE, dtype=object)
            for k in range(parameters.DB_SIZE):
                db[k] = c
            i+=3
            if len(method) == 1:
                return db
            res[j]=db
    return res