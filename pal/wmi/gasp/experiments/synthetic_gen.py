#!/usr/bin/env python3

import os
import time

from sympy import Symbol
from wmibench.synthetic.synthetic_pa import generate_benchmark, check_input_output, ModelGenerator, log10
from os import path
from pysmt.shortcuts import BOOL, LT, REAL, And, Bool, Ite, Not, Or, Plus, Pow, Real, Symbol, Times, is_sat

from wmibench.io import Density

N_PROBLEMS = 10
SEED = 666

MIN_BOOLS = 0
MAX_BOOLS = 0
MIN_REALS = 3
MAX_REALS = 5
MIN_DEPTH = 2
MAX_DEPTH = 6
MIN_EXPONENT = 2
MAX_EXPONENT = 4

def custom_generate_benchmark(output, n_reals, n_bools, depth, n_models, max_exponent, seedn=None):
    output_dir = "r{}_b{}_d{}_m{}_e{}_s{}".format(n_reals, n_bools, depth, n_models, max_exponent, seedn)
    output_dir = path.join(output, output_dir)

    check_input_output(output, output_dir)
    # init generator
    templ_bools = ModelGenerator.TEMPL_BOOLS
    templ_reals = ModelGenerator.TEMPL_REALS
    gen = ModelGenerator(n_reals, n_bools, seedn=seedn, templ_bools=templ_bools, templ_reals=templ_reals)
    gen.MAX_EXPONENT = max_exponent 

    # generate models
    print("Starting creating models")
    time_start = time.time()
    digits = int(log10(n_models)) + 1
    template = "r{r}_b{b}_d{d}_exp{e}_s{s}_{templ}.json".format(r=n_reals, b=n_bools, d=depth, e=max_exponent, s=seedn, templ="{n:0{d}}")
    for i in range(n_models):
        support, bounds = gen.generate_support_tree(depth)
        weight = gen.generate_weights_tree(depth, nonnegative=True)
        domain = {Symbol(v, REAL): bounds[v] for v in bounds}
        for v in support.get_free_variables():
            if v not in domain:
                domain[v] = None

        for v in weight.get_free_variables():
            if v not in domain:
                domain[v] = None
        
        density = Density(support, weight, domain)
        density_file = path.join(output_dir, template.format(n=i + 1, d=digits))
        density.to_file(density_file)
        print("\r" * 100, end="")
        print("Model {}/{}".format(i + 1, n_models), end="")

    print()
    time_end = time.time()
    seconds = time_end - time_start
    print("Done! {:.3f}s".format(seconds))


def main():
    exp_dir = "synthetic_exp"
    data_dir = os.path.join(exp_dir, "data")
    res_dir = os.path.join(exp_dir, "results")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    for n_bools in range(MIN_BOOLS, MAX_BOOLS + 1):
        for n_reals in range(MIN_REALS, MAX_REALS + 1):
            for depth in range(MIN_DEPTH, MAX_DEPTH + 1):
                for exponent in range(MIN_EXPONENT, MAX_EXPONENT + 1, 2):
                    custom_generate_benchmark(data_dir, n_reals, n_bools, depth, N_PROBLEMS, exponent, SEED)


if __name__ == "__main__":
    main()
