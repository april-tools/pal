from collections import namedtuple
from multiprocessing import Queue, Process
from queue import Empty as EmptyQueueError

import psutil
from pysmt.shortcuts import Real, Bool
#from pywmi import Domain as PywmiDomain, PyXaddEngine, XsddEngine, PyXaddAlgebra, FactorizedXsddEngine as FXSDD, \
#    RejectionEngine
#from pywmi.engines.algebraic_backend import SympyAlgebra
#from pywmi.engines.xsdd.vtrees.vtree import balanced

# add to imports
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from wmipa import WMI
from wmipa.integration import LatteIntegrator, VolestiIntegrator, SymbolicIntegrator
from gasp.torch.wmipa.numerical_symb_integrator_pa import NumericalSymbIntegratorPA
from wmipa.utils import is_pow

import torch
import torch.multiprocessing as mp

WMIResult = namedtuple("WMIResult", ["wmi_id",
                                     "value",
                                     "n_integrations",
                                     "parallel_integration_time",
                                     "sequential_integration_time"])


def compute_total_degree(node):

    if node.is_constant():
        return 0
    elif node.is_symbol():
        return 1
    elif is_pow(node):
        base, exponent = node.args()
        return compute_total_degree(base) * int(exponent.constant_value())
    elif node.is_ite():
        return max([compute_total_degree(c) for c in node.args()[1:]])
    elif node.is_plus():
        return max([compute_total_degree(c) for c in node.args()])
    elif node.is_times():
        return sum([compute_total_degree(c) for c in node.args()])

def get_wmi_id(mode, integrator):
    """Return a string identifying the pair <mode, integrator>."""
    integrator_str = "" if integrator is None else f"_{integrator.to_short_str()}"
    return f"{mode}{integrator_str}"


def get_integrators(args):
    """Returns the integrators to be used for the given command line arguments."""
    if args.mode not in WMI.MODES:
        return [None]
    if args.integrator == "latte":
        return [LatteIntegrator(n_threads=args.n_threads, stub_integrate=args.stub)]
    if args.integrator == "torch":
        import time
        time_start = time.time()
        base = 10*1024
        if args.monomials_use_float64:
            batch_size = base
        else:
            batch_size = base
        if args.total_degree > 40:
            batch_size = int(base / 2)
        integrator = NumericalSymbIntegratorPA(
            total_degree=args.total_degree,
            variable_map=args.variable_map,
            batch_size=batch_size,
            n_workers=args.n_threads,
            monomials_lower_precision=not args.monomials_use_float64,
            sum_seperately=args.sum_seperately,
            with_sorting=args.with_sorting,
        )
        # TODO: hack
        if "mlc" in args.input:
            integrator.set_device(torch.device("cuda:1"))
        else:
            integrator.set_device(torch.device("cuda:0"))
        time_end = time.time()
        print(f"Time to create integrator: {time_end - time_start}\n")
        return [integrator]
    elif args.integrator == "volesti":
        seeds = list(range(args.seed, args.seed + args.n_seeds))
        return [VolestiIntegrator(n_threads=args.n_threads, stub_integrate=args.stub,
                                  algorithm=args.algorithm, error=args.error, walk_type=args.walk_type,
                                  walk_length=args.walk_length, seed=seed, N=args.N) for seed in seeds]
    elif args.integrator == "symbolic":
        return [SymbolicIntegrator(n_threads=args.n_threads, stub_integrate=args.stub)]
    else:
        raise ValueError(f"Invalid integrator {args.integrator}")


def compute_wmi(args, domain, support, weight):
    """Computes the WMI for the given domain, support and weight, using the mode define by args. The result is put in
    the queue q to be retrieved by the main process.
    """

    if args.unweighted:
        weight = Real(1)

    real_vars = {v: b for v, b in domain.items() if v.symbol_type().is_real_type()}
    variable_map = {str(v.symbol_name()): i for i, v in enumerate(real_vars.keys())}
    args.variable_map = variable_map
    if "mlc" in args.input:
        args.total_degree = 1
    else:
        args.total_degree = compute_total_degree(weight)
    
    bool_vars = {v for v in domain if v.symbol_type().is_bool_type()}
    if args.mode in WMI.MODES:
        integrators = get_integrators(args)
        wmi = WMI(support, weight, integrator=integrators)
        results, n_ints = wmi.computeWMI(
            Bool(True),
            mode=args.mode,
            cache=args.cache,
            domA=bool_vars,
        )
        res = []
        for result, n_int, integrator in zip(results, n_ints, integrators):
            wmi_id = get_wmi_id(args.mode, integrator)
            wmi_result = WMIResult(wmi_id=wmi_id,
                                   value=float(result),
                                   n_integrations=int(n_int),
                                   parallel_integration_time=integrator.get_parallel_integration_time(),
                                   sequential_integration_time=integrator.get_sequential_integration_time())
            res.append(wmi_result)
    else:
        # get pywmi domain from wmibench domain
        pywmi_domain = PywmiDomain.make(
            boolean_variables=[v.symbol_name() for v in bool_vars],
            real_variables={v.symbol_name(): b for v, b in real_vars.items()},
        )
        if args.mode == "XADD":
            wmi = PyXaddEngine(domain=pywmi_domain, support=support, weight=weight)
        elif args.mode == "XSDD":
            wmi = XsddEngine(
                domain=pywmi_domain,
                support=support,
                weight=weight,
                algebra=PyXaddAlgebra(symbolic_backend=SympyAlgebra()),
                ordered=False,
            )
        elif args.mode == "FXSDD":
            wmi = FXSDD(
                domain=pywmi_domain,
                support=support,
                weight=weight,
                vtree_strategy=balanced,
                algebra=PyXaddAlgebra(symbolic_backend=SympyAlgebra()),
                ordered=False,
            )
        elif args.mode == "Rejection":
            wmi = RejectionEngine(
                domain=pywmi_domain,
                support=support,
                weight=weight,
                sample_count=10 ** 6
            )
        else:
            raise ValueError(f"Invalid mode {args.mode}")

        res = [WMIResult(wmi_id=get_wmi_id(args.mode, None),
                         value=wmi.compute_volume(add_bounds=False),
                         n_integrations=None,
                         parallel_integration_time=0,
                         sequential_integration_time=0)]

    return res

def _wrapper(q, fn, *args, **kwargs):
    _res = fn(*args, **kwargs)
    q.put(_res)


def run_fn_with_timeout(fn, timeout, *args, **kwargs):
    """Run compute_wmi with a timeout. If the computation exceeds the timeout, a TimeoutError is raised."""
    is_torch = args[0].integrator == "torch"
    if is_torch:
        q = mp.Queue()
    else:
        q = Queue()

    # def _wrapper(*args, **kwargs):
    #     _res = fn(*args, **kwargs)
    #     q.put(_res)

    # if args[0].integrator == "torch":
    #     timed_proc = mp.Process(target=_wrapper, args=args, kwargs=kwargs, daemon=True)
    # else:
    #     timed_proc = Process(target=_wrapper, args=args, kwargs=kwargs)
    if args[0].integrator == "torch":
        timed_proc = mp.Process(target=_wrapper, args=(q, fn) + args, kwargs=kwargs, daemon=True)
    else:
        timed_proc = Process(target=_wrapper, args=(q, fn) + args, kwargs=kwargs)
    timed_proc.start()
    timed_proc.join(timeout)
    if timed_proc.is_alive():
        # kill the process and its children
        pid = timed_proc.pid
        proc = psutil.Process(pid)
        for subproc in proc.children(recursive=True):
            try:
                subproc.kill()
            except psutil.NoSuchProcess:
                continue
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
        raise TimeoutError()
    else:
        try:
            res = q.get(block=False)
        except EmptyQueueError:
            # killed because of exceeding resources
            raise TimeoutError()
    return res


if __name__ == '__main__':

    from pysmt.shortcuts import *

    w = Real(666)
    print(f"weight: {w} \t\t\t TD: {compute_total_degree(w)}")

    w = Symbol("x", REAL)
    print(f"weight: {w} \t\t\t TD: {compute_total_degree(w)}")

    w = Plus(Symbol("x", REAL), Real(666))
    print(f"weight: {w} \t\t\t TD: {compute_total_degree(w)}")

    w = Plus(Symbol("x", REAL), Symbol("y", REAL))
    print(f"weight: {w} \t\t\t TD: {compute_total_degree(w)}")

    w = Times(Symbol("x", REAL), Real(666))
    print(f"weight: {w} \t\t\t TD: {compute_total_degree(w)}")

    w = Times(Symbol("x", REAL), Symbol("y", REAL))
    print(f"weight: {w} \t\t\t TD: {compute_total_degree(w)}")

    w = Plus(Times(Symbol("x", REAL), Symbol("y", REAL)), Times(Symbol("x", REAL), Real(666)))
    print(f"weight: {w} \t\t\t TD: {compute_total_degree(w)}")

    w = Pow(Times(Symbol("x", REAL), Symbol("y", REAL)), Real(666))
    print(f"weight: {w} \t\t\t TD: {compute_total_degree(w)}")

    w = Ite(LE(Real(3), Real(1)), Pow(Times(Symbol("x", REAL), Symbol("y", REAL)), Real(666)), Real(1337))
    print(f"weight: {w} \t\t\t TD: {compute_total_degree(w)}")
