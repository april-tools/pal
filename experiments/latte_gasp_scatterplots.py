import argparse

import json
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use("ggplot")

DEF_FOLDER = "latte_gasp/results/"
INTEGRATORS = ["latte", "torch"]

MIN_EXP = -2
TIMEOUT_VAL = 3600
MISSING_VAL = TIMEOUT_VAL

TITLE = "" #"Runtime of SAE4WMI(latte) vs. SAE4WMI(torch)"
XLABEL = "SAE4WMI(latte) [seconds]"
YLABEL = "SAE4WMI(gasp!) [seconds]"
MARKER = "x"
COLOR = "blue"
ALPHA = 0.9
TIMEOUT_COLOR = "red"
DIAGONAL_COLOR = "grey"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate scatterplot for the LattE vs. GASP! experiment.")
    parser.add_argument("-i", "--input", nargs="+", default=[DEF_FOLDER], help="Folder containing result files.")
    parser.add_argument("-o", "--output", default=os.path.join(os.getcwd(), "scatter.pdf"), help="Output path.")
    parser.add_argument("--title", type=str, default=None, help="Title to plot")    
    args = parser.parse_args()

    # retrieving results
    results_files = []
    for folder in args.input:
        for sub in os.listdir(folder):
            fullpath = os.path.join(folder, sub)
            if os.path.isfile(fullpath) and fullpath.endswith(".json"):
                results_files.append(fullpath)
            else:
                results_files.extend(
                    [os.path.join(fullpath, f)
                     for f in os.listdir(fullpath)
                     if os.path.isfile(os.path.join(fullpath, f))
                     and f.endswith(".json")]
                )

    print(f"Found {len(results_files)} results files.")

    # parsing results
    points = dict()
    for res_file in results_files:

        with open(res_file, "r") as f:
            res_dict = json.load(f)
            int_name = res_dict["integrator"]["name"]
            assert(int_name in INTEGRATORS), f"{int_name} unsupported."
            int_id = INTEGRATORS.index(int_name)

            for entry in res_dict["results"]:

                p = entry["filename"]
                if p not in points:
                    points[p] = [None, None]

                runtime = entry["parallel_time"]
                points[p][int_id] = runtime
    
    print(f"Parsed {len(points)} points.")

    # preprocessing results
    def process(val):
        #if val is None: return MISSING_VAL
        return val or MISSING_VAL
    
    xs, ys = [], []
    latte_faster = 0
    min_val = np.inf
    for xy in points.values():
        x, y = map(process, xy)
        xs.append(x)
        ys.append(y)
        if x < y: latte_faster += 1
        if x < min_val: min_val = x
        if y < min_val: min_val = y

    print(f"LattE is faster on {latte_faster}/{len(points)} instances.")

    # plotting results
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(TITLE)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_aspect("equal")

    ax.set_xlim([10 ** MIN_EXP, TIMEOUT_VAL + 10**3])
    ax.set_ylim([10 ** MIN_EXP, TIMEOUT_VAL + 10**3])
    ax.set_xscale("log")
    ax.set_yscale("log")
        
    ax.axhline(TIMEOUT_VAL, 0, TIMEOUT_VAL, linestyle="--", color=TIMEOUT_COLOR) # timeout lines and label
    ax.axvline(TIMEOUT_VAL, 0, TIMEOUT_VAL, linestyle="--", color=TIMEOUT_COLOR)
    ax.annotate("timeout", (0.02, 0.9), xycoords='axes fraction', color=TIMEOUT_COLOR)

    ax.plot([10 ** MIN_EXP, TIMEOUT_VAL], [10 ** MIN_EXP, TIMEOUT_VAL], linestyle="--", color=DIAGONAL_COLOR) # diagonals
    for i in range(1, 6):
        ax.plot([10 ** (MIN_EXP + i), TIMEOUT_VAL], [10 ** MIN_EXP, 10**(np.log10(TIMEOUT_VAL) - i)], linestyle="--", color=DIAGONAL_COLOR, alpha=(1 - 0.1*i))
        ax.plot([10 ** MIN_EXP, 10**(np.log10(TIMEOUT_VAL) - i)], [10 ** (MIN_EXP + i), TIMEOUT_VAL], linestyle="--", color=DIAGONAL_COLOR, alpha=(1 - 0.2*i))
    
    ax.scatter(xs, ys, marker=MARKER, color=COLOR, alpha=ALPHA)
    plt.savefig("scatter.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()
            

            
            
        




