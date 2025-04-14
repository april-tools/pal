import argparse

import json
import matplotlib.pyplot as plt
import numpy as np
import os

# plt.style.use("ggplot")

DEF_FOLDER = "latte_gasp/results/"
INTEGRATORS = ["latte", "torch"]

MIN_EXP = -2
TIMEOUT_VAL = 3600
MISSING_VAL = TIMEOUT_VAL

TITLE = "" #"Runtime of SAE4WMI(latte) vs. SAE4WMI(torch)"
XLABEL = "SAE4WMI(latte) [seconds]"
YLABEL = "SAE4WMI(gasp!) [seconds]"
MARKER = "x"
# get the first color in default color cycle

first_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

COLOR = first_color
ALPHA = 0.9
TIMEOUT_COLOR = "red"
# get the grey color of the grid from matplotlib
grid_color = plt.rcParams['grid.color']

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
                if points[p][int_id] is None or runtime < points[p][int_id]:
                    points[p][int_id] = runtime
    
    print(f"Parsed {len(points)} points.")

    # preprocessing results
    def process(val):
        #if val is None: return MISSING_VAL
        return val or MISSING_VAL
    
    xs, ys = [], []
    latte_faster = 0
    min_val = np.inf
    for key, xy in points.items():
        if xy[0] is None:
            print(f"Missing value for {key} for latte.")
        if xy[1] is None:
            print(f"Missing value for {key} for torch.")
        x, y = map(process, xy)
        xs.append(x)
        ys.append(y)
        if x < y:
            latte_faster += 1
            print(f"LattE is faster on {key}.")
        if x < min_val: min_val = x
        if y < min_val: min_val = y

    print(f"LattE is faster on {latte_faster}/{len(points)} instances.")

    # plotting results
    fig = plt.figure(figsize=(2.5, 2.5))
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
        ax.plot([10 ** (MIN_EXP + i), TIMEOUT_VAL], [10 ** MIN_EXP, 10**(np.log10(TIMEOUT_VAL) - i)], linestyle="--", color=DIAGONAL_COLOR, alpha=(1 - 0.2*i))
        ax.plot([10 ** MIN_EXP, 10**(np.log10(TIMEOUT_VAL) - i)], [10 ** (MIN_EXP + i), TIMEOUT_VAL], linestyle="--", color=DIAGONAL_COLOR, alpha=(1 - 0.2*i))
    
    ax.scatter(xs, ys, marker=MARKER, color=COLOR, alpha=ALPHA)

    # plot x and y grid lines, but only major ticks
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.xaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=15))
    ax.yaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=15))
    # set scientific notation for ticks
    # ax.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'$10^{{{"+" if int(np.log10(x)) >= 0 else ""}{int(np.log10(x))}}}$' if x != 0 else '0'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'$10^{{{"+" if int(np.log10(y)) >= 0 else ""}{int(np.log10(y))}}}$' if y != 0 else '0'))
    ax.grid(which="major", axis="both", linestyle="--")
    ax.set_axisbelow(True)

    # x ticks rotated
    # plt.xticks(rotation=45)

    # skip the first x tick label
    x_ticks = ax.xaxis.get_major_ticks()
    x_ticks[0].label1.set_visible(False) ## set first x tick label invisible

    # from matplotlib.ticker import MaxNLocator
    # ax.xaxis.set_major_locator(MaxNLocator(prune='lower'))


    plt.savefig("scatter.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()
            

            
            
        




