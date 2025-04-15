import os
import sys
current_folder = os.path.dirname(__file__)
pal_folder = os.path.abspath(os.path.join(current_folder, ".."))

if os.path.isdir(pal_folder):
    sys.path.insert(-1, pal_folder)
    print(f"added {pal_folder} to sys path")