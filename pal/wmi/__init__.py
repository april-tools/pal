# add the gasp/ folder to sys path
import os
import sys

current_folder = os.path.dirname(__file__)
gasp_folder = os.path.join(current_folder, "gasp")
if os.path.isdir(gasp_folder):
    sys.path.insert(-1, gasp_folder)