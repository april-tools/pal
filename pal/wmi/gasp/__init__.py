# add the gasp/ folder to sys path
import os
import sys

# 2. get the current folder
current_folder = os.path.dirname(__file__)
# 3. add the gasp/ folder to sys path
if os.path.isdir(current_folder):
    sys.path.insert(0, current_folder)