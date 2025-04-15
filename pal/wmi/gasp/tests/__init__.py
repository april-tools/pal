# import gasp.torch.wmipa.wmipa_monkeypatch  # noqa

import os
import sys
current_folder = os.path.dirname(__file__)
gasp_folder = os.path.abspath(os.path.join(current_folder, ".."))

if os.path.isdir(gasp_folder):
    sys.path.insert(-1, gasp_folder)
    # print(f"added {gasp_folder} to sys path")