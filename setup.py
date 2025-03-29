from setuptools import setup, find_packages

setup(
    name="gasp-wmi",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Main dependency from Git
        # "wmipa @ git+https://github.com/unitn-sml/wmi-pa.git@dd0600c5e2018589f55c657fde6374a981cf4179",
        "wmipa @ git+https://github.com/LeanderK/wmi-pa.git@405ee9a0d88d8c1014da327aca68297c6f716c9e",
        # Explicitly specify sub-dependencies
        "networkx",
        "numpy",
        "PySMT>=0.9.6.dev53",
        # Install sympy>=1.13 but allow overriding wmipa's restrictive dependency
        "sympy>=1.13",
        "torch>=2.6.0",
        # test
        "pytest",
        "pytest-runner",

    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
