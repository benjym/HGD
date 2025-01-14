# Heterarchical Granular Dynamics

A python package for simulating the motion of a granular material as a result of the motion of voids. In particular, it (currently) models non-inertial problems quite well. In particular, segregation is well modelled for several systems.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Installation

1.  Download and unzip (or clone) this repository
2.  Install python 3.11 or newer
3.  Install the required python packages with `pip install -e .`
4.  Set up your pre-commit hooks (so that when you make changes things are kept nice and clean) by running `pre-commit install`
5.  Run the code with `python HGD/main.py json/collapse.json5`. The parameters for this specific case are stored in `json/collapse.json5`. Change that file name to a different `json5` file to use those values instead.

# Documentation
You can [read the docs here](https://pages.github.sydney.edu.au/bmar5496/void-migration/).

# Authors
- [Benjy Marks](mailto:benjy.marks@sydney.edu.au)
- [Shivakumar Athani](mailto:shivakumar.athani@sydney.edu.au)
- [Jiahuan Li](mailto:jiahuan.li@sydney.edu.au)
