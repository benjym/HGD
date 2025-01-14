#!/bin/bash
# Run this script to generate the figures and tables for the Kinematic SLM paper
# This script will run the main.py script for each of the json5 files in the json directory
# It will then run the post processing script for each of the json5 files in the json directory
# Usage:
# In the terminal, navigate to the root directory of the repository (the one with `README.md` in it) and run the following command:
# ./generate_paper.sh

python void_migration/main.py papers/Kinematic_SLM/json/collapse_stress.json5 # 1
python papers/Kinematic_SLM/post_process/collapse_stress.py
echo "Collapse stress done"

python void_migration/main.py papers/Kinematic_SLM/json/collapse_fill.json5 # 3
python papers/Kinematic_SLM/post_process/collapse_fill.py
echo "Collapse fill done"

python void_migration/main.py papers/Kinematic_SLM/json/collapse_bi.json5 # 3
python papers/Kinematic_SLM/post_process/collapse_bi.py
echo "Collapse bi done"

python void_migration/main.py papers/Kinematic_SLM/json/collapse_poly.json5 # 3
python papers/Kinematic_SLM/post_process/collapse_poly.py
echo "Collapse poly done"

python void_migration/main.py papers/Kinematic_SLM/json/silo_alpha.json5 # 3
python papers/Kinematic_SLM/post_process/silo_alpha.py
echo "Silo alpha done"

python void_migration/main.py papers/Kinematic_SLM/json/collapse_angles.json5 # 5 x 2
python papers/Kinematic_SLM/post_process/collapse_angles.py
echo "Collapse angles done"