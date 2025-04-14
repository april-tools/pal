#!/bin/bash

SYN_DIR=synthetic_exp

INPUT=$SYN_DIR/results/r[3-5]_b0_d[3-5]*
OUTPUT=$SYN_DIR/plots/

mkdir -p $OUTPUT

#python3 plot.py $INPUT -o $OUTPUT -f r3-5_b0_d2-6  --legend-pos 4 # --timeout 200
#python3 plot.py $INPUT -o $OUTPUT -f r3-5_b0_d2-6_cactus --cactus --legend-pos 2 #--timeout 200

python3 plot.py $INPUT -o $OUTPUT -f r3_b0_d2-6  --legend-pos 4 # --timeout 200
python3 plot.py $INPUT -o $OUTPUT -f r3_b0_d2-6_cactus --cactus --legend-pos 2 #--timeout 200
