#!/bin/bash

LATTE_THREADS=20
SYN_DIR=synthetic_exp
DATA_DIR=$SYN_DIR/data

for dir in $(ls -d $DATA_DIR/*)
do
	res_dir=$(sed "s+data+results+g" <<< $dir)
	mkdir -p $res_dir
	echo Evaluating $dir

	echo Mode SAE4WMI latte
	python3 evaluateModels.py  $dir -o $res_dir  --n-threads $LATTE_THREADS -m SAE4WMI latte

	echo Mode SAE4WMI torch
	python3 evaluateModels.py $dir -o $res_dir -m SAE4WMI torch --monomials_use_float64 --sum_seperately --with_sorting


	error=0.1
	for N in 100 1000 10000
	do
	    echo "Mode SAE4WMI volesti, N $N, 5 seeds"
	    python3 evaluateModels.py $dir -o $res_dir -m SAE4WMI volesti -e $error -N $N --seed 666 --n-seeds 5
	done
done
