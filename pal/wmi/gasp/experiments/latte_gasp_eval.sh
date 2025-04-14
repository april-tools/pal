#!/bin/bash

LATTE_THREADS=20
SYN_DIR=latte_gasp
DATA_DIR=$SYN_DIR/data

for dir in $(ls -d $DATA_DIR/*)
do
	res_dir=$(sed "s+data+results+g" <<< $dir)
	mkdir -p $res_dir
	echo Evaluating $dir

	echo Mode SAE4WMI latte
	python3 evaluateModels.py  $dir -o $res_dir  --n-threads $LATTE_THREADS -m SAE4WMI latte

	echo Mode SAE4WMI torch
	python3 evaluateModels.py $dir -o $res_dir --n-threads $LATTE_THREADS -m SAE4WMI torch --monomials_use_float64 --sum_seperately --with_sorting

done
