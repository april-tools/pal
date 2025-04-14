#!/bin/bash

MLC_DIR=mlc

for dir in $(ls -d $MLC_DIR/data/*)
do
	res_dir=$(sed "s+data+results+g" <<< $dir)
  mkdir -p $res_dir
	echo Evaluating $dir
	# for mode in XSDD XADD FXSDD "PA latte" "SAPA latte" "SAE4WMI latte"
        for mode in "SAE4WMI torch" "SAE4WMI latte" "SAE4WMI volesti"
        do
                echo Mode $mode
                python3 evaluateModels.py $dir -o $res_dir --timeout 1200 --n-threads 50 -m $mode
        done
done
