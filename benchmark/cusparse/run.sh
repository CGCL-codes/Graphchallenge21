#!/bin/bash
cp -f Makefile.volta Makefile
make clean;make -j 

echo "Starting Benchmark"
date

#export DATASET=/home/vsm2/SpDNN_Challenge2020/iostream/dataset
export DATASET=//home/xinjie/xinjie/graph_challenge/data
#export DATASET=/home/vsm2/dataset

#1024 4096 16384 65536
#export NEURON=65536
#-0.3 -0.35 -0.4 -0.45
#export BIAS=-0.45
#6374505 25019051 98858913 392191985
#export INPUT=392191985

#120 480 1920
#export LAYER=1920
export BATCH=60000

export BLOCKSIZE=256
export BUFFER=24

export OMP_NUM_THREADS=16


for neuron in 1024 4096 16384
do 
	for layer in 120 480 1920
	do 
		if [[ $neuron -eq 1024 ]]
		then 
			export BIAS=-0.3
			export INPUT=6374505
		fi
		if [[ $neuron -eq 4096 ]]
		then 
			export BIAS=-0.35
			export INPUT=25019051
		fi
		if [[ $neuron -eq 16384 ]]
		then 
			export BIAS=-0.4
			export INPUT=98858913
		fi
		if [[ $neuron -eq 65536 ]]
		then 
			export BIAS=-0.45
			export INPUT=392191985
		fi

		export NEURON=$neuron
		export LAYER=$layer

		echo $LAYER
		echo $NEURON
		echo $BIAS
		echo $INPUT
		echo $DATASET
		./inference

	done 

done


#for l in 120 480 1920
#do
#  export LAYER=$l
#  jsrun -n1 -a1 -g1 -c7 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
#  jsrun -n1 -a3 -g3 -c21 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
#  jsrun -n1 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
#  jsrun -n2 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
#  jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info ./inference
#done
#
date