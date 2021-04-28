#!/bin/bash 

DATA_SET_DIR="/home/xinjie/data/graph_challenge"

if [[ -z "${DATA_SET_DIR}" ]]; then 
       echo "ERROR: Please Set Data Set Dir Variable"	
       exit 0
fi




if [ $1 == "1024" ]; then 
    echo "Downloading Categories"
    wget -P $DATA_SET_DIR https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron1024-l120-categories.tsv 

    echo "Downloading Spase Images"

    wget -P $DATA_SET_DIR https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/mnist/sparse-images-1024.tsv.gz

    echo "Downloading Weights"
    wget -P $DATA_SET_DIR https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron1024.tar.gz

fi


