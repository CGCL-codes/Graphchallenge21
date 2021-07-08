#!/bin/bash
for col_blk in 1024 512 256 128 64 32 16
do 
    for blockDim in 1024 512 256 128 64
    do
        for((blockx=1; blockx<=blockDim; blockx+=blockx))
        do
            blocky=`expr $blockDim / $blockx`
            echo "Run Config"
            echo $col_blk 
            echo $blockx 
            echo $blocky
            ./bf 1024 1000 1 2 $col_blk $blockx $blocky
        done
    done
done