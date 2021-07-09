# SpDNN Graph_challenge
source code for Sparse Deep Neural Network Graph Challenge (more detail:http://graphchallenge.mit.edu/challenges).


## Get Start
First, clone the project and download the dataset.
```
git clone https://github.com/CGCL-codes/Graphchallenge21.git
cd Graphchallenge21
mkdir data/
wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/dnn/neuron1024.tar.gz
wget https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019/mnist/sparse-images-1024.tsv.gz
tar –xzf neuron1024.tar.gz
tar –xzf sparse-images-1024.tsv.gz
```
Then, compile and run on single GPU version.
```
cd src/
nvcc -std=c++11 -O3 -o single.out network.cpp ./microbenchmark/all_network.cu
./single.out 1024 6000 120
```

