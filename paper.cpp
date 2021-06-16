

int Batch, Neuron, nnzs;

void foo(){

int** C, A, B;

int ReLU(int a);
int bias;


// simple dataflow
for(int b = 0; b != Batch; ++b) {
  for(int n = 0; n != Neuron; ++n) {
    int nnzs = GetNNZs(W, n);
    for(int k = 0; k < nnzs; ++k) {
      int idx = GetIdx(W, n, k);
      float val = GetIdx(W, n, k);
      C(b, n) += A(b, idx) * val;       
    }
  }
}
for(int b = 0; b != Batch; ++b) {
  for(int n = 0; n != Neuron; ++n) {
    C(b, n) = ReLU(C(b, n) + bias);
  }
}


// fused dataflow
for(int b = 0; b != Batch; ++b) {
  for(int n = 0; n != Neuron; ++n) {
    int nnzs = GetNNZs(W, n);
    for(int k = 0; k < nnzs; ++k) {
      int idx = GetIdx(W, n, k);
      float val_w = GetIdx(W, n, k);
      float val_a = A(b, idx);
      C(b, n) += val_a * val_w;       
    }
    C(b, n) = ReLU(C(b, n) + bias);
  }
}

// Tiled dataflow
for(int b = 0; b != Batch; b += TileBB) {
  for(int n = 0; n != Neuron; n += TileBN) { 
    for(int bb = b; bb < b + TileBB; bb += TileTB) {
      for(int nn = n; nn < n + TileBN; nn += TileTN) { 
        for(int bbb = bb; bbb < bb + TileTB; ++bbb) {
          for(int nnn = nn; nnn < nn + TileTN; ++nnn) {
            int nnzs = GetNNZs(W, nnn);
            for(int k = 0; k < nnzs; ++k) {
              int idx = GetIdx(W, nnn, k);
              float val_w = GetIdx(W, nnn, k);
              float val_a = A(bbb, idx);
              C(bbb, nnn) += val_a * val;       
            }
            C(bbb, nnn) = ReLU(C(bbb, nnn) + bias);
          }
        }
      }
    }
  }
}

// Tiled dataflow, output parallel
for(int b = 0; b != Batch; b += TileBB) { // block.x
  for(int n = 0; n != Neuron; n += TileBN) {  // block.y
    for(int nn = n; nn < n + TileBN; nn += TileTN) { 
      for(int nnn = nn; nnn < nn + TileTN; ++nnn) { // thread.x
        int nnzs = GetNNZs(W, nnn);
        for(int k = 0; k < nnzs; ++k) {
          int idx = GetIdx(W, nnn, k);
          float val_w = GetIdx(W, nnn, k); // W reuse 
          for(int bb = b; bb < b + TileBB; bb += TileTB) { 
            for(int bbb = bb; bbb < bb + TileTB; ++bbb) {
              float val_a = A(bbb, idx);
              C(bbb, nnn) += val_a * val;       
            }
            C(bbb, nnn) = ReLU(C(bbb, nnn) + bias);
          }
        }
      }
    }
  }
}


// Tiled dataflow, output parallel
...
for(int nnn = nn; nnn < nn + TileTN; ++nnn) { // thread.x
  ...
  for(int k = 0; k < nnzs; ++k) {
    int idx = GetIdx(W, nnn, k);
    float val_w = GetIdx(W, nnn, k); // W reuse 
    ...
    for(int bbb = bb; bbb < bb + TileTB; ++bbb) {
      float val_a = A(bbb, idx);
      C(bbb, nnn) += val_a * val;       
    }

...
for(int bbb = bb; bbb < bb + TileTB; ++bbb) { // thread.x
  for(int k = 0; k < nnzs; ++k) {
      ...
      for(int nnn = nn; nnn < nn + TileTN; ++nnn) { 
        int nnzs = GetNNZs(W, nnn);
        int idx = GetIdx(W, nnn, k);
        float val_w = GetIdx(W, nnn, k); // Register reuse
        float val_a = A(bbb, idx);
        C(bbb, nnn) += val_a * val;       
      }
      C(bbb, nnn) = ReLU(C(bbb, nnn) + bias);


// Tiled dataflow, batch parallel
for(int b = 0; b != Batch; b += TileBB) { // block.x
  for(int n = 0; n != Neuron; n += TileBN) {  // block.y
    for(int bb = b; bb < b + TileBB; bb += TileTB) { 
      for(int bbb = bb; bbb < bb + TileTB; ++bbb) { // thread.x
        for(int nn = n; nn < n + TileBN; nn += TileTN) { 
          for(int k = 0; k < N; ++k) {
            float val_a = A(bbb, k);
            for(int nnn = nn; nnn < nn + TileTN; ++nnn) { 
              float val_w = W(nnn, k)
              C(bbb, nnn) += val_a * val;       
            }
            C(bbb, nnn) = ReLU(C(bbb, nnn) + bias);
          }
        }
      }
    }
  }
}


   
for(int b = 0; b != Batch; b += TileBB) {
  for(int n = 0; n != Neuron; n += TileBN) {
    Cache A(b:b+TileBB, --) in Shared Memory
    Cache W(n:n+TileBN, --) in Shared Memory
    for(int bb = b; bb < b + TileBB; bb += TileTB) {
      for(int nn = n; nn < n + TileBN; nn += TileTN) {
        for(int bbb = bb; bbb < bb + TileTB; ++bbb) 
          for(int nnn = nn; nnn < nn + TileTN; ++nnn) {
            int nnzs = GetNNZs(W, nnn);
            for(int k = 0; k < nnzs; ++k) {
              int idx = GetIdx(W, nnn, k);
              float val = GetIdx(W, nnn, k);
              C(bbb, nnn) += A(bbb, idx) * val;    
            }
            C(bbb, nnn) = ReLU(C(bbb, nnn) + bias);
          }       
      }
    }
  }
}


for(int b = 0; b != Batch; b += TileBB) {
  for(int n = 0; n != Neuron; n += TileBN) {
    Cache A(b:b+TileBB, --) in Shared Memory
    Cache W(n:n+TileBN, --) in Shared Memory
    for(int bb = b; bb < b + TileBB; bb += TileTB) {
      for(int nn = n; nn < n + TileBN; nn += TileTN) {
        for(int nnn = nn; nnn < nn + TileTN; ++nnn) {
          int nnzs = GetNNZs(W, nnn);
          for(int k = 0; k < nnzs; ++k) {
            int idx = GetIdx(W, nnn, k);
            float val = GetIdx(W, nnn, k);
            for(int bbb = bb; bbb < bb + TileTB; ++bbb) {
              C(bbb, nnn) += A(bbb, idx) * val;    
            }
            C(bbb, nnn) = ReLU(C(bbb, nnn) + bias);
          }       
        }
      }
    }
  }
}
        



// dataflow fuse



for(int b = 0; b != Batch; b += TileBB) {
  for(int n = 0; n != Neuron; n += TileBN) { 
    for(int bb = b; bb < b + TileBB; bb += TileTB) {
      for(int nn = n; nn < n + TileBN; nn += TileTN) { 
        for(int bbb = bb; bbb < bb + TileTB; ++bbb) {
          for(int nnn = nn; nnn < nn + TileTN; ++nnn) {
            int nnzs = GetNNZs(W1, nnn);
            for(int k = 0; k < nnzs; ++k) {
              int idx = GetIdx(W1, nnn, k);
              float val_w = GetIdx(W1, nnn, k);
              float val_a = Y1(bbb, idx);
              Y2(bbb, nnn) += val_a * val;       
            }
            Y2(bbb, nnn) = ReLU(Y1(bbb, nnn) + bias);



for(int b = 0; b != Batch; b += TileBB) {
  for(int n = 0; n != Neuron; n += TileBN) { 
    for(int bb = b; bb < b + TileBB; bb += TileTB) {
      for(int nn = n; nn < n + TileBN; nn += TileTN) { 
        for(int bbb = bb; bbb < bb + TileTB; ++bbb) {
          for(int nnn = nn; nnn < nn + TileTN; ++nnn) {
            int nnzs = GetNNZs(W2, nnn);
            for(int k = 0; k < nnzs; ++k) {
              int idx = GetIdx(W2, nnn, k);
              float val_w = GetIdx(W2, nnn, k);
              float val_a = Y2(bbb, idx);
              Y3(bbb, nnn) += val_a * val;       
            }
            Y3(bbb, nnn) = ReLU(Y2(bbb, nnn) + bias);



for(int b = 0; b != Batch; b += TileBB) {
  for(int n = 0; n != Neuron; n += TileBN) { 
    for(int bb = b; bb < b + TileBB; bb += TileTB) {
      for(int nn = n; nn < n + TileBN; nn += TileTN) { 
        for(int bbb = bb; bbb < bb + TileTB; ++bbb) {
          for(int nnn = nn; nnn < nn + TileTN; ++nnn) {
            int nnzs = GetNNZs(W1, nnn);
            for(int k = 0; k < nnzs; ++k) {
              int idx = GetIdx(W1, nnn, k);
              float val_w = GetIdx(W1, nnn, k);
              float val_a = Y1(bbb, idx);
              Y2(bbb, nnn) += val_a * val;       
            }
            Y2(bbb, nnn) = ReLU(Y1(bbb, nnn) + bias);

for(int b = 0; b != Batch; b += TileBB) {
  for(int n = 0; n != Neuron; n += TileBN) { 
    for(int bb = b; bb < b + TileBB; bb += TileTB) {
      for(int nn = n; nn < n + TileBN; nn += TileTN) { 
        for(int bbb = bb; bbb < bb + TileTB; ++bbb) {
          for(int nnn = nn; nnn < nn + TileTN; ++nnn) {
            int nnzs = GetNNZs(W2, nnn);
            for(int k = 0; k < nnzs; ++k) {
              int idx = GetIdx(W2, nnn, k);
              float val_w = GetIdx(W2, nnn, k);
              float val_a = Y2(bbb, idx);
              Y3(bbb, nnn) += val_a * val;       
            }
            Y3(bbb, nnn) = ReLU(Y2(bbb, nnn) + bias);


}













///////////////////////////////////////////////////

for(int l = 0; l < layer, l++) {
  ...
  spmm_kerenl<<<...>>>(I[l], O[l], W[l], active_d, ...)
  cudaMemcpy(active, active_d, ...)
  ...
}





for(int b = 0; b != Batch; b++) 
  for(int n = 0; n != Neuron; n++) 
    for(int k = 0; k != Neuron; k++) 
      R(b, n) += Y(b, k) * W(k, n); 


for(int n = 0; n != Neuron; n++) 
  for(int b = 0; b != Batch; b++) 
    for(int k = 0; k != Neuron; k++) 
      R(b, n) += Y(b, k) * W(k, n); 


for(int b = 0; b != Batch; b += TileBB) 
  for(int n = 0; n != Neuron; n += TileBN)   
    for(int bb = b; bb != b + TileBB; bb += TileTB) 
      for(int nn = n; nn != n + TileBN; nn += TileTN) 
        for(int k = 0; k != Neuron; k += TileBK) 
          for(int kk = k; k != k + TileBK; k += TileTK) 
            for(int bbb = bb; bbb != bb + TileTB; ++bbb) 
              for(int nnn = nn; nnn != nn + TileTN; ++nnn) 
                for(int kkk = kk; kkk != kk + TileTK; ++kkk) 
                  R(bbb, nnn) += Y(bbb, kkk) * W(kkk, nnn);


for(int b = 0; b != Batch; b++) --------------- blockIdx.x
  for(int n = 0; n != Neuron; n += TileBN) 
    for(int k = 0; k != Neuron; k++) ---------- threadIdx.y
      for(int nn = n; nn != n + TileBN; nn++)-- threadIDx.x
        R(b, nn) +=  Y(b, k) * W(k, nn);


for(int b = 0; b != Batch; b++) --------------- blockIdx.x
  for(int n = 0; n != Neuron; n += TileBN)
    for(int k = 0; k != Neuron; k++) ---------- threadIdx.y
      tile_begin = rowOff[..] + threadIdx.x --- threadIDx.x
      tile_end = rowOff[.. + 1]
      for(int nn = tile_begin; nn != tile_end; nn++)
        R(b, nn) +=  Y(b, k) * W(k, nn);



for(int b = 0; b != Batch; b += TileBB) -------------------- blockIdx.x
  for(int n = 0; n != Neuron; n += TileBN)   --------------- blockIdx.y
    for(int bb = b; bb != b + TileBB; bb += TileTB) 
      for(int nn = n; nn != n + TileBN; nn += TileTN) 
        for(int k = 0; k != Neuron; k += TileBK) 
          for(int kk = k; k != k + TileBK; k += TileTK) 
            for(int bbb = bb; bbb != bb + TileTB; ++bbb)---- threadIDx.x 
            or
            for(int nnn = nn; nnn != nn + TileTN; ++nnn)---- threadIDx.x
              for(int kkk = kk; kkk != kk + TileTK; ++kkk) 
                R(bbb, nnn) += Y(bbb, kkk) * W(kkk, nnn);


for(int b = 0; b != Batch; b += TileBB) -----------  blockIdx.x
  for(int n = 0; n != Neuron; n += TileBN)   ------- blockIdx.y
    for(int bbb = bb; bbb != bb + TileTB; ++bbb)---- threadIDx.x

      for(int kkk = kk; kkk != kk + TileTK; ++kkk) 
        R(bbb, nnn) += Y(bbb, kkk) * W(kkk, nnn);


for(int b = 0; b != Batch; b += TileBB) -----------  blockIdx.x
  for(int n = 0; n != Neuron; n += TileBN) --------- blockIdx.y
    for(int bbb = b; bbb != b + TileBB; ++bbb) ----- threadIDx.x
      for(int k = 0; k != Neuron; ++k) ------------- compact? Tiled?
        val_I = I(bbb, k);
        for(int nnn = n; nnn != n + TileBN; ++nnn)-- compact? Tiled?
          O(bbb, nnn) += val_I * W(k, nnn);


for(int b = 0; b != Batch; b += TileBB) -----------  blockIdx.x
  for(int n = 0; n != Neuron; n += TileBN) --------- blockIdx.y
    for(int nnn = n; nnn != n + TileBN; ++nnn) ----- threadIDx.x
      for(int k = 0; k != Neuron; ++k) ------------- compact? Tiled?
        val_W = W(k, nnn);
        for(int bbb = b; b != bb + TileBB; ++bbb) 
          O(bbb, nnn) += I(bbb, k) * val_W;


// 20 champion dataflow
for(int b = 0; b != Batch; b += TileBB) { // blockIdx.x
  for(int n = 0; n != Neuron; n += TileBN) { // blockIdx.y
    for(int nn = n; nn != n + TileBN; nn += TileTN) { 
      for(int nnn = nn; nnn != nn + TileTN; ++nnn) { // threadIdx.x
        for(int k = 0; k != Neuron; ++k) {
          val_W = W(k, nnn);
          for(int bb = b; bb != b + TileBB; bb += TileTB) { 
            for(int bbb = bb; bbb != bb + TileTB; ++bbb) {
              O(bbb, nnn) += I(bbb, k) * val_W;
            }
          }
        }
      }
    }
  }
}


// 20 champion dataflow - modify
for(int b = 0; b != Batch; b += TileBB) { // blockIdx.x
  for(int n = 0; n != Neuron; n += TileBN) { // blockIdx.y
    for(int nn = n; nn != n + TileBN; nn += TileTN) { 
      for(int k = 0; k != Neuron; ++k) {
        for(int bb = b; bb != b + TileBB; bb += TileTB) { 
          for(int bbb = bb; bbb != bb + TileTB; ++bbb) {
            val_I = I(bbb, k)
            for(int nnn = nn; nnn != nn + TileTN; ++nnn) { // threadIdx.x
              O(bbb, nnn) += val_I * W(k, nnn);
            }
          }
        }
      }
    }
  }
}

// batch parallel dataflow
for(int b = 0; b != Batch; b += TileBB) { // blockIdx.x
  for(int n = 0; n != Neuron; n += TileBN) { // blockIdx.y

    for(int bbb = b; bbb != b + TileBB; ++bbb) { // threadIdx.x
        for(int k = 0; k != Neuron; ++k) {
          val_I = I(bbb, k);
          for(int nn = n; nn != n + TileBN; nn += TileTN) { 
            for(int nnn = nn; nnn != nn + TileTN; ++nnn) { 
              O(bbb, nnn) += val_I * W(k, nnn);
            }
          }
        }
      }
    }
  }
}


/*
nnn : compact(k), 3 种
bbb : compact(n), 
*/
for(int b = 0; b != Batch; b += TileBB) {
  for(int n = 0; n != Neuron; n += TileBN) {


    for(int bb = b; bb != b + TileBB; bb += TileTB) { 
      for(int nn = n; nn != n + TileBN; nn += TileTN) { 

        for(int bbb = bb; bbb != bb + TileTB; ++bbb) {
          
          for(int nnn = nn; nnn != nn + TileTN; ++nnn) {

            for(int k = 0; k != Neuron; k += TileBK) {
              for(int kk = k; k != k + TileBK; k += TileTK) {
                for(int kkk = kk; kkk != kk + TileTK; ++kkk) {
                  O(bbb, nnn) += I(bbb, kkk) * W(kkk, nnn);
                }


              }
            }
          }
        }
      }
    }
  }
}


// fused

__global__ void layer_fuse_spmm_relu() {
  __shared__ float input_s[SIZE];
  for(int i = 0; i < inread_num[blockIdx.x]; ++i) {
    for(int j = 0; j < TileBB; ++j) {
      input_s[i * TileBB + j] = currfeat[];
    }
  }
  __syncthreads();
  float res[TileBB];
  int nnzs = weight_nnzs[];
  for(int i = 0; i < k; ++i) {
    for(int b = 0; b < TileBB; ++b) {
      res[b] += 
    }
  }


}