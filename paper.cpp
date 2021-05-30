

int Batch, Neuron, nnzs;

void foo(){

int** C, A, B;

int ReLU(int a);
int bias;

for(int b = 0; b != Batch; ++b)
  for(int n = 0; n != Neuron; ++n) {
    int nnzs = GetNNZs(W, n);
    for(int k = 0; k < nnzs; ++k) {
      int idx = GetIdx(W, n, k);
      float val = GetIdx(W, n, k);
      C(b, n) += A(b, idx) * val;       
    }
    C(b, n) = ReLU(C(b, n) + bias);
  }

// dataflow 1

for(int b = 0; b != Batch; b += TileBB) 
  for(int n = 0; n != Neuron; n += TileBN) 
    for(int bb = b; bb < b + TileBB; bb += TileTB) 
      for(int nn = n; nn < n + TileBN; nn += TileTN) 
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
                
            
        



// dataflow 2

}