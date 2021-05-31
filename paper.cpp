

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

// Tiled dataflow, batch parallel
for(int b = 0; b != Batch; b += TileBB) { // block.x
  for(int n = 0; n != Neuron; n += TileBN) {  // block.y
    for(int bb = b; bb < b + TileBB; bb += TileTB) { 
      for(int bbb = bb; bbb < bb + TileTB; ++bbb) { // thread.x
        for(int k = 0; k < nnzs; ++k) {
          for(int nn = n; nn < n + TileBN; nn += TileTN) { 
            for(int nnn = nn; nnn < nn + TileTN; ++nnn) { 
              int nnzs = GetNNZs(W, nnn);
              int idx = GetIdx(W, nnn, k);
              float val_w = GetIdx(W, nnn, k); // Register reuse
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
        



// dataflow 2

}