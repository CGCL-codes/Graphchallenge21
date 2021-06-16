# ----- Make Macros -----

CXX = mpicxx
CXXFLAGS = -std=c++11 -fopenmp
OPTFLAGS = -O3

NVCC = nvcc
NVCCFLAGS = -lineinfo -O3 -std=c++11 -gencode arch=compute_60,code=sm_60 -ccbin=mpicxx -Xcompiler -fopenmp

LD_FLAGS = -ccbin=mpicxx -Xcompiler -fopenmp -lcusparse_static

TARGETS = inference
OBJECTS = main.o

# ----- Make Rules -----

all:	$(TARGETS)

%.o : %.cu vars.h
	${NVCC} ${NVCCFLAGS} $< -c -o $@

inference: $(OBJECTS)
	$(NVCC) -o $@ $(OBJECTS) $(LD_FLAGS)

clean:
	rm -f $(TARGETS) *.o *.o.* *.txt *.bin core *.html *.xml