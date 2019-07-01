ARGS = -arch sm_50

all: reduceInt

reduceInt: reduceInt.cu
	nvcc $(ARGS) reduceInt.cu -o reduceInt

clean:
	rm -rf reduceInt


