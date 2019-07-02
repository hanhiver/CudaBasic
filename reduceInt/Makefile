CU_APPS=reduceInt

C_APPS=

CU_ARGS = -arch sm_50 -O2
C_ARGS = -O2

all: $(CU_APPS) $(C_APPS)

%: %.cu
	nvcc $(CU_ARGS) -o $@ $<

%: %.c
	gcc $(C_ARGS) -o $@ $<

clean:
	rm -rf $(CU_APPS) $(C_APPS)

#reduceInt: reduceInt.cu
#	nvcc $(ARGS) reduceInt.cu -o reduceInt

#clean:
#	rm -rf reduceInt


