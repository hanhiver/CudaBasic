CU_APPS=gpuSleep

CU_ARGS = -arch sm_30 

all: $(CU_APPS) 

%: %.cu
	nvcc $(CU_ARGS) -o $@ $<

clean:
	rm -rf $(CU_APPS) 



