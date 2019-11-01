CU_APPS=nestedHelloWorld

C_APPS=

CU_ARGS = -arch sm_50 -O2 -rdc=true 
C_ARGS = -O2

all: $(CU_APPS) $(C_APPS)

%: %.cu
	nvcc $(CU_ARGS) -o $@ $< -lcudadevrt

%: %.c
	gcc $(C_ARGS) -o $@ $<

clean:
	rm -rf $(CU_APPS) $(C_APPS)


