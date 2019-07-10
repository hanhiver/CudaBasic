CU_APPS=transpose

C_APPS=

# CU_ARGS = -arch sm_50 -Xptxas -dlcm=ca 
CU_ARGS = -arch sm_50
C_ARGS = -O2

all: $(CU_APPS) $(C_APPS)

%: %.cu Makefile
	nvcc $(CU_ARGS) -o $@ $<

%: %.c Makefile
	gcc $(C_ARGS) -o $@ $<

clean:
	rm -rf $(CU_APPS) $(C_APPS)



