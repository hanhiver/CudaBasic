CC=gcc-7

all: checkDeviceInfo

ARGS = -arch sm_70 -ccbin /usr/bin/gcc-7 -lm 

checkDeviceInfo: checkDeviceInfo.cu
	nvcc $(ARGS) checkDeviceInfo.cu -o checkDeviceInfo

clean:
	rm -rf checkDeviceInfo


