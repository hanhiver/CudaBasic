all: checkDeviceInfo

ARGS = -arch sm_70

checkDeviceInfo: checkDeviceInfo.cu
	nvcc $(ARGS) checkDeviceInfo.cu -o checkDeviceInfo

clean:
	rm -rf checkDeviceInfo


