ARGS = -arch sm_50

all: host gpu checkDimension defineGridBlock checkThreadIndex

host: sumArrayOnHost.cu
	nvcc sumArrayOnHost.cu -o host

gpu: sumArrayOnGPU.cu
	nvcc $(ARGS) sumArrayOnGPU.cu -o gpu

checkDimension: checkDimension.cu
	nvcc checkDimension.cu -o checkDimension

defineGridBlock: defineGridBlock.cu
	nvcc defineGridBlock.cu -o defineGridBlock

checkThreadIndex: checkThreadIndex.cu
	nvcc $(ARGS) checkThreadIndex.cu -o checkThreadIndex

clean:
	rm -rf host gpu checkDimension defineGridBlock


