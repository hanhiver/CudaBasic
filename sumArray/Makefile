ARGS = -arch sm_70

all: host gpu matrix checkDimension defineGridBlock checkThreadIndex

host: sumArrayOnHost.cu
	nvcc sumArrayOnHost.cu -o host

gpu: sumArrayOnGPU.cu
	nvcc $(ARGS) sumArrayOnGPU.cu -o gpu

matrix: sumMatrixOnGPU-2d.cu
	nvcc $(ARGS) sumMatrixOnGPU-2d.cu -o matrix

checkDimension: checkDimension.cu
	nvcc checkDimension.cu -o checkDimension

defineGridBlock: defineGridBlock.cu
	nvcc defineGridBlock.cu -o defineGridBlock

checkThreadIndex: checkThreadIndex.cu
	nvcc $(ARGS) checkThreadIndex.cu -o checkThreadIndex

clean:
	rm -rf host gpu matrix checkDimension defineGridBlock checkThreadIndex


