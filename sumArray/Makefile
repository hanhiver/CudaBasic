all: host gpu checkDimension defineGridBlock

host: sumArrayOnHost.cu
	nvcc sumArrayOnHost.cu -o host

gpu: sumArrayOnGPU.cu
	nvcc sumArrayOnGPU.cu -o gpu

checkDimension: checkDimension.cu
	nvcc checkDimension.cu -o checkDimension

defineGridBlock: defineGridBlock.cu
	nvcc defineGridBlock.cu -o defineGridBlock

clean:
	rm -rf host gpu checkDimension defineGridBlock


