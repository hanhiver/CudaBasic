all: hello

hello: helloworld.cu
	nvcc -arch sm_30 helloworld.cu -o hello

clean:
	rm -rf hello


