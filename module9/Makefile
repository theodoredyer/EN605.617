all: npp_asg.cu
	nvcc npp_asg.cu -lcudart -lcuda -I common/inc -o assignment.exe

clean:
	rm assignment.exe