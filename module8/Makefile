all: asg8.cu
	nvcc asg8.cu -lcudart -lcuda -lcusparse -lcusolver -lcublas -lcurand -I common/inc -o assignment.exe

clean:
	rm assignment.exe