Execution instructions:
- Run "make"
- executable is then called "assignment.exe"

My implementation of adding each of these math operations was to take the (mod 5) of the input index for a data element to determine which operation to do on that data index, for example index 2 performs multiplication - which is checked within the kernel function and the corresponding value is filled in the same result storage as in the predefined example. 

Other changes included printing formatting changes to HelloWorld.cpp, and changing the program input argument to my assignment.cl file instead of the example "HelloWorld.cl", I kept the hello_kernel function signiture the same as requested in the assignment spec. 