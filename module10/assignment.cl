/*

Problem Spec: 
- Restructure the example program to include 5 basic math functions
	(add, sub, mul, div, pow) with the same signature as hello_kernel. 

Theodore Dyer
Introduction to GPU Programming Spring 2022 (EN605.617.81)
Chance Pascale
4/17/2022

*/


/*

hello_kernel()
- Implementation note, this function signature follows the same example as that in the
	original provided example, with one change being the addition of a flag variable.
	This this variable will determine which math operation to execute within the kernel 
	execution and will follow the key below: 

	flag = 0, perform addition
	flag = 1, perform subtraction
	flag = 2, perform multiplication
	flag = 3, perform division
	flag = 4, perform exponent operation 

*/
__kernel void hello_kernel(__global const float *a,
						__global const float *b,
						__global float *result)
{
    int gid = get_global_id(0);
	int flag = (gid % 5);

    result[gid] = a[gid] + b[gid];
	if(flag == 0) {
		// Addition (a + b)
		result[gid] = a[gid] + b[gid];
	} else if(flag == 1) {

		// Subtraction (b - a)
		result[gid] = b[gid] - a[gid];
	} else if(flag == 2) {

		// Multiplication (a * b)
		result[gid] = a[gid] * b[gid];
	} else if(flag == 3) {

		// Division (b / a)
		result[gid] = b[gid] / a[gid];
	} else if(flag == 4) {

		// Exponent operation (a squared)
		result[gid] = a[gid] * a[gid];
	}
}
