/*
Problem Spec: 
- Create a program that utilizes the Thrust library to perform the following
    operations on two vectors. 
    add() - add the value of two vectors placing res in 3rd vector
    subtract() - sub values and place in 3rd vector
    multiply() - same thing but multiplications
    modulo() - a mod b = c

Theodore Dyer
Introduction to GPU Programming Spring 2022 (EN605.617.81)
Chance Pascale
4/13/2022
*/

// Note - using thrust_example.cpp as a guideline. 

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>

int main(void) {

    // 3 integer long vector
    thrust::host_vector<int> HA(3);
    thrust::host_vector<int> HB(3);

    // Init vector A elements
    HA[0] = 8;
    HA[1] = 12;
    HA[2] = 16;

    // Init vector B elements
    HB[0] = 2;
    HB[1] = 4;
    HB[2] = 6;


    // Print contents of input vectors
    printf("Input vectors: \n");
    for(int i = 0; i < HA.size(); i++) {
        std::cout << "HA[" << i << "] = " << HA[i] << std::endl;
    }
    printf("\n");
    for(int i = 0; i < HA.size(); i++) {
        std::cout << "HB[" << i << "] = " << HB[i] << std::endl;
    }

    // Set up device vectors and copy data from host
    thrust::device_vector<int> DA = HA;
    thrust::device_vector<int> DB = HB;
    thrust::device_vector<int> DADD = HB;
    thrust::device_vector<int> DSUB = HB;
    thrust::device_vector<int> DMUL = HB;
    thrust::device_vector<int> DMOD = HB;

    // Perform operations
    for (int i = 0; i < DA.size(); i++) {
        DADD[i] = DA[i] + DB[i];
        DSUB[i] = DA[i] - DB[i];
        DMUL[i] = DA[i] * DB[i];
        DMOD[i] = DA[i] % DB[i];
    }

    // print addition results
    printf("\nResult of addition: \n");
    for(int i = 0; i < DA.size(); i++) {
        std::cout << "DC[" << i << "] = " << DADD[i] << std::endl;
    }

    // print subtraction results
    printf("\nResult of subtraction: \n");
    for(int i = 0; i < DA.size(); i++) {
        std::cout << "DC[" << i << "] = " << DSUB[i] << std::endl;
    }

    // print multiplication results
    printf("\nResult of multiplication: \n");
    for(int i = 0; i < DA.size(); i++) {
        std::cout << "DC[" << i << "] = " << DMUL[i] << std::endl;
    }

    // print modulo results
    printf("\nResult of modulus: \n");
    for(int i = 0; i < DA.size(); i++) {
        std::cout << "DC[" << i << "] = " << DMOD[i] << std::endl;
    }


}