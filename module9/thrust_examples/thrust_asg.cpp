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
    thrust::host_vector<int> H(3);

    // Init vector A elements
    HA[0] = 2;
    HA[1] = 4;
    HA[2] = 8;

    // Init vector B elements
    HB[0] = 8;
    HB[1] = 12;
    HB[2] = 16;


    // Print contents of input vectors
    for(int i = 0; i < H.size(); i++) {
        std::cout << "HA[" << i << "] = " << HA[i] << std::endl;
        std::cout << "HB[" << i << "] = " << HB[i] << std::endl;
    }

    thrust::device_vector<int> DA = HA;
    thrust::device_vector<int> DB = HB;
    thrust::device_vector<int> DC = HB;

    for (int i = 0; i < DA.size(); i++) {
        DC[i] = DA[i] + DB[i];
    }

    // print contents of D
    for(int i = 0; i < D.size(); i++) {
        std::cout << "DC[" << i << "] = " << DC[i] << std::endl;
    }


}