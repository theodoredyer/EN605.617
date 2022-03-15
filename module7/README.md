## Assignment 7
Theodore Dyer
Introduction to GPU Programming Spring 2022 (EN605.617.81)
Chance Pascale
3/14/2020

- Given the stretch problem for this module is to fix the provided file, my code for this is located in module7_stretch_problem.cu
- executable after running make should be assignment.exe

Note - One issue I'm encountering with my development is when I run each of the different type
of mathematical operation tests in separate program executions I'm getting distinct different execution times for each type of operation, but when I run them all in the same program execution they all give me the same answer, for example:

array size = 1024*1024*32

addition = (33.7-33.8) ms
subtraction = (33.7) ms
multiplication = (45) ms
division = (45) ms

but when I run them all in the same execution every single type of operation is stuck at 45 ms.
