#include <iostream>
#include <fstream>

double** generateMatrix(int N, double min, double max);
double** generateVector(int N, double min, double max);
double* computeOnCPU(double** matrix, double* vector, int N);
__global__ void computeOnGPU(double** matrix, double* vector, int N);

int main (int argc, char* argv[]) {
	return 0;
}
