#include <iostream>
#include <random>
#include <ctime>
#include <string>
#include <sstream>
#include <time.h>

using namespace std;

double* generateMatrix(int N, double min, double max);
double* generateVector(int N, double min, double max);
double* computeOnCPU(double* matrix, double* vector, int N);
timespec diff(timespec start, timespec end);
__global__ void computeOnGPU(double* matrix, double* vector, double* result,
		int* N);

int main(int argc, char* argv[]) {
	srand (time(NULL));
	int N;
	double min, max;

	string s(argv[1]);
	string s2(argv[2]);
	string s3(argv[3]);

	N = stoi(s);
	min = stof(s2);
	max = stof(s3);

	int doubleSize = sizeof(double);

	double* matrix = generateMatrix(N, min, max);
	double* vector = generateVector(N, min, max);

	cout << "Matrix" << endl;
	for (int x = 0; x < N; ++x) {
		for (int y = 0; y < N; ++y) {
			cout << matrix[y + x*N] << " ";
		}
		cout << endl;
	}

	cout << "Vector" << endl;
	for (int x = 0; x < N; ++x) {
		cout << vector[x] << endl;
	}

	double* gpu = new double[N];
	for (int i = 0; i < N; ++i) {
		gpu[i] = 0;
	}

	double* gpuMatrix;
	double* gpuVector;
	double* gpuResult;
	int* gpuN;

	cudaMalloc((void**) &gpuMatrix, N*N*doubleSize);
	cudaMalloc((void**) &gpuVector, N*doubleSize);
	cudaMalloc((void**) &gpuResult, N*doubleSize);
	cudaMalloc((void**) &gpuN, sizeof(int));

	cudaMemcpy(gpuMatrix, matrix, N*N*doubleSize, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuVector, vector, N*doubleSize, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuResult, gpu, N*doubleSize, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuN, &N, sizeof(int), cudaMemcpyHostToDevice);

	timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	double* res = computeOnCPU(matrix, vector, N);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	timespec differenceTime = diff(start, stop);
	std::cout << "CPU " << differenceTime.tv_sec << "." << differenceTime.tv_nsec << std::endl;

	cout << "CPU" << endl;
	for (int i = 0; i < N; ++i) {
		cout << res[i] << endl;
	}

	delete[] res;

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	computeOnGPU<<<N, 1>>>(gpuMatrix, gpuVector, gpuResult, gpuN);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	differenceTime = diff(start, stop);
	std::cout << "GPU " << differenceTime.tv_sec << "." << differenceTime.tv_nsec << std::endl;


	cudaMemcpy(gpu, gpuResult, N*doubleSize, cudaMemcpyDeviceToHost);

	cout << "GPU" << endl;
	for (int i = 0; i < N; ++i) {
		cout << gpu[i] << endl;
	}

	delete[] gpu;
	delete[] matrix;
	delete[] vector;

	cudaFree(gpuMatrix);
	cudaFree(gpuVector);
	cudaFree(gpuResult);
	cudaFree(gpuN);

	return 0;
}

double* generateMatrix(int N, double min, double max) {
	double * returnArray = new double[N*N];

	for (int x = 0; x < N; ++x) {
		for (int y = 0; y < N; ++y) {
			returnArray[y + x*N] = (min + (rand() % (int) (max - min + 1)));
		}
	}
	return returnArray;
}

double* generateVector(int N, double min, double max) {
	double * returnArray = new double[N];

	for (int x = 0; x < N; ++x) {
		returnArray[x] = (min + (rand() % (int) (max - min + 1)));
	}
	return returnArray;
}

double* computeOnCPU(double* matrix, double* vector, int N) {

	double* returnVector = new double[N];

	for (int x = 0; x < N; ++x) {
		returnVector[x] = 0;
		for (int y = 0; y < N; ++y) {
			returnVector[x] += vector[x] * matrix[y + x*N];
		}
	}
	return returnVector;
}

__global__ void computeOnGPU(double* matrix, double* vector, double* result,
		int* N) {
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	int elems = *N;

	if (row < elems && col < elems) {
		result[row] += (matrix[col + row*elems] * vector[col]);
	}

	__syncthreads();
}

timespec diff(timespec start, timespec end) {
	timespec temp;
	if ((end.tv_nsec - start.tv_nsec) < 0) {
		temp.tv_sec = end.tv_sec - start.tv_sec - 1;
		temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec - start.tv_sec;
		temp.tv_nsec = end.tv_nsec - start.tv_nsec;
	}
	return temp;
}
