#include <iostream>
#include <random>
#include <ctime>
#include <string>
#include <sstream>
#include <time.h>
#include <cstdio>

using namespace std;

int calcGridValue(int value);
double* computeOnCPU(double* matrix, double* vector, int N);
timespec diff(timespec start, timespec end);

__global__ void computeOnGPU(double* matrix, double* vector, double* result, size_t pitch,
		int* N) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < *N && y < *N) {
		printf("Matrix val %d for %d %d\n", (matrix[y + x*pitch]), x, y);
		printf("Vector val %d\n", (vector[x]));
		result[x + y*pitch] = (matrix[x + y*pitch] * vector[y]);
	}

//	for (int x = 0; x < *N; ++x) {
//		for (int y = 0; y < *N; ++y) {
//			printf("Matrix val %d ", (matrix[y + x*(*N)]));
//		}
//		printf("\n");
//	}
}

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

	double matrix[N][N];
	double vector[N];
	double cpuReturnVector[N];
	double gpuReturnVector[N][N];

	for (int x = 0; x < N; ++x) {
		for (int y = 0; y < N; ++y) {
			matrix[x][y]= (min + (rand() % (int) (max - min + 1)));
		}
		vector[x] = (min + (rand() % (int) (max - min + 1)));
	}

	cout << "Matrix" << endl;
	for (int x = 0; x < N; ++x) {
		for (int y = 0; y < N; ++y) {
			cout << matrix[x][y] << " ";
		}
		cout << endl;
	}

	cout << "Vector" << endl;
	for (int x = 0; x < N; ++x) {
		cout << vector[x] << endl;
	}

	timespec start, stop;
	timespec returnTime;

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	for (int x = 0; x < N; ++x) {
		cpuReturnVector[x] = 0;
		double sum = 0;
		for (int y = 0; y < N; ++y) {
			sum += vector[x] * matrix[x][y];
		}
		cpuReturnVector[x] = sum;
	}
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	returnTime = diff(start, stop);
	cout << "CPU TIME: " << returnTime.tv_sec<<"."<<returnTime.tv_nsec<< endl;

	//Tu zaczyna się CUDO-wanie

	//Ogarniane jest tutaj ile dana karta może wytrzymać wątków na jeden blok
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	int maxThreadsPerBlock = devProp.maxThreadsPerBlock;

	//Tu obliczana jest potrzebna liczba wątków
	int numberOfBlocks = (N*N) / maxThreadsPerBlock;
	if ((N*N) % maxThreadsPerBlock) {
		numberOfBlocks++;
	}


	//Poalokowane coby sobie nie robić problemów przy alokacji rozmiarów na karcie graficznej
	size_t doubleSize = sizeof(double);
	size_t sizeOfMatrix = N*N*doubleSize;
	size_t sizeOfVector = N*doubleSize;

	//Przedrostek d_ oznacza, że dana zmienna wykorzystywana jest nie na procku, ale na karcie
	double * d_matrix;
	double * d_vector;
	double * d_result;
	int* d_N;
	/*To je takie fajne cuś, co da info podczas alokowania pamięci na karcie, ile jest 'komórek'
	 *  pamięci w wierszu
	 */
	size_t d_pitch;

	//Zaalokuj pamięć na karcie na wektor do mnożenia
	cudaMalloc(&d_vector, sizeOfVector);
	//Skopiuj dane z pamięci komputra na pamięć karty graficznej
	cudaMemcpy(d_vector, &vector, sizeOfVector, cudaMemcpyHostToDevice);

	cudaMalloc(&d_N, sizeof(int));
	cudaMemcpy(d_N, &N, sizeof(int), cudaMemcpyHostToDevice);
	//Tutaj to samo tylko robimy to jakby dwuwymiarowo
	cudaMallocPitch(&d_matrix, &d_pitch, N * doubleSize, N);
	cudaMallocPitch(&d_result, &d_pitch, N * doubleSize, N);
	cudaMemcpy2D(d_matrix, d_pitch, matrix, N * doubleSize, N * doubleSize, N, cudaMemcpyHostToDevice);

	dim3 block (numberOfBlocks, numberOfBlocks);
	dim3 grid (calcGridValue(N/block.x), calcGridValue(N/block.y));


	cout << "Blocks: " << block.x << " " << block.y << " " << block.z << endl;
	cout << "Grid: " << grid.x << " " << grid.y << " " << grid.z << endl;

	clock_t calcGpuStart = clock();

	computeOnGPU<<<grid, block>>> (d_matrix, d_vector, d_result, d_pitch, d_N);

	cout<<"GPU time: " << ((clock() - calcGpuStart) / (double)(CLOCKS_PER_SEC / 1000)) <<endl;

	cudaDeviceSynchronize();
	cudaMemcpy2D(gpuReturnVector, N * sizeof(int), d_result, d_pitch, N * sizeof(int), N, cudaMemcpyDeviceToHost);

	cout << "Vector" << endl;
	for (int x = 0; x < N; ++x) {
		for (int y = 0; y < N; ++y) {
			cout << gpuReturnVector[x][y] << " ";
		}
		cout << endl;
	}

	cudaFree(d_matrix);
	cudaFree(d_vector);
	cudaFree(d_result);
	cudaFree(d_N);
	cudaDeviceReset();

	return 0;
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

int calcGridValue(int value) {
	if (value % 2 == 0) {
		return value;
	}
	else {
		return value + 1;
	}
}
