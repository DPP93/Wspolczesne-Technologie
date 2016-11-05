#include <iostream>
#include <random>
#include <ctime>
#include <string>
#include <sstream>
#include <time.h>
#include <cstdio>

using namespace std;

int calcGridValue(int value);
int* computeOnCPU(int* matrix, int* vector, int N);
timespec diff(timespec start, timespec end);

__global__ void computeOnGPU(int* matrix, int* vector, int* result, size_t pitch,
		int* N) {

	int size = (*N);
//	printf("Size %d\n", size);
	/* W zależności od tego na którym wątku którego bloku aktualnie działa
	 * kernel możemy pobrać adres w pamięci
	 *
	 */
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < size && y < size) {
		//To jest sposób z dokumentacji na to, żeby pobrać wskaźnik do KONKRETNEGO elementu
		int* matrixElem = (int*)((char*)matrix + x * pitch)+y;
		int* resultElem = (int*)((char*)result + x * pitch)+y;
		*resultElem = (*matrixElem) * vector[y];
	}

}

int main(int argc, char* argv[]) {

	int devCount;
	cudaGetDeviceCount(&devCount);
	if (devCount == 0) {
		printf("Can't compute due to lack of device compatible with CUDA\n");
		return 0;
	}

	srand (time(NULL));
	int N;
	int min, max;

	string s(argv[1]);
	string s2(argv[2]);
	string s3(argv[3]);

	N = stoi(s);
	min = stof(s2);
	max = stof(s3);

	int matrix[N][N];
	int vector[N];
	int cpuReturnVector[N][N];
	int gpuReturnVector[N][N];

	for (int x = 0; x < N; ++x) {
		for (int y = 0; y < N; ++y) {
			matrix[x][y]= (min + (rand() % (int) (max - min + 1)));
		}
		vector[x] = (min + (rand() % (int) (max - min + 1)));
	}

//	cout << "Matrix" << endl;
//	for (int x = 0; x < N; ++x) {
//		for (int y = 0; y < N; ++y) {
//			cout << matrix[x][y] << " ";
//		}
//		cout << endl;
//	}
//
//	cout << "Vector" << endl;
//	for (int x = 0; x < N; ++x) {
//		cout << vector[x] << endl;
//	}

	timespec start, stop;
	timespec returnTime;

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	for (int x = 0; x < N; ++x) {
		for (int y = 0; y < N; ++y) {
			cpuReturnVector[x][y] = vector[y] * matrix[x][y];
		}
	}
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	returnTime = diff(start, stop);
	cout << "CPU TIME: " << returnTime.tv_sec<<"."<<returnTime.tv_nsec<< endl;

//	cout << "Cpu Result" << endl;
//	for (int x = 0; x < N; ++x) {
//		for (int y = 0; y < N; ++y) {
//			cout << cpuReturnVector[x][y] << " ";
//		}
//		cout << endl;
//	}

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
	size_t intSize = sizeof(int);
	size_t sizeOfVector = N*intSize;

	//Przedrostek d_ oznacza, że dana zmienna wykorzystywana jest nie na procku, ale na karcie
	int * d_matrix;
	int * d_vector;
	int * d_result;
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
	cudaMallocPitch(&d_matrix, &d_pitch, sizeOfVector, N);
	cudaMallocPitch(&d_result, &d_pitch, sizeOfVector, N);
	cudaMemcpy2D(d_matrix, d_pitch, matrix, N * intSize, N * intSize, N, cudaMemcpyHostToDevice);

	dim3 thread (numberOfBlocks, numberOfBlocks);
	dim3 block (calcGridValue(N/thread.x), calcGridValue(N/thread.y));

	/*Tutaj printuję dane z bloków i wątków coby pokazać jak one wyglądają*/
	cout << "Thread: " << thread.x << " " << thread.y << " " << thread.z << endl;
	cout << "Block: " << block.x << " " << block.y << " " << block.z << endl;

	cudaEvent_t cudaStart, cudaStop;
	cudaEventCreate(&cudaStart);
	cudaEventCreate(&cudaStop);
	clock_t calcGpuStart = clock();

	cudaEventRecord(cudaStart);
	computeOnGPU<<<block, thread>>> (d_matrix, d_vector, d_result, d_pitch, d_N);
	//Poczekać na zakończenie wszystkiego
	cudaDeviceSynchronize();
	cudaEventRecord(cudaStop);


	cudaEventSynchronize(cudaStop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, cudaStart, cudaStop);
	cout<<"GPU time: " << milliseconds << "ms" <<endl;

	cudaMemcpy2D(gpuReturnVector, N * sizeof(int), d_result, d_pitch, N * sizeof(int), N, cudaMemcpyDeviceToHost);

//	cout << "Gpu Result" << endl;
//	for (int x = 0; x < N; ++x) {
//		for (int y = 0; y < N; ++y) {
//			cout << gpuReturnVector[x][y] << " ";
//		}
//		cout << endl;
//	}

	cudaFree(d_matrix);
	cudaFree(d_vector);
	cudaFree(d_result);
	cudaFree(d_N);
	cudaDeviceReset();

	return 0;
}


int* computeOnCPU(int* matrix, int* vector, int N) {

	int* returnVector = new int[N];

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
