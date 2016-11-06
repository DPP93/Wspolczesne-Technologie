#include <iostream>
#include <iomanip>
#include <random>
#include <ctime>
#include <string>
#include <sstream>
#include <time.h>
#include <cstdio>

using namespace std;

timespec diff(timespec start, timespec end);
void displayAll(int* matrix, int* vector, int* cpuReturnVector, int* gpuReturnVector, 
                int height, int width);

void computeOnCPU(int* matrix, int* vector, int* returnVector, int height, int width) {
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			returnVector[y] += matrix[y * width + x] * vector[x];
		}
	}
}

__global__ void computeOnGPU(int* matrix, int* vector, int* result, 
                             int* height, int* width, size_t pitch) {
	int lheight = (*height);
	int lwidth = (*width);

	/* W zależności od tego na którym wątku którego bloku aktualnie działa
	 * kernel możemy pobrać adres w pamięci
	 */
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < lwidth && y < lheight) {
		//To jest sposób z dokumentacji na to, żeby pobrać wskaźnik do KONKRETNEGO elementu
		int* matrixElem = (int*)((char*) matrix + y * pitch) + x;
        int* resultElem = result + y;
		atomicAdd(resultElem, (*matrixElem) * vector[x]);
	}

}

void generateRandomData(int* matrix, int* vector, int height, int width, int min, int max) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
			matrix[y * width + x] = (min + (rand() % (int) (max - min + 1)));
		}
    }
    for (int x = 0; x < width; ++x) {
		vector[x] = (min + (rand() % (int) (max - min + 1)));
	}
}

void solveOnCPU(int* matrix, int* vector, int* returnVector, int height, int width) {
	timespec start, stop;
	timespec returnTime;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    computeOnCPU(matrix, vector, returnVector, height, width);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	returnTime = diff(start, stop);
	cout << "CPU time: " << returnTime.tv_sec << "." 
         << setfill('0') << setw(9) << returnTime.tv_nsec << "s" << endl;
}

int main(int argc, char* argv[]) {
	int devCount;
	cudaGetDeviceCount(&devCount);
	if (devCount == 0) {
		printf("Can't compute due to lack of device compatible with CUDA\n");
		return 0;
	}

	srand (time(NULL));
	int height, width;
	int min, max;

	string s(argv[1]);
	string s2(argv[2]);
	string s3(argv[3]);

	height = width = stoi(s);
	min = stof(s2);
	max = stof(s3);

	int matrix[height][width];
	int vector[width];
	int cpuReturnVector[height];
	int gpuReturnVector[height];

    generateRandomData(matrix[0], vector, height, width, min, max);
    for (int y = 0; y < height; ++y) {
        cpuReturnVector[y] = 0;
        gpuReturnVector[y] = 0;
    }

    solveOnCPU(matrix[0], vector, cpuReturnVector, height, width);

    //solveOnGPU(matrix[0], vector, gpuReturnVector, height, width);

    // CUDA GO GO GO
	//Ogarniane jest tutaj ile dana karta może wytrzymać wątków na jeden blok
    int minGridSize;
    int blockSize;
    cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, &blockSize, (void*)computeOnGPU, 0, width * height
    );

    int blockHeight = blockSize;
    int blockWidth = 1;
    while (blockHeight > blockWidth) {
        blockHeight /= 2;
        blockWidth *= 2;
    }

    int gridHeight = (height + blockHeight - 1) / blockHeight;
    int gridWidth = (width + blockWidth - 1) / blockWidth;

	dim3 thread (blockWidth, blockHeight);
	dim3 block (gridWidth, gridHeight);

	//Tutaj printuję dane z bloków i wątków coby pokazać jak one wyglądają
	cout << "Block: " << block.x << " " << block.y << " " << block.z << endl;
	cout << "Thread: " << thread.x << " " << thread.y << " " << thread.z << endl;

	//Poalokowane coby sobie nie robić problemów przy alokacji rozmiarów na karcie graficznej
	size_t intSize = sizeof(int);
	size_t sizeOfVector = width * intSize;
	size_t sizeOfResult = height * intSize;

	//Przedrostek d_ oznacza, że dana zmienna wykorzystywana jest nie na procku, ale na karcie
	int* d_matrix;
	int* d_vector;
	int* d_result;
	int* d_height;
	int* d_width;

	/* To je takie fajne cuś, co da info podczas alokowania pamięci na karcie, ile jest 'komórek'
	 * pamięci w wierszu
	 */
	size_t d_pitch;

	//Zaalokuj pamięć na karcie na wektor do mnożenia
	cudaMalloc(&d_vector, sizeOfVector);
	//Skopiuj dane z pamięci komputra na pamięć karty graficznej
	cudaMemcpy(d_vector, &vector, sizeOfVector, cudaMemcpyHostToDevice);

	cudaMalloc(&d_result, sizeOfResult);
    cudaMemset(d_result, 0, sizeOfResult);

	cudaMalloc(&d_height, sizeof(int));
	cudaMalloc(&d_width, sizeof(int));
	cudaMemcpy(d_height, &height, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_width, &width, sizeof(int), cudaMemcpyHostToDevice);

	//Tutaj to samo tylko robimy to jakby dwuwymiarowo
	cudaMallocPitch(&d_matrix, &d_pitch, sizeOfVector, height);
	cudaMemcpy2D(d_matrix, d_pitch, matrix, sizeOfVector, sizeOfVector, height, 
                 cudaMemcpyHostToDevice);


	cudaEvent_t cudaStart, cudaStop;
	cudaEventCreate(&cudaStart);
	cudaEventCreate(&cudaStop);

	cudaEventRecord(cudaStart);
	computeOnGPU<<<block, thread>>> (d_matrix, d_vector, d_result, d_height, d_width, d_pitch);
	cudaEventRecord(cudaStop);

	//Poczekać na zakończenie wszystkiego
	cudaDeviceSynchronize();

	cudaEventSynchronize(cudaStop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, cudaStart, cudaStop);
	cout << "GPU time: " << milliseconds << "ms" << endl;

	cudaMemcpy(gpuReturnVector, d_result, sizeOfResult, cudaMemcpyDeviceToHost);

	cudaFree(d_matrix);
	cudaFree(d_vector);
	cudaFree(d_result);
	cudaFree(d_height);
	cudaFree(d_width);
	cudaDeviceReset();

    //displayAll(matrix[0], vector, cpuReturnVector, gpuReturnVector, height, width);
    int allOk = 0;
  	for (int y = 0; y < height; ++y) {
        if (gpuReturnVector[y] != cpuReturnVector[y]) {
            cout << "Fatalna omyłka! Wyniki nie zgadzają się na pozycji " << y << ", gdzie " 
                 << gpuReturnVector[y] << " != " << cpuReturnVector[y] << endl;
            ++allOk;
        }
  	}
	return allOk;
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

void displayAll(int* matrix, int* vector, int* cpuReturnVector, int* gpuReturnVector,
                int height, int width) {
    cout << "matrix = [" << endl;
    for (int y = 0; y < height; ++y) {
        cout << "  ";
        for (int x = 0; x < width; ++x) {
            cout << matrix[y * width + x] << " ";
        }
        cout << endl;
    }
    cout << "]" << endl;

    cout << "vector = [ ";
    for (int x = 0; x < width; ++x) {
        cout << vector[x] << " ";
    }
    cout << "]" << endl;

    cout << "cpuReturnVector = [ ";
    for (int y = 0; y < height; ++y) {
        cout << cpuReturnVector[y] << " ";
    }
    cout << "]" << endl;

    cout << "gpuReturnVector = [ ";
    for (int y = 0; y < height; ++y) {
        cout << gpuReturnVector[y] << " ";
    }
    cout << "]" << endl;

}
