/*
 ============================================================================
 Name        : Zadanie3.cu
 Author      : Majkelo-Pęczkowiniki
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */


#include <iostream>
#include <iomanip>
#include <random>
#include <ctime>
#include <string>
#include <sstream>
#include <cstdio>

using namespace std;

__global__ void bubble(int* N, int* tab)
{
	int ind, i, j, s;
	int a, b;
	ind = 2 * (threadIdx.x + blockDim.x*blockIdx.x);
	printf("%d\n", ind);
	for (int k = 0; k<(*N) - 1; k++)
	{

		s = (k % 2);
		i = ind + s;
		j = ind + 1 + s;
		if (j < (*N))
		{
			a = tab[i];
			b = tab[j];
			if (b<a)
			{
				tab[i] = b;
				tab[j] = a;
			}
		}
		__syncthreads();
	}
}

__global__ void bubble1(int* N, int* s, int* tab) {
	int ind, i, j;
	int a, b;
	ind = 2 * (threadIdx.x + blockDim.x*blockIdx.x);
	i = ind + (*s);
	j = ind + (*s) + 1;

	if (j < (*N)) {
		a = tab[i];
		b = tab[j];

		if (b < a) {
			tab[i] = b;
			tab[j] = a;
		}
	}
}

timespec diff(timespec start, timespec end);
void generateRandomData(int* tab, int N, int min, int max);
void computeBubbleOnCPU(int* tab, int N);

int main(int argc, char* argv[])
{
	srand(time(NULL));

	string s(argv[1]);
	string s2(argv[2]);
	string s3(argv[3]);

	int min, max;

	min = stof(s2);
	max = stof(s3);

	const int N = stoi(s);
	int tabBase[N]; //= new int[N];
	int tabCPU[N]; //= new int[N];
	int tabGPU[N]; //= new int[N];

	generateRandomData(tabBase, N, min, max);

	for (int i = 0; i < N; ++i) {
		tabCPU[i] = tabBase[i];
		tabGPU[i] = tabBase[i];
	}

	timespec start, stop;
	timespec returnTime;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

	computeBubbleOnCPU(tabCPU, N);


	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	returnTime = diff(start, stop);
	cout << "CPU time: " << returnTime.tv_sec << "."
         << setfill('0') << setw(9) << returnTime.tv_nsec << "s" << endl;

	//Ogarniane jest tutaj ile dana karta może wytrzymać wątków na jeden blok
    int minGridSize;
    int blockSize;
    cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, &blockSize, (void*)bubble, 0, N
    );

    int blockHeight = blockSize;
    int blockWidth = 1;
    while (blockHeight > blockWidth) {
        blockHeight /= 2;
        blockWidth *= 2;
    }

    int gridWidth = (N + blockWidth - 1) / blockWidth;

	dim3 thread (blockWidth);
	dim3 block (gridWidth);

	//Poalokowane coby sobie nie robiæ problemów przy alokacji rozmiarów na karcie graficznej
	size_t intSize = sizeof(int);
	size_t sizeOfTab = N * intSize;

	//Przedrostek d_ oznacza, ¿e dana zmienna wykorzystywana jest nie na procku, ale na karcie
	int* d_tab;
	int* d_N;
	int* d_s;

	//Zaalokuj pamiêæ na karcie na wektor do mno¿enia
	cudaMalloc(&d_tab, sizeOfTab);
	cudaMemcpy(d_tab, &tabGPU, sizeOfTab, cudaMemcpyHostToDevice);

	cudaMalloc(&d_N, sizeof(int));
	cudaMemcpy(d_N, &N, sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&d_s, sizeof(int));

	//Tutaj printujê dane z bloków i w¹tków coby pokazaæ jak one wygl¹daj¹
	cout << "Block: " << block.x << " " << block.y << " " << block.z << endl;
	cout << "Thread: " << thread.x << " " << thread.y << " " << thread.z << endl;

//	bubble <<< block, thread>>> (d_N, d_tab);


	cudaEvent_t cudaStart, cudaStop;
	cudaEventCreate(&cudaStart);
	cudaEventCreate(&cudaStop);

	float milliseconds = 0, temp = 0;


	for (int i = 0; i < N - 1; ++i) {
		int k = (i%2);
		cudaMemcpy(d_s, &k, sizeof(int), cudaMemcpyHostToDevice);
		cudaEventRecord(cudaStart);
		bubble1 <<< block, thread >>>(d_N, d_s, d_tab);
		cudaEventRecord(cudaStop);
		cudaEventSynchronize(cudaStop);
		cudaEventElapsedTime(&temp, cudaStart, cudaStop);
		milliseconds += temp;
	}


	cudaEventSynchronize(cudaStop);
	//Poczekaæ na zakoñczenie wszystkiego
	cudaDeviceSynchronize();

	cout << "GPU time: " << milliseconds << "ms" << endl;

	cudaMemcpy(tabGPU, d_tab, sizeOfTab, cudaMemcpyDeviceToHost);

	cudaFree(d_tab);
	cudaFree(d_N);
	cudaFree(d_s);

	cudaDeviceReset();

//	for (int i : tabCPU) {
//		std::cout << i << " ";
//	}
//	std::cout << std::endl;
//
//	for (int i : tabGPU) {
//		std::cout << i << " ";
//	}
//	std::cout << std::endl;

//	delete[] tabBase;
//	delete[] tabCPU;
//	delete[] tabGPU;

	for (int i = 0; i < N; ++i) {

//		cout << (i+1) << " - " << tabCPU[i] << "   :   " << tabGPU[i] << endl;
		if (tabCPU[i] != tabGPU[i]) {
			cout << "What a Terrible Failure" << endl;
			break;
		}
	}

//	getchar();
	return 0;
}

void generateRandomData(int* tab, int N, int min, int max) {
	for (int x = 0; x < N; ++x) {
		tab[x] = (min + (rand() % (int)(max - min + 1)));
	}
}

void computeBubbleOnCPU (int* tab, int N) {
	int counter  = N;

	do
	{
		for (int i = 0; i < counter - 1; ++i) {
			if (tab[i] > tab[i+1]) {
				int temp = tab[i];
				tab[i] = tab[i+1];
				tab[i+1] = temp;
			}
		}
		--counter;
	}while (counter > 1);

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
