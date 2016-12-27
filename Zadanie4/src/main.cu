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

#define _USE_MATH_DEFINES
#include <cmath>

#define G (6.67408*pow(10,-11))

//#define G 2

const double minStartedPositionValue = -10.0;
const double maxStartedPositionValue = 10.0;

const double maxStartedWeightValue = 10000.0;
const double maxStartedVelocityValue = 15.0;

const double timeDifference = 0.3;

using namespace std;

struct position {
	double x { };
	double y { };
	double z { };
};

struct physic_body {
	position r { };
	double v { };
	double m { };
};

__global__ void universeUpdateVelocity(int* N, physic_body* tab) {
	int index;
	index = 2 * (threadIdx.x + blockDim.x * blockIdx.x);

	if (index < (*N)) {
		double solution = 0;
		for (int j = 0; j < (*N); ++j) {
			if (j == index) {
				continue;
			}

			double dist = sqrt(
					pow(tab[index].r.x - tab[j].r.x, 2)
							+ pow(tab[index].r.y - tab[j].r.y, 2)
							+ pow(tab[index].r.z - tab[j].r.z, 2));

			solution += (tab[j].m / (pow(dist, 3))) * dist;
		}
		printf("Body %d velocity: %lf \n", index, tab[index].v);
		printf("Body %d velocity changing by: %lf \n", index,
				tab[index].v + G * solution * timeDifference);
		tab[index].v = tab[index].v + G * solution * timeDifference;
		printf("Body %d velocity changed to %lf \n", index, tab[index].v);

	}
}

__global__ void universeUpdatePosition(int* N, physic_body* tab) {
	int index;
	index = 2 * (threadIdx.x + blockDim.x * blockIdx.x);

	if (index < (*N)) {
		tab[index].r.x = tab[index].r.x + tab[index].v * timeDifference;
		tab[index].r.y = tab[index].r.y + tab[index].v * timeDifference;
		tab[index].r.z = tab[index].r.z + tab[index].v * timeDifference;
		printf("Body %d position changed to: %lf %lf %lf \n", index, tab[index].r.x,
				tab[index].r.y, tab[index].r.z);
	}
}

timespec diff(timespec start, timespec end);
double computeDistance(position positionA, position positionB);
void generateBodyValues(physic_body* bodies, int N);
void computeOnCPU(physic_body* bodies, int N, int iterations);

int main(int argc, char* argv[]) {
	srand(time(NULL));

	string s(argv[1]);
	string s2(argv[2]);

	const int iterations = stof(s2);

	const int N = stoi(s);

	physic_body cpuBodies[N];
	physic_body gpuBodies[N];
	generateBodyValues(cpuBodies, N);
	generateBodyValues(gpuBodies, N);

	cout << "G" << G << endl;

//	computeOnCPU(cpuBodies, N, iterations);

//NOW CUDA

//Ogarniane jest tutaj ile dana karta może wytrzymać wątków na jeden blok
	int minGridSize;
	int blockSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
			(void*) universeUpdateVelocity, 0, N / 2);

	int gridSize = (N / 2 + blockSize - 1) / blockSize;

	dim3 thread(blockSize);
	dim3 block(gridSize);

	size_t bodySize = sizeof(physic_body);
	size_t sizeOfTab = N * bodySize;

	//Przedrostek d_ oznacza, ¿e dana zmienna wykorzystywana jest nie na procku, ale na karcie
	physic_body* d_tab;
	int* d_N;

	cudaMalloc(&d_tab, sizeOfTab);
	cudaMemcpy(d_tab, &gpuBodies, sizeOfTab, cudaMemcpyHostToDevice);

	cudaMalloc(&d_N, sizeof(int));
	cudaMemcpy(d_N, &N, sizeof(int), cudaMemcpyHostToDevice);

	for (int iteration = 0; iteration < iterations; ++iteration) {
		universeUpdateVelocity<<< block, thread >>>(d_N, d_tab);
		universeUpdatePosition<<< block, thread >>>(d_N, d_tab);
	}

	cudaFree(d_tab);
	cudaFree(d_N);

	printf("End of all \n");

	return 0;
}

void generateBodyValues(physic_body* bodies, int N) {
	for (int i = 0; i < N; ++i) {
		bodies[i].r.x = minStartedPositionValue
				+ (rand()
						% (int) (maxStartedPositionValue
								- minStartedPositionValue + 1));
		bodies[i].r.y = minStartedPositionValue
				+ (rand()
						% (int) (maxStartedPositionValue
								- minStartedPositionValue + 1));
		bodies[i].r.z = minStartedPositionValue
				+ (rand()
						% (int) (maxStartedPositionValue
								- minStartedPositionValue + 1));
		bodies[i].v = (rand() % (int) (maxStartedVelocityValue + 1));
		bodies[i].m = (rand() % (int) (maxStartedWeightValue + 1));
	}
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

double computeDistance(position positionA, position positionB) {
	return sqrt(
			pow(positionA.x - positionB.x, 2)
					+ pow(positionA.y - positionB.y, 2)
					+ pow(positionA.z - positionB.z, 2));
}

void computeOnCPU(physic_body* bodies, int N, int iterations) {
	for (int iter = 0; iter < iterations; ++iter) {
		printf("Iteration %d \n", iter + 1);
		//Change velocity
//		printf ("Updating velocity\n");
		for (int i = 0; i < N; ++i) {
			double solution = 0;
			for (int j = 0; j < N; ++j) {
				if (j == i) {
					continue;
				}
				double dist = computeDistance(bodies[i].r, bodies[j].r);
				solution += (bodies[j].m / (pow(dist, 3))) * dist;
			}
//			printf ("Body %d velocity: %lf \n", i, bodies[i].v);
//			printf ("Body %d velocity changing by: %lf \n", i, bodies[i].v + G * solution * timeDifference);
			bodies[i].v = bodies[i].v + G * solution * timeDifference;
			printf("Body %d velocity changed to %lf \n", i, bodies[i].v);
		}

		//Change position
//		printf ("Updating position\n");
		for (int i = 0; i < N; ++i) {
//			printf ("Body %d position: %lf %lf %lf \n", i, bodies[i].r.x, bodies[i].r.y, bodies[i].r.z);
			bodies[i].r.x = bodies[i].r.x + bodies[i].v * timeDifference;
			bodies[i].r.y = bodies[i].r.y + bodies[i].v * timeDifference;
			bodies[i].r.z = bodies[i].r.z + bodies[i].v * timeDifference;
			printf("Body %d position changed to: %lf %lf %lf \n", i,
					bodies[i].r.x, bodies[i].r.y, bodies[i].r.z);
		}
	}
}
