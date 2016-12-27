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

const double EPSILON = 0.0001;

const double minStartedPositionValue = -10.0;
const double maxStartedPositionValue = 10.0;

const double maxStartedWeightValue = 10.0;
const double maxStartedVelocityValue = 15.0;

const double timeDifference = 0.0001;

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
	bool operator==(const physic_body& otherBody) const {
		if ((fabs(this->m - otherBody.m) < EPSILON)
				&& (fabs(this->r.x - otherBody.r.x) < EPSILON)
				&& (fabs(this->r.y - otherBody.r.y) < EPSILON)
				&& (fabs(this->r.z - otherBody.r.z) < EPSILON)
				&& (fabs(this->v - otherBody.v) < EPSILON)) {
			return true;
		}
		return false;
	}

	void print() {
		setprecision(10);
		cout << "V: " << this->v << endl;
		cout << "M: " << this->m << endl;
		cout << "RX: " << this->r.x << endl;
		cout << "RY: " << this->r.y << endl;
		cout << "RZ: " << this->r.z << endl;
	}

};

__global__ void universeUpdateVelocity(int* N, physic_body* tab) {
	int index;

	index = blockIdx.x * blockDim.x + threadIdx.x;

	printf("GPU Body %d index \n", index);

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
		printf("GPU Body %d velocity: %lf \n", index, tab[index].v);
		printf("GPU Body %d velocity changing by: %lf \n", index,
				tab[index].v + G * solution * timeDifference);
		tab[index].v = tab[index].v + G * solution * timeDifference;
		printf("GPU Body %d velocity changed to %lf \n", index, tab[index].v);

	}
}

__global__ void universeUpdatePosition(int* N, physic_body* tab) {
	int index;
	index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index < (*N)) {
		printf("GPU Body %d position before: %lf %lf %lf \n", index, tab[index].r.x,
				tab[index].r.y, tab[index].r.z);
		tab[index].r.x = tab[index].r.x + tab[index].v * timeDifference;
		tab[index].r.y = tab[index].r.y + tab[index].v * timeDifference;
		tab[index].r.z = tab[index].r.z + tab[index].v * timeDifference;
		printf("GPU Body %d position changed to: %lf %lf %lf \n", index,
				tab[index].r.x, tab[index].r.y, tab[index].r.z);
	}
}

timespec diff(timespec start, timespec end);
double computeDistance(position positionA, position positionB);
void generateBodyValues(physic_body* bodies, int N);
void computeOnCPU(physic_body* bodies, int N, int iterations);

int main(int argc, char* argv[]) {
	srand(time(NULL));
	setprecision(10);
	string s(argv[1]);
	string s2(argv[2]);

	const int iterations = stof(s2);

	const int N = stoi(s);

	physic_body cpuBodies[N];
	physic_body gpuBodies[N];
	generateBodyValues(cpuBodies, N);
	copy(cpuBodies, cpuBodies + N, gpuBodies);
//	generateBodyValues(gpuBodies, N);

	cout << "G" << G << endl;

	computeOnCPU(cpuBodies, N, iterations);

//NOW CUDA

//Ogarniane jest tutaj ile dana karta może wytrzymać wątków na jeden blok
	int minGridSize;
	int blockSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
			(void*) universeUpdateVelocity, 0, N);

	int gridSize = (N + blockSize - 1) / blockSize;

	dim3 thread(blockSize);
	dim3 block(gridSize);

	//Tutaj printuję dane z bloków i wątków coby pokazać jak one wyglądają
	cout << "Block: " << block.x << " " << block.y << " " << block.z << endl;
	cout << "Thread: " << thread.x << " " << thread.y << " " << thread.z
			<< endl;

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
		universeUpdateVelocity<<< block , thread >>>(d_N, d_tab);
		universeUpdatePosition<<< block, thread >>>(d_N, d_tab);
	}

	cudaMemcpy(gpuBodies, d_tab, sizeOfTab, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; ++i) {
		if (!(cpuBodies[i] == gpuBodies[i])) {
			cout << "WHAT A TERRIBLE FAILURE" << endl;
			cout << "CPU" << endl;
			cpuBodies[i].print();
			cout << "GPU" << endl;
			gpuBodies[i].print();
		}
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
		bodies[i].v = (rand() % (int) (maxStartedVelocityValue + 1)) + 1;
		bodies[i].m = (rand() % (int) (maxStartedWeightValue + 1)) + 1;
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
			printf ("Body %d velocity: %lf \n", i, bodies[i].v);
			printf ("Body %d velocity changing by: %lf \n", i, bodies[i].v + G * solution * timeDifference);
			bodies[i].v = bodies[i].v + G * solution * timeDifference;
//			printf("Body %d velocity changed to %lf \n", i, bodies[i].v);
		}

		//Change position
//		printf ("Updating position\n");
		for (int i = 0; i < N; ++i) {
			printf ("Body %d position: %lf %lf %lf \n", i, bodies[i].r.x, bodies[i].r.y, bodies[i].r.z);
			bodies[i].r.x = bodies[i].r.x + bodies[i].v * timeDifference;
			bodies[i].r.y = bodies[i].r.y + bodies[i].v * timeDifference;
			bodies[i].r.z = bodies[i].r.z + bodies[i].v * timeDifference;
			printf("Body %d position changed to: %lf %lf %lf \n", i,
					bodies[i].r.x, bodies[i].r.y, bodies[i].r.z);
		}
	}
}
