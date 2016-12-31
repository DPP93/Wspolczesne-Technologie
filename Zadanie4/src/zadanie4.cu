#include <iostream>
#include <random>
#include <ctime>

using namespace std;

inline cudaError_t checkCuda(cudaError_t result) {
	if (result != cudaSuccess) {
		cout << "CUDA Runtime Error: " << cudaGetErrorString(result) << endl;
	}
	return result;
}

__device__ float calcForce(float massJ, float distance, float g_const,
		float b_const) {
	if (distance == 0) {
		return 0;
	} else {
		return b_const * (g_const * (massJ) / ((distance * distance)));
	}
}

__global__ void KernelForces(unsigned int n, float deltaT, float* m, float3 *p,
		float3 *v, float3 *f) {
	int bd = blockDim.x * blockIdx.x + threadIdx.x;
	int numThreads = blockDim.x * gridDim.x;

	float G = 6.672 * 10e-11;
	float SOFTENING = 10e7;

	//compare all with all
	if (bd < n) {
		for (unsigned int ia = bd; ia < n; ia += numThreads) {
			float lfx = 0.0f;
			float lfy = 0.0f;
			float lfz = 0.0f;

			for (unsigned int ib = 0; ib < n; ib++) {
				//compute distance
				float dx = (p[ib].x - p[ia].x);
				float dy = (p[ib].y - p[ia].y);
				float dz = (p[ib].z - p[ia].z);

				float distance = sqrt((dx * dx) + (dy * dy) + (dz * dz));

				//compute force
				lfx += calcForce(m[ib], distance, G, SOFTENING) * dx;
				lfy += calcForce(m[ib], distance, G, SOFTENING) * dy;
				lfz += calcForce(m[ib], distance, G, SOFTENING) * dz;
			}

			f[ia] = make_float3(lfx, lfy, lfz);

			v[ia].x += deltaT * f[ia].x;
			v[ia].y += deltaT * f[ia].y;
			v[ia].z += deltaT * f[ia].z;

			printf("Update body %d velocity to %lf %lf %lf\n", ia, v[ia].x,
					v[ia].y, v[ia].z);
		}
	}
}

__global__ void KernelPositions(unsigned int n, float deltaT, float3 *p,
		float3 *v) {
	int bd = blockDim.x * blockIdx.x + threadIdx.x;
	int numThreads = blockDim.x * gridDim.x;

	//compare all with all
	if (bd < n) {
		for (unsigned int ia = bd; ia < n; ia += numThreads) {

			p[ia].x += deltaT * v[ia].x;
			p[ia].y += deltaT * v[ia].y;
			p[ia].z += deltaT * v[ia].z;

			printf("Update body %d position to %lf %lf %lf\n", ia, p[ia].x,
					p[ia].y, p[ia].z);
		}
	}
}

float randomFloat(float a, float b) {
	float random = ((float) rand()) / (float) RAND_MAX;
	float diff = b - a;
	float r = random * diff;
	return a + r;
}

int main(int argc, char* argv[]) {

	const int n = 15, iterations = 5;
	float deltaT = 0.01;

	float m[n];
	float3 p[n], v[n], f[n];

	/*
	 * m - masa
	 * p - wektor pozycji
	 * v - wektor prędkości
	 * f - wektor sił
	 * */

	float *_m;
	float3 *_p, *_v, *_f;

	srand(time(NULL));

	for (int i = 0; i < n; ++i) {
		m[i] = randomFloat(0, 5) + 0.01f;
		p[i] = make_float3(randomFloat(-10, 10), randomFloat(-10, 10),
				randomFloat(-10, 10));
		v[i] = make_float3(randomFloat(-2, 2), randomFloat(-2, 2),
				randomFloat(-2, 2));
		f[i] = make_float3(randomFloat(0, 0), randomFloat(0, 0),
				randomFloat(0, 0));
	}

	int floatSize = sizeof(float);
	int float3Size = sizeof(float3);
	int floatVectorSize = n * floatSize;
	int float3VectorSize = n * float3Size;

	checkCuda(cudaMalloc(&_m, floatVectorSize));
	checkCuda(cudaMalloc(&_p, float3VectorSize));
	checkCuda(cudaMalloc(&_v, float3VectorSize));
	checkCuda(cudaMalloc(&_f, float3VectorSize));

	checkCuda(cudaMemcpy(_m, m, floatVectorSize, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(_p, p, float3VectorSize, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(_v, v, float3VectorSize, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(_f, f, float3VectorSize, cudaMemcpyHostToDevice));

	int minGridSize;
	int blockSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
			(void*) KernelForces, 0, n);

	int gridSize = (n + blockSize - 1) / blockSize;

//Tutaj printuję dane z bloków i wątków coby pokazać jak one wyglądają
	dim3 thread(blockSize);
	dim3 block(gridSize);

	cout << "Block: " << block.x << " " << block.y << " " << block.z << endl;
	cout << "Thread: " << thread.x << " " << thread.y << " " << thread.z
			<< endl;

	for (int i = 1; i <= iterations; ++i) {
		printf("Iteration %d\n", i);
		KernelForces<<<block , thread>>> (n, deltaT, _m, _p, _v, _f);
		KernelPositions<<<block , thread>>> (n, deltaT, _p, _v);
	}

	cudaDeviceSynchronize();

	cudaFree(_m);
	cudaFree(_p);
	cudaFree(_v);
	cudaFree(_f);
	cudaThreadExit();
	cudaDeviceReset();
	cout << "END" << endl;
}
