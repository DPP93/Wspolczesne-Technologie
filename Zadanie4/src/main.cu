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

const double min = -1000;
const double max = 1000;

using namespace std;


struct position
{
	double x;
	double y;
	double z;
	position (double x, double y, double z)
	: x(x),
	  y(y),
	  z(z)
	{

	}
};

struct physic_body {
	position r;
	double v;
	double m;
	physic_body (position r, double v, double m)
	: r (r),
	  v (v),
	  m (m)
	{

	}
};

__global__ void universe(int* N, physic_body* tab)
{

}

timespec diff(timespec start, timespec end);
double computeDistance(position positionA, position positionB);
void generateBodyValues (physic_body* bodies, int N);

int main(int argc, char* argv[])
{
	srand(time(NULL));

	string s(argv[1]);
	string s2(argv[2]);

	const int iterations = stof(s2);

	const int N = stoi(s);

	cout << G << endl;

	return 0;
}

void generateBodyValues (physic_body* bodies, int N)
{

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
	return sqrt(pow(positionA.x - positionB.x, 2)
			+ pow(positionA.y - positionB.y, 2)
			+ pow(positionA.z - positionB.z, 2));
}
