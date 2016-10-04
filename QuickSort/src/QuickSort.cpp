//============================================================================
// Name        : QuickSort.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <vector>
#include <time.h>
#include <sys/time.h>
#include <random>

using namespace std;

constexpr int defaultCount = 100000;

void computeQuickSort(vector<int>& array, int firstIndex, int lastIndex);
int partition(vector<int>& array, int fristIndex, int lastIndex);
void setPesymisticElems(vector<int>& array, int count = defaultCount);
void setOptymisticElems(vector<int>& array, int count = defaultCount);
void setRandomElems(vector<int>&array, int count = defaultCount);

int main(int argc, char *argv[]) {


	vector<int> array { };
//	int x {};
//	for(int x; cin >> x;){
//		array.push_back(x);
//	}
	setRandomElems(array);
	cout << "Optymistic setted " << endl;
	struct timespec start, stop;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	computeQuickSort(array, 0, array.size() - 1);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	time_t timeS = (stop.tv_sec - start.tv_sec);
	long timeNS = (stop.tv_nsec - start.tv_nsec);
	cout << "TIME: " << timeNS << " ns" << endl;
	cout << "TIME: " << timeS << " s" << endl;
	return 0;
}

void computeQuickSort(vector<int>& array, int firstIndex, int lastIndex) {
	if (firstIndex < lastIndex) {
		int divideElement;
		divideElement = partition(array, firstIndex, lastIndex);
		computeQuickSort(array, firstIndex, divideElement - 1);
		computeQuickSort(array, divideElement + 1, lastIndex);
	}
}

int partition(vector<int>& array, int firstIndex, int lastIndex) {
	int x = array[lastIndex];
	int returnIndex = firstIndex - 1;
	for (int j = firstIndex; j < lastIndex; j++) {
		if (array[j] <= x) {
			returnIndex += 1;
			int temp = array[returnIndex];
			array[returnIndex] = array[j];
			array[j] = temp;
		}
	}
	int temp = array[returnIndex + 1];
	array[returnIndex + 1] = array[lastIndex];
	array[lastIndex] = temp;
	return ++returnIndex;
}

void setPesymisticElems(vector<int>& array, int count) {
	for (int i = 1; i <= count; i++) {
		array.push_back(i);
	}
}

void setOptymisticElems(vector<int>& array, int count) {
	for (int i = count; i >= 1; i--) {
		array.push_back(i);
	}
}

void setRandomElems(vector<int>&array, int count) {
	random_device rd;
	mt19937_64 gen(rd());
	uniform_int_distribution<> dis(1, defaultCount);
	for (int i = 1; i <= count; i++) {
		int x = dis(gen);
		array.push_back(x);
	}
}
