//============================================================================
// Name        : QuickSort.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <algorithm>
#include <vector>
#include <time.h>
#include <random>
#include <string>
#include "typedefs.h"
#include "QuickSort.h"

using namespace std;

void setSortType(char type, Sorting_Type& sortType);
void setPesymisticElems(vector<int>& array, int count, int uniqueValues);
void setRandomElems(vector<int>&array, int count, int uniqueValues);
timespec diff(timespec start, timespec end);

int main(int argc, char *argv[]) {
	vector<int> array { };
	QuickSort* qs { nullptr };

	Sorting_Type sortType = defaultSortingType;
	int repeatCount = defaultRepeatCount;
	int arraySize = defaultArraySize;
	int uniqueValues = defaultUniqueValues;

	if (argc == 2) {
		setSortType(*argv[1], sortType);
	} else if (argc == 3) {
		setSortType(*argv[1], sortType);
		repeatCount = stoi(argv[2]);
	} else if (argc == 4) {
		setSortType(*argv[1], sortType);
		repeatCount = stoi(argv[2]);
		arraySize = stoi(argv[3]);
	} else if (argc == 5) {
		setSortType(*argv[1], sortType);
		repeatCount = stoi(argv[2]);
		arraySize = stoi(argv[3]);
		uniqueValues = stoi(argv[4]);
	}
	for (int i = 0; i < repeatCount; i++) {

		array.clear();

		switch (sortType) {
		case Sorting_Type::sort_pesymistic:
			setPesymisticElems(array, arraySize, uniqueValues);
			break;
		case Sorting_Type::sort_random:
		default:
			setRandomElems(array, arraySize, uniqueValues);
			break;
		}

		qs = new QuickSort();
		timespec start, stop;
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
		qs->computeQuickSort(array, 0, array.size() - 1);
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
		timespec differenceTime = diff(start, stop);
		cout << differenceTime.tv_sec << "." << differenceTime.tv_nsec << endl;
		delete qs;
	}
	return 0;
}

void setSortType(char type, Sorting_Type& sortType) {
	switch (type) {
	case 'P':
	case 'p':
		sortType = Sorting_Type::sort_pesymistic;
		break;
	case 'R':
	case 'r':
		sortType = Sorting_Type::sort_random;
		break;
	}
}

void setPesymisticElems(vector<int>& array, int count, int uniqueValues) {
	if (uniqueValues < 1) {
		for (int i = 1; i <= count; i++) {
			array.push_back(i);
		}
	} else {
		random_device rd;
		mt19937 gen(rd());
		uniform_int_distribution<> dis(-numeric_limits<int>::max(), numeric_limits<int>::max());
		vector<int> uniques { };
		for (int i = 0; i < uniqueValues; i++) {
			bool end = false;
			while (!end) {
				int x = dis(gen);
				auto it = find(uniques.begin(), uniques.end(), x);
				if (it == uniques.end()) {
					uniques.push_back(x);
					end = true;
				}
			}
		}
		int last = static_cast<int>(uniques.size()) - 1;
		uniform_int_distribution<> d(0, last);
		for (int i = 0; i < count; i++) {
			int x = d(gen);
			array.push_back(uniques[x]);
		}
		sort(array.begin(), array.end());
	}
}

void setRandomElems(vector<int>&array, int count, int uniqueValues) {
	random_device rd;
	mt19937_64 gen(rd());
	uniform_int_distribution<> dis(-numeric_limits<int>::max(), numeric_limits<int>::max());

	if (uniqueValues < 1) {
		for (int i = 1; i <= count; i++) {
			int x = dis(gen);
			array.push_back(x);
		}
	} else {
		vector<int> uniques { };
		for (int i = 0; i < uniqueValues; i++) {
			bool end = false;
			while (!end) {
				int x = dis(gen);
				auto it = find(uniques.begin(), uniques.end(), x);
				if (it == uniques.end()) {
					uniques.push_back(x);
					end = true;
				}
			}
		}
		int last = static_cast<int>(uniques.size()) - 1;
		uniform_int_distribution<> d(0, last);
		for (int i = 0; i < count; i++) {
			int x = d(gen);
			array.push_back(uniques[x]);
		}
		shuffle(array.begin(), array.end(), gen);
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
