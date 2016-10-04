/*
 * QuickSort.cpp
 *
 *  Created on: Oct 4, 2016
 *      Author: dpp
 */

#include "QuickSort.h"

QuickSort::QuickSort() {
	// TODO Auto-generated constructor stub

}

QuickSort::~QuickSort() {
	// TODO Auto-generated destructor stub
}

void QuickSort::computeQuickSort(vector<int>& array, int firstIndex, int lastIndex) {
	if (firstIndex < lastIndex) {
		int divideElement;
		divideElement = partition(array, firstIndex, lastIndex);
		computeQuickSort(array, firstIndex, divideElement - 1);
		computeQuickSort(array, divideElement + 1, lastIndex);
	}
}

int QuickSort::partition(vector<int>& array, int firstIndex, int lastIndex) {
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
