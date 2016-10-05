/*
 * QuickSort.h
 *
 *  Created on: Oct 4, 2016
 *      Author: dpp
 */

#ifndef QUICKSORT_H_
#define QUICKSORT_H_

#include <vector>

using namespace std;

class QuickSort {
public:
	QuickSort();
	void computeQuickSort(vector<int>& array, int firstIndex, int lastIndex);
	virtual ~QuickSort();

private:
	int partition(vector<int>& array, int fristIndex, int lastIndex);

};

#endif /* QUICKSORT_H_ */
