#ifndef TYPEDEFS_H_
#define TYPEDEFS_H_

enum class Sorting_Type {sort_optimistic, sort_pesymistic, sort_random};

constexpr Sorting_Type defaultSortingType = Sorting_Type::sort_random;
constexpr int defaultRepeatCount = 10;
constexpr int defaultArraySize = 100000;
constexpr int defaultUniqueValues = 0;

#endif TYPEDEFS_H_
