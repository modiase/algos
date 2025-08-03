#pragma once

template <typename  T>
using Vector = std::vector<T>;
using Int = std::uint32_t;

#define __ABSURD std::cout << "Absurd state reached" << std::endl; exit(EXIT_FAILURE);

float find_median_sorted_arrays_Ologn(const Vector<Int> &a, const Vector<Int> &b);
float find_median_sorted_arrays_On(const Vector<Int> &a, const Vector<Int> &b);
