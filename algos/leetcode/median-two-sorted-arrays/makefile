CXX=g++
CXX_FLAGS=-std=c++11 -Wall -Wpedantic -Werror -I./include

solution: solution.cpp median-two-sorted-arrays
	$(CXX) $(CXX_FLAGS) solution.cpp .bin/median-two-sorted-arrays.o -o solution

timeit: timeit.cpp median-two-sorted-arrays
	$(CXX) $(CXX_FLAGS) timeit.cpp .bin/median-two-sorted-arrays.o -o timeit

median-two-sorted-arrays:
	mkdir -p .bin/
	$(CXX) $(CXX_FLAGS) -c median-two-sorted-arrays.cpp -o .bin/median-two-sorted-arrays.o

