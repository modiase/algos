#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "median-two-sorted-arrays.hpp"

#define check_error(name, f) { if(!f){ std::cout << "Failed to open file: " << name << std::endl; exit(EXIT_FAILURE); }}

template <typename  T> 
using Vector = std::vector<T>;
using Int = std::uint32_t;
using ClockType = std::chrono::steady_clock;
using Micros = std::chrono::microseconds;

int main(const int argc, const char* argv[]){

	if (argc < 3){
		std::cout << "program <input_filepath_1> <input_filepath_2>" << std::endl;
		return 1;
	}
	auto a = Vector<Int>({});
	auto b = Vector<Int>({});
	{
		std::string token;
		auto fp = std::ifstream(argv[1]);
		check_error(argv[1], fp);
		while (getline(fp, token,  ' ')){
			a.push_back(static_cast<Int>(std::stoi(token)));
		}
	}
	{
		std::string token;
		auto fp = std::ifstream(argv[2]);
		check_error(argv[2], fp);
		while (getline(fp, token,  ' ')){
			b.push_back(static_cast<Int>(std::stoi(token)));
		}
	}


	auto start = ClockType::now();
	find_median_sorted_arrays_Ologn(a, b);
	auto end = ClockType::now();
	auto duration_Ologn = std::chrono::duration_cast<Micros>(end - start).count();

	start = ClockType::now();
	find_median_sorted_arrays_On(a, b);
	end = ClockType::now();
	auto duration_On = std::chrono::duration_cast<Micros>(end - start).count();

	std::cout << duration_Ologn << " " << duration_On << std::endl;
	
	return 0;
}
