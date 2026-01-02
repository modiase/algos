#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstring>
#include <iomanip>

volatile int sink;

const size_t KB = 1024;
const size_t MB = 1024 * KB;
const size_t MAX_SIZE = 64 * MB;

double measure_sequential(size_t size_bytes, int* buffer)
{
    size_t count = size_bytes / sizeof(int);
    for (size_t i = 0; i < count; ++i) buffer[i] = i;

    size_t iterations = MAX_SIZE / size_bytes * 10;
    if (iterations < 100) iterations = 100;

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t k = 0; k < iterations; ++k) {
        for (size_t i = 0; i < count; i += 16) {
            sink = buffer[i];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> diff = end - start;

    return diff.count() / ((double)count/16.0 * iterations);
}

double measure_random(size_t size_bytes, int* buffer)
{
    size_t count = size_bytes / sizeof(int);
    std::vector<int> indices(count);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 g(42);
    std::shuffle(indices.begin(), indices.end(), g);

    for (size_t i = 0; i < count - 1; ++i) {
        buffer[indices[i]] = indices[i+1];
    }
    buffer[indices[count-1]] = indices[0];

    size_t iterations = MAX_SIZE / size_bytes * 10;
    if (iterations < 1000) iterations = 1000;
    size_t steps = count * iterations;
    if (steps > 100000000) steps = 100000000;

    int current = indices[0];
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < steps; ++i) {
        current = buffer[current];
    }
    auto end = std::chrono::high_resolution_clock::now();
    sink = current;

    std::chrono::duration<double, std::nano> diff = end - start;
    return diff.count() / steps;
}

int main()
{
    int* buffer = new int[MAX_SIZE / sizeof(int)];

    std::cout << "# Size(KB), Sequential(ns), Random(ns)" << std::endl;

    std::vector<size_t> sizes;
    for (size_t s = 1 * KB; s <= MAX_SIZE; s *= 1.4) {
        sizes.push_back(s);
    }

    for (size_t size : sizes) {
        double seq_lat = measure_sequential(size, buffer);
        double rand_lat = measure_random(size, buffer);

        std::cout << (size / KB) << ", "
                  << std::fixed << std::setprecision(2) << seq_lat << ", "
                  << std::fixed << std::setprecision(2) << rand_lat << std::endl;
    }

    delete[] buffer;
    return 0;
}
