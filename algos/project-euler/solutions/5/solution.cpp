#include <vector>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <utility>

using namespace std;

bool isPrime(int x);

int main()
{

    const int N = 20;
    vector<vector<int>> primeFactors;
    primeFactors.push_back(vector<int>());
    primeFactors.push_back(vector<int>());

    for (int i = 2; i <= N; i++) {
        vector<int> v;

        for (int j = 2; j <= i; j++) {
            if (i % j == 0 && isPrime(j)) {
                int quot = i;
                while (quot % j == 0) {
                    v.push_back(j);
                    quot = static_cast<int>(floor(quot / j));
                }
            }
        }

        primeFactors.push_back(v);
    }

    // for (int i = 0; i <= N; i++)
    // {
    //     for (int j = 0; j < primeFactors[i].size(); j++)
    //     {
    //         cout << i << "    " << primeFactors[i][j] << endl;
    //     }
    // }

    vector<pair<int, int>> accumulator;

    for (int i = 2; i < N; i++) {
        int max_count = 0;
        for (int j = 2; j < N; j++) {
            int inner_count = 0;
            for (auto x : primeFactors[j]) {
                if (x == i) {
                    inner_count++;
                }
            }
            max_count = max(max_count, inner_count);
        }
        accumulator.push_back(make_pair(i, max_count));
    }

    // for (int i = 0; i < accumulator.size(); i++)
    // {
    //     cout << i << "   " << accumulator[i].first << "    " << accumulator[i].second << endl;
    // }

    int prod = 1;
    for (auto x : accumulator) {
        prod *= static_cast<int>(pow(x.first, x.second));
    }
    cout << prod << endl;
}

bool isPrime(int x)
{
    const int limit = static_cast<int>(floor(sqrt(x)));
    for (int i = 2; i <= limit; i++) {
        if (x % i == 0) {
            return false;
        }
    }
    return true;
}