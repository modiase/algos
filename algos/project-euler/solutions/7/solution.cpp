#include <iostream>
#include <cmath>
#include <vector>

using namespace std;
bool isPrime(int x)
{
    const int limit = floor(sqrt(x));
    int index = 2;
    while (index <= limit) {
        if (x % index == 0) {
            return false;
        }
        index++;
    }
    return true;
}

int main()

{
    const int END = 6;
    int index = 2;
    vector<int> primes;
    while (primes.size() < END) {
        if (isPrime(index)) {
            primes.push_back(index);
        }
        index++;
    }

    cout << primes[END - 1] << endl;
}
