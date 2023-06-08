#include <random>
#include <iostream>
#include <cmath>

using namespace std;

int main()
{
    const int N = 200000000;
    int count = 0;
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double>
        dis(0.0, 1.0);
    for (int i = 0; i < N; i++)
    {
        double x = dis(gen);
        double y = dis(gen);
        if (pow(x - 0.5, 2.0) + pow(y - 0.5, 2.0) < 0.25)
        {
            count++;
        }
    }
    const double PI = 4.0 * count / (double)N;
    cout
        << PI << endl;
}
