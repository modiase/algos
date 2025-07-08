#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

int main()
{
    vector<int> nums;
    for (int i = 1; i <= 100; i++)
    {
        nums.push_back(i);
    }
    int sum_square = 0;
    int sum_of_squares = 0;

    for (auto i : nums)
    {
        sum_of_squares += static_cast<int>(pow(i, 2));
        sum_square += i;
    }
    sum_square *= sum_square;
    int diff = sum_square - sum_of_squares;
    cout << sum_square << endl;
    cout << sum_of_squares << endl;
    cout << diff << endl;
}