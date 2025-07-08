// ? If we list all the natural numbers below 10 that are multiples of 3 or 5, we get 3, 5, 6 and 9. The sum of these multiples is 23.
// ? Find the sum of all the multiples of 3 or 5 below 1000.
#include <iostream>
#include <vector>

int main()
{
    // Setup vector to hold numbers matching the predicate
    std::vector<int> nums;

    // Loop through the numbers below 1000
    for (int i = 0; i < 1000; i++)
    {
        if (i % 3 == 0 || i % 5 == 0)
        {
            nums.push_back(i);
        }
    }

    // Loop through the vector to calculate the sum.
    int sum = 0;
    for (const int &x : nums)
    {
        sum += x;
    }
    std::cout << sum << std::endl;
}