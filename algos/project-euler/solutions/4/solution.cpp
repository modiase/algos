#include <iostream>
#include <vector>
#include <algorithm>

bool isPalindrome(int x);

using namespace std;
int main()
{
    vector<int> palindromes;
    for (int i = 100; i < 1000; i++)
    {
        for (int j = 100; j <= i; j++)
        {
            int prod = i * j;
            if (isPalindrome(prod))
            {
                palindromes.push_back(prod);
                sort(palindromes.begin(), palindromes.end());
            }
        }
    }
    int res = -1;
    if (palindromes.size() > 0)
    {
        res = palindromes.back();
    }
    cout << res << endl;
}

bool isPalindrome(int x)
{
    vector<int> digits;
    int cpy = x;
    while (cpy != 0)
    {
        int digit = cpy % 10;
        digits.push_back(digit);
        cpy = cpy / 10;
    };
    int len = digits.size();
    for (int i = 0; i <= (len / 2); i++)
    {
        if (digits[i] != digits[len - 1 - i])
            return false;
    }
    return true;
}