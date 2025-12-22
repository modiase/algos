/*
 * Subtle bug when writing this resulted in i - window being smaller than 0 which for an unsigned integer results in rollover to a max value
 * thus i is by definition ALWAYS smaller than i - window.
 */
#include <vector>
#include <deque>
#include <iostream>
#include <utility>

template <typename T>
using Vector = std::vector<T>;
template <typename T>
using Deque = std::deque<T>;
template <typename T1, typename T2>
using Pair = std::pair<T1, T2>;
using Int = std::uint32_t;
using SizeT = std::size_t;

void max_subarray(const Vector<Int> &vec, const Int window)
{
    auto q = Deque<Pair<SizeT, Int>>({});
    SizeT i = 0;
    while (i < vec.size()) {
        if (q.empty()) q.push_back({i, vec[i]});
        else  {
            while(!q.empty() && i >= window && std::get<0>(q[0]) <= i - window) { // Added i >= window to avoid rollover problem (see top)
                q.pop_front();
            }
            if (!q.empty() && std::get<1>(q[0]) < vec[i]) {
                q.clear();
            }
            q.push_back({i, vec[i]});
        }
        if (i >= window - 1) {
            std::cout << i << " " << std::get<1>(q[0]) << std::endl;
        }
        i++;
    }
}

int main()
{
    auto vec = Vector<Int>({ 5, 3, 4, 5, 2, 3, 7, 2, 1, 0, 1, 10, 2, 3, 3 });
    max_subarray(vec, 3);
    return 0;
}
