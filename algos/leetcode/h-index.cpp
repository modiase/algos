#include <cstdint>
#include <vector>

using Int = std::uint32_t;
template <typename T> using Vector = std::vector<T>;


Int hIndex(Vector<Int>& citations)
{
    const Int N = citations.size();
    auto citationCount = Vector<Int>(N+1, 0);
    for (auto c : citations) {
        if (c >= N) citationCount[N]++;
        else citationCount[c]++;
    }

    Int cumulativeSum = 0;
    for (Int i = N; i >= 0; i--) {
        cumulativeSum += citationCount[i];
        if (cumulativeSum >= i) return i;
    }
    return cumulativeSum;
}
