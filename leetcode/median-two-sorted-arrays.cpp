#include <cstdint>
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#define __ABSURD std::cout << "Absurd state reached" << std::endl; exit(1);

template <typename  T> 
using Vector = std::vector<T>;
using Int = std::uint32_t;

float find_median_sorted_arrays(const Vector<Int> &a, const Vector<Int> &b){
	if (a.size() < b.size()) return find_median_sorted_arrays(b, a);

	const Int m = static_cast<Int>(a.size());
	const Int n = static_cast<Int>(b.size());
	const Int s = m + n;
	const Int c = floor(s/2)+1;
	Int step = floor(n/2)+1;
	Int nn = n;
	Int nm = c - nn;

	const Vector<Int> &M = a;
	const Vector<Int> &N = b;


    if (m == 0) {
        return static_cast<float>(0);
	}
	else if (n == 0) {
        if (s % 2 == 0) {
            return (M[nm-1]+M[nm-2])/2.0;
		}
        else{
            return M[nm-1];
		}
	}
	else if (m == 1) {
        return (M[0] + N[0]) / 2.0;
	}
	else  {
		while (1) {
			Int pm, pn;
			bool p, q;
			pm = nm - 1;
			pn = nn - 1;


			p = (pn != 0) ?  (N[pn-1] <= M[pm]) : true; 
			q = (pm == 0 or M[pm-1] <= N[pn]);
			
			if (p and q) {
            	Int lower_median = std::min(M[pm], N[pn]), upper_median;
                if (lower_median == M[pm]) {
					upper_median = N[pn];
					if (pm < m - 1) upper_median = std::min(N[pn], M[pm+1]);
				}
				else{
					upper_median = M[pm];
					if (pn < n - 1) upper_median = std::min(M[pm], N[pn+1]);
				}

				if (s % 2 == 0) return (lower_median + upper_median) / 2.0;
				else return upper_median;

			}
			else {
				if (not p){
					if (nn == 1){
						Int l = N[0];
						if (pm != 0) l = std::max(N[0], M[pm-1]);

						Int lower_median = std::min(l, M[pm]), upper_median;
						if (lower_median == M[pm]){
							upper_median = N[0];
							if (pm <  m - 1){
								upper_median = std::min(M[pm+1], N[0]);
							}
						}else{
							upper_median = M[pm];
						}
						if (s % 2 == 0) return (lower_median + upper_median) / 2.0;
						else return upper_median;
					}
					nn = std::max<Int>(nn - step, 1);
				}
				else if (not q){
					if (nn == n){
						Int l = std::max(N[pn],M[pm-1]);

						Int lower_median = std::min(l, M[pm]), upper_median;
						if (lower_median == M[pm]){
							upper_median = std::min(M[pm+1], N[pn]);
						}
						else upper_median = M[pm];
						if (s % 2 == 0) return (lower_median + upper_median) / 2.0;
						else return upper_median;
					}
					nn = std::min<Int>(nn + step, n);
				}
				else{
					__ABSURD
				}
				nm = c - nn;
				step = ceil(step/2);

			}
		}
	}
	__ABSURD
}

int main(const int argc, const char* argv[]){

	if (argc < 3){
		std::cout << "program <nums1> <nums2>" << std::endl;
		return 1;
	}
	auto a = Vector<Int>({});
	auto b = Vector<Int>({});
	{
		std::string token;
		auto ss = std::stringstream(argv[1]);
		while (getline(ss, token,  ' ')){
			a.push_back(std::stoi(token));
		}
	}
	{
		std::string token;
		auto ss = std::stringstream(argv[2]);
		while (getline(ss, token,  ' ')){
			b.push_back(std::stoi(token));
		}
	}


	auto result = find_median_sorted_arrays(a, b);
	std::cout << result << std::endl;
	
	return 0;
}
