#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>

#include "median-two-sorted-arrays.hpp"

float find_median_sorted_arrays_On(const Vector<Int> &a, const Vector<Int> &b){
	auto m = Vector<Int>({});
	const Int s = static_cast<Int>(a.size() + b.size());
	Int pa = 0;
	Int pb = 0;

	while (pa < a.size() && pb < b.size()){
		if (a[pa] < b[pb]) {
			m.push_back(a[pa]);
			pa++;
		}else{
			m.push_back(b[pb]);
			pb++;
		}
	}
	while (pa < a.size()){
			m.push_back(a[pa]);
			pa++;
	}
	while (pb < b.size()){
			m.push_back(b[pb]);
			pb++;
	}
	if (s % 2 == 0){
		return (m[s/2] + m[(s/2)-1])/2.0;
	}else{
		return static_cast<float>(m[std::floor(s / 2)]);
	}

}

float find_median_sorted_arrays_Ologn(const Vector<Int> &a, const Vector<Int> &b){
	if (a.size() < b.size()) return find_median_sorted_arrays_Ologn(b, a);

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

