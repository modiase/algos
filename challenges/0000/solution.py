import collections


class BruteForceStrategy:
    def __init__(self):
        pass

    def all_ways(self, l: list) -> list:
        def _all_ways(_l):
            if len(_l) == 0:
                return []
            if len(_l) == 1:
                return [[1]]
            if len(_l) == 2:
                return [[1, 0], [0, 1]]
            res = []
            rec = self.all_ways(l[2:])
            for r in rec:
                if r[0] == 0:
                    res.append([0, 1] + r)
                    res.append([1, 0] + r)
                else:
                    res.append([1, 0] + r)
            return res

        return _all_ways(l)

    def execute(self, list):
        max_sum_found = -1
        res = []
        ptr_arrays = self.all_ways(list)
        for l in ptr_arrays:
            s = sum([a * b for (a, b) in zip(list, l)])
            if s > max_sum_found:
                res = l
                max_sum_found = s
        return (max_sum_found, res)

    def __call__(self, list):
        return self.execute(list)

    def __repr__(self):
        return '<BruteForceStrategy>'


def solve(l, prefered_strategy=None):
    if not prefered_strategy:
        strategy = BruteForceStrategy()
    else:
        strategy = prefered_strategy
    return strategy(l)


if __name__ == '__main__':
    l = [5, 1, 1, 5]
    solution = solve(l)
