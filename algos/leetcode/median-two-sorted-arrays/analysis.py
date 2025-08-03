import subprocess
from typing import List, Tuple

from tabulate import tabulate


def main():
    results: List[Tuple[str, str, str]] = [("n", "Ologn", "On")]
    for i in range(1, 9):
        logn_run = 0
        n_run = 0
        for _ in range(5):
            s1, s2 = (
                subprocess.run(
                    f"./timeit .tmp/a_{i}.txt .tmp/b_{i}.txt",
                    shell=True,
                    check=True,
                    capture_output=True,
                )
                .stdout.decode("utf8")
                .split(" ")
            )
            logn_run += int(s1)
            n_run += int(s2)

        results.append((str(f"10^{i}"), str(logn_run / 5.0), str(n_run / 5.0)))

    print("Average of 5 runs (microseconds)")
    print(tabulate(results, headers="firstrow"))


if __name__ == "__main__":
    main()
