from typing import Iterator


def count_distinct_letters(s: str) -> int:
    """Returns the number of distinct letters in the argument string"""
    return len(set(s))


def all_of_len(s: str) -> Iterator[str]:
    """Returns all the strings of length k in the string s"""
    l = len(s)
    for k in range(l, 0, -1):
        for j in range(0, l - k + 1):
            yield (s[j : j + k])


def main(s: str, k: int):
    gen = all_of_len(s)
    for sub in gen:
        if count_distinct_letters(sub) == k:
            return sub


if __name__ == "__main__":
    res = main("abcba", 2)
    print(res)
