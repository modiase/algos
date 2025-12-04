import re


def is_palindrome_permutated(s: str) -> bool:
    s = re.sub(r"[^A-Za-z]", "", s.lower())
    acc = set()
    for c in s:
        if c in acc:
            acc.remove(c)
        else:
            acc.add(c)
    return len(acc) == len(s) % 2


assert is_palindrome_permutated("tact coa")
assert not is_palindrome_permutated("abc")
assert is_palindrome_permutated("aabb")
assert is_palindrome_permutated("aabbc")
assert is_palindrome_permutated("")
assert is_palindrome_permutated("a")
assert not is_palindrome_permutated("akfdjlsdododf fosododsasjsj")
