from solution import main, all_of_len, count_distinct_letters


def test_count_distinct_letters():
    s = "ababba"
    assert count_distinct_letters(s) == 2
    s2 = "abcabcddbaba"
    assert count_distinct_letters(s2) == 4


def test_given():
    assert main("abcba", 2) == "bcb"
