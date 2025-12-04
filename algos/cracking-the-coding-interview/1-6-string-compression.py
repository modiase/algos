def string_compress(s: str) -> str:
    if not s:
        return ""
    current = s[0]
    count = 1
    result = []
    for c in s[1:]:
        if c == current:
            count += 1
        else:
            result.append(str(count) + current)
            current = c
            count = 1
    result.append(str(count) + current)

    compressed = "".join(result)
    return s if len(s) < len(compressed) else compressed


assert string_compress("aaaabbbb") == "4a4b"
assert string_compress("aabcccccaaa") == "2a1b5c3a"
assert string_compress("") == ""
assert string_compress("a") == "a"
