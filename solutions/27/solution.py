


def main(input_string: str) -> bool:
    open_braces_stack = []
    map_closing_to_opening_brace = {
        ')': '(',
        '}': '{',
        ']':'['
    }

    try:
        for char in input_string:
            if char in ('{','(','['):
                open_braces_stack.append(char)
            else:
                assert char in ('}',')',']')
                if not open_braces_stack[-1] == \
                    map_closing_to_opening_brace[char]:
                    return False
                open_braces_stack.pop()
    except IndexError:
        return False
    
    return len(open_braces_stack) == 0
