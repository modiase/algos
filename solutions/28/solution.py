

from typing import List, Tuple


def main(words: List[str], k: int) -> List[str]:
    lines = []
    current_line = ''
    while len(words) != 0:
        word = words.pop(0)
        if len(current_line + word) > k - 1:
            current_line = current_line.rstrip()
            lines.append(justify(current_line, k))
            current_line = word
        else:
            current_line += (word + ' ')
    return lines


def justify(line: str, k: int) -> str:
    current_line_len = len(line)
    while current_line_len < k:
        line = pad_spaces_from_leftmost_evenly(line, k)
        current_line_len = len(line)
    return line


def pad_spaces_from_leftmost_evenly(line: str, k: int) -> str:
    index_and_size_of_spaces = compute_space_indices_and_size(line)

    number_of_padding_spaces_needed = k - len(line)
    index_and_size_of_required_spaces = add_spaces_evenly(index_and_size_of_spaces,
                                                          number_of_padding_spaces_needed)

    required_sizes = [x[1] for x in index_and_size_of_required_spaces]
    sizes = [x[1] for x in index_and_size_of_spaces]

    diffs = [x[0] - x[1] for x in zip(required_sizes, sizes)]
    indices = [x[0] for x in index_and_size_of_spaces]

    reversed_indices = reversed(indices)
    reversed_diffs = reversed(diffs)

    line_as_list = [char for char in line]
    for index, diff in zip(reversed_indices, reversed_diffs):
        line_as_list.insert(index, ' ' * diff)
    return ''.join(line_as_list)


def compute_space_indices_and_size(line: str) -> List[Tuple[int, int]]:
    spaces_by_size_and_start_index: List[Tuple[int, int]] = []
    try:
        while idx := line.index(' '):
            trimmed = line[idx:]
            size = len(trimmed) - len(trimmed.lstrip())
            spaces_by_size_and_start_index.append((idx, size))
            line = trimmed.lstrip()
    except ValueError:
        pass
    return spaces_by_size_and_start_index


def add_spaces_evenly(index_and_size_of_spaces: List[Tuple[int, int]],
                      number_of_padding_spaces_needed: int) -> List[Tuple[int, int]]:
    index_and_size_of_spaces = index_and_size_of_spaces[:]
    while number_of_padding_spaces_needed > 0:
        size_of_first_space = index_and_size_of_spaces[0][1]
        if (all([x[1] == size_of_first_space
                 for x in index_and_size_of_spaces])):
            pair = index_and_size_of_spaces[0]
            index_and_size_of_spaces[0] = (pair[0], pair[1] + 1)
        else:
            for i in range(1, len(index_and_size_of_spaces)):
                if index_and_size_of_spaces[i][1] != size_of_first_space:
                    pair = index_and_size_of_spaces[i]
                    index_and_size_of_spaces[i] = (pair[0], pair[i] + 1)
                    break
        number_of_padding_spaces_needed -= 1

    return index_and_size_of_spaces


if __name__ == '__main__':
    test_input = ["the", "quick", "brown", "fox",
                  "jumps", "over", "the", "lazy", "dog"]
    main(test_input, 16)
