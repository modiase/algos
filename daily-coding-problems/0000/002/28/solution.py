
import logging
from typing import List, Tuple

logging.basicConfig(
    format='[%(levelname)s - %(funcName)s - %(lineno)d]:'
    ' %(message)s', level=logging.INFO)


def main(words: List[str], k: int) -> List[str]:
    words = words[:]
    lines = []
    current_line = ''

    while len(words) != 0:
        word = words.pop(0)
        if len(current_line + word) > k - 1:
            current_line = current_line.rstrip()

            logging.debug(f"Appending line: {current_line}.")
            lines.append(justify(current_line, k))

            current_line = word + " "
        else:
            logging.debug(f'Current line is: "{current_line}".')

            logging.debug(f'Adding word: "{word}".')
            current_line += (word + ' ')

            logging.debug(f'Current line is now: "{current_line}".')

    current_line = current_line.rstrip()
    lines.append(justify(current_line, k))

    logging.debug(f'Computed lines {lines}')
    return lines


def justify(line: str, k: int) -> str:
    input_line = line
    current_line_len = len(line)
    while current_line_len < k:
        line = pad_spaces_by_adding_evenly_from_leftmost(line, k)
        current_line_len = len(line)

    logging.debug(f'Returning justified line: "{input_line}" -> "{line}"')
    return line


def pad_spaces_by_adding_evenly_from_leftmost(line: str, k: int) -> str:
    start_index_and_size_of_spaces = compute_start_index_and_size_of_each_space(
        line)

    number_of_padding_spaces_needed = k - len(line)
    index_and_size_of_required_spaces = \
        compute_required_start_index_and_size_of_each_space(
            start_index_and_size_of_spaces,
            number_of_padding_spaces_needed)

    required_sizes = [x[1] for x in index_and_size_of_required_spaces]
    sizes = [x[1] for x in start_index_and_size_of_spaces]

    diffs = [x[0] - x[1] for x in zip(required_sizes, sizes)]
    indices = [x[0] for x in start_index_and_size_of_spaces]

    logging.debug(f"Indices to add spaces: {indices}.")
    reversed_indices = reversed(indices)
    reversed_diffs = reversed(diffs)

    line_as_list = [char for char in line]
    for index, diff in zip(reversed_indices, reversed_diffs):
        line_as_list.insert(index, ' ' * diff)
    return ''.join(line_as_list)


def compute_start_index_and_size_of_each_space(
        line: str) -> List[Tuple[int, int]]:
    spaces_by_start_index_and_size: List[Tuple[int, int]] = []
    offset = 0
    try:
        while idx := line.index(' ') + offset:
            trimmed = line[idx:]
            size = len(trimmed) - len(trimmed.lstrip())

            logging.debug(f'Index is: "{idx}".')
            logging.debug(f'Trimmed is: "{trimmed}".')
            logging.debug(f'Trimmed lstripped is: "{trimmed.lstrip()}".')

            spaces_by_start_index_and_size.append((idx, size))

            offset += len(line) - len(trimmed.lstrip())
            line = trimmed.lstrip()

    except ValueError:
        pass

    if len(spaces_by_start_index_and_size) == 0:
        spaces_by_start_index_and_size.append((len(line), 0))

    logging.debug(f'Returning spaces: {spaces_by_start_index_and_size}.')
    return spaces_by_start_index_and_size


def compute_required_start_index_and_size_of_each_space(
        start_index_and_size_of_spaces: List[Tuple[int, int]],
        number_of_padding_spaces_needed: int) -> List[Tuple[int, int]]:
    required_start_index_and_size_of_spaces = start_index_and_size_of_spaces[:]

    while number_of_padding_spaces_needed > 0:
        size_of_first_space = required_start_index_and_size_of_spaces[0][1]

        if (all([space_size == size_of_first_space
                 for (_, space_size)
                 in required_start_index_and_size_of_spaces])):

            space_index, space_size = required_start_index_and_size_of_spaces[0]
            required_start_index_and_size_of_spaces[0] = (
                space_index, space_size + 1)
            number_of_padding_spaces_needed -= 1

        else:
            for i in range(1, len(required_start_index_and_size_of_spaces)):
                if number_of_padding_spaces_needed == 0:
                    break

                if required_start_index_and_size_of_spaces[i][1] == \
                        size_of_first_space:
                    continue
                else:
                    pair = required_start_index_and_size_of_spaces[i]
                    required_start_index_and_size_of_spaces[i] = (
                        pair[0], pair[i] + 1)
                    number_of_padding_spaces_needed -= 1

    return required_start_index_and_size_of_spaces
