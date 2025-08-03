from typing import List


def main(input_map: List[int]) -> int:
    fill_level = input_map[0]
    total_fill = 0

    for wall_height in input_map:
        if wall_height < fill_level:
            total_fill += fill_level - wall_height
        else:
            fill_level = wall_height

    max_height = fill_level

    if input_map[-1] < fill_level:
        n_fill_level = input_map[-1]
        for wall_height in reversed(input_map):
            if wall_height == max_height:
                break
            elif wall_height > n_fill_level:
                n_fill_level = wall_height
                total_fill -= max_height - n_fill_level
            else:
                total_fill -= fill_level - n_fill_level

    return total_fill
