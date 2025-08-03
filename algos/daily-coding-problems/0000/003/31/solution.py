import argparse


def main(first_word: str, second_word: str) -> int:
    distance = abs(len(first_word) - len(second_word))

    shortest_word_length = min(len(first_word), len(second_word))

    for i in range(0, shortest_word_length):
        if not first_word[i] == second_word[i]:
            distance += 1

    return distance


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("first_word", type=str)
    parser.add_argument("second_word", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    result = main(args.first_word, args.second_word)

    print(result)
