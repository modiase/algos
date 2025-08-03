choices = [1, 2]


def take_steps(N: int) -> int:
    def _take_steps(M: int) -> int:
        if M == 0 or M == 1:
            return 1
        valid_choices = [x for x in choices if x <= M]
        return sum([_take_steps(M - x) for x in valid_choices])

    return _take_steps(N)


def main():
    print(take_steps(4))


if __name__ == "__main__":
    main()
