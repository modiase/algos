class OrderStore:
    def __init__(self):
        self._q = []

    def record(self, order_id: int) -> None:
        self._q.append(order_id)

    def get_last(self, i: int) -> int:
        return self._q[-i]


def main(): ...


if __name__ == "__main__":
    main()
