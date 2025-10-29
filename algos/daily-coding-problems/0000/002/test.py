from solution import main, Node, LinkedList


def test_solution():
    n10 = Node(10)
    n8 = Node(8, n10)
    n1 = Node(1, n8)
    n99 = Node(99, n1)
    n7 = Node(7, n8)
    n3 = Node(3, n7)

    l1 = LinkedList(n3)
    l2 = LinkedList(n99)

    assert main({"l1": l1, "l2": l2}) == 8
