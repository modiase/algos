class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def is_unival(self):
        if self.left:
            left = self.left.value == self.value and self.left.is_unival()
        else:
            left = True
        if self.right:
            right = self.right.value == self.value and self.right.is_unival()
        else:
            right = True

        return left and right


def count_univals(root):
    def _recursive_count_univals(node):
        if not node:
            return 0
        return (
            int(node.is_unival())
            + _recursive_count_univals(node.left)
            + _recursive_count_univals(node.right)
        )

    return _recursive_count_univals(root)
