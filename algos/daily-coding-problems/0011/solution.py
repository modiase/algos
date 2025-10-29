from typing import Dict, List, Tuple, Union


def take_first(s: str) -> Tuple[str, str]:
    if len(s) == 1:
        return (s, "")
    return (s[0], s[1:])


def prepare_tree(strings: List[str]) -> Dict[str, Union[Dict]]:
    children = {}
    tups = map(take_first, strings)
    for tup in tups:
        first = tup[0]
        if first in children.keys() and tup[1]:
            children[first].append(tup[1])
        elif tup[1]:
            children[first] = [tup[1]]
        else:
            children[first] = []
    res = {}
    for first_letter, child_list in children.items():
        res[first_letter] = prepare_tree(child_list)
    return res


def return_endings(tree: Dict[str, Dict]) -> List[str]:
    def _return_endings(sub_tree: Dict[str, Dict]) -> List[str]:
        if not sub_tree.keys():
            return [""]
        res = []
        for k in sub_tree.keys():
            endings = _return_endings(sub_tree[k])
            res = res + [k + ending for ending in endings]
        return res

    return _return_endings(tree)


def walk_tree(string: str, tree: Dict[str, Dict]) -> List[str]:
    def _walk_tree(sub_tree, index) -> Tuple[str, List[str]]:
        if index == len(string):
            return (string, return_endings(sub_tree))
        if string[index] not in sub_tree.keys():
            return (string, [])
        return _walk_tree(sub_tree[string[index]], index + 1)

    return _walk_tree(tree, 0)[1]


def main(partial_s: str, strings: List[str]) -> List[str]:
    tree = prepare_tree(strings)
    result = walk_tree(partial_s, tree)
    return [partial_s + res for res in result]


if __name__ == "__main__":
    t = "de"
    s = ["deer", "deal", "love"]
    main(t, s)
