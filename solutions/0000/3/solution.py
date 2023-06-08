from given import Node
from typing import Union
import re


node_regex = re.compile(r'\((.*?),(.*?)\)')


def serialize(node):
    """Serializes a Node object"""
    def _serialize(sub_node: Union[Node, str, None], node_id: int) -> list:
        if isinstance(sub_node, Node):
            return [(node_id, sub_node.val)] + _serialize(sub_node.left, node_id * 2 + 1) + _serialize(sub_node.right, node_id * 2 + 2)
        elif sub_node:
            return [(node_id, sub_node)]
        else:
            return []
    return "::".join([f'({x[0]},{x[1]})' for x in _serialize(node, 0)])


def deserialize(node_str):
    """Deserializes a Node object"""
    tokens = node_str.split("::")
    map = {}
    for token in tokens:
        mo = node_regex.match(token)
        if not mo:
            continue
        id = int(mo.group(1))
        payload = mo.group(2)
        map[id] = Node(payload)
    indices = map.keys()
    assert 0 in indices, "No root node found."
    map_list = list(map.items())
    for n in map_list:
        if 2 * n[0] + 1 in indices:
            map[n[0]].left = map[2 * n[0] + 1]
        if 2 * n[0] + 2 in indices:
            map[n[0]].right = map[2 * n[0] + 2]
    return map[0]
