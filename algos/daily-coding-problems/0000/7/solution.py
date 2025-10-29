import collections

N_MAX = 27


def consume_next(tokens):
    """Returns a list of the next possible states where each state is 2-tuple of: 1. The extracted tokens, 2. The remaining tokens."""
    res = []
    len_tokens = len(tokens)
    if len_tokens == 0:
        return res
    if len_tokens == 1:
        return [(tokens, [])]
    first_solution = (tokens[0], tokens[1:])
    res.append(first_solution)
    first_two_tokens_int_val = int(tokens[0] + tokens[1])
    if first_two_tokens_int_val > N_MAX:
        return res
    second_solution = (tokens[0:2], tokens[2:])
    res.append(second_solution)
    return res


def count_decodings(input):
    tokens = [c for c in input]

    count = 0
    queue = collections.deque()
    init = consume_next(tokens)
    for t in init:
        if len(t[1]) != 0:
            queue.appendleft(t[1])

    while len(queue) != 0:
        next_item = queue.pop()
        next_states = consume_next(next_item)
        for state in next_states:
            if len(state[1]) == 0:
                count += 1
            else:
                queue.appendleft(state[1])
    return count
