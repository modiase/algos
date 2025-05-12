"""
This problem was asked by Amazon.

The stable marriage problem is defined as follows:

Suppose you have N men and N women, and each person has ranked their prospective opposite-sex partners in order of preference.

For example, if N = 3, the input could be something like this:

guy_preferences = {
    'andrew': ['caroline', 'abigail', 'betty'],
    'bill': ['caroline', 'betty', 'abigail'],
    'chester': ['betty', 'caroline', 'abigail'],
}

gal_preferences = {
    'abigail': ['andrew', 'bill', 'chester'],
    'betty': ['bill', 'andrew', 'chester'],
    'caroline': ['bill', 'chester', 'andrew']
}
Write an algorithm that pairs the men and women together in such a way that no two people of opposite sex would both rather be with each other than with their current partners.
"""

import os
import random
import time
from collections.abc import Mapping, MutableSet, Sequence

from bidict import bidict


def gale_shapley(
    proposers: Mapping[str, Sequence[str]], acceptors: Mapping[str, Sequence[str]]
) -> Mapping[str, str] | None:
    """
    Time complexity: O(N^2)
    Space complexity: O(N)
    """
    proposer_to_acceptor = bidict({})
    free_proposers: MutableSet[str] = set(proposers.keys())
    free_acceptors: MutableSet[str] = set(acceptors.keys())

    while len(free_proposers) != 0:
        proposer = free_proposers.pop()
        proposer_preferences = proposers[proposer]
        for acceptor in proposer_preferences:
            if acceptor in free_acceptors:
                proposer_to_acceptor[proposer] = acceptor
                free_acceptors.remove(acceptor)
                break
            else:
                proposer_rank = (
                    acceptors[acceptor].index(proposer)
                    if proposer in acceptors[acceptor]
                    else None
                )
                if proposer_rank is None:
                    continue
                if proposer_rank < acceptors[acceptor].index(
                    current_proposer := proposer_to_acceptor.inverse[acceptor]
                ):
                    proposer_to_acceptor.pop(current_proposer)
                    free_proposers.add(current_proposer)
                    proposer_to_acceptor[proposer] = acceptor
                    break

        else:
            return None

    return dict(proposer_to_acceptor)


if __name__ == "__main__":
    seed = int(os.environ.get("SEED", time.time()))
    random.seed(seed)
    print(f"Seed: {seed}")
    N = 5
    proposers = {
        f"proposer_{i}": [f"acceptor_{j}" for j in random.sample(range(N), N)]
        for i in range(N)
    }
    acceptors = {
        f"acceptor_{i}": [f"proposer_{j}" for j in random.sample(range(N), N)]
        for i in range(N)
    }
    print(proposers)
    print(acceptors)
    print(gale_shapley(proposers, acceptors))
