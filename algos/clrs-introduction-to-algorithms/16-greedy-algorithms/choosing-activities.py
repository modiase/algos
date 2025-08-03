import bisect
import os
import random as rn
from collections.abc import Collection
from typing import NamedTuple, NewType, Sequence


class Activity(NamedTuple):
    start: int
    end: int


ActivitiesByEnd = NewType("ActivitiesByEnd", Sequence[Activity])


def choose_activities_dp(activities: Collection[Activity]) -> ActivitiesByEnd:
    def _insort(activities: ActivitiesByEnd, activity: Activity) -> ActivitiesByEnd:
        cpy = activities[:]
        bisect.insort(cpy, activity, key=lambda a: a.start)
        return cpy

    activities.sort(key=lambda a: a.end)
    N = len(activities)
    c = [[], []]
    for i in range(N):
        current_activity = activities[i]
        # This step loops through all activities that are compatible with the current activity
        # and inserts the current activity into the list of compatible activities which takes O(n) time
        # therefore the overall time complexity is O(n^2)
        c1, c2 = (
            _insort(
                [
                    activity
                    for activity in c
                    if activity.start >= current_activity.end
                    or activity.end <= current_activity.start
                ],
                current_activity,
            )
            for c in (c[i + 1], c[i])
        )
        c.append(c1 if len(c1) > len(c2) else c2)
    return c[c.index(max(c, key=len))]


def choose_activities(activities: Collection[Activity]) -> ActivitiesByEnd:
    if not activities:
        return []
    activities.sort(key=lambda a: a.end)
    chosen = [activities[0]]
    for activity in activities[1:]:
        # This step checks if the current activity is compatible with the last chosen activity
        # and appends the current activity to the list of chosen activities if it is
        # therefore the overall time complexity is O(n)
        if activity.start >= chosen[-1].end:
            chosen.append(activity)
    return chosen


if __name__ == "__main__":
    seed = int(os.getenv("SEED", 42))
    rn.seed(seed)
    print(f"{seed=}")
    activities = [
        Activity(
            start := rn.randint(0, 20), rn.randint(start + 1, start + rn.randint(1, 10))
        )
        for _ in range(10)
    ]
    print(f"{activities=}")
    print(f"{choose_activities(activities)=}")
    print(f"{choose_activities_dp(activities)=}")
