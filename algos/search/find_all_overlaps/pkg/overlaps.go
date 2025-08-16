package overlaps

import "sort"

type Interval struct {
	Key   int
	Start int
	End   int
}

func FindAllOverlaps(intervals []Interval) map[int][]*Interval {
	active := make(map[int]Interval)
	overlaps := make(map[int][]*Interval, 0)

	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i].Start < intervals[j].Start
	})

	for _, interval := range intervals {
		for _, other := range active {
			if other.End < interval.Start {
				delete(active, other.Key)
				continue
			}
			overlaps[interval.Key] = append(overlaps[interval.Key], &other)
		}
		active[interval.Key] = interval
	}
	return overlaps
}
