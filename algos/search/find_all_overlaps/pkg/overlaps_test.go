package overlaps

import (
	"reflect"
	"testing"
)

func TestFindAllOverlaps(t *testing.T) {
	tests := []struct {
		name      string
		intervals []Interval
		expected  map[int][]*Interval
	}{
		{
			name:      "empty intervals",
			intervals: []Interval{},
			expected:  map[int][]*Interval{},
		},
		{
			name: "single interval",
			intervals: []Interval{
				{Key: 1, Start: 1, End: 5},
			},
			expected: map[int][]*Interval{},
		},
		{
			name: "two non-overlapping intervals",
			intervals: []Interval{
				{Key: 1, Start: 1, End: 3},
				{Key: 2, Start: 4, End: 6},
			},
			expected: map[int][]*Interval{},
		},
		{
			name: "two overlapping intervals",
			intervals: []Interval{
				{Key: 1, Start: 1, End: 5},
				{Key: 2, Start: 3, End: 7},
			},
			expected: map[int][]*Interval{
				2: {&Interval{Key: 1, Start: 1, End: 5}},
			},
		},
		{
			name: "three intervals with overlaps",
			intervals: []Interval{
				{Key: 1, Start: 1, End: 5},
				{Key: 2, Start: 3, End: 7},
				{Key: 3, Start: 6, End: 10},
			},
			expected: map[int][]*Interval{
				2: {&Interval{Key: 1, Start: 1, End: 5}},
				3: {&Interval{Key: 2, Start: 3, End: 7}},
			},
		},
		{
			name: "intervals with same start time",
			intervals: []Interval{
				{Key: 1, Start: 1, End: 5},
				{Key: 2, Start: 1, End: 3},
			},
			expected: map[int][]*Interval{
				2: {&Interval{Key: 1, Start: 1, End: 5}},
			},
		},
		{
			name: "intervals with same end time",
			intervals: []Interval{
				{Key: 1, Start: 1, End: 5},
				{Key: 2, Start: 3, End: 5},
			},
			expected: map[int][]*Interval{
				2: {&Interval{Key: 1, Start: 1, End: 5}},
			},
		},
		{
			name: "completely contained intervals",
			intervals: []Interval{
				{Key: 1, Start: 1, End: 10},
				{Key: 2, Start: 3, End: 7},
			},
			expected: map[int][]*Interval{
				2: {&Interval{Key: 1, Start: 1, End: 10}},
			},
		},
		{
			name: "multiple overlaps for single interval",
			intervals: []Interval{
				{Key: 1, Start: 1, End: 5},
				{Key: 2, Start: 2, End: 4},
				{Key: 3, Start: 3, End: 6},
			},
			expected: map[int][]*Interval{
				2: {&Interval{Key: 1, Start: 1, End: 5}},
				3: {&Interval{Key: 1, Start: 1, End: 5}, &Interval{Key: 2, Start: 2, End: 4}},
			},
		},
		{
			name: "intervals with zero length",
			intervals: []Interval{
				{Key: 1, Start: 1, End: 1},
				{Key: 2, Start: 1, End: 1},
			},
			expected: map[int][]*Interval{
				2: {&Interval{Key: 1, Start: 1, End: 1}},
			},
		},
		{
			name: "negative start times",
			intervals: []Interval{
				{Key: 1, Start: -5, End: 0},
				{Key: 2, Start: -3, End: 2},
			},
			expected: map[int][]*Interval{
				2: {&Interval{Key: 1, Start: -5, End: 0}},
			},
		},
		{
			name: "large numbers",
			intervals: []Interval{
				{Key: 1, Start: 1000, End: 2000},
				{Key: 2, Start: 1500, End: 2500},
			},
			expected: map[int][]*Interval{
				2: {&Interval{Key: 1, Start: 1000, End: 2000}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := FindAllOverlaps(tt.intervals)
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("findAllOverlaps() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestFindAllOverlapsEdgeCases(t *testing.T) {
	t.Run("nil intervals", func(t *testing.T) {
		result := FindAllOverlaps(nil)
		if result == nil {
			t.Error("FindAllOverlaps(nil) should return empty map, not nil")
		}
		if len(result) != 0 {
			t.Errorf("FindAllOverlaps(nil) should return empty map, got %v", result)
		}
	})

	t.Run("intervals with duplicate keys", func(t *testing.T) {
		intervals := []Interval{
			{Key: 1, Start: 1, End: 5},
			{Key: 1, Start: 2, End: 6}, // Same key as first
		}
		result := FindAllOverlaps(intervals)
		// This test will help identify if the function handles duplicate keys correctly
		t.Logf("Result with duplicate keys: %v", result)
	})

	t.Run("intervals with invalid ranges (start > end)", func(t *testing.T) {
		intervals := []Interval{
			{Key: 1, Start: 5, End: 1}, // Invalid: start > end
			{Key: 2, Start: 2, End: 4},
		}
		result := FindAllOverlaps(intervals)
		// This test will help identify if the function handles invalid ranges correctly
		t.Logf("Result with invalid ranges: %v", result)
	})
}
