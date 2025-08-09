// algos/data-structures/trees/test/testSegmentTree.go
package test

import (
	segmenttree "algos/data-structures/trees/segment-tree/pkg"
	"testing"
)

func sumInt(a, b int) int { return a + b }

func TestSegmentTree_QueryBasic(t *testing.T) {
	arr := []int{1, 3, 5, 7, 9, 11}
	st := segmenttree.NewSegmentTree[int](arr, sumInt, 0, 0)

	tests := []struct {
		l, r int
		want int
	}{
		{0, 0, 1},
		{0, 2, 9},  // 1+3+5
		{1, 3, 15}, // 3+5+7
		{3, 5, 27}, // 7+9+11
		{0, 5, 36}, // sum all
		{2, 2, 5},  // single element
	}

	for _, tt := range tests {
		got := st.Query(tt.l, tt.r)
		if got != tt.want {
			t.Fatalf("Query(%d,%d) = %d, want %d", tt.l, tt.r, got, tt.want)
		}
	}
}

func TestSegmentTree_UpdateAndQuery(t *testing.T) {
	arr := []int{1, 3, 5, 7, 9, 11}
	st := segmenttree.NewSegmentTree[int](arr, sumInt, 0, 0)

	// Update index 1 from 3 -> 10
	st.Update(1, 10)

	tests := []struct {
		l, r int
		want int
	}{
		{0, 2, 1 + 10 + 5},              // 16
		{1, 3, 10 + 5 + 7},              // 22
		{0, 5, 1 + 10 + 5 + 7 + 9 + 11}, // 43
	}

	for _, tt := range tests {
		got := st.Query(tt.l, tt.r)
		if got != tt.want {
			t.Fatalf("after Update, Query(%d,%d) = %d, want %d", tt.l, tt.r, got, tt.want)
		}
	}

	// Another update: index 4 from 9 -> -1
	st.Update(4, -1)
	if got, want := st.Query(3, 5), 7+(-1)+11; got != want {
		t.Fatalf("after second Update, Query(3,5) = %d, want %d", got, want)
	}
	if got, want := st.Query(0, 5), 1+10+5+7-1+11; got != want {
		t.Fatalf("after second Update, Query(0,5) = %d, want %d", got, want)
	}
}

func TestSegmentTree_SingleElement(t *testing.T) {
	arr := []int{42}
	st := segmenttree.NewSegmentTree[int](arr, sumInt, 0, 0)

	if got := st.Query(0, 0); got != 42 {
		t.Fatalf("Query(0,0) = %d, want %d", got, 42)
	}

	st.Update(0, -5)
	if got := st.Query(0, 0); got != -5 {
		t.Fatalf("after Update, Query(0,0) = %d, want %d", got, -5)
	}
}
