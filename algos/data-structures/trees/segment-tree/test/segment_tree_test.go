// algos/data-structures/trees/test/testSegmentTree.go
package test

import (
	segmenttree "algos/data-structures/trees/segment-tree/pkg"
	"math"
	"testing"
)

func sumInt(a, b int) int { return a + b }

func TestSegmentTree_QueryBasic(t *testing.T) {
	arr := []int{1, 3, 5, 7, 9, 11}
	monoid := segmenttree.NewMonoid(sumInt, 0)
	st := segmenttree.NewSegmentTree(arr, monoid)

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
	monoid := segmenttree.NewMonoid(sumInt, 0)
	st := segmenttree.NewSegmentTree(arr, monoid)

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
	monoid := segmenttree.NewMonoid(sumInt, 0)
	st := segmenttree.NewSegmentTree(arr, monoid)

	if got := st.Query(0, 0); got != 42 {
		t.Fatalf("Query(0,0) = %d, want %d", got, 42)
	}

	st.Update(0, -5)
	if got := st.Query(0, 0); got != -5 {
		t.Fatalf("after Update, Query(0,0) = %d, want %d", got, -5)
	}
}

func TestSegmentTree_EmptyArray(t *testing.T) {
	var arr []int
	monoid := segmenttree.NewMonoid(sumInt, 0)
	st := segmenttree.NewSegmentTree(arr, monoid)

	if got := st.Query(0, 0); got != 0 {
		t.Fatalf("Query on empty tree = %d, want %d", got, 0)
	}
	if got := st.Query(-5, 100); got != 0 {
		t.Fatalf("Query(-5,100) on empty tree = %d, want %d", got, 0)
	}

	// Updates outside should be no-ops and not panic
	st.Update(0, 1)
	st.Update(-1, 1)
}

func TestSegmentTree_OutOfBoundsClamping(t *testing.T) {
	arr := []int{1, 3, 5, 7, 9, 11}
	monoid := segmenttree.NewMonoid(sumInt, 0)
	st := segmenttree.NewSegmentTree(arr, monoid)

	if got, want := st.Query(-5, 100), 36; got != want {
		t.Fatalf("Query(-5,100) = %d, want %d", got, want)
	}
	if got, want := st.Query(-5, 2), 1+3+5; got != want {
		t.Fatalf("Query(-5,2) = %d, want %d", got, want)
	}
	if got, want := st.Query(2, 100), 5+7+9+11; got != want {
		t.Fatalf("Query(2,100) = %d, want %d", got, want)
	}
}

func TestSegmentTree_QlGreaterThanQrEmpty(t *testing.T) {
	arr := []int{1, 2, 3}
	monoid := segmenttree.NewMonoid(sumInt, 0)
	st := segmenttree.NewSegmentTree(arr, monoid)

	if got := st.Query(2, 1); got != 0 {
		t.Fatalf("Query(2,1) = %d, want 0 (empty)", got)
	}
	if got := st.Query(10, -10); got != 0 {
		t.Fatalf("Query(10,-10) = %d, want 0 (empty)", got)
	}
}

func TestSegmentTree_MultipleUpdates(t *testing.T) {
	arr := []int{0, 0, 0, 0, 0, 0, 0}
	monoid := segmenttree.NewMonoid(sumInt, 0)
	st := segmenttree.NewSegmentTree(arr, monoid)

	cur := append([]int(nil), arr...)
	set := func(i, v int) {
		st.Update(i, v)
		cur[i] = v
	}
	sumRange := func(a []int, l, r int) int {
		if l < 0 {
			l = 0
		}
		if r >= len(a) {
			r = len(a) - 1
		}
		if l > r || len(a) == 0 {
			return 0
		}
		s := 0
		for i := l; i <= r; i++ {
			s += a[i]
		}
		return s
	}

	set(3, 5)
	set(0, 2)
	set(6, -1)
	set(3, 7)  // overwrite
	set(1, 10) // another

	cases := [][2]int{{0, 6}, {0, 0}, {3, 3}, {2, 4}, {1, 5}, {-5, 100}, {4, 1}}
	for _, c := range cases {
		l, r := c[0], c[1]
		got := st.Query(l, r)
		want := sumRange(cur, l, r)
		if got != want {
			t.Fatalf("Query(%d,%d) = %d, want %d; cur=%v", l, r, got, want, cur)
		}
	}
}

func TestSegmentTree_PointQueriesAfterUpdates(t *testing.T) {
	arr := []int{5, 4, 3, 2, 1}
	monoid := segmenttree.NewMonoid(sumInt, 0)
	st := segmenttree.NewSegmentTree(arr, monoid)

	st.Update(0, 9)
	st.Update(4, -3)
	st.Update(2, 100)

	tests := []struct {
		i, want int
	}{
		{0, 9},
		{1, 4},
		{2, 100},
		{3, 2},
		{4, -3},
	}
	for _, tt := range tests {
		if got := st.Query(tt.i, tt.i); got != tt.want {
			t.Fatalf("Query(%d,%d) = %d, want %d", tt.i, tt.i, got, tt.want)
		}
	}
}

func TestSegmentTree_ImmutabilityOfInput(t *testing.T) {
	arr := []int{1, 2, 3}
	monoid := segmenttree.NewMonoid(sumInt, 0)
	st := segmenttree.NewSegmentTree(arr, monoid)

	arr[0] = 999 // mutate caller slice after building
	if got := st.Query(0, 0); got != 1 {
		t.Fatalf("tree reflects mutated input; got %d want %d", got, 1)
	}
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func TestSegmentTree_MinCombineExample(t *testing.T) {
	arr := []int{5, 1, 9, 3, 7}
	// empty as MaxInt (sentinel for empty)
	monoid := segmenttree.NewMonoid(minInt, math.MaxInt)
	st := segmenttree.NewSegmentTree(arr, monoid)

	tests := []struct {
		l, r int
		want int
	}{
		{0, 4, 1},
		{1, 3, 1},
		{2, 2, 9},
		{-10, 10, 1},
	}
	for _, tt := range tests {
		if got := st.Query(tt.l, tt.r); got != tt.want {
			t.Fatalf("min Query(%d,%d) = %d, want %d", tt.l, tt.r, got, tt.want)
		}
	}

	// Empty range should return empty sentinel
	if got := st.Query(3, 2); got != math.MaxInt {
		t.Fatalf("min Query(3,2) empty = %d, want %d", got, math.MaxInt)
	}
}
