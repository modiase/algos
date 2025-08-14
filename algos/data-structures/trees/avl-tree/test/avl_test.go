package avl_test

import (
	"strconv"
	"strings"
	"testing"

	avl "avl-tree/pkg"
)

func intCmp(a, b int) int    { return a - b }
func revIntCmp(a, b int) int { return b - a }
func strCmp(a, b string) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	default:
		return 0
	}
}

func TestInorder_EmptyTree(t *testing.T) {
	tree := avl.NewTree(intCmp)
	got := tree.Inorder()
	if got != "" {
		t.Fatalf("expected empty inorder traversal, got %q", got)
	}
}

func TestInorder_SingleInsert(t *testing.T) {
	tree := avl.NewTree(intCmp)
	tree.Insert(42)
	got := tree.Inorder()
	want := "42"
	if got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}

func TestInorder_SortedAfterVariousInsertOrders_Ints(t *testing.T) {
	cases := []struct {
		name   string
		input  []int
		expect []int
	}{
		{name: "ascending (RR case path)", input: []int{1, 2, 3}, expect: []int{1, 2, 3}},
		{name: "descending (LL case path)", input: []int{3, 2, 1}, expect: []int{1, 2, 3}},
		{name: "LR rotation path", input: []int{3, 1, 2}, expect: []int{1, 2, 3}},
		{name: "RL rotation path", input: []int{1, 3, 2}, expect: []int{1, 2, 3}},
		{name: "mixed", input: []int{10, 20, 5, 4, 15, 30, 25, 16, 14, 13, 12}, expect: []int{4, 5, 10, 12, 13, 14, 15, 16, 20, 25, 30}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tree := avl.NewTree(intCmp)
			for _, v := range tc.input {
				tree.Insert(v)
			}
			got := tree.Inorder()
			want := intsToSortedString(tc.expect)
			if got != want {
				t.Fatalf("inorder mismatch\nwant: %q\ngot:  %q", want, got)
			}
		})
	}
}

func TestInorder_DuplicateInsertionsIgnored(t *testing.T) {
	tree := avl.NewTree(intCmp)
	input := []int{2, 2, 1, 3, 3, 2}
	for _, v := range input {
		tree.Insert(v)
	}
	got := tree.Inorder()
	want := intsToSortedString([]int{1, 2, 3})
	if got != want {
		t.Fatalf("expected %q with duplicates ignored, got %q", want, got)
	}
}

func TestGeneric_StringKeys(t *testing.T) {
	tree := avl.NewTree[string](strCmp)
	vals := []string{"delta", "alpha", "charlie", "bravo"}
	for _, s := range vals {
		tree.Insert(s)
	}
	got := tree.Inorder()
	want := strings.Join([]string{"alpha", "bravo", "charlie", "delta"}, " ")
	if got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}

func TestCustomComparator_DescendingOrder(t *testing.T) {
	tree := avl.NewTree(revIntCmp)
	for _, v := range []int{1, 2, 3, 4, 5} {
		tree.Insert(v)
	}
	got := tree.Inorder()
	want := intsToSortedString([]int{5, 4, 3, 2, 1})
	if got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}

func TestStress_InsertMany_NoPanicsAndSortedOutput(t *testing.T) {
	tree := avl.NewTree(intCmp)
	var input []int
	for i := 100; i >= 1; i-- {
		input = append(input, i)
	}
	for _, v := range input {
		tree.Insert(v)
	}
	got := tree.Inorder()
	var asc []int
	for i := 1; i <= 100; i++ {
		asc = append(asc, i)
	}
	want := intsToSortedString(asc)
	if got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}

func intsToSortedString(arr []int) string {
	parts := make([]string, len(arr))
	for i, v := range arr {
		parts[i] = strconv.Itoa(v)
	}
	return strings.Join(parts, " ")
}
