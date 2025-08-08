package test

import (
	bidxtree "algos/data-structures/trees/binary-indexed-tree/pkg"
	"testing"
)

var intSumGroup = bidxtree.NewGroup(
	func(a, b int) int { return a + b },
	func(a int) int { return -a },
	0,
)

// TestNew tests the creation of a new Binary Indexed Tree.
func TestNew(t *testing.T) {
	tests := []struct {
		name string
		size int
		want int
	}{
		{"small BIT", 5, 5},
		{"medium BIT", 100, 100},
		{"large BIT", 1000, 1000},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bit := bidxtree.New(tt.size, intSumGroup)
			if bit.Size() != tt.want {
				t.Errorf("New(%d).Size() = %d, want %d", tt.size, bit.Size(), tt.want)
			}

			// Verify all elements are initially zero
			for i := 1; i <= tt.size; i++ {
				if got := bit.Get(i); got != 0 {
					t.Errorf("Initial value at index %d = %d, want 0", i, got)
				}
			}
		})
	}
}

// TestNewPanicsOnInvalidSize tests that New panics with invalid sizes.
func TestNewPanicsOnInvalidSize(t *testing.T) {
	tests := []int{0, -1, -100}

	for _, size := range tests {
		t.Run("invalid size", func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("New(%d) did not panic", size)
				}
			}()
			bidxtree.New(size, intSumGroup)
		})
	}
}

// TestNewFromArray tests the creation of a BIT from an existing array.
func TestNewFromArray(t *testing.T) {
	tests := []struct {
		name string
		arr  []int
	}{
		{"single element", []int{5}},
		{"multiple elements", []int{1, 3, 5, 7, 9}},
		{"with zeros", []int{0, 2, 0, 4, 0}},
		{"negative numbers", []int{-1, -3, 5, -7, 9}},
		{"all same", []int{2, 2, 2, 2}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bit := bidxtree.NewFromArray(tt.arr, intSumGroup)

			if bit.Size() != len(tt.arr) {
				t.Errorf("Size() = %d, want %d", bit.Size(), len(tt.arr))
			}

			// Verify all elements match the original array
			for i, expected := range tt.arr {
				if got := bit.Get(i + 1); got != expected { // Convert to 1-based indexing
					t.Errorf("Get(%d) = %d, want %d", i+1, got, expected)
				}
			}
		})
	}
}

// TestUpdate tests the update operation extensively.
func TestUpdate(t *testing.T) {
	t.Run("single update", func(t *testing.T) {
		bit := bidxtree.New(5, intSumGroup)
		bit.Update(3, 10)

		expected := []int{0, 0, 10, 0, 0} // index 3 should be 10
		for i := 1; i <= 5; i++ {
			if got := bit.Get(i); got != expected[i-1] {
				t.Errorf("After Update(3, 10), Get(%d) = %d, want %d", i, got, expected[i-1])
			}
		}
	})

	t.Run("multiple updates same index", func(t *testing.T) {
		bit := bidxtree.New(5, intSumGroup)
		bit.Update(2, 5)
		bit.Update(2, 3)
		bit.Update(2, -2)

		expected := 5 + 3 - 2 // Should be 6
		if got := bit.Get(2); got != expected {
			t.Errorf("After multiple updates to index 2, Get(2) = %d, want %d", got, expected)
		}
	})

	t.Run("multiple updates different indices", func(t *testing.T) {
		bit := bidxtree.New(6, intSumGroup)
		updates := []struct {
			index int
			delta int
		}{
			{1, 1}, {2, 3}, {3, 5}, {4, 7}, {5, 9}, {6, 11},
		}

		for _, update := range updates {
			bit.Update(update.index, update.delta)
		}

		expected := []int{1, 3, 5, 7, 9, 11}
		for i := 1; i <= 6; i++ {
			if got := bit.Get(i); got != expected[i-1] {
				t.Errorf("Get(%d) = %d, want %d", i, got, expected[i-1])
			}
		}
	})

	t.Run("negative updates", func(t *testing.T) {
		bit := bidxtree.NewFromArray([]int{10, 20, 30, 40, 50}, intSumGroup)
		bit.Update(3, -25) // 30 - 25 = 5

		if got := bit.Get(3); got != 5 {
			t.Errorf("After Update(3, -25), Get(3) = %d, want 5", got)
		}
	})
}

// TestUpdatePanicsOnInvalidIndex tests that Update panics with invalid indices.
func TestUpdatePanicsOnInvalidIndex(t *testing.T) {
	bit := bidxtree.New(5, intSumGroup)
	invalidIndices := []int{0, -1, 6, 100}

	for _, index := range invalidIndices {
		t.Run("invalid index", func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("Update(%d, 1) did not panic", index)
				}
			}()
			bit.Update(index, 1)
		})
	}
}

// TestPrefixSum tests prefix sum queries extensively.
func TestPrefixSum(t *testing.T) {
	// Create a BIT with known values: [1, 3, 5, 7, 9, 11]
	arr := []int{1, 3, 5, 7, 9, 11}
	bit := bidxtree.NewFromArray(arr, intSumGroup)

	tests := []struct {
		index    int
		expected int
	}{
		{0, 0},                        // Before array
		{1, 1},                        // First element
		{2, 1 + 3},                    // First two elements
		{3, 1 + 3 + 5},                // First three elements
		{4, 1 + 3 + 5 + 7},            // First four elements
		{5, 1 + 3 + 5 + 7 + 9},        // First five elements
		{6, 1 + 3 + 5 + 7 + 9 + 11},   // All elements
		{7, 1 + 3 + 5 + 7 + 9 + 11},   // Beyond array (should clamp)
		{100, 1 + 3 + 5 + 7 + 9 + 11}, // Way beyond array
	}

	for _, tt := range tests {
		t.Run("prefix sum", func(t *testing.T) {
			if got := bit.PrefixSum(tt.index); got != tt.expected {
				t.Errorf("PrefixSum(%d) = %d, want %d", tt.index, got, tt.expected)
			}
		})
	}
}

// TestPrefixSumAfterUpdates tests prefix sums after dynamic updates.
func TestPrefixSumAfterUpdates(t *testing.T) {
	bit := bidxtree.NewFromArray([]int{1, 2, 3, 4, 5}, intSumGroup)

	// Initial prefix sums: [1, 3, 6, 10, 15]
	initialPrefixSums := []int{1, 3, 6, 10, 15}
	for i, expected := range initialPrefixSums {
		if got := bit.PrefixSum(i + 1); got != expected {
			t.Errorf("Initial PrefixSum(%d) = %d, want %d", i+1, got, expected)
		}
	}

	// Update index 3 by adding 10 (3 becomes 13)
	bit.Update(3, 10)

	// New prefix sums: [1, 3, 16, 20, 25]
	updatedPrefixSums := []int{1, 3, 16, 20, 25}
	for i, expected := range updatedPrefixSums {
		if got := bit.PrefixSum(i + 1); got != expected {
			t.Errorf("After update, PrefixSum(%d) = %d, want %d", i+1, got, expected)
		}
	}
}

// TestRangeSum tests range sum queries extensively.
func TestRangeSum(t *testing.T) {
	// Create a BIT with known values: [2, 4, 6, 8, 10]
	arr := []int{2, 4, 6, 8, 10}
	bit := bidxtree.NewFromArray(arr, intSumGroup)

	tests := []struct {
		name  string
		start int
		end   int
		want  int
	}{
		{"single element", 1, 1, 2},
		{"single element middle", 3, 3, 6},
		{"two elements", 2, 3, 4 + 6},
		{"three elements", 2, 4, 4 + 6 + 8},
		{"all elements", 1, 5, 2 + 4 + 6 + 8 + 10},
		{"last two elements", 4, 5, 8 + 10},
		{"invalid range (start > end)", 4, 2, 0},
		{"range with same start and end", 2, 2, 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := bit.RangeSum(tt.start, tt.end); got != tt.want {
				t.Errorf("RangeSum(%d, %d) = %d, want %d", tt.start, tt.end, got, tt.want)
			}
		})
	}
}

// TestGet tests individual element retrieval.
func TestGet(t *testing.T) {
	arr := []int{10, 20, 30, 40, 50}
	bit := bidxtree.NewFromArray(arr, intSumGroup)

	for i, expected := range arr {
		if got := bit.Get(i + 1); got != expected {
			t.Errorf("Get(%d) = %d, want %d", i+1, got, expected)
		}
	}
}

// TestSet tests setting individual elements.
func TestSet(t *testing.T) {
	bit := bidxtree.NewFromArray([]int{1, 2, 3, 4, 5}, intSumGroup)

	// Set index 3 to 100
	bit.Set(3, 100)
	if got := bit.Get(3); got != 100 {
		t.Errorf("After Set(3, 100), Get(3) = %d, want 100", got)
	}

	// Verify other elements are unchanged
	expected := []int{1, 2, 100, 4, 5}
	for i, want := range expected {
		if got := bit.Get(i + 1); got != want {
			t.Errorf("Get(%d) = %d, want %d", i+1, got, want)
		}
	}

	// Verify prefix sums are updated correctly
	if got := bit.PrefixSum(5); got != 1+2+100+4+5 {
		t.Errorf("PrefixSum(5) after Set = %d, want %d", got, 1+2+100+4+5)
	}
}

// TestBITOperationsIntegration tests multiple operations working together.
func TestBITOperationsIntegration(t *testing.T) {
	bit := bidxtree.New(10, intSumGroup)

	// Perform a series of operations
	operations := []struct {
		op    string
		args  []int
		check func() bool
	}{
		{"update", []int{1, 5}, func() bool { return bit.Get(1) == 5 }},
		{"update", []int{5, 10}, func() bool { return bit.Get(5) == 10 }},
		{"update", []int{10, 15}, func() bool { return bit.Get(10) == 15 }},
		{"set", []int{3, 7}, func() bool { return bit.Get(3) == 7 }},
		{"range_sum", []int{1, 5}, func() bool { return bit.RangeSum(1, 5) == 5+0+7+0+10 }},
		{"prefix_sum", []int{10}, func() bool { return bit.PrefixSum(10) == 5+0+7+0+10+0+0+0+0+15 }},
	}

	for i, op := range operations {
		switch op.op {
		case "update":
			bit.Update(op.args[0], op.args[1])
		case "set":
			bit.Set(op.args[0], op.args[1])
		case "range_sum":
			// This is a check operation
		case "prefix_sum":
			// This is a check operation
		}

		if !op.check() {
			t.Errorf("Operation %d (%s) failed verification", i, op.op)
		}
	}
}
