package leftist

import (
	"testing"
)

// TestNewNode tests the creation of a new node with proper initialization
func TestNewNode(t *testing.T) {
	node := NewNode(42, 42)
	if node.Key != 42 {
		t.Errorf("Expected key 42, got %v", node.Key)
	}
	if node.Value != 42 {
		t.Errorf("Expected value 42, got %v", node.Value)
	}
	if node.Left != nil {
		t.Errorf("Expected Left to be nil, got %v", node.Left)
	}
	if node.Right != nil {
		t.Errorf("Expected Right to be nil, got %v", node.Right)
	}
	if node.Rank != 0 {
		t.Errorf("Expected Rank to be 0, got %d", node.Rank)
	}
}

// TestNewHeap tests the creation of a new heap with proper initialization
func TestNewHeap(t *testing.T) {
	less := func(a, b int) bool { return a < b }
	heap := NewHeap(less)
	if heap.less == nil {
		t.Error("Expected less function to be set")
	}
	if heap.root != nil {
		t.Errorf("Expected root to be nil, got %v", heap.root)
	}
}

// TestHeapInsert tests inserting elements into the heap
func TestHeapInsert(t *testing.T) {
	less := func(a, b int) bool { return a < b }
	heap := NewHeap(less)

	heap.Insert(5, 5)
	if heap.root == nil {
		t.Error("Expected root to be set after insert")
	}
	if heap.root.Key != 5 {
		t.Errorf("Expected root key to be 5, got %v", heap.root.Key)
	}

	heap.Insert(3, 3)
	heap.Insert(7, 7)

	if heap.root.Key != 3 {
		t.Errorf("Expected root key to be 3, got %v", heap.root.Key)
	}
}

// TestHeapMerge tests merging heaps and edge cases
func TestHeapMerge(t *testing.T) {
	less := func(a, b int) bool { return a < b }
	heap := NewHeap(less)

	result := heap.Merge(nil, nil)
	if result != nil {
		t.Error("Expected merge of nil heaps to return nil")
	}

	node1 := NewNode(5, 5)
	result = heap.Merge(nil, node1)
	if result != node1 {
		t.Error("Expected merge of nil with node to return node")
	}

	result = heap.Merge(node1, nil)
	if result != node1 {
		t.Error("Expected merge of node with nil to return node")
	}

	node2 := NewNode(3, 3)
	result = heap.Merge(node1, node2)

	if result.Key != 3 {
		t.Errorf("Expected merged root to be 3, got %v", result.Key)
	}

	if result.Rank != 0 {
		t.Errorf("Expected rank to be 0, got %d", result.Rank)
	}
}

// TestHeapMergeMaintainsLeftistProperty verifies the leftist property is maintained
func TestHeapMergeMaintainsLeftistProperty(t *testing.T) {
	less := func(a, b int) bool { return a < b }
	heap := NewHeap(less)

	heap.Insert(10, 10)
	heap.Insert(5, 5)
	heap.Insert(15, 15)
	heap.Insert(3, 3)
	heap.Insert(7, 7)

	if heap.root.Left != nil && heap.root.Right != nil {
		if heap.root.Left.Rank < heap.root.Right.Rank {
			t.Error("Leftist property violated: left rank < right rank")
		}
	}
}

// TestHeapInsertionOrder tests insertion order independence
func TestHeapInsertionOrder(t *testing.T) {
	less := func(a, b int) bool { return a < b }
	heap := NewHeap(less)

	heap.Insert(9, 9)
	heap.Insert(8, 8)
	heap.Insert(7, 7)
	heap.Insert(6, 6)
	heap.Insert(5, 5)

	if heap.root.Key != 5 {
		t.Errorf("Expected root to be 5, got %v", heap.root.Key)
	}
}

// TestHeapWithStringKeys tests the heap with string keys
func TestHeapWithStringKeys(t *testing.T) {
	less := func(a, b string) bool { return a < b }
	heap := NewHeap(less)

	heap.Insert("zebra", "zebra")
	heap.Insert("apple", "apple")
	heap.Insert("banana", "banana")

	if heap.root.Key != "apple" {
		t.Errorf("Expected root to be 'apple', got %v", heap.root.Key)
	}
}

// TestHeapWithCustomStruct tests the heap with custom struct types
func TestHeapWithCustomStruct(t *testing.T) {
	type Person struct {
		Name string
		Age  int
	}

	less := func(a, b Person) bool { return a.Age < b.Age }
	heap := NewHeap(less)

	heap.Insert(Person{"Alice", 30}, Person{"Alice", 30})
	heap.Insert(Person{"Bob", 25}, Person{"Bob", 25})
	heap.Insert(Person{"Charlie", 35}, Person{"Charlie", 35})

	if heap.root.Key.Age != 25 {
		t.Errorf("Expected root age to be 25, got %d", heap.root.Key.Age)
	}
	if heap.root.Key.Name != "Bob" {
		t.Errorf("Expected root name to be 'Bob', got %s", heap.root.Key.Name)
	}
}

// TestHeapRankCalculation tests rank calculation correctness
func TestHeapRankCalculation(t *testing.T) {
	less := func(a, b int) bool { return a < b }
	heap := NewHeap(less)

	heap.Insert(10, 10)
	heap.Insert(5, 5)
	heap.Insert(15, 15)

	if heap.root.Rank != 1 {
		t.Errorf("Expected root rank to be 1, got %d", heap.root.Rank)
	}

	if heap.root.Left != nil && heap.root.Left.Rank != 0 {
		t.Errorf("Expected left child rank to be 0, got %d", heap.root.Left.Rank)
	}

	if heap.root.Right != nil && heap.root.Right.Rank != 0 {
		t.Errorf("Expected right child rank to be 0, got %d", heap.root.Right.Rank)
	}
}

// TestHeapMergeComplex tests complex merging scenarios
func TestHeapMergeComplex(t *testing.T) {
	less := func(a, b int) bool { return a < b }
	heap := NewHeap(less)

	heap1 := NewHeap(less)
	heap1.Insert(5, 5)
	heap1.Insert(10, 10)

	heap2 := NewHeap(less)
	heap2.Insert(3, 3)
	heap2.Insert(7, 7)

	merged := heap.Merge(heap1.root, heap2.root)

	if merged.Key != 3 {
		t.Errorf("Expected merged root to be 3, got %v", merged.Key)
	}

	if merged.Left == nil || merged.Right == nil {
		t.Error("Expected merged heap to have both children")
	}
}

// BenchmarkHeapInsert1k benchmarks insertion of 1000 elements
func BenchmarkHeapInsert1k(b *testing.B) {
	benchmarkHeapInsert(b, 1000)
}

// BenchmarkHeapInsert5k benchmarks insertion of 5000 elements
func BenchmarkHeapInsert5k(b *testing.B) {
	benchmarkHeapInsert(b, 5000)
}

// BenchmarkHeapInsert10k benchmarks insertion of 10000 elements
func BenchmarkHeapInsert10k(b *testing.B) {
	benchmarkHeapInsert(b, 10000)
}

// BenchmarkHeapInsert50k benchmarks insertion of 50000 elements
func BenchmarkHeapInsert50k(b *testing.B) {
	benchmarkHeapInsert(b, 50000)
}

// BenchmarkHeapInsert100k benchmarks insertion of 100000 elements
func BenchmarkHeapInsert100k(b *testing.B) {
	benchmarkHeapInsert(b, 100000)
}

// BenchmarkHeapInsert500k benchmarks insertion of 500000 elements
func BenchmarkHeapInsert500k(b *testing.B) {
	benchmarkHeapInsert(b, 500000)
}

// BenchmarkHeapInsert1M benchmarks insertion of 1000000 elements
func BenchmarkHeapInsert1M(b *testing.B) {
	benchmarkHeapInsert(b, 1000000)
}

// BenchmarkHeapInsert5M benchmarks insertion of 5000000 elements
func BenchmarkHeapInsert5M(b *testing.B) {
	benchmarkHeapInsert(b, 5000000)
}

// benchmarkHeapInsert is a helper function for benchmarking heap insertions
func benchmarkHeapInsert(b *testing.B, size int) {
	less := func(a, b int) bool { return a < b }

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		heap := NewHeap(less)
		for j := 0; j < size; j++ {
			heap.Insert(j, j)
		}
	}
}
