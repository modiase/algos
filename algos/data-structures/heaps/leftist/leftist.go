package leftist

type Node[T any] struct {
	Key   T
	Value T
	Left  *Node[T]
	Right *Node[T]
	Rank  int
}

func NewNode[T any](key, value T) *Node[T] {
	return &Node[T]{
		Key:   key,
		Value: value,
		Left:  nil,
		Right: nil,
		Rank:  0,
	}
}

type LessFunc[T any] func(a, b T) bool

type Heap[T any] struct {
	less LessFunc[T]
	root *Node[T]
}

func NewHeap[T any](less LessFunc[T]) *Heap[T] {
	return &Heap[T]{
		less: less,
		root: nil,
	}
}

func (h *Heap[T]) Insert(key, value T) {
	h.root = h.Merge(h.root, NewNode(key, value))
}

func (h *Heap[T]) Merge(heap1, heap2 *Node[T]) *Node[T] {
	if heap1 == nil {
		return heap2
	}
	if heap2 == nil {
		return heap1
	}

	if h.less(heap2.Key, heap1.Key) {
		heap1, heap2 = heap2, heap1
	}

	heap1.Right = h.Merge(heap1.Right, heap2)

	if heap1.Left == nil || (heap1.Right != nil && heap1.Left.Rank < heap1.Right.Rank) {
		heap1.Left, heap1.Right = heap1.Right, heap1.Left
	}

	if heap1.Right != nil {
		heap1.Rank = heap1.Right.Rank + 1
	} else {
		heap1.Rank = 0
	}

	return heap1
}

func (h *Heap[T]) ExtractMin() T {
	if h.root == nil {
		var zero T
		return zero
	}

	minValue := h.root.Value
	h.root = h.Merge(h.root.Left, h.root.Right)
	return minValue
}

type NodeDepth[T any] struct {
	Node  *Node[T]
	Depth int
}

func (h *Heap[T]) Traverse() []NodeDepth[T] {
	if h.root == nil {
		return []NodeDepth[T]{}
	}

	var result []NodeDepth[T]
	queue := []NodeDepth[T]{{Node: h.root, Depth: 0}}

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		result = append(result, current)

		if current.Node.Left != nil {
			queue = append(queue, NodeDepth[T]{Node: current.Node.Left, Depth: current.Depth + 1})
		}
		if current.Node.Right != nil {
			queue = append(queue, NodeDepth[T]{Node: current.Node.Right, Depth: current.Depth + 1})
		}
	}

	return result
}

func (h *Heap[T]) TraverseGeneric() []any {
	traversal := h.Traverse()
	result := make([]any, len(traversal))
	for i, nd := range traversal {
		result[i] = nd
	}
	return result
}
