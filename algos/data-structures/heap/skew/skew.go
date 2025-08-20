package skew

type Node[T any] struct {
	Key   T
	Value T
	Left  *Node[T]
	Right *Node[T]
}

func NewNode[T any](key, value T) *Node[T] {
	return &Node[T]{
		Key:   key,
		Value: value,
		Left:  nil,
		Right: nil,
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

func (h *Heap[T]) ExtractMin() (T, T, bool) {
	if h.root == nil {
		var zeroKey, zeroValue T
		return zeroKey, zeroValue, false
	}

	key := h.root.Key
	value := h.root.Value

	h.root = h.Merge(h.root.Left, h.root.Right)

	return key, value, true
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
	heap1.Left, heap1.Right = heap1.Right, heap1.Left

	return heap1
}
