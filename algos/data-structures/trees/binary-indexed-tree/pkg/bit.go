package bit

import "fmt"

type Combine[T any] func(a, b T) T
type Inverse[T any] func(a T) T

type Group[T any] struct {
	combine  Combine[T]
	inverse  Inverse[T]
	identity T
}

// NewGroup constructs a Group with the provided combine, inverse, and identity functions/values.
// The returned Group can be used from external packages without accessing unexported fields.
func NewGroup[T any](combine Combine[T], inverse Inverse[T], identity T) Group[T] {
	return Group[T]{
		combine:  combine,
		inverse:  inverse,
		identity: identity,
	}
}

type BinaryIndexedTree[T any] struct {
	tree     []T
	size     int
	combine  Combine[T]
	inverse  Inverse[T]
	identity T
}

func NewGeneric[T any](size int, g Group[T]) *BinaryIndexedTree[T] {
	if size <= 0 {
		panic("BIT size must be positive")
	}
	return &BinaryIndexedTree[T]{
		tree:     make([]T, size+1),
		size:     size,
		combine:  g.combine,
		inverse:  g.inverse,
		identity: g.identity,
	}
}

func NewFromArrayGeneric[T any](arr []T, g Group[T]) *BinaryIndexedTree[T] {
	bit := NewGeneric[T](len(arr), g)
	for i, v := range arr {
		bit.Update(i+1, v)
	}
	return bit
}

func lsb(x int) int { return x & -x }

func (b *BinaryIndexedTree[T]) Update(index int, delta T) {
	if index <= 0 || index > b.size {
		panic(fmt.Sprintf("index %d out of bounds [1, %d]", index, b.size))
	}
	for index <= b.size {
		b.tree[index] = b.combine(b.tree[index], delta)
		index += lsb(index)
	}
}

func (b *BinaryIndexedTree[T]) PrefixSum(index int) T {
	if index <= 0 {
		return b.identity
	}
	if index > b.size {
		index = b.size
	}
	res := b.identity
	for index > 0 {
		res = b.combine(res, b.tree[index])
		index -= lsb(index)
	}
	return res
}

func (b *BinaryIndexedTree[T]) RangeSum(l, r int) T {
	if l > r {
		return b.identity
	}
	return b.combine(b.PrefixSum(r), b.inverse(b.PrefixSum(l-1)))
}

func (b *BinaryIndexedTree[T]) Get(index int) T {
	return b.RangeSum(index, index)
}

func (b *BinaryIndexedTree[T]) Set(index int, value T) {
	cur := b.Get(index)
	delta := b.combine(b.inverse(cur), value)
	b.Update(index, delta)
}

func (b *BinaryIndexedTree[T]) Size() int { return b.size }

func (b *BinaryIndexedTree[T]) String() string {
	res := "BIT Internal Tree: ["
	for i := 1; i <= b.size; i++ {
		if i > 1 {
			res += ", "
		}
		res += fmt.Sprintf("%v", b.tree[i])
	}
	res += "]\nLogical Array: ["
	for i := 1; i <= b.size; i++ {
		if i > 1 {
			res += ", "
		}
		res += fmt.Sprintf("%v", b.Get(i))
	}
	res += "]"
	return res
}
