package bit

import "fmt"

// Group represents an algebraic structure with an associative binary operation and inverse.
type Group[T any] interface {
	Combine(a, b T) T
	Inverse(a T) T
	Identity() T
}

type BinaryIndexedTree[T any] struct {
	group Group[T]
	size  int
	tree  []T
}

func New[T any](size int, group Group[T]) *BinaryIndexedTree[T] {
	if size <= 0 {
		panic("BIT size must be positive")
	}
	return &BinaryIndexedTree[T]{
		group: group,
		size:  size,
		tree:  make([]T, size+1),
	}
}

func NewFromArray[T any](arr []T, group Group[T]) *BinaryIndexedTree[T] {
	bit := New(len(arr), group)
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
		b.tree[index] = b.group.Combine(b.tree[index], delta)
		index += lsb(index)
	}
}

func (b *BinaryIndexedTree[T]) PrefixSum(index int) T {
	if index <= 0 {
		return b.group.Identity()
	}
	if index > b.size {
		index = b.size
	}
	res := b.group.Identity()
	for index > 0 {
		res = b.group.Combine(res, b.tree[index])
		index -= lsb(index)
	}
	return res
}

func (b *BinaryIndexedTree[T]) RangeSum(l, r int) T {
	if l > r {
		return b.group.Identity()
	}
	return b.group.Combine(b.PrefixSum(r), b.group.Inverse(b.PrefixSum(l-1)))
}

func (b *BinaryIndexedTree[T]) Get(index int) T {
	return b.RangeSum(index, index)
}

func (b *BinaryIndexedTree[T]) Set(index int, value T) {
	cur := b.Get(index)
	delta := b.group.Combine(b.group.Inverse(cur), value)
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

// NewGroup creates a concrete group implementation for a given type.
func NewGroup[T any](combine func(a, b T) T, inverse func(a T) T, identity T) Group[T] {
	return &groupImpl[T]{combine: combine, inverse: inverse, identity: identity}
}

type groupImpl[T any] struct {
	combine  func(a, b T) T
	inverse  func(a T) T
	identity T
}

func (g *groupImpl[T]) Combine(a, b T) T {
	return g.combine(a, b)
}

func (g *groupImpl[T]) Inverse(a T) T {
	return g.inverse(a)
}

func (g *groupImpl[T]) Identity() T {
	return g.identity
}
