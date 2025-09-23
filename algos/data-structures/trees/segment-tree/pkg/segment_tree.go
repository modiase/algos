package segmenttree

// Monoid represents an algebraic structure with an associative binary operation and identity element.
type Monoid[T any] interface {
	Combine(a, b T) T
	Empty() T
}

type SegmentTree[T any] struct {
	n      int
	arr    []T
	tree   []T
	monoid Monoid[T]
}

// NewSegmentTree builds a segment tree over arr using the provided monoid.
func NewSegmentTree[T any](arr []T, monoid Monoid[T]) *SegmentTree[T] {
	n := len(arr)
	st := &SegmentTree[T]{
		n:      n,
		arr:    append([]T(nil), arr...),
		tree:   make([]T, 2*(n+1)),
		monoid: monoid,
	}
	if n > 0 {
		st.build(1, 0, n-1)
	}
	return st
}

// build constructs the segment tree for the interval [l, r] at the given node.
func (st *SegmentTree[T]) build(node, l, r int) {
	if l == r {
		st.tree[node] = st.arr[l]
		return
	}
	mid := (l + r) / 2
	st.build(node*2, l, mid)
	st.build(node*2+1, mid+1, r)
	st.tree[node] = st.monoid.Combine(st.tree[node*2], st.tree[node*2+1])
}

// Query returns the aggregated value over the inclusive range [ql, qr].
// Out-of-range bounds are clamped; empty ranges and empty trees return empty.
func (st *SegmentTree[T]) Query(ql, qr int) T {
	if st.n == 0 || ql > qr {
		return st.monoid.Empty()
	}
	if ql < 0 {
		ql = 0
	}
	if qr >= st.n {
		qr = st.n - 1
	}
	return st.query(1, 0, st.n-1, ql, qr)
}

// query is the recursive implementation of Query over [ql, qr] within node interval [l, r].
func (st *SegmentTree[T]) query(node, l, r, ql, qr int) T {
	if ql <= l && r <= qr {
		return st.tree[node]
	}
	if r < ql || qr < l {
		return st.monoid.Empty()
	}
	mid := (l + r) / 2
	return st.monoid.Combine(
		st.query(node*2, l, mid, ql, qr),
		st.query(node*2+1, mid+1, r, ql, qr),
	)
}

// Update assigns arr[idx] = val and updates the segment tree.
// Indices outside [0, n) are ignored.
func (st *SegmentTree[T]) Update(idx int, val T) {
	if idx < 0 || idx >= st.n {
		return
	}
	st.arr[idx] = val
	st.update(1, 0, st.n-1, idx, val)
}

// update is the recursive implementation of Update for position idx within [l, r].
func (st *SegmentTree[T]) update(node, l, r, idx int, val T) {
	if l == r {
		st.tree[node] = val
		return
	}
	mid := (l + r) / 2
	if idx <= mid {
		st.update(node*2, l, mid, idx, val)
	} else {
		st.update(node*2+1, mid+1, r, idx, val)
	}
	st.tree[node] = st.monoid.Combine(st.tree[node*2], st.tree[node*2+1])
}

// NewMonoid creates a concrete monoid implementation for a given type.
func NewMonoid[T any](combine func(a, b T) T, empty T) Monoid[T] {
	return &monoidImpl[T]{combine: combine, empty: empty}
}

type monoidImpl[T any] struct {
	combine func(a, b T) T
	empty   T
}

func (m *monoidImpl[T]) Combine(a, b T) T {
	return m.combine(a, b)
}

func (m *monoidImpl[T]) Empty() T {
	return m.empty
}
