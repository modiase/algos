package segmenttree

// Combine is an associative binary operation used to aggregate two values.
type Combine[T any] func(a, b T) T

type SegmentTree[T any] struct {
	n        int
	arr      []T
	tree     []T
	combine  Combine[T]
	identity T
	empty    T
}

// NewSegmentTree builds a segment tree over arr using combine and identity.
// identity must satisfy: combine(x, identity) == x and combine(identity, x) == x.
// empty is returned by Query for empty ranges or an empty tree.
func NewSegmentTree[T any](arr []T, combine Combine[T], identity T, empty T) *SegmentTree[T] {
	n := len(arr)
	st := &SegmentTree[T]{
		n:        n,
		arr:      append([]T(nil), arr...),
		tree:     make([]T, 2*(n+1)),
		combine:  combine,
		identity: identity,
		empty:    empty,
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
	st.tree[node] = st.combine(st.tree[node*2], st.tree[node*2+1])
}

// Query returns the aggregated value over the inclusive range [ql, qr].
// Out-of-range bounds are clamped; empty ranges and empty trees return empty.
func (st *SegmentTree[T]) Query(ql, qr int) T {
	if st.n == 0 {
		return st.empty
	}
	if ql < 0 {
		ql = 0
	}
	if qr >= st.n {
		qr = st.n - 1
	}
	if ql > qr {
		return st.empty
	}
	return st.query(1, 0, st.n-1, ql, qr)
}

// query is the recursive implementation of Query over [ql, qr] within node interval [l, r].
// It uses identity for non-overlap so partial results combine correctly.
func (st *SegmentTree[T]) query(node, l, r, ql, qr int) T {
	if ql <= l && r <= qr {
		return st.tree[node]
	}
	if r < ql || qr < l {
		return st.identity
	}
	mid := (l + r) / 2
	left := st.query(node*2, l, mid, ql, qr)
	right := st.query(node*2+1, mid+1, r, ql, qr)
	return st.combine(left, right)
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
	st.tree[node] = st.combine(st.tree[node*2], st.tree[node*2+1])
}
