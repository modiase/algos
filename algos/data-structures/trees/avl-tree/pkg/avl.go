package avl

import (
	"fmt"
	"strings"
)

type node[T any] struct {
	value  T
	height int
	left   *node[T]
	right  *node[T]
}

/* Compare returns a negative value if a < b, a positive value if a > b, and 0 iff a == b. */
type Compare[T any] func(a, b T) int

type Tree[T any] struct {
	root    *node[T]
	compare Compare[T]
}

func height[T any](n *node[T]) int {
	if n == nil {
		return -1
	}
	return n.height
}

/* Returns the balance factor of the given node. */
func balance[T any](n *node[T]) int {
	return height(n.right) - height(n.left)
}

func updateHeight[T any](n *node[T]) {
	n.height = 1 + max(height(n.left), height(n.right))
}

/* Rotates the tree rooted at n to the left. Returns the new root of the subtree. */
func (n *node[T]) leftRotate() *node[T] {
	r := n.right
	n.right = r.left
	r.left = n
	updateHeight(n)
	updateHeight(r)
	return r
}

/* Rotates the tree rooted at n to the right. Returns the new root of the subtree. */
func (n *node[T]) rightRotate() *node[T] {
	l := n.left
	n.left = l.right
	l.right = n
	updateHeight(n)
	updateHeight(l)
	return l
}

/* Returns the new root of the subtree after insertion. */
func (n *node[T]) nodeInsert(value T, compare Compare[T]) *node[T] {
	if compare(value, n.value) < 0 {
		if n.left == nil {
			n.left = &node[T]{value: value, height: 0, left: nil, right: nil}
		} else {
			n.left = n.left.nodeInsert(value, compare)
		}
	} else if compare(value, n.value) > 0 {
		if n.right == nil {
			n.right = &node[T]{value: value, height: 0, left: nil, right: nil}
		} else {
			n.right = n.right.nodeInsert(value, compare)
		}
	}
	updateHeight(n)
	balance := balance(n)
	if balance > 1 {
		if compare(value, n.right.value) < 0 {
			n.right = n.right.rightRotate()
		}
		return n.leftRotate()
	} else if balance < -1 {
		if compare(value, n.left.value) > 0 {
			n.left = n.left.leftRotate()
		}
		return n.rightRotate()
	}
	return n
}

func (t *Tree[T]) Insert(value T) {
	if t.root == nil {
		t.root = &node[T]{value: value, height: 0, left: nil, right: nil}
		return
	} else {
		t.root = t.root.nodeInsert(value, t.compare)
	}
}

func (t *Tree[T]) Inorder() string {
	if t.root == nil {
		return ""
	}

	var result []string
	stack := make([]*node[T], 0)
	current := t.root

	for current != nil || len(stack) > 0 {
		for current != nil {
			stack = append(stack, current)
			current = current.left
		}

		current = stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		result = append(result, fmt.Sprint(current.value))

		current = current.right
	}

	return strings.Join(result, " ")
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func NewTree[T any](compare Compare[T]) *Tree[T] {
	return &Tree[T]{
		root:    nil,
		compare: compare,
	}
}
