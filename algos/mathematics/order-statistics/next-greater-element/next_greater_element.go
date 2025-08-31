package next_greater_element

// Optimised helpfer function. It is unsafe because it assumes that the stack is
// not empty without checking.
func unsafeStackPop(stack *[]int) int {
	stack_pointer := len(*stack) - 1
	topIndex := (*stack)[stack_pointer]
	*stack = (*stack)[:stack_pointer]
	return topIndex
}

// NextGreaterElements finds the index of the next greater element to the right
// for each element in the input slice. Returns -1 if no greater element exists.
// Time complexity: O(n), Space complexity: O(n)
func NextGreaterElements(arr []int) []int {
	n := len(arr)
	if n == 0 {
		return []int{}
	}

	result := make([]int, n)
	stack := make([]int, 0) // Stack to store indices

	// Initialize all positions to -1 (no greater element found)
	for i := range result {
		result[i] = -1
	}

	// Process each element
	for i := range result {
		// While stack is not empty and current element is greater than
		// the element at the index stored at top of stack
		for len(stack) > 0 && arr[i] > arr[stack[len(stack)-1]] {
			// Pop from stack and set result for that index
			result[unsafeStackPop(&stack)] = i
		}

		// Push current index to stack
		stack = append(stack, i)
	}

	return result
}

// NextGreaterElementsCircular finds the next greater element in a circular array
// Time complexity: O(n), Space complexity: O(n)
func NextGreaterElementsCircular(arr []int) []int {
	n := len(arr)
	if n == 0 {
		return []int{}
	}

	result := make([]int, n)
	stack := make([]int, 0)

	// Initialize all positions to -1
	for i := range result {
		result[i] = -1
	}

	// Process array twice to handle circular nature
	for i := 0; i < 2*n; i++ {
		currentIndex := i % n

		for len(stack) > 0 && arr[currentIndex] > arr[stack[len(stack)-1]] {
			result[unsafeStackPop(&stack)] = currentIndex
		}

		// Only push during first iteration to avoid duplicates
		if i < n {
			stack = append(stack, currentIndex)
		}
	}

	return result
}
