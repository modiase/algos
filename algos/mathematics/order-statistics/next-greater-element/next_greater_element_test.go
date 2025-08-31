package next_greater_element

import (
	"fmt"
	"reflect"
	"testing"
)

func TestNextGreaterElements(t *testing.T) {
	// Test case 1: Basic case
	arr1 := []int{4, 5, 2, 25}
	expected1 := []int{1, 3, 3, -1}
	result1 := NextGreaterElements(arr1)
	assertEqual(result1, expected1, "Test 1 failed")

	// Test case 2: Another basic case
	arr2 := []int{1, 3, 2, 4}
	expected2 := []int{1, 3, 3, -1}
	result2 := NextGreaterElements(arr2)
	assertEqual(result2, expected2, "Test 2 failed")

	// Test case 3: Decreasing sequence
	arr3 := []int{5, 4, 3, 2, 1}
	expected3 := []int{-1, -1, -1, -1, -1}
	result3 := NextGreaterElements(arr3)
	assertEqual(result3, expected3, "Test 3 failed")

	// Test case 4: Increasing sequence
	arr4 := []int{1, 2, 3, 4, 5}
	expected4 := []int{1, 2, 3, 4, -1}
	result4 := NextGreaterElements(arr4)
	assertEqual(result4, expected4, "Test 4 failed")
}

func TestNextGreaterElementsCircular(t *testing.T) {
	// Test case 1: Circular array
	arr1 := []int{1, 2, 1}
	expected1 := []int{1, -1, 1}
	result1 := NextGreaterElementsCircular(arr1)
	assertEqual(result1, expected1, "Circular test 1 failed")

	// Test case 2: All elements can find greater in circular
	arr2 := []int{5, 4, 3, 2, 1}
	expected2 := []int{-1, 0, 0, 0, 0}
	result2 := NextGreaterElementsCircular(arr2)
	assertEqual(result2, expected2, "Circular test 2 failed")
}

func TestEdgeCases(t *testing.T) {
	// Empty array
	emptyResult := NextGreaterElements([]int{})
	assertEqual(emptyResult, []int{}, "Empty array test failed")

	// Single element
	singleResult := NextGreaterElements([]int{5})
	assertEqual(singleResult, []int{-1}, "Single element test failed")

	// Two elements - ascending
	twoAscResult := NextGreaterElements([]int{1, 2})
	assertEqual(twoAscResult, []int{1, -1}, "Two ascending elements test failed")

	// Two elements - descending
	twoDescResult := NextGreaterElements([]int{2, 1})
	assertEqual(twoDescResult, []int{-1, -1}, "Two descending elements test failed")
}

func assertEqual(actual, expected []int, message string) {
	if !reflect.DeepEqual(actual, expected) {
		panic(fmt.Sprintf("%s: expected %v, got %v", message, expected, actual))
	}
}
