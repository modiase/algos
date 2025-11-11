package main

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"sync"

	"gopkg.in/alecthomas/kingpin.v2"
)

type Matrix [][]float64

type BlockTask struct {
	rowStart, rowEnd int
	colStart, colEnd int
}

type BlockResult struct {
	rowStart, rowEnd int
	colStart, colEnd int
	data             Matrix
}

func multiplyBlock(a, b Matrix, rowStart, rowEnd, colStart, colEnd int) Matrix {
	rows := rowEnd - rowStart
	cols := colEnd - colStart
	result := make(Matrix, rows)
	for i := range result {
		result[i] = make([]float64, cols)
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			sum := 0.0
			for k := 0; k < len(a[0]); k++ {
				sum += a[rowStart+i][k] * b[k][colStart+j]
			}
			result[i][j] = sum
		}
	}
	return result
}

func matrixMultiplyParallel(a, b Matrix, blockSize, numWorkers int) Matrix {
	if len(a[0]) != len(b) {
		panic("Matrix dimensions incompatible for multiplication")
	}

	fmt.Printf("Multiplying (%d×%d) × (%d×%d) with %d workers\n",
		len(a), len(a[0]), len(b), len(b[0]), numWorkers)
	fmt.Printf("Block size: %d×%d\n", blockSize, blockSize)

	result := make(Matrix, len(a))
	for i := range result {
		result[i] = make([]float64, len(b[0]))
	}

	rowBlocks := []int{}
	for i := 0; i < len(a); i += blockSize {
		rowBlocks = append(rowBlocks, i)
	}
	if rowBlocks[len(rowBlocks)-1] < len(a) {
		rowBlocks = append(rowBlocks, len(a))
	}

	colBlocks := []int{}
	for j := 0; j < len(b[0]); j += blockSize {
		colBlocks = append(colBlocks, j)
	}
	if colBlocks[len(colBlocks)-1] < len(b[0]) {
		colBlocks = append(colBlocks, len(b[0]))
	}

	tasks := make(chan BlockTask, (len(rowBlocks)-1)*(len(colBlocks)-1))
	results := make(chan BlockResult, (len(rowBlocks)-1)*(len(colBlocks)-1))

	for i := 0; i < len(rowBlocks)-1; i++ {
		for j := 0; j < len(colBlocks)-1; j++ {
			tasks <- BlockTask{
				rowStart: rowBlocks[i],
				rowEnd:   rowBlocks[i+1],
				colStart: colBlocks[j],
				colEnd:   colBlocks[j+1],
			}
		}
	}
	close(tasks)

	totalTasks := (len(rowBlocks) - 1) * (len(colBlocks) - 1)
	fmt.Printf("Created %d×%d grid = %d tasks\n",
		len(rowBlocks)-1, len(colBlocks)-1, totalTasks)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for task := range tasks {
				blockResult := multiplyBlock(a, b, task.rowStart, task.rowEnd, task.colStart, task.colEnd)
				results <- BlockResult{
					rowStart: task.rowStart,
					rowEnd:   task.rowEnd,
					colStart: task.colStart,
					colEnd:   task.colEnd,
					data:     blockResult,
				}
			}
		}()
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	taskNum := 0
	for res := range results {
		taskNum++
		for i := 0; i < res.rowEnd-res.rowStart; i++ {
			for j := 0; j < res.colEnd-res.colStart; j++ {
				result[res.rowStart+i][res.colStart+j] = res.data[i][j]
			}
		}
		fmt.Printf("Completed block [%d:%d, %d:%d] (%d/%d)\n",
			res.rowStart, res.rowEnd, res.colStart, res.colEnd, taskNum, totalTasks)
	}

	return result
}

func matrixMultiplySequential(a, b Matrix) Matrix {
	if len(a[0]) != len(b) {
		panic("Matrix dimensions incompatible for multiplication")
	}

	result := make(Matrix, len(a))
	for i := range result {
		result[i] = make([]float64, len(b[0]))
	}

	for i := 0; i < len(a); i++ {
		for j := 0; j < len(b[0]); j++ {
			sum := 0.0
			for k := 0; k < len(a[0]); k++ {
				sum += a[i][k] * b[k][j]
			}
			result[i][j] = sum
		}
	}
	return result
}

func main() {
	app := kingpin.New("parallel-matrix-multiplication", "Block-based parallel matrix multiplication")

	demoCmd := app.Command("demo", "Run demo with random matrices")
	demoSize := demoCmd.Flag("size", "Size of square matrices").Short('n').Default("128").Int()
	demoBlockSize := demoCmd.Flag("block-size", "Block size for partitioning").Short('b').Default("32").Int()
	demoWorkers := demoCmd.Flag("workers", "Number of workers (default: CPU count)").Short('w').Int()

	testCmd := app.Command("test", "Run test suite")

	switch kingpin.MustParse(app.Parse(os.Args[1:])) {
	case demoCmd.FullCommand():
		workers := *demoWorkers
		if workers == 0 {
			workers = runtime.NumCPU()
		}
		runDemo(*demoSize, *demoBlockSize, workers)
	case testCmd.FullCommand():
		runTests()
	}
}

func runDemo(size, blockSize, workers int) {
	fmt.Println("======================================================================")
	fmt.Println("MULTITHREADED MATRIX MULTIPLICATION")
	fmt.Println("======================================================================")

	rng := rand.New(rand.NewSource(42))
	a := make(Matrix, size)
	b := make(Matrix, size)
	for i := 0; i < size; i++ {
		a[i] = make([]float64, size)
		b[i] = make([]float64, size)
		for j := 0; j < size; j++ {
			a[i][j] = float64(rng.Intn(10))
			b[i][j] = float64(rng.Intn(10))
		}
	}

	fmt.Printf("\nMatrix dimensions: %d×%d\n", size, size)
	fmt.Printf("Workers: %d\n", workers)
	fmt.Printf("Block size: %d×%d\n\n", blockSize, blockSize)

	fmt.Println("Starting parallel multiplication...")
	result := matrixMultiplyParallel(a, b, blockSize, workers)

	fmt.Printf("\nCompleted! Result shape: (%d×%d)\n", len(result), len(result[0]))
}

func runTests() {
	fmt.Println("Running tests...")

	tests := []struct {
		name     string
		a        Matrix
		b        Matrix
		expected Matrix
	}{
		{
			name:     "2x2 basic",
			a:        Matrix{{1, 2}, {3, 4}},
			b:        Matrix{{5, 6}, {7, 8}},
			expected: Matrix{{19, 22}, {43, 50}},
		},
		{
			name:     "2x2 identity",
			a:        Matrix{{1, 0}, {0, 1}},
			b:        Matrix{{5, 6}, {7, 8}},
			expected: Matrix{{5, 6}, {7, 8}},
		},
		{
			name:     "2x3 × 3x2",
			a:        Matrix{{1, 2, 3}, {4, 5, 6}},
			b:        Matrix{{7, 8}, {9, 10}, {11, 12}},
			expected: Matrix{{58, 64}, {139, 154}},
		},
		{
			name:     "3x3 identity",
			a:        Matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			b:        Matrix{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			expected: Matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
		},
	}

	passed := 0
	for _, tt := range tests {
		result := matrixMultiplyParallel(tt.a, tt.b, 1, 2)
		if matrixEqual(result, tt.expected) {
			fmt.Printf("✓ %s\n", tt.name)
			passed++
		} else {
			fmt.Printf("✗ %s\n", tt.name)
			fmt.Printf("  Expected: %v\n", tt.expected)
			fmt.Printf("  Got:      %v\n", result)
		}
	}

	fmt.Printf("\nParallel vs Sequential tests...\n")
	seqTests := []struct {
		name string
		a    Matrix
		b    Matrix
	}{
		{
			name: "2x2",
			a:    Matrix{{1, 2}, {3, 4}},
			b:    Matrix{{5, 6}, {7, 8}},
		},
		{
			name: "2x3 × 3x2",
			a:    Matrix{{1, 2, 3}, {4, 5, 6}},
			b:    Matrix{{7, 8}, {9, 10}, {11, 12}},
		},
	}

	for _, tt := range seqTests {
		resultParallel := matrixMultiplyParallel(tt.a, tt.b, 1, 2)
		resultSeq := matrixMultiplySequential(tt.a, tt.b)
		if matrixEqual(resultParallel, resultSeq) {
			fmt.Printf("✓ %s\n", tt.name)
			passed++
		} else {
			fmt.Printf("✗ %s\n", tt.name)
		}
	}

	fmt.Printf("\nBlock size tests...\n")
	a := Matrix{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}
	b := Matrix{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}
	resultSeq := matrixMultiplySequential(a, b)
	resultBlock1 := matrixMultiplyParallel(a, b, 1, 2)
	resultBlock2 := matrixMultiplyParallel(a, b, 2, 2)

	if matrixEqual(resultBlock1, resultSeq) {
		fmt.Println("✓ block_size=1")
		passed++
	} else {
		fmt.Println("✗ block_size=1")
	}

	if matrixEqual(resultBlock2, resultSeq) {
		fmt.Println("✓ block_size=2")
		passed++
	} else {
		fmt.Println("✗ block_size=2")
	}

	fmt.Printf("\n%d tests passed\n", passed)
}

func matrixEqual(a, b Matrix) bool {
	if len(a) != len(b) || len(a[0]) != len(b[0]) {
		return false
	}
	for i := range a {
		for j := range a[i] {
			if abs(a[i][j]-b[i][j]) > 1e-9 {
				return false
			}
		}
	}
	return true
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
