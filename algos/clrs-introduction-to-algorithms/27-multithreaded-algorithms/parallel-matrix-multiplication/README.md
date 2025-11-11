# Parallel Matrix Multiplication

Go implementation of block-based parallel matrix multiplication following the same logic as the Python version.

## Usage

### Help
```bash
go run main.go --help
go run main.go demo --help
```

### Run Demo
```bash
# Using long flags
go run main.go demo --size 128 --block-size 32 --workers 4

# Using short flags
go run main.go demo -n 128 -b 32 -w 4

# Using defaults (workers = CPU count)
go run main.go demo
```

Options:
- `-n, --size`: Size of square matrices (default: 128)
- `-b, --block-size`: Block size for partitioning (default: 32)
- `-w, --workers`: Number of workers (default: CPU count)

### Run Tests
```bash
go run main.go test
```

## Implementation

Uses goroutines and channels for parallel block-based matrix multiplication:
- Divides result matrix into a grid of blocks
- Each worker computes blocks from a task queue
- Workers communicate results via channels
- Main goroutine assembles the final result

CLI built with kingpin for nice help text and both short/long flag options.
