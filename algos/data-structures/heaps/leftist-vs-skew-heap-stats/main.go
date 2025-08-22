// @WIP Review this code
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"time"

	"leftist"
	"skew"
)

type Heap[T any] interface {
	Insert(key, value T)
	ExtractMin() T
	TraverseGeneric() []any
}

type IntHeap = Heap[int]

type NodeDepth[T any] struct {
	Node  *Node[T]
	Depth int
}

type Node[T any] struct {
	Left  *Node[T]
	Right *Node[T]
	Value T
}

type HeapFactory func() IntHeap

type HeapImplementation struct {
	Name    string
	Factory HeapFactory
}

type HeapRegistry struct {
	Implementations map[string]*HeapImplementation
}

func NewHeapRegistry(less func(int, int) bool) *HeapRegistry {
	registry := &HeapRegistry{
		Implementations: make(map[string]*HeapImplementation),
	}

	registry.Implementations["skew"] = &HeapImplementation{
		Name: "skew",
		Factory: func() IntHeap {
			return skew.NewHeap(less)
		},
	}

	registry.Implementations["leftist"] = &HeapImplementation{
		Name: "leftist",
		Factory: func() IntHeap {
			return leftist.NewHeap(less)
		},
	}

	return registry
}

type Command interface {
	Type() string
}

type InsertCommand struct {
	Value int
}

func (c InsertCommand) Type() string { return "insert" }

type ExtractMinCommand struct{}

func (c ExtractMinCommand) Type() string { return "extract_min" }

type HeapInterrogator struct {
	Commands []Command
}

func (h *HeapInterrogator) Snapshot(heap IntHeap) HeapStatistics {
	size := len(heap.TraverseGeneric())
	minDepth, maxDepth := 0, 0

	if size > 0 {
		var leafDepths []int

		for _, nodeDepth := range heap.TraverseGeneric() {
			switch nd := nodeDepth.(type) {
			case skew.NodeDepth[int]:
				if nd.Node.Left == nil && nd.Node.Right == nil {
					leafDepths = append(leafDepths, nd.Depth)
				}
			case leftist.NodeDepth[int]:
				if nd.Node.Left == nil && nd.Node.Right == nil {
					leafDepths = append(leafDepths, nd.Depth)
				}
			}
		}

		if len(leafDepths) > 0 {
			minDepth, maxDepth = leafDepths[0], leafDepths[0]
			for _, depth := range leafDepths {
				if depth < minDepth {
					minDepth = depth
				}
				if depth > maxDepth {
					maxDepth = depth
				}
			}
		}
	}

	return HeapStatistics{
		MinDepth: minDepth,
		MaxDepth: maxDepth,
		Size:     size,
		Skew:     0,
	}
}

type HeapStatistics struct {
	MinDepth int `json:"min_depth"`
	MaxDepth int `json:"max_depth"`
	Size     int `json:"size"`
	Skew     int `json:"skew"`
}

type StatisticsReport struct {
	Timestamp  time.Time      `json:"timestamp"`
	Step       int            `json:"step"`
	WorkerID   int            `json:"worker_id"`
	HeapType   string         `json:"heap_type"`
	Statistics HeapStatistics `json:"statistics"`
}

type InterrogatorWorker struct {
	ID            int
	Registry      *HeapRegistry
	Heaps         map[string]IntHeap
	Interrogators map[string]*HeapInterrogator
	StatsChannel  chan<- StatisticsReport
	RNG           *rand.Rand
}

func NewInterrogatorWorker(id int, registry *HeapRegistry, statsChannel chan<- StatisticsReport, rng *rand.Rand) *InterrogatorWorker {
	heaps := make(map[string]IntHeap)
	interrogators := make(map[string]*HeapInterrogator)

	for name, impl := range registry.Implementations {
		heaps[name] = impl.Factory()
		interrogators[name] = &HeapInterrogator{
			Commands: make([]Command, 0, 1000),
		}
	}

	return &InterrogatorWorker{
		ID:            id,
		Registry:      registry,
		Heaps:         heaps,
		Interrogators: interrogators,
		StatsChannel:  statsChannel,
		RNG:           rng,
	}
}

func (w *InterrogatorWorker) Run(N int, gamma int, s int, wg *sync.WaitGroup) {
	defer wg.Done()

	for i := range N {
		for name, heap := range w.Heaps {
			interrogator := w.Interrogators[name]

			size := len(heap.TraverseGeneric())
			extractProb := float64(size) / float64(size+gamma)

			if w.RNG.Float64() < extractProb && size > 0 {
				interrogator.Commands = append(interrogator.Commands, ExtractMinCommand{})
				heap.ExtractMin()
			} else {
				value := w.RNG.Intn(1000)
				interrogator.Commands = append(interrogator.Commands, InsertCommand{Value: value})
				heap.Insert(value, value)
			}
		}

		if (i+1)%s == 0 {
			w.takeSnapshots(i + 1)
		}
	}

	w.takeSnapshots(N)
}

func (w *InterrogatorWorker) takeSnapshots(step int) {
	for name, heap := range w.Heaps {
		interrogator := w.Interrogators[name]
		stats := interrogator.Snapshot(heap)

		w.StatsChannel <- StatisticsReport{
			Timestamp:  time.Now(),
			Step:       step,
			WorkerID:   w.ID,
			HeapType:   name,
			Statistics: stats,
		}
	}
}

func statsWriter(statsChannel <-chan StatisticsReport, filename string, done chan<- bool) {
	file, err := os.OpenFile(filename, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		fmt.Printf("Error opening file: %v\n", err)
		done <- true
		return
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	for report := range statsChannel {
		if err := encoder.Encode(report); err != nil {
			fmt.Printf("Error writing to file: %v\n", err)
		}
	}

	done <- true
}

func main() {
	var (
		N        = flag.Int("n", 1000, "Number of operations to perform")
		seed     = flag.Int64("seed", 42, "Random seed")
		gamma    = flag.Int("gamma", 1000, "Gamma parameter for extract probability")
		s        = flag.Int("s", 10, "Snapshot interval")
		filename = flag.String("output", "heap_statistics.jsonl", "Output file for statistics")
	)
	flag.Parse()

	numWorkers := runtime.NumCPU()
	fmt.Printf("Starting %d worker goroutines\n", numWorkers)
	fmt.Printf("Running heap statistics with N = %d per worker and snapshot interval s = %d\n", *N, *s)
	fmt.Printf("Using seed = %d and gamma = %d\n", *seed, *gamma)
	fmt.Printf("Output file: %s\n", *filename)
	fmt.Println()

	statsChannel := make(chan StatisticsReport, 1000)
	writerDone := make(chan bool)

	go statsWriter(statsChannel, *filename, writerDone)

	var wg sync.WaitGroup
	less := func(a, b int) bool { return a < b }

	registry := NewHeapRegistry(less)

	for workerID := 0; workerID < numWorkers; workerID++ {
		wg.Add(1)

		workerSeed := *seed + int64(workerID)
		rng := rand.New(rand.NewSource(workerSeed))

		worker := NewInterrogatorWorker(workerID, registry, statsChannel, rng)
		go worker.Run(*N, *gamma, *s, &wg)
	}

	wg.Wait()
	close(statsChannel)
	<-writerDone

	fmt.Println("All workers completed. Statistics written to", *filename)
}
