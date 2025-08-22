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

	"./leftist"
	"./skew"
)

type CommandType int

type CommandInsertOperands[T any] struct{ Value T }
type CommandExtractMinOperands struct{}

type CommandInsert[T any] = CommandInsertOperands[T]
type CommandExtractMin = CommandExtractMinOperands

type HeapInterface[T any] interface {
	Insert(key, value T)
	ExtractMin() T
	Size() int
}

type HeapInterrogator struct {
	Commands []interface{}
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
	ID                  int
	SkewHeap            *skew.Heap[int]
	LeftistHeap         *leftist.Heap[int]
	SkewInterrogator    *HeapInterrogator
	LeftistInterrogator *HeapInterrogator
	StatsChannel        chan<- StatisticsReport
	RNG                 *rand.Rand
}

func (h *HeapInterrogator) SnapshotSkew(skewHeap *skew.Heap[int]) HeapStatistics {
	size := skewHeap.Size()

	minDepth := 0
	maxDepth := 0

	if size > 0 {
		traversal := skewHeap.Traverse()
		var leafDepths []int

		for _, nodeDepth := range traversal {
			if nodeDepth.Node.Left == nil && nodeDepth.Node.Right == nil {
				leafDepths = append(leafDepths, nodeDepth.Depth)
			}
		}

		if len(leafDepths) > 0 {
			minDepth = leafDepths[0]
			maxDepth = leafDepths[0]

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

func (h *HeapInterrogator) SnapshotLeftist(leftistHeap *leftist.Heap[int]) HeapStatistics {
	size := leftistHeap.Size()

	minDepth := 0
	maxDepth := 0

	if size > 0 {
		traversal := leftistHeap.Traverse()
		var leafDepths []int

		for _, nodeDepth := range traversal {
			if nodeDepth.Node.Left == nil && nodeDepth.Node.Right == nil {
				leafDepths = append(leafDepths, nodeDepth.Depth)
			}
		}

		if len(leafDepths) > 0 {
			minDepth = leafDepths[0]
			maxDepth = leafDepths[0]

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

func (w *InterrogatorWorker) Run(N int, gamma int, s int, wg *sync.WaitGroup) {
	defer wg.Done()

	for i := 0; i < N; i++ {
		skewSize := w.SkewHeap.Size()
		leftistSize := w.LeftistHeap.Size()

		skewProb := float64(skewSize) / float64(skewSize+gamma)
		leftistProb := float64(leftistSize) / float64(leftistSize+gamma)

		if w.RNG.Float64() < skewProb && skewSize > 0 {
			w.SkewInterrogator.Commands = append(w.SkewInterrogator.Commands, CommandExtractMinOperands{})
			w.SkewHeap.ExtractMin()
		} else {
			value := w.RNG.Intn(1000)
			w.SkewInterrogator.Commands = append(w.SkewInterrogator.Commands, CommandInsertOperands[int]{Value: value})
			w.SkewHeap.Insert(value, value)
		}

		if w.RNG.Float64() < leftistProb && leftistSize > 0 {
			w.LeftistInterrogator.Commands = append(w.LeftistInterrogator.Commands, CommandExtractMinOperands{})
			w.LeftistHeap.ExtractMin()
		} else {
			value := w.RNG.Intn(1000)
			w.LeftistInterrogator.Commands = append(w.LeftistInterrogator.Commands, CommandInsertOperands[int]{Value: value})
			w.LeftistHeap.Insert(value, value)
		}

		if (i+1)%s == 0 {
			skewStats := w.SkewInterrogator.SnapshotSkew(w.SkewHeap)
			leftistStats := w.LeftistInterrogator.SnapshotLeftist(w.LeftistHeap)

			w.StatsChannel <- StatisticsReport{
				Timestamp:  time.Now(),
				Step:       i + 1,
				WorkerID:   w.ID,
				HeapType:   "skew",
				Statistics: skewStats,
			}

			w.StatsChannel <- StatisticsReport{
				Timestamp:  time.Now(),
				Step:       i + 1,
				WorkerID:   w.ID,
				HeapType:   "leftist",
				Statistics: leftistStats,
			}
		}
	}

	skewFinalStats := w.SkewInterrogator.SnapshotSkew(w.SkewHeap)
	leftistFinalStats := w.LeftistInterrogator.SnapshotLeftist(w.LeftistHeap)

	w.StatsChannel <- StatisticsReport{
		Timestamp:  time.Now(),
		Step:       N,
		WorkerID:   w.ID,
		HeapType:   "skew",
		Statistics: skewFinalStats,
	}

	w.StatsChannel <- StatisticsReport{
		Timestamp:  time.Now(),
		Step:       N,
		WorkerID:   w.ID,
		HeapType:   "leftist",
		Statistics: leftistFinalStats,
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

	for workerID := 0; workerID < numWorkers; workerID++ {
		wg.Add(1)

		workerSeed := *seed + int64(workerID)
		rng := rand.New(rand.NewSource(workerSeed))

		worker := &InterrogatorWorker{
			ID:                  workerID,
			SkewHeap:            skew.NewHeap(less),
			LeftistHeap:         leftist.NewHeap(less),
			SkewInterrogator:    &HeapInterrogator{Commands: make([]interface{}, 0, *N)},
			LeftistInterrogator: &HeapInterrogator{Commands: make([]interface{}, 0, *N)},
			StatsChannel:        statsChannel,
			RNG:                 rng,
		}

		go worker.Run(*N, *gamma, *s, &wg)
	}

	wg.Wait()
	close(statsChannel)
	<-writerDone

	fmt.Println("All workers completed. Statistics written to", *filename)
}
