set terminal pngcairo size 1024,768 enhanced font 'Verdana,10'
set output 'memory_hierarchy.png'

set title "Memory Hierarchy Latency: Sequential vs Random Access"
set xlabel "Buffer Size (KB)"
set ylabel "Latency (ns)"

set logscale x
set grid xtics mxtics ytics mytics

set datafile separator ","

set key left top

plot 'results.csv' using 1:2 with linespoints linewidth 2 title 'Sequential (Prefetching)', \
     'results.csv' using 1:3 with linespoints linewidth 2 title 'Random (Pointer Chasing)'

