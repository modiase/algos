#! /usr/bin/env nix-shell
#! nix-shell -i bash -p meson ninja gnuplot gcc chafa

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
pushd "$SCRIPT_DIR" > /dev/null

trap 'popd > /dev/null' EXIT

if [ ! -d "build" ]; then
    meson setup build
fi

meson compile -C build

echo "Running benchmark..."
./build/bench > results.csv
echo "Benchmark complete."

gnuplot plot.gp
echo "Plot generated: memory_hierarchy.png"
