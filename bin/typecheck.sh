#!/bin/bash
set -e

if [ $# -eq 0 ]; then
    echo "No Python files to typecheck"
    exit 0
fi

# Filter out files with problematic characters for nix-shell
filtered_files=()
for file in "$@"; do
    if [[ "$file" =~ [\(\)] ]]; then
        echo "Skipping file with special characters: $file"
        continue
    fi
    filtered_files+=("$file")
done

if [ ${#filtered_files[@]} -eq 0 ]; then
    echo "No valid Python files to typecheck after filtering"
    exit 0
fi

nix-shell -p mypy --run "mypy ${filtered_files[*]}"
