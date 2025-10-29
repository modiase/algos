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
    # Skip files in directories whose names start with a digit (mypy package-name issue)
    IFS='/' read -ra parts <<< "$file"
    skip_numeric_dir=false
    for part in "${parts[@]}"; do
        if [[ "$part" =~ ^[0-9] ]]; then
            skip_numeric_dir=true
            break
        fi
    done
    if [ "$skip_numeric_dir" = true ]; then
        echo "Skipping file in numeric-named directory: $file"
        continue
    fi
    filtered_files+=("$file")
done

if [ ${#filtered_files[@]} -eq 0 ]; then
    echo "No valid Python files to typecheck after filtering"
    exit 0
fi

nix-shell -p mypy --run "mypy --ignore-missing-imports --namespace-packages ${filtered_files[*]}"
