# Algorithms repo task runner

set shell := ["bash", "-euo", "pipefail", "-c"]

# List available recipes
default:
    @just --list

# Run all pre-commit checks on staged files
pre-commit:
    #!/usr/bin/env bash
    set -uo pipefail
    staged=$(git diff --cached --name-only --diff-filter d)
    [[ -z "$staged" ]] && echo "No staged files." && exit 0

    has_py=false; has_nix=false; has_c=false; has_go=false
    while IFS= read -r f; do
        case "$f" in
            *.py)                                          has_py=true  ;;
            *.nix)                                         has_nix=true ;;
            *.c|*.cpp|*.cc|*.cxx|*.h|*.hpp|*.hh|*.hxx)    has_c=true   ;;
            *.go)                                          has_go=true  ;;
        esac
    done <<< "$staged"

    pids=()
    labels=()

    run_recipe() {
        just "$1" &
        pids+=($!)
        labels+=("$1")
    }

    if $has_py;  then run_recipe pre-commit-python;    fi
    if $has_py;  then run_recipe pre-commit-typecheck; fi
    if $has_nix; then run_recipe pre-commit-nix;       fi
    if $has_c;   then run_recipe pre-commit-c;         fi
    if $has_go;  then run_recipe pre-commit-go;        fi
    run_recipe pre-commit-whitespace

    failed=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "  ✓ ${labels[$i]}"
        else
            echo "  ✗ ${labels[$i]}"
            ((failed++)) || true
        fi
    done

    if [[ $failed -gt 0 ]]; then
        echo ""
        echo "$failed recipe(s) failed — re-stage and retry."
        exit 1
    fi

    git diff --cached --name-only --diff-filter d | xargs -r git add

# Lint and format staged Python files
pre-commit-python:
    #!/usr/bin/env bash
    set -euo pipefail
    files=$(git diff --cached --name-only --diff-filter d -- '*.py')
    [[ -z "$files" ]] && exit 0
    ruff check --fix $files
    ruff format $files

# Typecheck staged Python files via pyright
pre-commit-typecheck:
    #!/usr/bin/env bash
    set -euo pipefail
    files=$(git diff --cached --name-only --diff-filter d -- '*.py')
    [[ -z "$files" ]] && exit 0
    ./bin/typecheck.sh $files

# Format staged Nix files
pre-commit-nix:
    #!/usr/bin/env bash
    set -euo pipefail
    files=$(git diff --cached --name-only --diff-filter d -- '*.nix')
    [[ -z "$files" ]] && exit 0
    nixfmt $files

# Format staged C/C++ files
pre-commit-c:
    #!/usr/bin/env bash
    set -euo pipefail
    files=$(git diff --cached --name-only --diff-filter d -- '*.c' '*.cpp' '*.cc' '*.cxx' '*.h' '*.hpp' '*.hh' '*.hxx')
    [[ -z "$files" ]] && exit 0
    astyle --style=linux --suffix=none $files

# Format staged Go files
pre-commit-go:
    #!/usr/bin/env bash
    set -euo pipefail
    files=$(git diff --cached --name-only --diff-filter d -- '*.go')
    [[ -z "$files" ]] && exit 0
    gofmt -s -w $files

# Strip trailing whitespace from staged files
pre-commit-whitespace:
    #!/usr/bin/env bash
    set -euo pipefail
    files=$(git diff --cached --name-only --diff-filter d | grep -v -E '\.(png|jpg|jpeg|gif|ico|bmp|pdf|woff2?|ttf|otf|eot|pyc)$' || true)
    [[ -z "$files" ]] && exit 0
    echo "$files" | xargs sed -i'' -e 's/[[:space:]]*$//'

# Check WIP markers require WIP: commit message prefix (commit-msg hook)
pre-commit-wip *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    marker="@""WIP"
    commit_msg_file="${@: -1}"
    if git diff --cached 2>/dev/null | grep '^+' | grep -q "$marker" 2>/dev/null; then
        commit_msg=$(head -n1 "$commit_msg_file")
        if ! echo "$commit_msg" | grep -q "^WIP:"; then
            echo "ERROR: Found $marker in staged changes."
            echo "Either remove $marker markers or prefix your commit message with WIP:"
            exit 1
        fi
    fi

# Clean up merged branches (post-checkout hook)
post-checkout:
    #!/usr/bin/env bash
    set -euo pipefail
    default_branch=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || echo "main")
    git branch --merged "$default_branch" \
        | grep -v -E "^\*|^\s*(main|master|develop)$" \
        | xargs -r git branch -d
