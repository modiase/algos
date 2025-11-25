<script lang="ts">
    import type { Constraint, Operator } from "./types";

    interface Props {
        constraint: Constraint;
        numVars: number;
        onUpdate: (constraint: Constraint) => void;
        onRemove: () => void;
    }

    let { constraint, numVars, onUpdate, onRemove }: Props = $props();

    function updateCoefficient(index: number, value: string) {
        const newCoefficients = [...constraint.coefficients];
        newCoefficients[index] = parseFloat(value) || 0;
        onUpdate({ ...constraint, coefficients: newCoefficients });
    }

    function updateOperator(value: string) {
        onUpdate({ ...constraint, operator: value as Operator });
    }

    function updateRhs(value: string) {
        onUpdate({ ...constraint, rhs: parseFloat(value) || 0 });
    }
</script>

<div
    class="flex items-center gap-2 p-4 bg-base-200 rounded-lg border border-base-300 hover:shadow-md transition-shadow"
>
    <div class="flex items-center gap-2 flex-wrap flex-1">
        {#each Array(numVars) as _, i}
            <div class="relative">
                <input
                    type="number"
                    value={constraint.coefficients[i] || 0}
                    oninput={(e) => updateCoefficient(i, e.currentTarget.value)}
                    class="input input-bordered input-sm w-14 pr-7"
                    step="any"
                />
                <span
                    class="absolute right-1.5 top-1/2 -translate-y-1/2 text-xs text-base-content/60 pointer-events-none"
                >
                    x<sub>{i + 1}</sub>
                </span>
            </div>
            {#if i < numVars - 1}
                <span class="text-sm font-medium">+</span>
            {/if}
        {/each}

        <div
            class="flex items-center gap-1 bg-base-100 border border-base-300 rounded-lg px-2 py-1"
        >
            <select
                value={constraint.operator}
                onchange={(e) => updateOperator(e.currentTarget.value)}
                class="select select-sm border-0 focus:outline-none w-12 p-0"
            >
                <option value="<=">≤</option>
                <option value=">=">≥</option>
                <option value="=">=</option>
            </select>
            <input
                type="number"
                value={constraint.rhs}
                oninput={(e) => updateRhs(e.currentTarget.value)}
                class="input input-sm border-0 focus:outline-none w-16 p-1"
                step="any"
            />
        </div>
    </div>

    <button
        onclick={onRemove}
        class="btn btn-error btn-sm btn-circle"
        title="Remove constraint"
    >
        <svg
            xmlns="http://www.w3.org/2000/svg"
            class="h-4 w-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
        >
            <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M6 18L18 6M6 6l12 12"
            />
        </svg>
    </button>
</div>
