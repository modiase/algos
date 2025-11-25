<script lang="ts">
    import "katex/dist/katex.min.css";
    import ConstraintEditor from "./lib/ConstraintEditor.svelte";
    import MathDisplay from "./lib/MathDisplay.svelte";
    import type {
        LPProblem,
        Constraint,
        FormType,
        DisplayMode,
        Objective,
    } from "./lib/types";
    import { toStandardForm, toSlackForm } from "./lib/conversions";
    import { formatCompact, formatFull } from "./lib/latex";

    const SESSION_KEY = "lp-visualizer-state";

    interface SavedState {
        numVars: number;
        objective: Objective;
        objectiveCoefficients: number[];
        constraints: Constraint[];
        formType: FormType;
        displayMode: DisplayMode;
    }

    function loadState(): SavedState | null {
        try {
            const saved = sessionStorage.getItem(SESSION_KEY);
            return saved ? JSON.parse(saved) : null;
        } catch {
            return null;
        }
    }

    function saveState(state: SavedState) {
        try {
            sessionStorage.setItem(SESSION_KEY, JSON.stringify(state));
        } catch {}
    }

    const savedState = loadState();

    let numVars = $state(savedState?.numVars ?? 2);
    let objective: Objective = $state(savedState?.objective ?? "maximize");
    let objectiveCoefficients = $state(
        savedState?.objectiveCoefficients ?? [1, 1]
    );
    let constraints: Constraint[] = $state(
        savedState?.constraints ?? [
            { id: "1", coefficients: [1, 1], operator: "<=", rhs: 10 },
            { id: "2", coefficients: [2, 1], operator: "<=", rhs: 15 },
        ]
    );
    let formType: FormType = $state(savedState?.formType ?? "given");
    let displayMode: DisplayMode = $state(savedState?.displayMode ?? "compact");

    $effect(() => {
        saveState({
            numVars,
            objective,
            objectiveCoefficients,
            constraints,
            formType,
            displayMode,
        });
    });

    function updateNumVars(value: string) {
        const newNum = parseInt(value) || 2;
        numVars = newNum;
        objectiveCoefficients = Array(newNum)
            .fill(0)
            .map((_, i) => objectiveCoefficients[i] || 1);
        constraints = constraints.map((c) => ({
            ...c,
            coefficients: Array(newNum)
                .fill(0)
                .map((_, i) => c.coefficients[i] || 0),
        }));
    }

    function updateObjectiveCoef(index: number, value: string) {
        objectiveCoefficients[index] = parseFloat(value) || 0;
    }

    function inferUnrestrictedVariables(
        constraints: Constraint[],
        numVars: number
    ): boolean[] {
        const restricted = new Array(numVars).fill(false);

        for (const constraint of constraints) {
            const nonZeroCoeffs = constraint.coefficients
                .map((c, i) => ({ coeff: c, index: i }))
                .filter(({ coeff }) => coeff !== 0);

            if (nonZeroCoeffs.length === 1) {
                const { coeff, index } = nonZeroCoeffs[0];

                if (
                    constraint.operator === ">=" &&
                    constraint.rhs >= 0 &&
                    coeff > 0
                ) {
                    restricted[index] = true;
                }
            }
        }

        return restricted.map((r) => !r);
    }

    function addConstraint() {
        constraints = [
            ...constraints,
            {
                id: Date.now().toString(),
                coefficients: Array(numVars).fill(0),
                operator: "<=",
                rhs: 0,
            },
        ];
    }

    function removeConstraint(id: string) {
        constraints = constraints.filter((c) => c.id !== id);
    }

    function updateConstraint(id: string, updated: Constraint) {
        constraints = constraints.map((c) => (c.id === id ? updated : c));
    }

    const currentProblem = $derived.by(
        (): LPProblem => ({
            objective,
            objectiveCoefficients,
            constraints,
            variableNames: Array(numVars)
                .fill(0)
                .map((_, i) => `x_{${i + 1}}`),
            unrestrictedVariables: inferUnrestrictedVariables(
                constraints,
                numVars
            ),
        })
    );

    const standardProblem = $derived(toStandardForm(currentProblem));
    const slackProblem = $derived(toSlackForm(standardProblem));

    const displayProblem = $derived(
        formType === "given"
            ? currentProblem
            : formType === "standard"
              ? standardProblem
              : slackProblem
    );

    const latex = $derived(
        displayMode === "compact"
            ? formatCompact(displayProblem)
            : formatFull(displayProblem)
    );
</script>

<div
    class="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50"
    data-theme="lpvisualizer"
>
    <div class="container mx-auto px-4 py-8 max-w-7xl">
        <div class="text-center mb-12 pt-8">
            <h1
                class="text-3xl font-bold bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 bg-clip-text text-transparent p-3"
            >
                Linear Programming Equation Forms
            </h1>
        </div>

        <div class="grid grid-cols-1 xl:grid-cols-2 gap-8">
            <div class="space-y-6">
                <div class="card bg-base-100 shadow-xl">
                    <div class="card-body">
                        <h2 class="card-title text-xl mb-4">
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
                                    d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"
                                />
                            </svg>
                            Problem Definition
                        </h2>

                        <div class="grid grid-cols-2 gap-4">
                            <div class="form-control">
                                <label for="num-vars" class="label">
                                    <span class="label-text font-semibold"
                                        >Variables</span
                                    >
                                </label>
                                <input
                                    id="num-vars"
                                    type="number"
                                    value={numVars}
                                    oninput={(e) =>
                                        updateNumVars(e.currentTarget.value)}
                                    min="1"
                                    max="10"
                                    class="input input-bordered input-primary"
                                />
                            </div>

                            <div class="form-control">
                                <label for="objective-select" class="label">
                                    <span class="label-text font-semibold"
                                        >Objective</span
                                    >
                                </label>
                                <select
                                    id="objective-select"
                                    value={objective}
                                    onchange={(e) =>
                                        (objective = e.currentTarget
                                            .value as Objective)}
                                    class="select select-bordered select-primary"
                                >
                                    <option value="maximize">Maximize</option>
                                    <option value="minimize">Minimize</option>
                                </select>
                            </div>
                        </div>

                        <div class="form-control mt-4">
                            <div class="label">
                                <span class="label-text font-semibold"
                                    >Objective Coefficients</span
                                >
                            </div>
                            <div class="flex items-center gap-2 flex-wrap">
                                {#each objectiveCoefficients as coef, i}
                                    <div class="relative">
                                        <input
                                            type="number"
                                            value={coef}
                                            oninput={(e) =>
                                                updateObjectiveCoef(
                                                    i,
                                                    e.currentTarget.value
                                                )}
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
                                        <span class="text-sm font-medium"
                                            >+</span
                                        >
                                    {/if}
                                {/each}
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card bg-base-100 shadow-xl">
                    <div class="card-body">
                        <div class="flex justify-between items-center mb-4">
                            <h2 class="card-title text-xl">
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
                                        d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
                                    />
                                </svg>
                                Constraints
                            </h2>
                            <button
                                onclick={addConstraint}
                                class="btn btn-success btn-sm gap-2"
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
                                        d="M12 4v16m8-8H4"
                                    />
                                </svg>
                                Add
                            </button>
                        </div>

                        <div class="space-y-3 max-h-96 overflow-y-auto pr-2">
                            {#each constraints as constraint (constraint.id)}
                                <ConstraintEditor
                                    {constraint}
                                    {numVars}
                                    onUpdate={(c) =>
                                        updateConstraint(constraint.id, c)}
                                    onRemove={() =>
                                        removeConstraint(constraint.id)}
                                />
                            {/each}
                        </div>
                    </div>
                </div>
            </div>

            <div class="space-y-6">
                <div class="card bg-base-100 shadow-xl">
                    <div class="card-body">
                        <h2 class="card-title text-xl mb-4">
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
                                    d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
                                />
                                <path
                                    stroke-linecap="round"
                                    stroke-linejoin="round"
                                    stroke-width="2"
                                    d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                                />
                            </svg>
                            Display Options
                        </h2>

                        <div class="space-y-6">
                            <div class="form-control">
                                <div class="label">
                                    <span class="label-text font-semibold"
                                        >Form Type</span
                                    >
                                </div>
                                <div class="flex gap-2">
                                    <button
                                        onclick={() => (formType = "given")}
                                        class="btn btn-sm flex-1 {formType ===
                                        'given'
                                            ? 'btn-primary shadow-md'
                                            : 'btn-outline btn-primary'}"
                                    >
                                        Given
                                    </button>
                                    <button
                                        onclick={() => (formType = "standard")}
                                        class="btn btn-sm flex-1 {formType ===
                                        'standard'
                                            ? 'btn-primary shadow-md'
                                            : 'btn-outline btn-primary'}"
                                    >
                                        Standard
                                    </button>
                                    <button
                                        onclick={() => (formType = "slack")}
                                        class="btn btn-sm flex-1 {formType ===
                                        'slack'
                                            ? 'btn-primary shadow-md'
                                            : 'btn-outline btn-primary'}"
                                    >
                                        Slack
                                    </button>
                                </div>
                            </div>

                            <div class="form-control">
                                <div class="label">
                                    <span class="label-text font-semibold"
                                        >Display Mode</span
                                    >
                                </div>
                                <div class="flex gap-2">
                                    <button
                                        onclick={() =>
                                            (displayMode = "compact")}
                                        class="btn btn-sm flex-1 {displayMode ===
                                        'compact'
                                            ? 'btn-secondary shadow-md'
                                            : 'btn-outline btn-secondary'}"
                                    >
                                        Full
                                    </button>
                                    <button
                                        onclick={() => (displayMode = "full")}
                                        class="btn btn-sm flex-1 {displayMode ===
                                        'full'
                                            ? 'btn-secondary shadow-md'
                                            : 'btn-outline btn-secondary'}"
                                    >
                                        Matrix
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card bg-base-100 shadow-xl">
                    <div class="card-body">
                        <div class="flex justify-between items-start mb-2">
                            <h2 class="card-title text-xl">
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
                                        d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z"
                                    />
                                </svg>
                                {formType === "given"
                                    ? "Given Form"
                                    : formType === "standard"
                                      ? "Standard Form"
                                      : "Slack Form"}
                            </h2>
                            <div
                                class="text-xs bg-base-200 px-3 py-1 rounded-full"
                            >
                                <span class="font-semibold"
                                    >{displayProblem.variableNames.length}</span
                                >
                                vars ·
                                <span class="font-semibold"
                                    >{displayProblem.constraints.length}</span
                                > constraints
                            </div>
                        </div>
                        <div class="divider my-0"></div>
                        <MathDisplay {latex} />
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
